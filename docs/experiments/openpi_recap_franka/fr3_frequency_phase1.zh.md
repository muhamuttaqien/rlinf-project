# FR3 Peg Insertion：10 Hz 数据集准备

本报告说明频率一致的 OpenPI π₀.₅ 数据集及拟议 SFT 配置，供你审查后选择训练输入。现有 100 条演示支持离线转换为 10 Hz；训练和物理评估仍需单独启动。

## 原始记录与实际频率

源数据位于 `/data1/Muttaqien/fr3_peg_100`，包含 2026-11-09 采集的 `episode_0001` 至 `episode_0100`。审计遍历全部 state 和 command 记录，检查原始操作员 outcome review，并将既有训练标签与转换 provenance 对照。

| 记录流 | 样本数 | 间隔中位数 | 最大间隔 |
| --- | ---: | ---: | ---: |
| SpaceMouse normalized Twist request | 212,623 | 9.999813 ms | 13.844603 ms |
| 保存的 joint state | 21,234 | 100.030117 ms | 102.280906 ms |
| 保存的 TCP pose | 21,234 | 100.023062 ms | 102.933725 ms |
| 保存的 external wrench | 21,234 | 100.034575 ms | 103.327353 ms |
| 保存的物理 gripper state | 21,234 | 100.007965 ms | 103.347516 ms |
| 保存的 base RGB 源帧 | 21,234 | 101.179728 ms | 104.181191 ms |
| 保存的 wrist RGB 源帧 | 21,234 | 101.344827 ms | 104.165103 ms |
| 完整 observation 记录 | 21,234 | 100.126373 ms | 107.106894 ms |

各 episode 的 Twist 频率为 99.9921–100.0086 Hz，保存的 state 平均约为 9.988 Hz。所有流的时间戳均严格递增。200 个 gripper request 是离散按钮事件，每条演示两个；事件发生率不代表采样频率。

`fr3_data_collect.py` 默认以 10 Hz 组装 observation，相机配置为 30 Hz：base 序列号 `337122075473`，分辨率 640×480；wrist 序列号 `218622277827`，分辨率 848×480。`fr3_raw_capture.py` 保存选中的 BGR 图像和 state snapshot，随后记录完成时间。因此，原始文件没有保留中间相机帧或 100 Hz robot state。review 引用的 NUC 文件 `fr3_telemetry_isolated_4c2t7/recordings/stationary_telemetry.jsonl` 修改时间为 2026-11-07，内容使用 steady-clock 时间戳，且没有可用的逐 episode 时钟对应关系，无法据此恢复 11 月 9 日演示的同步高频 state。

同步质量均按工作站 monotonic 接收时间计算：

| 同步指标 | 中位数 | 99 分位数 | 最大值 |
| --- | ---: | ---: | ---: |
| base/wrist 接收时间差 | 9.386874 ms | 30.247218 ms | 33.973369 ms |
| 四路 state topic 接收时间跨度 | 0.916603 ms | 2.047988 ms | 3.302566 ms |
| 六路 modality 接收时间跨度 | 23.780458 ms | 34.037181 ms | 34.710843 ms |
| 最后一次接收至记录完成 | 10.035497 ms | 14.497150 ms | 17.241544 ms |

相机时间戳表示主机读取完成时间，不是经校准的曝光时间。ROS header 与主机 UNIX 时间之差同时包含时钟偏差和传输延迟，不能直接视为单向延迟。物理 controller acceptance 和 tool-tip calibration 也无法由这些记录证明。

## Action 语义与重采样

所选标签来自 `/franka_controller/target_cartesian_velocity_percent`，是已完成轴映射的六维 normalized held SpaceMouse request，顺序为 linear `x,y,z` 和 angular `x,y,z`。记录明确标注 `normalized_controller_input_not_m_per_s` 与 `normalized_controller_input_not_rad_per_s`。保留原始符号和 1.0 scale；controller 的映射、限制和 acceptance logic 位于下游。

每个精确 100 ms 区间输出 held request 的时间加权均值：

`action[k] = integral(request(t), t[k], t[k] + 0.1 s) / 0.1 s`。

该转换保留所保留区间的 normalized request-seconds。由于 controller 可能非线性，输入积分相同不能证明实际机器人轨迹相同。`fr3_observed_action_proxy.py` 的 measured-TCP delta 属于另一类标签，本次没有使用。若处理 pose delta，平移应累加、旋转应组合，不能直接平均逐帧 delta。

第七维是事件驱动的 normalized gripper request：`0` 为关闭，`1` 为打开。每个区间使用起点之前最后一次按钮值。state 第 17 个下标继续保存以弧度表示的物理 knuckle angle。首次 gripper request 之前的前缀会被裁剪，不根据关节测量猜测初始命令。

每条演示使用固定 100 ms 网格，网格起点须晚于对应 observation 的完整记录时间。要求各起点最近可用的 tuple 恰好对应原始 tuple，所有 modality 均已接收，最早接收值的 age 不超过 100 ms。跳帧、重复 tuple、时间倒序、request 覆盖缺失、gripper 事件合并或丢失均会使转换失败；尾部不足一个完整区间的记录会被裁剪。

旧数据集已声明为 10 Hz。其 15,366 个标签按 67.344–102.902 ms 的相机间隔求 request 均值，间隔中位数为 101.245 ms，随后却使用均匀 100 ms 的 dataset 时间戳。此重新定时导致单个 episode 的 request 积分最多变化 0.0433713 normalized request-seconds。全部 21,234 个原始 tuple 中，有 21,228 个包含晚于旧相机 anchor 的 state receipt，最多相差 33.761892 ms。审计字段 `old_labels_with_future_state` 统计的是原始 tuple，也包含旧导出范围以外的记录。本次转换修正区间和因果对齐问题；它不证明存在十倍训练频率错误，也不能单独证明 weak movement 已解决。

## 数据集与验证

独立输出目录为：

`/data1/Muttaqien/datasets/fr3_peg_request_10hz_phase1_v1`

| Split | 保留的原始 episode 身份数 | 帧数 | 完整十步 action horizon 起点数 |
| --- | ---: | ---: | ---: |
| `train` | 80 | 12,240 | 11,520 |
| `validation` | 20 | 3,065 | 2,885 |

episode 分配直接读取既有 `fr3_peg_request_lerobot_{train,validation}_v1` provenance。共保留 15,305 帧，比旧导出少 61 帧。每条演示为 124–183 帧，即 12.4–18.3 秒。裁剪来自未知 gripper 初值的前缀和不足完整区间的尾部。保留的 observation 连续，空间轨迹保持原样；位置、姿态、关节和图像均不做插值。

证据与数据保存在一起；精简副本见仓库中的 [fr3_frequency_phase1_evidence.json](fr3_frequency_phase1_evidence.json)：

| Artifact | 用途 |
| --- | --- |
| `audit.json` | 原始频率分布、逐 episode 边界、源文件 SHA-256、最终导出检查与完成标记 |
| `resampled/episode_*.jsonl` | 精确区间、原始帧身份、接收时间、完整记录时间、state 和 action |
| `{train,validation}/frequency_provenance.json` | 原始 episode 与导出 episode 的对应关系 |
| `{train,validation}/meta/info.json` | LeRobot 10 Hz metadata 与 feature contract |
| `norm/norm_stats.json` | 仅用 float32 training rows 重新计算并 padding 至 32D 的统计量 |
| `norm/provenance.json` | 参与统计的 80 条 training episode；validation 样本数为零 |

验证会从 Parquet 读回每个 state/action 和全部 RGB 像素，与源数据逐项比较。原始 BGR 只转换一次为无损 RGB，两路视图仍使用 `image` 和 `extra_view_image`。原始 frame index 和 monotonic 网格时间额外以 int64 feature 保存。验证也检查 episode 长度和末帧 `done`；末帧 `is_success` 来自原始操作员确认的成功 episode，不代表自动推断的插入完成时刻。

保留样本中最旧 modality 的 age 不超过 75.606503 ms。200 个 gripper 事件全部保留，量化延迟为 0–99.194223 ms。request 积分的 float64 最大误差小于 `1.03e-14`，float32 导出后小于 `3.99e-8`。保留 TCP 位置与源数据一致。转换前后校验原始记录 metadata 的哈希；源记录和旧数据集保持原样。

离线验证共通过 25 个 unit test，覆盖新 converter 及既有 alignment、normalization、export 和 policy contract。通过 `build_official_openpi_sft_dataloader` 执行的 model-free smoke check 在两个 split 上均通过：state 为 `(1,32)`，action 为 `(1,10,32)`，每路有效 RGB view 为 `(1,3,224,224)`，完整 horizon 数与上表一致。从导出的 training rows 独立重算的统计量与保存值一致。EN/ZH Sphinx build 和针对本次文件的 pre-commit 检查均通过。

## 复现离线转换

从仓库根目录执行审计，并指定新的可写目录：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python \
  -m toolkits.realworld_check.fr3_frequency_dataset \
  --output /tmp/fr3_frequency_audit_new --audit-only
```

该命令读取原始数据和既有两个 split，检查时间与标签语义，并输出审计证据和重采样 JSONL。去掉 `--audit-only` 并选择另一个新目录，即可导出 LeRobot 数据、读回所有像素和标签，并计算 training-only 统计量。工具拒绝已存在或与源数据重叠的输出目录，且不会初始化 robot 或 training worker。

## 拟议 SFT 配置

启动训练前审查 [`realworld_sft_openpi_fr3_peg_request_10hz_phase1.yaml`](../../../examples/sft/config/realworld_sft_openpi_fr3_peg_request_10hz_phase1.yaml)。该配置使用新的 train、validation 和 norm 路径，保留原有 π₀.₅ base model、padding 至 32D 的 19D measured state、两路 RGB、padding 至 32D 的七维 normalized request、十步 action horizon、seed、optimizer 和拟议的 3,000-step schedule，每 500 step 做一次 validation。十步 horizon 对应一秒；official OpenPI loader 只使用同一 episode 内的完整 horizon。

原始 task prompt 是 **`FR3 peg insertion`**，由 `prompt_from_task=True` 提供。该数据集继续使用此文本。既有 native evaluation 配置使用了不同 prompt，比较未来 checkpoint 前应另行统一；本次准备保持 native evaluation 配置不变。

频率 contract 为：有真实记录时使用 100 Hz raw request 和 raw state，training target 与 policy action 均为 10 Hz，执行端以 100 Hz 刷新 held policy request。刷新不意味着将 action 乘以十，也不引入 0.15 scale。现有记录在所测 host-receipt 边界内支持可靠的 10 Hz 训练准备；曝光时间校准、追溯 100 Hz state 或实际执行等效性仍需单独证据。Phase 1 未启动 SFT 或物理评估。
