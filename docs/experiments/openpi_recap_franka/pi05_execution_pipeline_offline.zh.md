# π₀.₅ 执行流水线：离线验证

本报告说明 2026-11-18 实现的有界命令刷新机制、完整 mapping 门槛、离线时序证据，以及下一次单独获得操作员批准的实机测试所需的检查。先阅读时序结果，再查看请求单位、缩放比较与验证范围。[English](pi05_execution_pipeline_offline.md)

后续的 [Episode 8 停止处理报告](pi05_episode8_shutdown.zh.md)说明了终止事件竞态修正，以及完成资格验证后使用 1.0 SpaceMouse mapping 的准备步骤。

## 时序与命令所有权

已确认的间隔来自 `fr3_pi05_supervised_rollout.py` 的串行流程：等待 `chunk_done`，采集相机与状态，转换观测，运行推理，反归一化，验证，保存观测，然后提交下一组动作。原 ROS executor 每隔 100 ms 发布一个动作，共十个；在一秒边界发布一次 neutral，随后等待。stdin 的 `readline()` 还可能因不完整消息而阻塞。后者是已确认的阻塞路径，但没有证据表明它造成了已记录的间隔。

executor 进程现在独立按 10 ms 周期发布。每组仍然只有十个 policy 动作。动作 `i` 仅在该组接收后对应的 100 ms 时段内有效；重复发布不会推进 policy、生成新预测，或增加 gripper 的确认计数。一秒 horizon 结束后，executor 在等待新推理期间持续刷新 **neutral**，不会用最后一个运动请求填补空档。

输入读取限制为 64 KiB，并且不会等待完整行。stop、quit 和父进程 EOF 均在运动发布前处理。调度或 arm 发布间隔超过 50 ms 时，系统会先报错退出，再拒绝恢复运动；不会突发重放错过的动作。发布前的最后一次检查还会拒绝在消息构造期间过期的动作。arm publisher 只保留一个排队请求。这些是应用端约束，并非硬实时或 DDS 端到端交付保证。

父进程独立于推理监测操作员输入、EOF 和 episode 截止时间。stop 会撤销后续动作提交权限，并要求 executor 取消当前动作组。executor 处理 stop 或 fault 后只保留 neutral 清理请求，停止发送 gripper goal 和运动请求。arm neutral 不会取消已经发出的 gripper goal。原有硬件 E-stop 要求仍然有效。

## 完整幅度门槛与延迟优化

`identity_verified` 现在将 `decoded_policy_action[:6]` 原样发送，保留六个轴的顺序和符号。默认 `legacy_limited` 仍将这些值乘以 0.15。两种模式都通过真实 executor 循环和假的 ROS endpoint 测试全部十二个带符号轴向用例。已有 100 Hz scheduler 保留。

完整幅度执行同时要求 `--verified-request-mapping` 和 `--mapping-verification <record.json>`。父进程先验证记录，再进行 model 或硬件设置；每个 episode 还要求输入 `RUN VERIFIED <record SHA-256 的前 12 个字符>`。子进程在 import/初始化 ROS 前独立验证记录与 digest，并在 `begin_episode` 时再次检查该 digest。仅设置 mode 环境变量无法启用完整幅度。本任务不创建物理批准记录；合成 attestation 仅存在于临时测试 fixture 中。

验证记录必须满足以下 contract：

| 字段 | 所需证据 |
| --- | --- |
| `schema`, `mapping_mode` | `fr3_pi05_mapping_verification_v1`, `identity_verified` |
| `arm_axes` | 按原顺序排列的六个 `ARM_AXES` 名称 |
| `policy_period_ns`, `publish_period_ns` | `100000000`, `10000000` |
| `operator`, `verified_at_utc` | 审核人姓名与实际已完成的 UTC 验证时间 |
| `controller_binary_sha256`, `controller_config_sha256` | 通过物理验证的安装文件的小写 SHA-256 fingerprint |
| `base_frame`, `tool_frame` | 经物理审查的 base frame 与已标定的 tool/TCP frame |
| `checks` | 以下 17 项检查各自具有 `passed: true`、`evidence_path` 和 `evidence_sha256` |

检查项包括每个轴名加 `_positive` 和 `_negative` 后缀，共十二项，再加上 `controller_acceptance`、`tool_frame_alignment`、`timing_10hz_100hz`、`neutral_stop` 和 `gripper_open_close`。证据文件必须存在、非空且 SHA-256 匹配。相对路径以记录所在目录为基准。操作员逐项审核后，一份同步 trace 可以支持多项检查。Hash 只能确认完整性，无法认证物理测量或检查正在运行的 controller。操作员在每次同意执行前必须确认当前运行的 binary/configuration 和 frame 与记录一致；安装发生变化时需重新验证。批准的记录路径和 digest 会保存在 manifest 中。

已测得的一个软件瓶颈是提交前压缩观测。在工作站 CPU 上，将 Episode 6 的第一份观测（480×640 与 480×848 RGB，加 state19）离线写入十次，耗时中位数为 **125.001 ms**，p95 为 **125.900 ms**，最大为 **126.417 ms**。格式化、写入并 flush 十组已保存的 command/state，耗时中位数为 **0.536 ms**，最大为 **0.682 ms**。所有 benchmark 写入均位于临时目录。这些测量仅隔离磁盘/CPU 工作，不代表实时采集或 GPU 推理耗时。

父进程现在先提交通过验证的动作，再压缩原样观测，使归档与一秒命令窗口重叠。归档失败会立即撤销当前与后续动作，并保留部分 executor event。失败发生前可能已经发送初始请求；该故障 attempt 不能算作可用 outcome。归档阻塞期间 stop 仍独立生效。`pipeline_timing.jsonl` 记录采集、推理、提交、归档完成和 wait 返回时间，供下一次批准的验证使用。较小的 command/state 日志写入仍保持同步。

采集与推理仍在上一组动作结束后进行。用更早的观测预测下一组动作会改变训练时的 observation/action 对齐，因此不引入此类 prefetch、采样步数修改或人为插值。其实时耗时尚未测量，剩余等待由 neutral 刷新覆盖。

## 修改前后的证据

历史比较读取全部 100 个 `/data1/Muttaqien/fr3_peg_100/episode_*/operator_commands.jsonl` 文件，以及 `/data1/Muttaqien/fr3_pi05_real_rollouts_20/episode_06/policy_command_log.jsonl` 中的 110 个 action event。相邻 receipt/event 的时间差分别在各 episode 内计算。修改后的结果来自真实 executor 循环、假的 ROS endpoint 和确定性时钟，其中包括两组动作之间 484 ms 的 neutral 等待。

| 指标 | 已保存的基线 | 修改后的离线结果 |
| --- | --- | --- |
| SpaceMouse receipt | 212,623 条；100.00 Hz；最大间隔 13.845 ms | 仅作参考 |
| 组内 policy 动作间隔 | 约 100 ms | 100 ms；每组十个动作 |
| 组间 action event 间隔 | 平均 468.403 ms；最大 484.156 ms | 推理等待期间保持 neutral；不补造动作 |
| 有效 policy action event 频率 | Episode 6 全程为 7.474 Hz | 仍取决于采集、推理和保存耗时 |
| arm 发布间隔 | policy 动作为 100 ms；每个边界一次 neutral | 确定性 executor 测试中为 10 ms，包含推理等待 |
| 超过 250 ms 的间隔 | 已保存的 **action** 日志中有十次 | 确定性发布轨迹中为零 |
| arm 倍率 | 0.15 | Identity 1.0 受物理证据与操作员门槛约束；保留 legacy 0.15 |

已保存的 policy action 日志没有记录 neutral 请求和 controller receipt，因此不能仅凭十次长 action 间隔认定 controller 发生了十次 freshness 超时。实时传输时序、命令接受情况与物理响应尚未测试。即使数值增益不变，更高的刷新频率也可能改变 controller 响应；继续收集 outcome 前必须完成验证。

## 训练单位与缩放

标签路径先经过 `fr3_command_request_alignment.py` 中的 `_mean_held_twist()`，再进入 `fr3_request_lerobot_export.py`。它对观测窗口内已经完成轴映射的归一化 Twist 请求计算时间加权平均。第七个通道是锁存的 gripper 按键请求。标签既不是 SI 速度，也不是实测机器人运动。rollout 解码仍先用仅从训练集得到的 OpenPI `Unnormalize`，再取七个请求通道。只读 deployment audit 通过了来源 episode 互斥的 80/20 划分和仅用训练集归一化的检查。

可重复运行的只读 audit 比较了全部 100 份 demonstration 中的 212,623 条 Twist 和 200 条按键请求，以及 Episode 6 的 110 个已保存动作。Legacy replay 与 identity replay 相对各自 contract 的最大绝对 mapping 误差均为零。已保存的 gripper 日志与当前离线 debounce replay 均包含 109 次 hold 和一次 open goal。

下表比较全部 SpaceMouse 记录、Episode 6 预测和实际记录的发送值的平均绝对幅度。这些是来自不同轨迹、按样本等权统计的值，不能用来判断匹配状态下的预测精度或完成物理标定。

| Twist 轴 | SpaceMouse 记录 | 解码预测 | Legacy 发送 | Identity 离线发送 |
| --- | ---: | ---: | ---: | ---: |
| linear X | 0.1899 | 0.1296 | 0.0194 | 0.1296 |
| linear Y | 0.2097 | 0.3596 | 0.0539 | 0.3596 |
| linear Z | 0.3023 | 0.1850 | 0.0278 | 0.1850 |
| angular X | 0.0835 | 0.1325 | 0.0199 | 0.1325 |
| angular Y | 0.1227 | 0.1190 | 0.0179 | 0.1190 |
| angular Z | 0.1151 | 0.1218 | 0.0183 | 0.1218 |

Episode 6 中每条 arm 发送值都等于解码预测乘以 0.15，最大绝对残差为零。这确认了训练后的衰减，但不能证明移除衰减是安全或充分的。默认模式仍为 `legacy_limited`；加强后的 `identity_verified` 门槛见上文。不会自动改变物理增益或 mapping。Request data config 不包含 `DeltaActions`/`AbsoluteActions`，output adapter 仅截取七个请求通道。六个 arm 维度在校验后保留解码值，不重复应用 HID 变换，也不进行坐标旋转或 SI 单位换算。后续的[夹爪校验修正](pi05_gripper_validation.zh.md)仅限制第七维，并记录原始预测。

后续 mapping 审查应先固定部署二进制与源码对应关系、安装及工具坐标系、轴方向和更新语义。随后单独取得操作员批准，进行有界、仅 arm 的带符号轴向标定，同时记录 controller receipt/reference 诊断与 TCP 实测响应。先在当前增益下验证 neutral、stop 和保持时序，再考虑另行批准的增益修改。一次 X 轴 pulse 或 topic 名称一致不足以验证六个轴。现有门槛见 [mapping review](pi05_real_rollout_mapping_review.md)。先前 X-only pulse 报告了 +0.383 mm X 和 -0.539 mm Z；尚未解释的跨轴响应与未标定的 flange-to-peg TCP，使离线检查无法确认物理方向兼容。代码不会据此猜测轴重映射。Gripper debounce 也与直接按键控制存在有意差异：需要三个置信度足够的 policy 动作，相邻 goal 至少间隔 0.8 s；100 Hz arm 刷新不会增加确认次数。

## 离线验证与范围

在仓库根目录执行以下 CPU 检查，每个测试文件启动独立 pytest 进程：

```bash
for component in fr3_pi05_ros_executor fr3_pi05_request_adapter \
  fr3_pi05_deployment_audit fr3_rollout_outcomes \
  fr3_command_request_alignment fr3_supervised_arm_pulse fr3_multimodal \
  fr3_peg_request_policy fr3_request_norm_stats fr3_request_lerobot_export \
  fr3_request_sft_offline_eval fr3_request_live_shadow fr3_request_chunk_eval; do
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
    "tests/unit_tests/test_${component}.py" || exit 1
done
```

这些检查在进程边界用假的 endpoint 测试实际 scheduler 与 ROS 循环，并通过真实子进程管道验证异步 stop 和故障。覆盖 300/484/1500 ms 推理等待、动作组重新开始、观测与请求 receipt 过期、不完整 IPC、父进程死亡、调用方不活动期间的操作员 stop、截止时间 stop、损坏 event、独占控制或 subscriber 丢失、publisher 故障、消息构造延迟、pose 过期、位移和时长限制。完整的针对性测试在工作站 CPU、Python 3.11.15 上通过，**13 个文件共 205 个测试通过**，每个文件独立运行 pytest 进程；除已有时序与安全回归外，还包含两种 mapping 模式、全部带符号轴向、证据与同意失败、归档重叠与故障，以及归档缓慢时的操作员 stop。未运行 policy 推理、ROS discovery、controller 激活或硬件 e2e。OpenPI input/output transform 与 normalization 测试仅操作 CPU 数组。已有 TorchScript deprecation warning 与本次修改无关。

观测年龄现在使用四个状态来源与两个相机中最早的 host receipt，并在推理与 IPC 中保留；验证不会用新时间戳替换它。接收动作组时，观测年龄上限仍为一秒，请求发出年龄上限仍为 300 ms。已接收动作组具有一秒的有界 horizon，期间持续检查实时 pose；这两种 freshness 约束含义不同。250 ms pose watchdog、8 cm 实测位移限制、20 s executor 时限、1–15 s rollout 时限、请求范围、gripper 阈值和独占检查数值均保持不变。controller timeout、增益、IK、torque、安全配置、policy 权重和既有采集数据均未修改。

`executor_events.jsonl` 保存部分动作和终止故障。`commands_sent` 与旧 manifest 字段 `robot_commands_sent` 仍统计 **policy 动作**；`policy_actions_sent` 显式注明这一含义。episode-end event 和 manifest 中的 `arm_publications` 与 `max_publish_gap_ns` 描述传输，包含 end acknowledgement 前的 neutral 刷新，不包含退出清理。两类计数均不能证明 controller 接受了命令。

修改的 Python 文件通过 Ruff lint/format、Python 编译和空白检查。本分支没有 `.github/workflows/ci-tests.yml`，提交上游前需恢复覆盖规则。docs symbol checker 在未修改的 Sphinx 页面中报告了 24 个既有问题。RST markup checker 因缺少 `docutils` 无法运行；环境没有 Sphinx，且这两份实验 Markdown 位于两个 Sphinx 文档树之外，因此跳过 EN/ZH Sphinx 构建。本次修改由 AI assistant 实现并检查。

不使用 ROS 或推理，即可重复完整的已记录请求比较：

```bash
.venv/bin/python -m toolkits.realworld_check.fr3_pi05_deployment_audit \
  --raw-root /data1/Muttaqien/fr3_peg_100 \
  --episode /data1/Muttaqien/fr3_pi05_real_rollouts_20/episode_06
```

该命令验证仅用训练集归一化和 80/20 划分，审查记录的范围、单位与时序，并 replay 两种软件 mapping 和 gripper debounce。Train/validation export preflight 也通过了只读检查，分别包含 12,288/3,078 个标签。[汇总证据](pi05_execution_compatibility_evidence.json) 保存来源 hash 和离线归档 profile，不是物理批准记录。

下一步是进行 **一次单独获得操作员批准的实机验证**，记录真实发布与接收间隔、neutral 切换、controller 接受情况和实测响应。在物理 mapping 证据满足 identity 门槛前保持 legacy 模式；之后的完整幅度执行仍需每次同意。审查该结果后，才能继续收集 20 个 RECAP outcome。本任务止于离线验证，新流水线尚未通过实机验证。
