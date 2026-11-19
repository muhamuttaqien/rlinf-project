# FR3 π₀.₅ 物理 Mapping 验证

本清单用于补齐 scaling 1.0 所需的物理证据，再由操作员批准一次监督 rollout。**当前状态：17 项物理检查通过 0 项，禁止执行。** 已有软件证据可复用，但 X 方向响应尚未明确，不能批准方向映射。本次工作止于准备，不部署、不激活控制器，也不执行标定或 policy。[English](pi05_physical_mapping_qualification.md)

## 可复用的证据

[Mapping review](pi05_real_rollout_mapping_review.md)、[控制器调查](fr3_controller_investigation.md)、[执行报告](pi05_execution_pipeline_offline.zh.md)、[夹爪报告](pi05_gripper_validation.zh.md) 和 [Episode 8 报告](pi05_episode8_shutdown.zh.md) 确立了以下边界。[证据清单](pi05_mapping_qualification_evidence.json) 保存了当前文件哈希和本次离线结果。

- 训练标签是已完成 HID 轴映射的归一化 Twist 请求在时间窗内的平均值，轴顺序为 `[linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]`。回放不再做 HID 转换或 SI 单位换算；现有 SpaceMouse 接口保持原样。
- Identity 路径逐值保留六个解码后的机械臂请求。Episode 6 和 Episode 8 使用 0.15 gain。Episode 3 的日志显示最大 TCP 位移为 5.479 mm，不能据此认定 mapping 或控制器接受检查通过。
- 已报告的 +X 脉冲到达了控制器，产生有效 IK 和参考更新。最近的最终位移 `[+0.001,+0.001,-0.003]` mm 和较早的 `[+0.383,-0.002,-0.539]` mm 均不足以确认可分辨、独立的 +X 响应。本证据包没有该物理实验的同步原始轨迹。
- 准备时旧诊断库的哈希为 `fc5128dd…3925744`。Release 库 `3b160839…7752b4` 通过了 9 项 CPU 控制器测试，但当时仅完成暂存，未投入运行。两者均为历史身份，验证前必须检查实际进程映射；不能把暂存库的哈希当作运行库的哈希。
- 保存的源码采用线性偏移 `0.007 * R_mountᵀ * request` m 和后乘的 flange RPY 旋转 `0.03 * request` rad。IK 以实测位姿为基准，频率 50 Hz；参考更新标称 1 kHz，timeout 为 0.25 s。记录的 mounting RPY 为零。这解决了该源码中 0.03/0.06 的文档矛盾，不能证明身份未知的运行库也相同。
- Episode 8 在 0.15 gain 下记录了 504 次发布，最大间隔为 11.806 ms。这支持传输可行性，不能证明控制器接收时序或 identity 物理验证通过。夹爪日志和 filter 测试证明的是请求语义，尚未证明开、合目标均被接受。

## 17 项必需检查

下表使用 `MAPPING_CHECKS` 中的准确 key。“部分”表示存在辅助证据；[待完成记录](pi05_mapping_verification.pending.json) 中所有检查仍为 `passed=false`。

| # | 必需 key | 可复用证据 | 仍需物理证据 |
| --- | --- | --- | --- |
| 1 | `linear_x_positive` | 部分：已报告脉冲；软件符号测试 | 同步记录接受过程和位姿，确认 +X 响应超过基线噪声 |
| 2 | `linear_x_negative` | 软件符号测试；CPU IK fixture | 独立 −X 响应 |
| 3 | `linear_y_positive` | 软件符号测试 | 独立 +Y 响应 |
| 4 | `linear_y_negative` | 软件符号测试 | 独立 −Y 响应 |
| 5 | `linear_z_positive` | 软件符号测试 | 独立 +Z 响应 |
| 6 | `linear_z_negative` | 软件符号测试 | 独立 −Z 响应 |
| 7 | `angular_x_positive` | 软件符号测试 | flange X 轴右手正向旋转 |
| 8 | `angular_x_negative` | 软件符号测试 | flange X 轴负向旋转 |
| 9 | `angular_y_positive` | 软件符号测试 | flange Y 轴正向旋转 |
| 10 | `angular_y_negative` | 软件符号测试 | flange Y 轴负向旋转 |
| 11 | `angular_z_positive` | 软件符号测试 | flange Z 轴正向旋转 |
| 12 | `angular_z_negative` | 软件符号测试 | flange Z 轴负向旋转 |
| 13 | `controller_acceptance` | 部分：已报告接收、IK 和参考计数 | 同步记录参考、关节、力矩、robot mode/errors 和实测响应 |
| 14 | `tool_frame_alignment` | 源码为 URDF `base` → `fr3_link8`；广播 `O_T_EE` | 实际 base/mount、flange → EE → Robotiq peg TCP 变换和工具身份 |
| 15 | `timing_10hz_100hz` | CPU 调度测试；legacy Episode 8 发布间隔 | 实际 100 ms 样本保持、10 ms 发布/接收间隔、等待期间 neutral |
| 16 | `neutral_stop` | 源码保留最后关节参考；CPU 停止测试 | neutral、过期和操作员停止后的残余运动，以及批准的硬件停止检查 |
| 17 | `gripper_open_close` | 二值标签；debounce/越界测试；已记录 open 请求 | 开合目标接受、实测行程、hold/不重复发目标和停止行为 |

## 操作员仍需完成的动作

现已准备[标定接口与操作流程](pi05_calibration_operator.zh.md)，支持逐项六轴/夹爪命令、同步采集和显式证据评审。上方清单保留 11 月 18 日的证据快照，物理检查仍无新增通过项。

先检查之后是否已采集新日志。若轨迹明确对应相同 binary、配置、工具、位姿和请求时序，并满足下述标准，即可复用，仅补做缺失项目。历史脉冲本身尚不满足这些条件。

1. **固定安装与工作区。** 现场操作员检查自由空间、mounting、Robotiq/peg 几何、相机角色、初始位姿、硬件停止装置及批准的停止流程。保存当前 `/proc/<actual-arm-pid>/maps`、映射 `.so` 的 SHA-256、选定的 launch/config/URDF、实际控制器参数、active controller 和 7 个 claimed effort interface。用 manifest 保存配置文件哈希、运行时默认参数、mounting 和工具变换，`controller_config_sha256` 对该 manifest 的原始字节计算。存档 K/D 为 `[600,600,600,600,250,150,50]` / `[30,30,30,30,10,10,5]`；保持 gain、limit、timeout 和 scale 不变。通过原接口的正常流程停止竞争的 SpaceMouse/policy publisher，保留其代码和配置。如需部署诊断库，按已有[部署流程](../../../outputs/fr3_step3_prepare_20261118/deployment_and_rollback.txt)另行作出维护决定；存在 Release 产物并不构成部署理由。
2. **运动前明确 frame。** 按实际 URDF/mount 确认物理 base 轴。在 mounting 为 `[0,0,0]` 时，线性请求沿 base X/Y/Z；旋转采用后乘，因此正角请求绕当前 flange X/Y/Z 按右手规则旋转。确认 `O_T_EE`、flange 和 peg TCP 的区别或其已标定变换，并用变换从 EE 实测位姿恢复 flange 运动。旋转符号用 `log(R_flange_startᵀ R_flange_now)` 判断；旋转导致的 peg 平移可能来自工具杠杆臂。不能从未标定 TCP 的平移推断轴交换。
3. **建立基线并采集。** 每项前后至少记录 2 秒完整位姿、实测及机器人期望关节/力矩、robot mode/errors 和 `FR3_DIAG_V1`。沿用部署流程的 telemetry bag topic 列表。核实安装版本中 `tau_J`、`tau_J_d` 和 broadcaster 字段的含义；机器人 `q_d` 不是控制器私有的 `q_desired_`。记录发送端及控制器本地 monotonic 时间，不直接相减未同步主机的时钟。操作员在运动前确定噪声/分辨率、允许的跨轴响应、残余平移/旋转和稳定时间。阈值未知或运动无法分辨时，结果为 **inconclusive**，不能通过。向命令 topic 添加 recorder 会增加 subscriber，破坏 X 工具恰好一个 subscriber 的检查，应使用发送日志和控制器接收证据。
4. **先解决 +X，再停下评审。** 按新操作流程执行只读 preflight，再单独确认一次 `linear_x_positive` case。接口保留 0.05 请求、0.5 s 持续时间和 10 mm 平移上限，新增旋转检查，并采集新鲜基线、完整 pose/joint/state 和诊断。五个默认 100 ms slot 在同一带符号轴上使用 `[0.030,0.024,0.018,0.024,0.030]`，按 100 Hz 刷新；请求偏移不保证物理行程。响应仍无法分辨或存在跨轴问题时，结束验证并调查轨迹，不自动加大或重复。原有仅 X 轴、20 Hz 工具保留为独立诊断，不能证明 policy 传输时序。
5. **+X 评审后逐项补齐方向。** 按 `+X, −X, +Y, −Y, +Z, −Z, +Rx, −Rx, +Ry, −Ry, +Rz, −Rz` 单独确认各 case，复用已有效的轨迹。[操作流程](pi05_calibration_operator.zh.md)列出了准确 case 名称及命令。每项只改变一个请求轴，机械臂方向 case 不操作夹爪，随后观察 neutral。默认最大角偏移为 0.0009 rad（0.052°），分辨率不足时判为 inconclusive。保持硬件停止监督，每项后先评审再继续。不能绕过 identity record 或用 policy 预测作标定。
6. **复用这些项目的接受、时序和 neutral 证据。** 经评审的标定接口以明确的 100 ms slot 保持样本，每 10 ms 刷新，不增加 gain。记录实际发布与控制器接收间隔、50 Hz IK/reference gate 活动、最后 slot 结束后的 neutral、至少 0.5 s 的模拟推理 neutral 等待，以及保持样本期间的操作员停止。保留 executor 的 50 ms 发布 deadline，超过即为 timing 失败。按原有 250 ms timeout 核查新鲜度，确认不回放、不追赶样本、处理 stop 后不再发移动请求，残余运动符合操作员预设净空。短脉冲不能证明所有生产 deadline。接受检查需结合计数差值、移动 gate 内的 joint-error/PD/总力矩峰值、各关节机器人期望与实测力矩、关节变化和状态；计数器或独立标量峰值不足以通过。Neutral/timeout 保留最后参考，机器人可能继续跟踪，不能代替 E-stop。按本地批准流程完成硬件停止测试/检查，不绕过安全机制或人为诱发硬件 fault。出现 fault 或异常运动时执行批准的硬件停止流程并结束验证。
7. **机械臂保持 neutral 时验证夹爪。** 检查 domain 70 的 Float32 bridge 和 domain 71 的 Robotiq controller/action endpoint，不作修改。在手指活动范围清空后，通过操作员批准的标定发送 3 个连续、10 Hz 的明确 open 样本（`1`，第 3 个触发目标），保持，再于距前一目标至少 0.8 s 后发送 close 样本（`0`）。核查目标接受/完成、实测开合、实际型号/单位及空载净空。验证中间值如 `0.5` 为 hold、稳定 latch 不重复发目标、100 Hz 机械臂刷新不增加确认次数、处理 stop 后不发新目标。机械臂 neutral 和当前 executor stop 不会取消在途 Robotiq 目标，需检查并批准其完成/独立停止行为。仅直接操作 bridge 不能验证 policy filter；使用新接口中单独确认的 open/close/hold case，采集实测/status 证据；工具不会取消在途目标。
8. **完成记录并独立评审。** 仅根据已完成证据填写 pending JSON：真实操作员姓名、过去的 UTC 验证时间、已验证 base/tool frame、实际映射 binary 哈希及配置 manifest 哈希。17 项检查分别附上非空测量/评审文件及真实 SHA-256，`evidence_path` 相对于记录文件；文件应标明原始轨迹及哈希、实验时间段、测量值、标准和操作员结论。确实覆盖多个检查时可复用同一文件。仅将已评审通过的项改为 `passed=true`，未完成记录继续阻止执行。保留 pending 快照，将完整批准的记录另存为新文件。Loader 仅校验完整性和文件哈希，不能认证操作员、检查运行库、验证测量真实性或作安全认证。

## 先验证记录，再执行一次 Rollout

连接 rollout 前，先对已完成记录运行以下只读校验。Parent 和 child 必须使用相同记录及证据字节；修改后原 consent 失效。使用本包 pending record 时，应因缺少操作员/物理批准而被拒绝。

```bash
export OPERATOR_VERIFIED_MAPPING_JSON=docs/experiments/openpi_recap_franka/pi05_mapping_verification.pending.json
.venv/bin/python - <<'PY'
import os
from toolkits.realworld_check.fr3_pi05_request_adapter import load_mapping_verification
r = load_mapping_verification(os.environ['OPERATOR_VERIFIED_MAPPING_JSON'])
print(r['record_sha256'])
print('Required consent: RUN VERIFIED ' + r['record_sha256'][:12])
PY
```

只有 17/17 检查通过、现场安装匹配且获得单次执行批准后，才将该环境变量改为完整记录。下方未来命令沿用 Episode 6 记录的 checkpoint，请求一次 3 秒、gain 1.0 的监督 episode；本次未执行。操作员需先确认训练起始布置，并批准 `[-1,1]` 请求范围内的 policy 运动净空。小幅标定证明方向与接口工作，不能证明任意 policy 轨迹安全。Executor 的 8 cm 检查只监测平移，旋转与接触净空仍由操作员在现有硬件保护下监督。

```bash
export PI05_CHECKPOINT='/data1/Muttaqien/logs/20261111-04:55:26-realworld_sft_openpi_fr3_peg_request_full3000_wandb/fr3_peg_pi05_request_offline_full3000_wandb/checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt'
PYTHONPATH=. .venv/bin/python -m toolkits.realworld_check.fr3_pi05_supervised_rollout \
  --checkpoint "$PI05_CHECKPOINT" \
  --root /data1/Muttaqien/fr3_pi05_real_rollouts_20 \
  --verified-request-mapping \
  --mapping-verification "$OPERATOR_VERIFIED_MAPPING_JSON" \
  --max-episode-seconds 3
```

在之后的 episode 提示中输入准确的 `RUN VERIFIED <12-character digest>`。Ready/event 日志必须显示 `identity_verified`、`arm_gain=1.0`、100 ms policy step、10 ms refresh 及批准的 record digest。ENTER 可在推理期间停止新命令，始终保留硬件停止监督。现有检查还涵盖最旧 observation receipt ≤1 s、issue age ≤300 ms、live pose ≤250 ms、机械臂/夹爪 publisher 独占与 subscriber 存在、请求范围、一秒 chunk、child 20 s 时限，以及 CLI 1–15 s 时限。验证失败不启用 full-strength fallback。实际安装仍需人工核对，loader 不会比较实时 `/proc` maps/configuration。

停止后检查 `executor_events.jsonl`、`pipeline_timing.jsonl`、命令/状态日志及 `outcome.json`。每个已发机械臂样本应等于解码请求，不能有 stale 或接受中断区间，并需确认 shutdown 和操作员审核的 outcome。将实测 flange/peg 运动、基线噪声、关节/力矩响应及任务进展与 legacy 证据比较；轨迹不同，不能视为仅改变 gain 的受控实验。只有实测运动具有意义并经操作员审核，才能判断“barely move”已解决，不能按请求数量判断。结果不明确时保留 attempt，停止后续 rollout。保留已有 Episode 1–8 文件。

## 本次离线验证

11 月 18 日的工作站 CPU 快照共 **269 项通过**，每个文件单独启动 pytest：request adapter 95、ROS executor 136、deployment audit 9、outcomes 19、X pulse 10。测试使用 fake ROS/进程边界和 pipe，没有 ROS discovery、控制器激活、新推理或运动。Unit test 的合成 approval 仅用于测试，未复制到待完成的物理记录。可用以下命令复现测试和数据审计：

```bash
for component in fr3_pi05_request_adapter fr3_pi05_ros_executor \
  fr3_pi05_deployment_audit fr3_rollout_outcomes fr3_supervised_arm_pulse; do
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
    "tests/unit_tests/test_${component}.py" || exit 1
done
.venv/bin/python -m toolkits.realworld_check.fr3_pi05_deployment_audit \
  --raw-root /data1/Muttaqien/fr3_peg_100 \
  --episode /data1/Muttaqien/fr3_pi05_real_rollouts_20/episode_06
```

本次审计通过了 80/20 source split 不相交、train-only normalization 及两种 mode 的记录请求回放检查，仍保持 `physical_identity_execution_approved=false`。证据哈希和 pending record 的预期拒绝另行检查。本包由 AI assistant 准备，未修改生产代码、控制器设置、模型权重或 SpaceMouse 接口。

本地链接、17 个 checklist key 及 EN/ZH 命令块一致性检查均通过。两个语言的 Sphinx build 均通过仓库 harness，在其默认 autodoc-import 过滤下报告零 warning；这些 Markdown 文件位于 Sphinx source tree 之外。全局 symbol 检查仍有 24 项发现，RST markup 检查有 61 条 warning、零 error，均来自未修改页面。日志保存在 `outputs/fr3_mapping_qualification_20261118/`。

新接口及当前回归验证边界见[操作流程](pi05_calibration_operator.zh.md)，17 项物理检查仍全部待完成。
