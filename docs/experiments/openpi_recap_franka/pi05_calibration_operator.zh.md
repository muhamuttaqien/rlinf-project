# FR3 Mapping 标定操作流程

按本流程逐项预览和执行有界标定，再依据 [17 项验证清单](pi05_physical_mapping_qualification.zh.md)评审证据。运动执行完成了离线验证，只读技术 preflight 也已在实际 NUC 上检查；物理 mapping 检查仍全部待完成。Scaling 1.0 能否解决弱运动问题，仍需物理实验证明。[English](pi05_calibration_operator.md)

## 准备安装证据

使用已经运行、经操作员批准的控制器。这些工具不会启动、激活、重启、调参或恢复控制器。先阅读[控制器调查](fr3_controller_investigation.md)和[待完成记录](pi05_mapping_verification.pending.json)。已有 +X 报告说明了请求接收、IK 和参考活动，但缺少同步且可分辨的物理方向轨迹。存档 Release build 不能证明其已部署。若后来已有符合标准的轨迹，应先复用，再考虑补做运动。

将实际进程映射的 controller library、选定的 launch/config/URDF、运行时参数、mounting 和工具标定证据复制到本地证据目录。人工核对 `/proc/<actual-pid>/maps` 与现场配置；复制或计算哈希不会检查 NUC 的当前进程。用真实文件生成配置 manifest：

```bash
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_evidence config-manifest \
  --output /tmp/fr3_calibration/config_manifest.json \
  /tmp/fr3_calibration/controllers.yaml \
  /tmp/fr3_calibration/live_parameters.yaml \
  /tmp/fr3_calibration/robot.urdf \
  /tmp/fr3_calibration/tool_calibration.json
sha256sum /tmp/fr3_calibration/copied_controller.so /tmp/fr3_calibration/config_manifest.json
```

根据 [deployment 模板](pi05_calibration_deployment.example.json)创建 `/tmp/fr3_calibration/deployment.json`。其中的 null 和 false 会阻止执行。填写真实操作员、frame 名称、diagnostic logger、Robotiq joint/bridge 身份及 domain 71 的 action status topic，并填入上一步得到的两个哈希。平移与旋转软件检查值均须由操作员批准，为正数且分别不超过 **0.01 m / 0.01 rad**。平移上限沿用原有标定上限；新增的旋转检查不改变控制器 limit。

三个变换均采用 `[x,y,z,qx,qy,qz,qw]`：`base_to_pose` 从控制器 base 指向 pose topic header 的 frame；`ee_to_flange` 从广播的 EE 指向受控 flange；`ee_to_tcp` 从该 EE 指向 Robotiq peg TCP。必须根据实际安装和标定填写。运行时会核查 `f_t_ee * ee_to_flange` 是否为单位变换，并监测 EE、flange 和 TCP 位移，但不能认证真实工具几何。检查历史错误、清空工作区、核查批准的硬件停止装置，并匹配当前安装后，才将相应声明设为 true。保持 gain、limit、weight 和 SpaceMouse 配置不变。

## 预览与只读检查

默认命令仅显示准确的请求 slot，不导入 ROS。每项实验都先预览：

```bash
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_calibration \
  --case linear_x_positive
```

五个 100 ms 机械臂 slot 在同一带符号轴上依次使用 `0.030, 0.024, 0.018, 0.024, 0.030`，其余轴保持零。不同数值可让发送日志暴露错误的样本保持行为。这些值是归一化请求，不增加 gain，不再做 HID 映射或 SI 换算；最大值仍为建议的 0.03。CLI 沿用最大幅度 0.05、最长 0.5 s 的限制，持续时间须为整数个 100 ms slot。夹爪 open/close 至少需要三个样本才能产生目标。响应弱时，不自动加大请求或重复实验。

在之后的操作员会话中，打开新的工作站 shell，source Jazzy 及匹配的 message definitions，包括 `franka_msgs` 和 `controller_manager_msgs`。通过正常的拥有者接口停止竞争 publisher。以下 preflight 只订阅 telemetry，并调用只读 controller-list service，**不创建命令 publisher**：

```bash
source toolkits/realworld_check/fr3_calibration_env.sh /tmp/fr3_calibration/jazzy_msgs/install
/usr/bin/python3 -m toolkits.realworld_check.fr3_mapping_calibration \
  --case linear_x_positive \
  --deployment /tmp/fr3_calibration/deployment.json --preflight
```

该 helper source Jazzy 和已验证的 message overlay，保留两者的 Python 路径，加入仓库路径，并在不初始化 ROS 的情况下检查 native type support。使用 `/usr/bin/python3`（Python 3.12）并保留该环境，不能用 `PYTHONPATH=.` 覆盖 ROS 路径。`/tmp/fr3_calibration/jazzy_msgs` 是仅包含接口的本地 workspace，包含与 NUC 匹配的 `franka_msgs` 和 `controller_manager_msgs`；自定义接口从 Humble 已安装包复制，并为 Jazzy 本地编译。Jazzy 原生 `ControllerState` 多出字段，因此 controller service 包也须匹配。Overlay 不改变系统 ROS 安装或 NUC。

在新的 shell 中用现有源码及 manifest 证据重建，避免下载不同版本：

```bash
source /opt/ros/jazzy/setup.bash
(
  cd /tmp/fr3_calibration/jazzy_msgs
  /usr/bin/colcon build --base-paths src --packages-select franka_msgs controller_manager_msgs \
    --executor sequential --cmake-args -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=/usr/bin/python3
)
```

`--preflight` 检查 artifact 哈希、已配置的 base/EE 变换和实时 telemetry，允许运动准备字段保持未填写，输出待完成项及 `motion_authorized=false`，并禁止创建命令 publisher。执行路径仍须加载完整 deployment、核查批准与已标定 TCP，并获得逐 case 确认。技术 preflight 通过后不要补造批准。

两个命令 topic 都须恰好有一个类型正确的 subscriber，且无竞争 writer；pose、joint、完整 robot state 和夹爪位置必须新鲜。Joint-impedance controller 须处于 active 状态并 claim 七个 effort interface，robot mode/current errors 须允许标定，还须收到两条新鲜诊断报告。`FR3_DIAG_V1` 必须包含已有 joint-error/PD/总力矩 peak 字段。缺少消息定义、诊断或 telemetry 时停止；工具不会安装、激活或修改任何组件来补齐它们。命令 topic 的 recorder 会增加 subscriber，导致严格图检查失败，应使用发送端事件和控制器诊断采集证据。

## 单独批准后，一次执行一项

先完成 +X，并评审实测响应，再考虑其他方向。下方是未来的操作员命令，**本次准备未执行**：

```bash
/usr/bin/python3 -m toolkits.realworld_check.fr3_mapping_calibration \
  --case linear_x_positive \
  --deployment /tmp/fr3_calibration/deployment.json \
  --output /tmp/fr3_calibration/x_positive_01 --execute
```

终端会显示新的 `CALIBRATE <case> <digest>` token，绑定此次 plan 和 deployment。阅读预览并核对安装后，准确输入 token；管道输入的 approval 会被拒绝。确认仅覆盖一次有界 case 及其 neutral 清理，包括 100 Hz 刷新。新的机械臂或夹爪 case 必须重新交互确认。本地独占锁和实时 DDS 检查共同限制竞争控制。

确认后，工具重新采集至少 2.1 s 连续新鲜基线，再执行最长 0.5 s 请求，并观察至少 2.1 s 的 neutral 阶段。ENTER/EOF、SIGINT 或 SIGTERM 会撤销新运动/目标；在 active sample 期间按 ENTER 可记录停止切换。Abort 路径会独立于证据存储尝试发布五次 neutral。Neutral 保持最后关节参考，仍可能产生残余跟踪运动；它不会取消在途夹爪目标，也不能代替批准的硬件停止装置。已进入 middleware 的发布无法撤回。

循环会复查控制权、telemetry、完整姿态和工具几何。Pose/joint/robot/gripper state 时效上限为 200 ms，controller-list 状态为 1 s，诊断为 2 s。Action status 是事件流，按 action status QoS 接收，不作为周期性 heartbeat；目标后的接受/完成事件仍须评审。传输间隔超过 50 ms 或 100 ms slot 过期时 abort，不追赶样本。证据写入使用有界队列；队列溢出或写入失败会停止 case，neutral 清理仍独立运行。工具不重试 fault，也不重启 case。

解决 +X 后，使用下表的单项名称逐项运行，每项单独确认并使用新输出目录。Mounting RPY 为 `[0,0,0]` 时，线性方向沿控制器 base，角度方向遵循当前 flange 的右手轴；纯旋转可通过杠杆臂引起 peg 平移。默认最大 IK 偏移为 0.21 mm 和 0.0009 rad（0.052°），不保证实际行程。运动前约定分辨率、跨轴容差和可接受残余运动；无法分辨时判为 inconclusive。

| 轴 | 正向 case | 负向 case |
| --- | --- | --- |
| Base X | `linear_x_positive` | `linear_x_negative` |
| Base Y | `linear_y_positive` | `linear_y_negative` |
| Base Z | `linear_z_positive` | `linear_z_negative` |
| Flange X | `angular_x_positive` | `angular_x_negative` |
| Flange Y | `angular_y_positive` | `angular_y_negative` |
| Flange Z | `angular_z_positive` | `angular_z_negative` |

`gripper_open`、`gripper_close` 和 `gripper_hold` 是独立 case，机械臂保持 neutral。Open/close 在三个明确的 10 Hz 样本后只发一个目标，稳定 latch 不重复发目标，hold 使用请求 0.5；机械臂刷新不增加夹爪确认次数。每次基线/后续观察使顺序运行间隔超过原有 0.8 s 目标间隔。记录 domain 71 action status 和实测 joint，bridge 发布不代表接受。检查 Robotiq 的独立停止/完成行为。`neutral` case 只发送零机械臂请求，不能单独证明运动中的停止行为。

## 检查与评审真实测量

每次运行创建 `session.json`、`events.jsonl`、`telemetry.jsonl`、`summary.json`、安装输入副本和带哈希的 `manifest.json`。混合 `/joint_states` 消息在更新接收时间前按配置中的物理 joint 名称筛选，normalized surrogate 不会覆盖实测样本或刷新其 watchdog。采集内容包括控制器 `/rosout` 诊断、pose、实测/期望 joint、完整 robot state、只读 controller 状态及 domain 71 夹爪 state/status。完整 robot-state 证据最多按 100 Hz 采样并保留接收计数，不等于完整 1 kHz 记录。Host monotonic receipt 用于对齐本地流，ROS header 时钟另行保存。

```bash
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_evidence verify-session \
  /tmp/fr3_calibration/x_positive_01
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_evidence analyze \
  /tmp/fr3_calibration/x_positive_01
```

命令检查哈希并报告实际发送间隔、带符号 frame 运动、计数差值和峰值，**不会自动通过物理检查**。Fake capture 不能批准 mapping。`FR3_DIAG_V1` 只有汇总接收计数和独立最大值，无法测量逐命令控制器接收间隔、硬件接受力矩或准确的 request/reference 对应关系。缺乏充分且真实的控制器侧时序证据时，`timing_10hz_100hz` 保持待完成；工具不会为补齐证据而新增控制器 instrumentation。

根据 [review 模板](pi05_calibration_review.example.json)填写真实评审。绑定 analysis 中的 session-manifest/scope 哈希，准确命名 17 项检查之一，记录实测值、标准、采集完成后的过去 UTC 评审时间、notes 和带哈希的非空附件。只有物理评审通过后才选择 `pass`，否则使用 `inconclusive` 或 `fail`。Scope 不匹配和嵌套文件改动均会被拒绝。一条有效轨迹可供多项独立评审复用，例如方向与操作员停止，无需重复运动。

对于 `gripper_open_close`，还须提供 `counterpart_session_manifest: {"path": "close_session/manifest.json", "sha256": "<actual hash>"}`，指向另一完整 case。两个 session 都须为真实 ROS 采集、安装 scope 相同、具有实测/status 数据，且目标在同一 host boot clock 上相隔至少 0.8 s，并附上两次目标的接受/完成证据。Frame 检查附上真实 TCP 标定；接受检查对比实测/期望 joint effort、mode/errors、诊断和运动；neutral/stop 检查附上残余运动和批准的硬件停止证据。

将一项评审应用到新的 partial record，另行输入 `REVIEW <check> <digest>`：

```bash
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_evidence apply-review \
  --record docs/experiments/openpi_recap_franka/pi05_mapping_verification.pending.json \
  --session /tmp/fr3_calibration/x_positive_01 \
  --review /tmp/fr3_calibration/x_positive_review.json \
  --output /tmp/fr3_calibration/mapping_partial_01.json
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_evidence audit-record \
  /tmp/fr3_calibration/mapping_partial_01.json
```

新记录附带可移动的证据 bundle，后续评审使用最新 partial record，不覆盖既有 outcome 或证据。即使个别检查通过，partial record 仍不具有最终批准。只有 17 项真实评审全部完成后，最终操作员才能输入 `APPROVE MAPPING <digest>`：

```bash
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_evidence finalize \
  --record /tmp/fr3_calibration/mapping_partial_17.json \
  --operator '<actual operator name>' \
  --output /tmp/fr3_calibration/mapping_verified.json
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_evidence verify-record \
  /tmp/fr3_calibration/mapping_verified.json
```

工具验证完整性和传递依赖的哈希，不能认证操作员身份或证明物理安全。Finalize 只满足 record gate。之后执行一次 scaling 1.0 rollout 仍须核对实际安装，并单独输入 `RUN VERIFIED <digest>`，具体命令见验证清单。先停止并评审这一次 outcome，再考虑后续 rollout。

## 离线验证边界

在独立 CPU pytest 进程中运行 `test_fr3_mapping_calibration.py` 和 `test_fr3_mapping_evidence.py`，以及原有 request-adapter、executor、outcomes、deployment-audit 和 X-pulse 测试。新测试通过 fake hardware 边界运行实际状态机和 ROS 发布方法。合成 review fixture 只存在于 pytest 临时目录，不会成为物理记录。此分支省略了上游 CI workflow，提交上游前须恢复 path-filter coverage。AI assistant 准备并测试了该接口；本次没有操作控制器、执行物理运动、新推理、修改 gain/limit/weight 或编辑 SpaceMouse。

本次离线运行共 **9 个组件文件、379 项测试通过**，另以 Python 3.12 的实际 Jazzy message class 完成两项 fake-hardware case。EN/ZH build、本地链接与命令一致性、Ruff lint/format 及空白检查均通过。全局 docs 检查保留未修改页面中的 24 项 symbol 发现和 61 条 RST warning（零 error）。源码哈希及限制见[离线证据记录](pi05_calibration_offline_evidence.json)。

[Preflight 修复证据](pi05_calibration_preflight_evidence.json)记录了接口兼容性和实际只读验证结果，不代表物理 mapping 或运动批准。
