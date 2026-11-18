# π₀.₅ 停止处理与 SpaceMouse 执行兼容性

本报告说明 Episode 8 的停止处理修正，以及下一次单独获得批准的 rollout 应使用的已验证 1.0 请求路径。内容依次介绍故障证据、policy 与已安装 SpaceMouse publisher 的接口比较、离线验证和操作员准备步骤。[English](pi05_episode8_shutdown.md)

## Episode 8 与丢失的停止确认

Episode 8 在执行 36 个 policy 动作后达到时限。executor 记录了一条 `episode_ended`，其中包含 36 个动作、504 次机械臂发布，最大发布间隔为 11,805,514 ns。父进程只保存了前 30 条动作和状态记录；最后六个动作仍保留在 `executor_events.jsonl`。四个观测文件已经保存，但没有 `outcome.json`。

原来的 `Executor.wait("chunk_done")` 会从共享队列中移除所有收到的事件，包括并非本次等待目标的事件。时限停止过程中，它可能先取出并丢弃 `episode_ended`，随后发现停止标志并抛出 `RolloutStopped`。后续 `wait("episode_ended")` 因而超时，尽管审计日志中已有确认事件。使用修正前代码和管道子进程的离线复现得到了原始的两个异常。

新的等待逻辑只消费匹配事件，并保留终止确认，供重复清理调用使用。在返回成功确认前仍会报告 fault。停止请求和撤销新 chunk 提交权限使用同一个写锁。父进程只发送一次 `end_episode`；executor 即使收到重复停止消息，也只产生一次确认。ROS 循环在下一次发布前处理停止输入并清空当前 schedule；处理停止后不再发布运动或夹爪目标。

## 部分结果与超时处理

父进程现在先完成停止握手，再归档被中止 chunk 的最终动作事件。这样可以保留部分执行的全部动作，避免再次像 Episode 8 那样漏掉最后六条记录。动作和状态日志会 flush，退出时保留完整 executor 事件流。

chunk 超时会触发停止请求。如果停止确认超时，或 executor 报错、退出，父进程会在结果复核前关闭指令进程，保留已收到的部分事件，并记录尚未由操作员复核的 `aborted` 结果及停止原因、错误详情。`shutdown_confirmed` 表示是否收到了确认；缺失的发布计数保留为 `null`，不会估算。正常的时限停止或操作员停止仍要求人工填写结果。结果输入遇到 EOF 或中断时，保留未复核的中止记录。

新的结果文件使用独占创建，不能覆盖已有结果。registrar 可以保留待复核的中止记录，包括已有空日志的零动作尝试，但不会将其计入 RECAP。新 manifest 还记录 `execution_interface_qualified`：此字段为 `false` 的 legacy 诊断结果即使经人工复核，也不会计入 20 个有效结果。历史 manifest 保留原有解释。

离线验证前后核对了 Episode 8 的八个现存文件，哈希均未变化。没有补写其缺失的结果，也没有改写不完整的动作日志。[证据 JSON](pi05_episode8_shutdown_evidence.json) 保存了文件哈希、终止事件、SpaceMouse 源文件标识和测试摘要。

## SpaceMouse 与 Policy 请求

已安装的 SpaceMouse publisher 与工作区源文件逐字节一致，SHA-256 为 `2ed3e166966b31778264889301ef8e6dccba4c51a0ffed0ac0524fe46a801c87`。它每隔 0.01 s 发布一次归一化 `Twist`。HID 映射为 `linear=(-y, x, z)`、`angular=(-roll, -pitch, -yaw)`，根据操作员位置配置进行的符号调整也在发布前完成。

训练对齐代码读取这些已发布的 Twist 字段，并在 10 Hz 观测窗口内计算时间加权均值。FR3 OpenPI 数据变换保留六个轴的值和顺序，为模型补齐维度，并使用训练请求的归一化统计。rollout 对动作只执行一次反归一化，再取七个请求维度，不重复应用 HID 映射、坐标旋转或 SI 单位转换。第七维使用单独记录的[有界夹爪适配](pi05_gripper_validation.zh.md)。

| 属性 | 正常工作的 SpaceMouse | 已验证的 π₀.₅ 路径 |
| --- | --- | --- |
| 机械臂 endpoint | `/franka_controller/target_cartesian_velocity_percent` | 相同 endpoint |
| 消息与单位 | 归一化 `Twist`，六个请求字段 | 相同字段和请求单位 |
| 机械臂发布 | 100 Hz | 100 Hz |
| 请求乘数 | 1.0 | `identity_verified` 下为 1.0 |
| 轴顺序与符号 | HID 映射后发布的 Twist 轴 | 原样保留解码轴 |
| 动作节奏 | 每个 timer tick 读取当前 HID 样本 | 训练时的 10 Hz 动作，每个样本仅在有效的 100 ms 时段内保持 |
| 间隔 | 持续读取设备样本 | 一秒 chunk 过期后刷新 neutral |
| Controller | 现有 `JointImpedanceIKController` | 相同 controller 和配置 |

Episode 8 使用了 `legacy_limited`：ready 事件报告增益为 0.15，每个已保存机械臂指令都恰好等于对应请求乘以 0.15，残差为零。因此该次运行没有测试完整幅度接口。identity 路径已经实现 1.0 增益；本次保留该实现，并将其明确为完成资格验证后应使用的路径。默认仍是衰减的 legacy 诊断模式。单位和发布节奏相同不代表轨迹相同，因为 policy 预测的是窗口均值，动作幅度也可能与操作员不同。

## 离线验证

CPU 验证通过假的 ROS endpoint 运行真实 executor 循环，通过管道子进程运行真实父进程 IPC 代码，不加载 checkpoint 或激活 controller。全部 259 个组件测试通过，覆盖 Episode 8 竞态、停止处理前先收到终止事件、重复停止、部分 chunk、缺失确认、子进程 fault/退出、人工结果保持不变，以及结果文件独占创建。

同一套测试还验证了 1.0 增益下全部十二个带符号轴向用例、record/fingerprint/consent 门槛、每次 100 Hz 刷新与当前不同的 10 Hz 动作是否匹配、消息构造期间过期、推理间隔、陈旧请求、neutral 停止、watchdog 和独占发布。在仓库根目录运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/unit_tests/test_fr3_pi05_ros_executor.py \
  tests/unit_tests/test_fr3_rollout_outcomes.py \
  tests/unit_tests/test_fr3_pi05_request_adapter.py \
  tests/unit_tests/test_fr3_pi05_deployment_audit.py

.venv/bin/ruff check --preview \
  toolkits/realworld_check/fr3_pi05_supervised_rollout.py \
  toolkits/realworld_check/fr3_pi05_ros_executor.py \
  toolkits/realworld_check/fr3_rollout_outcomes.py \
  tests/unit_tests/test_fr3_pi05_ros_executor.py \
  tests/unit_tests/test_fr3_rollout_outcomes.py

.venv/bin/ruff format --check \
  toolkits/realworld_check/fr3_pi05_supervised_rollout.py \
  toolkits/realworld_check/fr3_pi05_ros_executor.py \
  toolkits/realworld_check/fr3_rollout_outcomes.py \
  tests/unit_tests/test_fr3_pi05_ros_executor.py \
  tests/unit_tests/test_fr3_rollout_outcomes.py

git diff --check
```

Ruff 和 diff 检查通过。仓库 docs harness 在默认过滤 autodoc import 警告的设置下，两种语言构建均通过。这些 Markdown 报告位于 Sphinx 源目录之外，因此另行核对了本地链接、命令和中英文内容一致性。

## 准备一次另行批准的 Rollout

只有完成物理资格验证后才能使用 `identity_verified`。现有记录必须覆盖全部带符号轴向、controller 接收、坐标系、10 Hz/100 Hz 时序、neutral 停止和夹爪行为，并包含证据哈希及 controller binary/configuration 指纹。操作员需要将运行中的安装与记录核对，停止竞争的 SpaceMouse publisher，清空工作区，并在批准的 E-stop 位置监督。本次工作不授予物理资格，也不修改验证记录、controller 设置、checkpoint 或安全限制。

以下命令仅为之后操作员批准的会话准备，本次未执行。将 `PI05_CHECKPOINT` 设为现有最终 `global_step_3000/actor/model_state_dict/full_weights.pt`，将 `OPERATOR_VERIFIED_MAPPING_JSON` 设为实际完成资格验证的记录：

```bash
PYTHONPATH=. .venv/bin/python -m toolkits.realworld_check.fr3_pi05_supervised_rollout \
  --checkpoint "$PI05_CHECKPOINT" \
  --root /data1/Muttaqien/fr3_pi05_real_rollouts_20 \
  --verified-request-mapping \
  --mapping-verification "$OPERATOR_VERIFIED_MAPPING_JSON"
```

父进程在 model/hardware 设置前验证记录，子进程独立核对 digest。逐次同意口令仍为 `RUN VERIFIED <record SHA-256 的前 12 个字符>`。已有尝试目录会被跳过。下一次 episode 只有在接口完成资格验证、结果经操作员复核后，才可计入 RECAP。本次工作在离线验证后停止；机械臂运动过弱是否解决，仍需通过实体测试判断。
