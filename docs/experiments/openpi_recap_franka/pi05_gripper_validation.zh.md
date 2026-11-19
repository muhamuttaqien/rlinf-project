# π₀.₅ 夹爪预测校验

本报告说明 Episode 7 因夹爪预测校验中止后的离线修正，依次核对训练标签、反归一化、已有预测中的越界情况，以及修正范围和回归验证。[English](pi05_gripper_validation.md)

## 中止原因与原始记录

Episode 7 在 100 Hz 机械臂指令刷新下执行了 30 个 policy 动作，随后一个 chunk 未通过校验。首个夹爪预测为 `-0.05276369364422559`，超出原有 `[-0.05, 1.05]` 接纳范围。第二个预测更低，为 `-0.05716913694965839`；仅放行首个值仍会导致整个 chunk 被拒绝。该 chunk 的六个机械臂维度均在原有限制内。

原始结果保持为 `aborted`，`stop_reason=invalid_policy_action`，记录了 30 个 policy 动作、456 次机械臂发布，最大发布间隔为 11,727,168 ns。被拒绝的 chunk 从未执行。离线回放后再次核对 Episode 1–7 所有现存文件的哈希，均未变化。Episode 3 有日志和观测文件，但缺少 `outcome.json`；本次工作保留这一状态。

## 训练标签与解码

第七维估计最近一次操作员按钮请求：`0=close`、`1=open`。它表示锁存请求，而非夹爪宽度或测得的关节位置。`fr3_command_request_alignment.py` 取观测窗口结束前最后一次请求，`fr3_request_lerobot_export.py` 保留该标签。首次请求前的未知区间会被排除，不会根据测量位置补造标签。

读取已导出 Parquet 文件的动作列，得到以下标签分布：

| 数据集 | 动作行数 | 闭合（`0`） | 张开（`1`） | 其他值 |
| --- | ---: | ---: | ---: | ---: |
| 训练集 | 12,288 | 8,191 | 4,097 | 0 |
| 验证集 | 3,078 | 2,031 | 1,047 | 0 |

已保存的训练归一化统计中，夹爪 `q01=0`、`q99=1`，均值为 `0.3334147135416667`，标准差为 `0.4714332851375357`。`fr3_request_norm_stats.py` 要求标签为二值。rollout 对模型生成的动作执行一次 OpenPI `Unnormalize(..., use_quantiles=True)`，随后取七个请求维度。已安装版本的反变换为：

```text
decoded = (generated + 1) / 2 * (q99 - q01 + 1e-6) + q01
```

这个仿射变换没有输出边界。连续模型预测可能超出二值目标范围，已有预测也证实了这种情况经常出现。归一化中的微小 epsilon 无法解释约 0.057 的越界量。此次中止发生在二值请求解释之前：原有校验直接拒绝了超出接纳范围的连续估计。

## 已有预测与接纳边界

本次只读分析使用已保存的预测，没有运行模型推理或访问机器人。[证据 JSON](pi05_gripper_validation_evidence.json) 记录了源路径、SHA-256 哈希、统计量、反归一化参数、被拒绝的 chunk 和回放结果。

| 来源 | 预测数 | 最小值 | 最大值 | 超出 `[0, 1]` | 超出原有范围 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 离线验证 shadow，300 个重叠 chunk | 3,000 | -0.02103265 | 1.04768977 | 741（24.7%） | 0 |
| 实时传感器 shadow v2 的已保存预测 | 50 | -0.02261593 | 1.04127464 | 33 | 0 |
| Episode 1–7 已执行动作日志 | 430 | -0.04907386 | 1.04712043 | 239 | 0 |
| Episode 7 被拒绝的 chunk | 10 | -0.05716914 | 1.01893515 | 6 | 2 |

首个实时 shadow 目录的预测文件为空，不计入样本。已执行动作日志只包含通过原有校验的预测，无法据此估计被拒绝预测的频率。离线预测的动作窗口存在重叠，也不能视为独立试验。

新的接纳范围为 `[-0.06, 1.06]`。0.06 的允许越界量来自已保存预测中的最大越界量 `0.05716913694965839`，按单位标签跨度的 0.01 向外取整。这个较窄的经验边界覆盖了完整的失败 chunk，并不保证未来模型输出都落在范围内。超出此范围仍会中止，不引入通用的 0.1 容差或无限制裁剪。

## 修正与安全契约

`validate_action_chunk()` 先拒绝非数值和非有限值，并检查原有机械臂限制及新的夹爪接纳边界。对于通过校验的值，仅将第七维饱和到 `[0, 1]`，返回新的 chunk。六个机械臂值保持不变，包括带符号的零。即使无效值位于 chunk 末尾，也会在发布任何动作之前拒绝整个 chunk。

父进程 rollout 现在使用校验函数返回的 chunk。已接纳动作的日志通过 `raw_predicted_7d` 保留原始解码行，通过 `requested_7d` 记录适配后的请求。无效 chunk 仍在 `invalid_policy_action.json` 中保留原始表示，结果仍自动标为 `aborted`。executor 使用同一个 adapter 独立校验收到的 chunk。

夹爪过滤器继续使用 `<=0.30` 闭合阈值、`>=0.70` 张开阈值、连续三次确认，以及 0.8 s 最小目标间隔。对端点附近的越界值进行饱和不会改变阈值分类。机械臂缩放、mapping 模式、100 Hz 刷新、controller 代码、观测新鲜度检查、位移和时长限制及停止处理均保持原有行为。checkpoint 和归一化文件仅作只读核对，没有加载或修改权重。

## 离线验证

工作站验证使用 Python 3.11 和 CPU，通过假的 ROS 消息及 publisher 接口与管道子进程测试运行，不访问实体 ROS endpoint。在仓库根目录运行以下组件测试及静态检查：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/unit_tests/test_fr3_pi05_request_adapter.py \
  tests/unit_tests/test_fr3_pi05_ros_executor.py \
  tests/unit_tests/test_fr3_pi05_deployment_audit.py

.venv/bin/ruff check --preview \
  toolkits/realworld_check/fr3_pi05_request_adapter.py \
  toolkits/realworld_check/fr3_pi05_supervised_rollout.py \
  tests/unit_tests/test_fr3_pi05_request_adapter.py \
  tests/unit_tests/test_fr3_pi05_ros_executor.py

.venv/bin/ruff format --check \
  toolkits/realworld_check/fr3_pi05_request_adapter.py \
  toolkits/realworld_check/fr3_pi05_supervised_rollout.py \
  tests/unit_tests/test_fr3_pi05_request_adapter.py \
  tests/unit_tests/test_fr3_pi05_ros_executor.py

git diff --check
```

工作目录和基于远端最新提交的独立 worktree 均通过全部 218 个测试。Ruff lint、Ruff format 和 `git diff --check` 均通过。

仓库 docs build harness 在默认过滤 autodoc import 警告的设置下，英文和中文构建均报告零警告。RST markup checker 报告零错误，以及未修改页面中已有的 61 条警告。实验报告位于 Sphinx 源目录之外，因此另外核对了本地 Markdown 链接及两种语言中相同的命令块。

测试覆盖有效值、正负越界及边界值、更大误差、每个维度的 NaN 和正负无穷、机械臂值及限制保持不变、夹爪阈值和去抖时序，以及 Episode 7 保存的失败 chunk。假的 ROS 测试还确认：后续无效 chunk 会触发 fault 并发布机械臂中性指令，不会重试；夹爪饱和生效时，位移越界仍会停止执行。原有操作员停止、watchdog、独占发布、调度和归档失败测试一并运行。

已保存预测的回放接纳了全部 3,490 个被检查的预测，包括 Episode 7 被拒绝的十个预测。六个机械臂维度在适配前后逐字节一致，两种现有机械臂增益下的结果也一致。对原先已通过校验的数据，每个十动作 chunk 使用新建过滤器按 10 Hz 回放，所有 open/close/hold 决策均保持一致。这些结果验证的是离线转换契约，不代表实体任务结果。

本次工作在离线验证后停止。下一次实体 episode 必须另行获得操作员批准，并通过现有逐次同意和安全检查。Episode 7 不会自动重试。
