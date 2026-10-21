# OpenPI + RECAP：Panda StackCube-v1

在加载模型权重前，验证本地 Panda 示范数据到 OpenPI 的数据流程。继续使用 `branch-openpi-recap`，保留现有数据和 50 步评估上限。[English](STACKCUBE_RECAP.md)

## 已验证的数据流程

2026-10-21，Milestone 1 在 RTX 4070 Ti SUPER 上通过。查看 `outputs/stackcube_recap/pipeline_v1/validation.json`，获取源文件 SHA-256、runtime 路径、轨迹长度和 batch shape。结果覆盖数据加载和推理预处理；模型加载及训练仍待验证。

- 现有专家 RGB replay 包含 10 条成功记录、1,084 个 action，seed 为 0–9。
- 源文件：`outputs/stackcube_recap/raw/StackCube-v1/motionplanning/stackcube_expert.rgb.pd_ee_delta_pose.physx_cpu.h5`。
- 转换后的数据位于 `outputs/stackcube_recap/lerobot_stackcube_v1`，采用 LeRobot v3 和 20 FPS。action、本体状态和 success 标记均与源数据一致。
- 两路视频均完整解码 1,084 帧。每条轨迹起点和终点的共 20 个 action window 保持对齐，并在 episode 边界使用最后一个 action 填充。
- 新采集的一条 107-action 专家轨迹在 `outputs/stackcube_recap/milestone1_fresh_demo_20261021` 中通过 RGB replay 和转换检查。该轨迹再次使用 seed 0，不构成独立评估结果。

## Observation 和 Action 约定

使用 `pi05_stackcube`，将九维 `qpos` 与九维 `qvel` 按顺序拼接。不使用物体位姿或其他仅在模拟器中可见的状态。七维 action 沿用 `pd_ee_delta_pose` 的控制命令，包括 gripper。保留单位和控制参考系，不再根据 state 进行额外的 delta 变换。

将 `observation.images.base_camera` 和 `observation.images.hand_camera` 分别映射到 base 和 left wrist 输入。第三路 camera 使用零填充并设置 mask。语言指令直接读取 LeRobot 的 `task` 字符串；固定版本 OpenPI 的 task-index helper 使用旧版 task table 格式。

该 config 使用 10-action horizon、32 维 padded state/action 和 200 个语言 token，并启用 discrete state input。真实 RLinf SFT batch 包含 `[1, 10, 32]` 的 actions、`[1, 32]` 的 state、每路 `[1, 3, 224, 224]` 的 image 和 `[1, 200]` 的 tokens。训练与实时推理预处理均已通过。

使用 `examples/embodiment/config/env/maniskill_stack_cube.yaml` 配置匹配的 RLinf observation 接口。显式选择的 `proprio` 模式输出按顺序拼接的 `qpos` 和 `qvel`、两路 camera 和任务文本，seed 从 1000 开始。现有 `simple` 模式包含额外模拟器状态，不适用于本数据集的 statistics。

## Normalization 和评估

使用 `outputs/stackcube_recap/pipeline_v1/norm_stats.json`。检查脚本基于全部 1,084 个示范帧计算均值、总体标准差及 NumPy 精确的第 1/99 百分位数，在 padding 前使用 OpenPI 的 `state` 和 `actions` key。π₀.₅ 使用 quantile normalization。当前没有声明独立验证集，现有 LeRobot statistics 保持不变。

评估继续使用 20 Hz、最多 50 个模拟器控制步。真实 RLinf wrapper 检查确认第 50 步发生 truncation。10-action horizon 仅定义预测窗口，不改变 episode budget。

示范轨迹长 94–120 步，首次成功出现在第 90–115 步。已安装的 ManiSkill motion planner 在 truncation 后继续运行，并根据最终 success 筛选；replay 默认接受超时后成功的轨迹，除非启用 timeout rejection。因此，示范记录成功不能证明 policy 能在 50 步内成功。后续 SFT 和 RECAP 必须使用相同的独立 seed 和 budget；修改评估协议前需获得批准。

## 复现数据验证

数据验证 runtime 组合使用仓库 `.venv`、`.venv-recap-openpi` 中的 `rlinf-openpi==0.1.1`，以及相邻 `../openpi` checkout 的 client 源码和 JAX 依赖。package overlay 来自 `uv pip install --target <fresh-directory> --no-deps rlinf-openpi==0.1.1`，两个原有环境均未重装。Torch 2.10.0 和 Transformers 4.57.6 与 OpenPI 声明的训练依赖版本不同。RLinf PyTorch model 使用本地实现；预训练 checkpoint 转换和单步训练仍需验证。

在仓库根目录为验证进程选择以下源路径：

```bash
task_openpi_root="$(realpath ../openpi)"
export PYTHONPATH="$PWD:$PWD/.venv/lib/python3.11/site-packages:$PWD/.venv-recap-openpi:$task_openpi_root/packages/openpi-client/src:$task_openpi_root/.venv/lib/python3.11/site-packages"
export HF_DATASETS_CACHE="$PWD/outputs/stackcube_recap/hf_datasets_cache"
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -B -m scripts.check_stackcube_recap \
  --output outputs/stackcube_recap/pipeline_v2 --check-loader --check-env
```

该命令比较源数据与转换后的数据，解码视频，生成新的 OpenPI statistics，检查真实 RLinf 环境及推理预处理，并通过已组合的 SFT config 读取一个 batch。每次均需选择新的输出目录。模拟器渲染需要在 sandbox 外访问本机 GPU。

在同一 runtime 中运行针对性的 CPU 测试：

```bash
.venv/bin/python -B -m pytest -q -o addopts= -p no:cacheprovider \
  tests/unit_tests/test_stackcube_recap.py
```

四项测试均已通过，覆盖转换对应关系与 normalization、错误 action 对齐、错误控制频率，以及当前和旧版输入映射。查看 `pipeline_v1/trajectory_0_frames.png`，检查第 0、50、107 步的 base 和 wrist 样本图像。

## 16 GB SFT 配置

本地 PyTorch π₀.₅ 使用 Gemma 2B vision-language trunk 和 300M action expert。全量微调需要为完整模型分配 gradients 和 Adam states。Smoke config 设置 `actor.model.openpi.train_expert_only: True`，加载权重后冻结 SigLIP 和 Gemma expert 0，只训练 action expert 和 action projections。此 RLinf Pi0 实现支持 `actor.fsdp_config.gradient_checkpointing: True`；SFT forward path 现在会保留该设置。首次运行保持 micro-batch 和 global batch 为 1。

当前 PyTorch path 没有经过验证的 4-bit 或 QLoRA 权重加载流程。Gemma module 含有部分 LoRA 支持，但 attention LoRA 尚未实现，因此 smoke test 使用 expert-only fine-tuning。若实测 peak memory 仍不足，可启用 FSDP CPU offload。缓存中的 JAX/Orbax `pi05_base` 约 12 GB，目标 Torch checkpoint 尚不存在，repository filesystem 仅剩约 20 GB；创建另一份完整 checkpoint 前需核对可用空间和转换过程峰值占用。

## 后续阶段

组合后的 `examples/sft/config/stackcube_sft_openpi_pi05.yaml` 使用已验证的接口、batch size 1、gradient checkpointing 和单步 smoke budget。转换本地 checkpoint 后，Milestone 3 已经验证了 forward，但 backward、optimizer step 和训练 checkpoint 保存尚未成功。

### 16 GB Smoke Test 结果

2026-10-21，RLinf 格式的 π₀.₅ checkpoint 已在 RTX 4070 Ti SUPER 上成功加载。Worker 读取了 StackCube LeRobot batch 和 normalization statistics，并在启用 expert-only training 与 gradient checkpointing 后进入 backward。随后发生 CUDA out of memory：PyTorch 报告已分配 12.65 GiB、剩余 188 MiB，且无法分配 128 MiB。Optimizer step 和训练 checkpoint 均未完成。该配置仅验证到数据和模型前向路径，不能视为成功的 SFT smoke test；需在显存更大的硬件上重试，或先验证更节省显存的训练方案。

完成 SFT 验证和 baseline rollout 后，依次执行离线 RECAP 的 returns、value-model fitting、advantage labeling 和 CFG policy fitting，入口见 [RECAP 指南](source-zh/rst_source/examples/embodied/recap.rst)。保留 episode outcome，并使用相同的独立评估条件。每个主要阶段均需获得监督者批准。
