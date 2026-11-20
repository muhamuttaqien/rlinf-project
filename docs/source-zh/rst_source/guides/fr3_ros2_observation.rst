通过 ROS 2 读取 FR3 和 Robotiq 状态
===================================

通过只读 robot parts，将 RLinf 接入现有的 FR3 和 Robotiq 2F-85 状态 publisher。你可以读取测量关节、位姿、外力估计和夹爪状态，硬件连接仍由现有 ROS controller 持有。先运行有时限的检查，再组合 parts 并检查新鲜度 metadata。

运行有时限的观测检查
--------------------

订阅需要在兼容 ROS 的 Python 进程中运行。已审计工作站上的 Jazzy 使用系统 Python 3.12，RLinf 使用 Python 3.11。adapter 在各个子进程中加载 ``/opt/ros/jazzy/setup.bash``，显式选择机械臂 domain 70 和夹爪 domain 71。运行前需保持现有状态 publisher 在线。

在仓库根目录运行：

.. code-block:: bash

   PYTHONPATH="$PWD" python toolkits/realworld_check/observe_fr3_ros2.py --duration 5

该命令订阅状态 topic，等待完整观测，采样五秒，并在 ``finally`` 中断开连接。日志记录不完整读数、最大本地接收年龄、源 header 和 frame 标识。采样期间只要出现不完整读数，检查就以状态码 1 退出。通过 ``--ros-python`` 和 ``--ros-setup`` 选择其他兼容安装，通过 ``--max-age`` 设置以秒为单位的接收年龄阈值。

.. warning::

   实机检查应直接使用这些 observation parts。支持运动的 ``FrankaEnv`` 在构造和 reset 时可能移动机器人。

组合只读 Robot Parts
--------------------

原生 robot tree 允许独立组合可观测和可控制的 parts。将这两个 parts 直接传入 ``FrankaRobot``，即可使用原生连接和观测生命周期。这条路径绕过控制 backend builder，不需要机械臂 IP 地址或串口连接。

.. code-block:: python

   from rlinf.robotics import FrankaRobot
   from rlinf.robotics.parts.ros2_observation import (
       FR3ROS2Observation,
       RobotiqROS2Observation,
   )

   arm = FR3ROS2Observation(domain_id=70, max_age=0.25)
   gripper = RobotiqROS2Observation(domain_id=71, max_age=0.25)
   robot = FrankaRobot(arm=arm, end_effector=gripper)
   try:
       robot.connect()
       arm.wait_for_observation(timeout=10)
       gripper.wait_for_observation(timeout=10)
       observations = robot.get_observation()
       arm_state = observations["arm"]["state"]
       arm_status = observations["arm"]["ros2"]
   finally:
       robot.disconnect()

构造只记录设置；``connect()`` 为每个 part 启动一个订阅进程。该调用确认 worker 已就绪，随后通过 ``wait_for_observation()`` 等待发现 publisher 和接收首批消息。所选字段完整时，该方法返回观测；状态缺失或过期时抛出 ``TimeoutError``，worker 失败时抛出 ``ConnectionError``。``get_observation()`` 在有超时限制的 IPC 请求后返回当前观测，允许数据不完整。``disconnect()`` 请求 worker 退出、关闭 IPC、回收进程并移除临时日志，可以重复调用。重新连接时显式执行 disconnect/connect。

两个 parts 均继承 ``RobotPart``，原生 ``reset()`` 为空操作，不暴露 action 字段或控制方法。它们通过同一个 ``PartGroup`` contract 与其他可观测 parts 组合。其 contract 比 ``BaseArm`` 和 ``BaseGripper`` 更窄：后两者分别包含当前不可用的机械臂字段和经过标定的夹爪开口宽度。

理解所选测量值
--------------

每个观测包含 ``state`` 和 ``ros2``。关节按名称选择，额外关节不会改变 FR3 输出顺序。位姿四元数保留 ROS 的 ``x,y,z,w`` 顺序。所选 topic 固定为测量状态：

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Stream / Domain
     - Topic 和类型
     - State 字段
   * - joints / 70
     - ``/franka/joint_states`` (``sensor_msgs/msg/JointState``)
     - ``arm_joint_position``、``arm_joint_velocity``、``arm_joint_effort``；``fr3_joint1`` 至 ``fr3_joint7``；rad、rad/s、N m
   * - pose / 70
     - ``/franka_robot_state_broadcaster/current_pose`` (``geometry_msgs/msg/PoseStamped``)
     - ``tcp_pose``；位置单位为米，四元数顺序为 x,y,z,w
   * - wrench_base / 70
     - ``/franka_robot_state_broadcaster/external_wrench_in_base_frame`` (``geometry_msgs/msg/WrenchStamped``)
     - ``base_force``、``base_torque``；N、N m
   * - wrench_stiffness / 70
     - ``/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame`` (``geometry_msgs/msg/WrenchStamped``)
     - ``tcp_force``、``tcp_torque``；N、N m；保留原始 header
   * - knuckle / 71
     - ``/joint_states`` (``sensor_msgs/msg/JointState``)，``robotiq_85_left_knuckle_joint``
     - ``knuckle_position_rad``
   * - normalized / 71
     - ``/joint_states`` (``sensor_msgs/msg/JointState``)，``robotiq_normalized_joint``
     - ``normalized_closure``；0=张开，1=闭合

Wrench 是 broadcaster 输出的外力估计，其 header 决定坐标系。实机审计时 stiffness-frame wrench 的 frame 为 ``fr3_link2``。adapter 保留原值，不进行坐标变换，也不将 ``base`` 与 ``base_link`` 视为相同。它检查维度、有限数值、关节名称唯一性、归一化闭合量范围和单位四元数，不会静默归一化四元数。

缺失测量值为 ``None``。``ros2.unavailable_fields`` 列出所选 contract 之外的字段：机械臂的测量 TCP 速度和 Jacobian，以及夹爪的标定开口宽度、测量速度、夹持力和自定义 typed status。夹爪标准 stream 中由 driver 固定填充的零速度被排除。原始 knuckle 角度和归一化闭合量是独立读数，不能互换为宽度。工作站 ROS 安装无需 Franka 或 Robotiq 自定义消息包即可读取所选标准消息。

检查连接、新鲜度与错误
----------------------

通过 metadata 判断读数是否可用。``is_connected`` 遵循原生的已打开 handle contract；``ros2.bridge_alive`` 表示 worker 是否存活。每个 stream 的 ``connected`` 标记来自最近一次每 0.5 秒执行的 graph 检查，表示存在类型匹配的 publisher。发现 publisher 并不等于已接收到数据。

``ros2.complete`` 要求每个所选 stream 都有有效样本、匹配的 publisher，接收年龄不超过 ``max_age``，且每个所选字段均存在。完整性仅描述这个子集，不表示原生机械臂或夹爪控制状态完整。通过 ``missing_streams``、``stale_streams``、``invalid_streams``、``missing_fields``、顶层 ``error`` 以及各 stream 的 ``error`` 或 ``graph_error`` 查找原因。格式错误的消息会清空对应 stream 的样本；过期但有效的样本仍保留，并带有过期标记。

在 ``ros2.streams[stream]["sample"]`` 中，``header`` 保留 ROS 的 ``sec``、``nanosec`` 和 ``frame_id``，``received_monotonic_ns`` 单独记录本地 callback 接收时间。``age_s`` 使用该单调时钟，衡量接收新鲜度，不表示采集年龄或时钟同步。源 timestamp 重复以及持续发布均不能证明每次都获得了新的串口测量。这些最新样本独立接收，不是同步的机器人快照。

worker 检查 publisher 宣告的类型后，使用 best-effort、volatile、keep-last-depth-5 订阅。每个 stream 只保留一个解码样本。私有 Unix ``SOCK_SEQPACKET`` socket 传输 version-1 JSON 请求和响应，每个 packet 上限为 64 KiB。支持 ``snapshot``、``inspect_status`` 和 ``shutdown`` 请求。状态检查使用固定的只读白名单查询 ROS graph、控制器列表和参数；状态数据按需返回。父进程最多保留 16 KiB 的 stderr。启动和 IPC 默认超时为五秒，失败时关闭会话并报告诊断；父进程 EOF 也会触发 worker 退出。

worker 仅在所选解释器中导入 ROS 包，不建立硬件控制连接。它订阅状态数据，并可在请求状态检查时创建临时只读诊断 service client。标准 rclpy 基础设施可能发布 parameter events 或提供本地 type-description service；adapter 不创建 command publisher、action client，也不发送改变状态的 service request。短时检查验证了这些标准 stream 的 Humble/Jazzy 通信，采集前仍需在目标负载下验证时序。

验证离线采集配置
----------------

专用的 ``fr3_offline`` 环境使用带合成标记的数据包和现有离线动作模拟器，复用上述观测部件。Hydra 配置为合成数据记录选择原生环境、相机和输入设备字段。物理相机角色、运行参数、TCP/夹爪标定及 SpaceMouse 映射仍待确认。

在仓库根目录执行以下命令，可以验证配置，而无需构造环境、启动 worker 或打开设备：

.. code-block:: bash

   PYTHONPATH="$PWD" python -m toolkits.realworld_check.validate_fr3_config
   PYTHONPATH="$PWD" python -m toolkits.realworld_check.validate_fr3_config \
     env.eval.seed=19 env.eval.max_episode_steps=2

JSON 结果包含 ``offline_synthetic``、固定的 ``fr3_offline`` 注册项及 ``resources_opened: false``。不受支持的执行模式或设备映射会被拒绝。物理资格状态保持为 ``NOT_QUALIFIED``，命令授权为 ``NOT_GRANTED``，资格门禁为 ``NO_GO``。

主配置文件为 ``examples/embodiment/config/realworld_fr3_collect_data.yaml``。数值载荷包含六个 FR3 归一化位姿偏移通道和一个 Robotiq 归一化闭合通道。环境 step 要求带类型的请求封装及显式合成授权，拒绝通用 TCP 位姿和采集脚本的数值数组。现有 ``collect_real_data.py`` 仍固定使用通用环境和数值动作记录方式；请使用下述专用离线入口，请勿用此配置运行 ``collect_data.sh``。配置结论与离线生命周期测试命令见仓库中的 ``docs/fr3_fresh/milestone_3_rlinf_configuration.md``。

记录并重新加载合成 Episode
----------------------------------------

离线记录器使用 ``TrajectoryStep``、``TrajectoryAccumulator`` 和 ``TrajectoryReplayBuffer`` 保存原生 replay 张量。JSON 附属文件分别保留脚本生成的原始输入、带类型的请求、验证与准入事件、模拟提交与结果，以及关联观测。稠密动作张量表示已提交的模拟载荷，``physical_execution`` 始终为 false。

在仓库根目录执行以下命令，将两个各四步的 episode 写入新目录，再验证哈希、schema、动作来源、时间戳及原生 buffer 重新加载结果：

.. code-block:: bash

   PYTHONPATH="$PWD" python -m examples.embodiment.collect_fr3_offline \
     --output /tmp/fr3-milestone4-synthetic \
     runner.num_data_episodes=2 env.eval.max_episode_steps=4 env.eval.seed=7
   PYTHONPATH="$PWD" python -m examples.embodiment.collect_fr3_offline \
     --output /tmp/fr3-milestone4-synthetic --validate-only

原生 ``demos`` 目录包含两个轨迹文件及 replay 索引。每个 episode 的 JSON/JSONL 附属文件与 ``manifest.json`` 记录其合成来源。输出目录已存在时会拒绝写入；只有经过验证且状态为 ``COMPLETE`` 的 manifest 才会被接受，中断或写入失败会保留未完成状态。

Fixture 的源时间与到达时间使用显式合成时钟，实际本地接收、动作和记录器时间戳保留独立的主机时钟标识。动作前最近采样策略会拒绝缺失、过期、延迟或时间偏差超限的 fixture，且不使用旧帧替代。固定 seed 可复现数值载荷及逻辑时间，主机接收时间和 session 标识会随运行变化。零奖励表示未评分 fixture，模拟夹爪完成也不代表实际抓取成功。物理资格仍为 ``NOT_QUALIFIED``，授权仍为 ``NOT_GRANTED``，资格门禁保持 ``NO_GO``。Schema、失败处理及验证结果见 ``docs/fr3_fresh/milestone_4_data_collection_integration.md``。

评估并导出合成数据集
--------------------

使用数据集评估入口，在独立目录中生成质量报告，查看每个 episode 和 transition 的诊断、各模态检查结果及时间偏差分布。可选导出通过 RLinf 原生 LeRobot writer 完成，并将原生文件原样保存在随附归档中，以保留 frame table 无法表达的字段。每条命令都应选择新的输出目录：

.. code-block:: bash

   PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=/tmp/fr3-milestone5-hf-cache \
     python -m toolkits.realworld_check.validate_fr3_dataset \
     --dataset /tmp/fr3-milestone4-synthetic \
     --output /tmp/fr3-milestone5-validation --export
   PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=/tmp/fr3-milestone5-hf-cache \
     python -m toolkits.realworld_check.validate_fr3_dataset \
     --dataset /tmp/fr3-milestone4-synthetic \
     --output /tmp/fr3-milestone5-independent-reload \
     --reload-export /tmp/fr3-milestone5-validation/export

第一条命令写入 ``quality.json`` 和新的导出包。第二条命令在独立进程中加载该包，验证原生归档的哈希，再逐字段比较 LeRobot 数据与原始记录。仅需评估时，省略 ``--export``。无效记录会保留供审计；报告分别标记未完成的 episode、无效 schema、被拒绝的动作及损坏的文件。验证和重新加载均不修改原始数据集。

合成数据的源采样年龄和跨流时间偏差上限分别为 50 ms 和 20 ms。这些值是带显式时钟域的 fixture 验证限制，不是经过资格验证的物理时序预算。导出的任务文本表示未评分的合成流水线验证；原生轨迹没有保留可恢复的任务指令。完整请求、准入及模拟结果日志保存在原生归档中。

LeRobot 重新加载成功只确认导出子集及其归档的兼容性。OpenPI pi0.5 兼容性仍未获得资格确认：FR3 state 包含 42 个值，现有 realworld transform 要求 19 个值；checkpoint 的动作语义、归一化、相机角色及 horizon 处理还需核实。RECAP 还需要经过资格验证的结果、rollout/checkpoint 标识、干预记录及数据划分信息。审计 schema 与验证结果见 ``docs/fr3_fresh/milestone_5_dataset_validation.md``。物理执行仍为 ``NO_GO / NOT_GRANTED``。
