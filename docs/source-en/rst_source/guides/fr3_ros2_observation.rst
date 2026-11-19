Read FR3 and Robotiq State Through ROS 2
========================================

Attach RLinf to existing FR3 and Robotiq 2F-85 state publishers with read-only
robot parts. You can inspect measured joints, pose, wrench, and gripper state
while the existing ROS controllers retain their hardware connections. Start
with a bounded check, then compose the parts and inspect their freshness metadata.

Run a Bounded Observation Check
-------------------------------

Use a ROS-compatible Python process for subscriptions. On the audited workstation,
Jazzy uses system Python 3.12 while RLinf uses Python 3.11. The adapter sources
``/opt/ros/jazzy/setup.bash`` in each child and explicitly selects domain 70 for
the arm and domain 71 for the gripper. Keep the existing state publishers running.

From the repository root, run:

.. code-block:: bash

   PYTHONPATH="$PWD" python toolkits/realworld_check/observe_fr3_ros2.py --duration 5

This attaches to state topics, waits for complete observations, samples for five
seconds, and disconnects in a ``finally`` block. The log reports incomplete
readings, maximum local receipt ages, source headers, and frame identifiers.
The check exits with status 1 if a sampled reading is incomplete. Use
``--ros-python`` and ``--ros-setup`` to select a different compatible installation;
``--max-age`` sets the receipt-age threshold, in seconds.

.. warning::

   Use these observation parts directly for live checks. Constructing the
   motion-capable ``FrankaEnv`` can move the robot during initialization and reset.

Compose Read-Only Robot Parts
-----------------------------

The native robot tree accepts observable parts independently of controllable
parts. Build ``FrankaRobot`` directly with these two parts to obtain the native
connection and observation lifecycle. This path bypasses its control backend
builder and requires neither an arm IP address nor a serial-port connection.

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

Construction records settings; ``connect()`` starts one subscriber process per
part. It confirms worker readiness, so allow discovery and initial delivery with
``wait_for_observation()``. That call returns an observation when the selected
fields are complete, raises ``TimeoutError`` for missing or stale state, and
raises ``ConnectionError`` when the worker fails. ``get_observation()`` returns
partial observations immediately after its bounded IPC request. ``disconnect()``
requests worker shutdown, closes IPC, reaps the process, and removes its temporary
logs; repeated disconnects are safe. Reconnect explicitly with disconnect/connect.

Both parts subclass ``RobotPart``. Their native ``reset()`` is a no-op, and they
expose no action fields or control methods. They compose with other observable
parts using the same ``PartGroup`` contract. They are deliberately narrower than
``BaseArm`` and ``BaseGripper``: those contracts include unavailable arm fields
and calibrated gripper opening widths respectively.

Interpret the Selected Measurements
-----------------------------------

Each observation contains ``state`` and ``ros2``. Joint values are selected by
name; extra joints never change the FR3 order. Pose quaternions retain the ROS
``x,y,z,w`` order. The selected topics are fixed to measured state:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Stream / Domain
     - Topic and Type
     - State Fields
   * - joints / 70
     - ``/franka/joint_states`` (``sensor_msgs/msg/JointState``)
     - ``arm_joint_position``, ``arm_joint_velocity``, ``arm_joint_effort``; ``fr3_joint1`` through ``fr3_joint7``; rad, rad/s, N m
   * - pose / 70
     - ``/franka_robot_state_broadcaster/current_pose`` (``geometry_msgs/msg/PoseStamped``)
     - ``tcp_pose``; position in metres, quaternion x,y,z,w
   * - wrench_base / 70
     - ``/franka_robot_state_broadcaster/external_wrench_in_base_frame`` (``geometry_msgs/msg/WrenchStamped``)
     - ``base_force``, ``base_torque``; N, N m
   * - wrench_stiffness / 70
     - ``/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame`` (``geometry_msgs/msg/WrenchStamped``)
     - ``tcp_force``, ``tcp_torque``; N, N m; original header retained
   * - knuckle / 71
     - ``/joint_states`` (``sensor_msgs/msg/JointState``), ``robotiq_85_left_knuckle_joint``
     - ``knuckle_position_rad``
   * - normalized / 71
     - ``/joint_states`` (``sensor_msgs/msg/JointState``), ``robotiq_normalized_joint``
     - ``normalized_closure``; 0=open, 1=closed

Wrenches are external-wrench estimates from the broadcaster. Their frame headers
remain authoritative: the audited live stiffness-frame wrench reported
``fr3_link2``. The adapter neither transforms it nor equates ``base`` with
``base_link``. It validates dimensions, finite values, unique joint names, normalized
closure range, and unit quaternions without silently normalizing them.

Missing measurements are ``None``. ``ros2.unavailable_fields`` names fields outside
the selected contract: measured TCP velocity and Jacobian for the arm; calibrated
opening width, measured velocity, grip force, and custom typed status for the
gripper. The standard gripper stream's driver-supplied zero velocity is excluded.
A raw knuckle angle and normalized closure are independent readings, not interchangeable
width measurements. The selected standard-message subset works without Franka or
Robotiq custom message packages in the workstation ROS installation.

Check Connection, Freshness, and Errors
---------------------------------------

Use the metadata to decide whether a reading is usable. ``is_connected`` follows
the native open-handle contract. ``ros2.bridge_alive`` reports worker liveness,
while each stream's ``connected`` flag reflects matching publishers at the latest
0.5-second graph check. Publisher discovery alone does not establish data delivery.

``ros2.complete`` requires every selected stream to have a valid sample, matching
publishers, and receipt age within ``max_age``, plus every selected field to be
present. Completeness describes this subset, not the full native arm or gripper
control state. Inspect ``missing_streams``, ``stale_streams``, ``invalid_streams``,
``missing_fields``, the top-level ``error``, and each stream's ``error`` or
``graph_error`` for the reason a reading is incomplete. A malformed message clears
that stream's sample; a stale valid sample remains available with its stale flag.

Under ``ros2.streams[stream]["sample"]``, ``header`` preserves ROS ``sec``,
``nanosec`` and ``frame_id``. ``received_monotonic_ns`` records local callback
receipt separately. ``age_s`` uses that monotonic clock; it measures receipt
freshness, not acquisition age or clock synchronization. Repeated source stamps
and fresh publications do not establish fresh serial measurements. These latest
samples are independently received and are not a synchronized robot snapshot.

Each worker uses best-effort, volatile, keep-last-depth-5 subscriptions after
checking the advertised publisher type. It retains one decoded sample per stream.
A private Unix ``SOCK_SEQPACKET`` socket carries version-1 JSON requests and
responses, limited to 64 KiB per packet. Only ``snapshot`` and ``shutdown`` requests
are accepted; state is sent on demand. Parent stderr retention is bounded to
16 KiB. Startup and IPC timeouts default to five seconds; failure closes the
session and reports diagnostics. The worker also exits on parent EOF.

The worker imports ROS packages only in its selected interpreter and never opens
a hardware-control connection. Its only application ROS endpoints are state
subscriptions. Standard rclpy infrastructure may advertise parameter events and
local type-description services; the adapter creates no command publisher,
action client, or service client. Mixed Humble/Jazzy communication was verified
for these standard streams in short checks; qualify timing under your intended
workload before using observations for collection.
