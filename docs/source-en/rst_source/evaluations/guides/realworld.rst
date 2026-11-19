Real-World Evaluation
=====================

RLinf supports evaluating and deploying VLA policies on Franka arms, including Bin-relocation (pick-and-place) tasks and a YAML-configurable generic environment (``FrankaEnv-v1``) for custom tasks.

Related training docs: :doc:`../../examples/embodied/franka_pi0_sft_deploy`, :doc:`../../examples/embodied/franka`, :doc:`../../examples/embodied/sft_dreamzero`

Environment Setup
-----------------

**Hardware**

- Franka Emika Panda arm + Intel RealSense cameras (default)
- One **GPU node** (training / rollout) and one **robot control node** (direct Franka and camera access, no GPU required)
- All nodes on the same LAN; the arm only needs to reach the control node

For ZED cameras or Robotiq grippers, see :doc:`../../examples/embodied/franka_zed_robotiq`.

**Dependencies**

Install dependencies separately on the control node and the GPU node:

.. code-block:: bash

   # Robot control node (Franka + cameras)
   bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

.. code-block:: bash

   # GPU / rollout node (π₀ evaluation)
   bash requirements/install.sh embodied --model openpi --env franka
   source .venv/bin/activate

The control node uses the default Franky backend. To drive the arm through ROS instead, follow the Legacy ROS Backend (Optional) section in :doc:`../../examples/embodied/franka`.

DreamZero real-robot evaluation also requires DreamZero dependencies on the GPU node; see :doc:`../../examples/embodied/sft_dreamzero`.

**Node topology**

Real-robot evaluation typically uses a **1 GPU node + 1 Franka control node** heterogeneous layout:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - ``RLINF_NODE_RANK``
     - Role
     - Notes
   * - rank 0
     - GPU / head
     - Runs ``rollout``; submit ``run_eval.sh`` on this node
   * - rank 1
     - Robot control
     - Runs ``env`` workers with direct Franka and camera access

``realworld_pnp_eval.yaml`` and ``realworld_pnp_eval_dreamzero.yaml`` use this two-node layout; ``realworld_eval.yaml`` (custom tasks) is a **single-node** layout with both ``env`` and ``rollout`` on the Franka node.

For full Ray cluster setup, firmware versions, and libfranka compatibility, see :doc:`../../examples/embodied/franka` and :doc:`../../guides/realworld_robot`.

Example Configs
---------------

The following examples live under ``evaluations/realworld/``:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Config file
     - Task
     - Model
   * - ``realworld_pnp_eval.yaml``
     - Bin-relocation (PnP)
     - π₀
   * - ``realworld_pnp_eval_dreamzero.yaml``
     - Bin-relocation (PnP)
     - DreamZero
   * - ``realworld_eval.yaml``
     - Custom task (``FrankaEnv-v1``)
     - π₀

If ``evaluations/realworld/<config>.yaml`` is missing, ``run_eval.sh`` falls back to the same name under ``examples/embodiment/config/`` (set ``runner.task_type: embodied_eval`` and ``runner.only_eval: True``). See :doc:`../reference/cli`.
Dual Franka deployment currently uses this fallback path with ``realworld_eval_dual_franka``.

Pre-flight Checks
-----------------

Before running evaluation, complete these checks in order:

1. **Camera connection** (control node):

   .. code-block:: bash

      python -m toolkits.realworld_check.test_franka_camera

   Record the camera serials and set ``env.eval.override_cfg.camera_serials``.

2. **Target pose** (PnP tasks, control node):

   .. code-block:: bash

      export FRANKA_ROBOT_IP=<robot_ip>
      python -m toolkits.realworld_check.test_franka_controller
      # Enter getpos_euler to read the target end-effector pose

   Set the result in ``env.eval.override_cfg.target_ee_pose``. For PnP, this pose is the lowest point in the motion workspace and is used for success checking and workspace clipping; see :doc:`../../examples/embodied/franka_pi0_sft_deploy`.

3. **Ray cluster** (all nodes):

   .. code-block:: bash

      ray status

   Both the GPU node and the Franka node should appear online.

4. **Dummy mode (optional):** See ``examples/embodiment/config/realworld_dummy_franka_sac_cnn.yaml`` and set ``is_dummy: True`` in ``override_cfg`` to verify cluster wiring without real robot motion.

.. warning::

   Verify workspace limits (``ee_pose_limit_min`` / ``ee_pose_limit_max``) and the emergency stop before real-robot evaluation. Use a small ``env.eval.rollout_epoch`` on the first run.

Starting the Ray Cluster
------------------------

On **each node**, before ``ray start``, align environment variables (you can use ``ray_utils/realworld/setup_before_ray.sh``):

.. code-block:: bash

   source ray_utils/realworld/setup_before_ray.sh
   export RLINF_NODE_RANK=<0|1>          # unique within the cluster
   # On multi-NIC hosts, pin the reachable interface:
   # export RLINF_COMM_NET_DEVICES=<network_interface>

On the control node, also activate the Franka environment (unless the setup script already does):

.. code-block:: bash

   source .venv/bin/activate

Then start Ray (let ``<head_ip>`` be the head node IP):

.. code-block:: bash

   # GPU node (rank 0, head)
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<head_ip>

   # Franka control node (rank 1)
   export RLINF_NODE_RANK=1
   ray start --address=<head_ip>:6379

.. important::

   ``ray start`` freezes the Python interpreter and environment variables at launch time. Complete venv and ``PYTHONPATH`` setup on every node **before** starting Ray.

End-to-End Workflow (PnP / π₀)
--------------------------------

**Step 1: Start the Ray cluster**

Start Ray on all nodes as above and confirm both nodes appear in ``ray status``.

**Step 2: Prepare the model**

π₀ PnP evaluation requires:

- ``rollout.model.model_path``: Pi0 base model directory containing ``<repo_id>/norm_stats.json`` (generated during SFT prep; see :doc:`../../examples/embodied/franka_pi0_sft_deploy`)
- ``runner.ckpt_path``: SFT-exported ``full_weights.pt``

**Step 3: Edit the config**

Update ``evaluations/realworld/realworld_pnp_eval.yaml`` at minimum:

.. code-block:: yaml

   cluster:
     node_groups:
       - label: "4090"
         node_ranks: 0          # GPU node rank
       - label: franka
         node_ranks: 1          # Franka control node rank
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 1

   runner:
     ckpt_path: /path/to/full_weights.pt

   env:
     eval:
       rollout_epoch: 20
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
         camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
         task_description: "pick up the object and place it into the container"

   rollout:
     model:
       model_path: /path/to/pi0-model
       openpi:
         config_name: "pi0_realworld"

``node_ranks`` and ``component_placement`` must match the actual ``RLINF_NODE_RANK`` on each machine.

**Step 4: Launch evaluation**

On the **GPU / head node**:

.. code-block:: bash

   bash evaluations/run_eval.sh realworld_pnp_eval

You can also override fields via Hydra without editing the YAML:

.. code-block:: bash

   bash evaluations/run_eval.sh realworld_pnp_eval \
     rollout.model.model_path=/path/to/pi0-model \
     runner.ckpt_path=/path/to/full_weights.pt

**Step 5: Check results**

The policy runs with ``runner.only_eval: True``; task metrics appear in the terminal. Logs are described in :doc:`../reference/results`.

.. _realworld-eval-config:

Evaluation Config Reference
---------------------------

The following applies to ``realworld_pnp_eval.yaml``; custom-task evaluation is covered below, and DreamZero in :doc:`../../examples/embodied/sft_dreamzero`.

Required fields
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Field
     - Location
     - Notes
   * - ``robot_ip``
     - ``cluster.node_groups[].hardware.configs``
     - Franka robot IP
   * - ``node_ranks`` / ``component_placement``
     - ``cluster``
     - Must match ``RLINF_NODE_RANK``; env on Franka node, rollout on GPU node
   * - ``target_ee_pose``
     - ``env.eval.override_cfg``
     - PnP target pose ``[x,y,z,rx,ry,rz]``; affects success checks and motion clipping
   * - ``camera_serials``
     - ``env.eval.override_cfg``
     - RealSense serial list (**not** a ``node_groups`` field)
   * - ``task_description``
     - ``env.eval.override_cfg``
     - Language instruction; must match SFT training
   * - ``model_path``
     - ``rollout.model``
     - Pi0 base model directory (with ``norm_stats.json``)
   * - ``ckpt_path``
     - ``runner``
     - SFT checkpoint (``full_weights.pt``)
   * - ``config_name``
     - ``rollout.model.openpi``
     - ``"pi0_realworld"`` for PnP

Key ``env.eval`` fields
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Notes
   * - ``rollout_epoch``
     - Number of evaluation rounds; default 20
   * - ``max_episode_steps``
     - Max steps per trajectory; default 200
   * - ``max_steps_per_rollout_epoch``
     - Steps per rollout round; **must be divisible by** ``rollout.model.num_action_chunks`` (default 10 for PnP)
   * - ``total_num_envs``
     - Parallel env count; typically 1 on real hardware
   * - ``teleop``
     - Device that takes over from the policy; usually ``none`` for eval

``run_eval.sh`` behavior
~~~~~~~~~~~~~~~~~~~~~~~~

- The ``realworld`` benchmark **does not** call ``setup_sim_env`` (no simulation env vars needed)
- Logs go to ``logs/<timestamp>-<config_name>/eval_embodiment.log``
- Submit the eval entry script on the **head (GPU) node**

For DreamZero real-robot evaluation (``realworld_pnp_eval_dreamzero.yaml``), see :doc:`../../examples/embodied/sft_dreamzero`.

Custom Real-Robot Task Evaluation
---------------------------------

This section is for **tasks you define yourself**, not the built-in Bin-relocation (PnP) task above.

RLinf ships a generic real-robot environment ``FrankaEnv-v1``: set ``task_description``, goal/reset poses, and workspace limits in YAML—no new Python env class required. It is commonly used to **deploy policies trained with supervised fine-tuning (SFT)** on your own demonstration data (see :doc:`../../examples/embodied/franka_pi0_sft_deploy`). The env template is ``examples/embodiment/config/env/realworld_franka_sft_env.yaml``; the eval config is ``evaluations/realworld/realworld_eval.yaml``.

**Node topology**

``realworld_eval.yaml`` is **single-node**: both ``env`` and ``rollout`` run on the Franka node (``num_nodes: 1``), suitable when GPU and control share one machine.

Key ``override_cfg`` fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   override_cfg:
     task_description: "pick up the object and place it into the container"
     target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]   # goal pose
     reset_ee_pose:  [0.5, 0.0, 0.2, -3.14, 0.0, 0.0]    # reset pose (above goal)
     max_num_steps: 300
     reward_threshold: [0.01, 0.01, 0.01, 0.2, 0.2, 0.2]
     action_scale: [1.0, 1.0, 1.0]
     ee_pose_limit_min: [0.4, -0.2, 0.05, -3.64, -0.5, -0.5]
     ee_pose_limit_max: [0.6,  0.2, 0.35, -2.64,  0.5,  0.5]

**Launch evaluation**

Replace ``ROBOT_IP`` and ``MODEL_PATH``, then run:

.. code-block:: bash

   bash evaluations/run_eval.sh realworld_eval

For data collection, SFT training, and deployment on custom tasks, see :doc:`../../examples/embodied/franka_pi0_sft_deploy`.

Dual Franka Deployment
----------------------

Dual Franka SFT deployment reuses the unified evaluation launcher with the
fallback config ``examples/embodiment/config/realworld_eval_dual_franka.yaml``.
Set ``rollout.model.model_path`` to the staged checkpoint directory and
``actor.model.openpi_data.repo_id`` to the repo id that contains
``norm_stats.json``.

.. code-block:: bash

   bash evaluations/run_eval.sh realworld_eval_dual_franka \
       rollout.model.model_path=/path/to/deploy/global_step_<N> \
       actor.model.openpi_data.repo_id=<repo_id>/tcp_rot6d_v1 \
       env.eval.override_cfg.task_description="handover the object"

For the full collection, SFT, checkpoint staging, and pedal-control workflow,
see :doc:`../../examples/embodied/dual_franka`.

Native FR3 Request Policy Evaluation
------------------------------------

Use ``FR3RequestEnv-v1`` for the 19D, two-camera OpenPI request-SFT checkpoint on an FR3 whose NUC ROS 2 controller is already active. The native environment attaches through ``FrankaRobot`` and a ROS 2 request arm; reset reads the current state. Run one supervised three-second episode with ``evaluations/realworld/realworld_fr3_pi05_request_eval.yaml``. This deployment-specific config pins the existing 3,000-step checkpoint, request normalization statistics, camera serials, and physical GPU 1.

Activate the ML environment, keep the matching Jazzy message overlay available, and follow the operator procedure in ``docs/experiments/openpi_recap_franka/pi05_native_evaluation.md``. On the configured workstation, launch:

.. code-block:: bash

   source .venv/bin/activate
   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.590.48.01:/usr/lib/x86_64-linux-gnu/libcuda.so.590.48.01 \
   RAY_ADDRESS=local \
   bash evaluations/run_eval.sh realworld realworld_fr3_pi05_request_eval

This selects process-local NVIDIA libraries, initializes isolated native evaluation workers, and asks for ``RUN FR3 3S`` before enabling requests. Each arm sample keeps its trained normalized Twist order and signs at scaling 1.0. Samples run at 10 Hz within each horizon and refresh at 100 Hz; inference gaps become neutral. A separate ROS process enforces the three-second wall-clock deadline and fresh-state/ownership checks. Use Ctrl-C or the printed session's ``STOP`` file to request neutral, and keep an operator at the physical emergency stop. Neutral holds the existing NUC joint reference.

Before the operator prompt, the rollout worker performs two discarded predictions with synthetic inputs and restores its random-generator state. Reset then acquires the rollout observation after confirmation. The one-second observation-to-chunk freshness check remains active. Use ``env_timing.jsonl`` and ``policy_timing.jsonl`` in the printed session directory to separate acquisition, transport, inference, and action-admission latency.

Inspect the normal evaluation logs and the session's ``ros_evidence.jsonl`` to compare policy requests, transmitted Twist values, controller diagnostics, and measured pose. The measured pose convention matches training; physical tool-tip calibration and peg-insertion success are separate observations. Use a new explicit operator confirmation for each run.

For offline preparation of the existing 100 FR3 peg-insertion demonstrations, use ``toolkits/realworld_check/fr3_frequency_dataset.py`` and the audit report ``docs/experiments/openpi_recap_franka/fr3_frequency_phase1.md``. The separate 10 Hz dataset preserves episode identities, averages held normalized requests over exact 100 ms intervals, and uses training-only statistics. Review ``examples/sft/config/realworld_sft_openpi_fr3_peg_request_10hz_phase1.yaml`` as a proposed SFT configuration; dataset preparation does not launch training or evaluation.

Viewing Results
---------------

- **Terminal metrics:** task success rate, return, etc. (exact fields vary by environment)
- **Logs:** ``logs/<timestamp>-<config_name>/eval_embodiment.log``
- **Videos:** when ``env.eval.video_cfg.save_video: True``, videos go to ``video_base_dir`` or ``<log_path>/video/eval/``

See :doc:`../reference/results` for details.

FAQ
---

- **Safety:** Verify workspace limits and emergency stop before evaluation; use a small ``rollout_epoch`` on the first run.
- **Node topology:** ``env`` workers must run on nodes with direct Franka access; ``node_ranks`` must match ``RLINF_NODE_RANK``. PnP and Dual Franka use two nodes; custom-task eval uses one.
- **Cameras not found:** Run ``python -m toolkits.realworld_check.test_franka_camera`` on the control node and verify ``camera_serials``.
- **Abnormal actions:** Check that ``norm_stats.json`` is under ``model_path/<repo_id>/`` and that ``openpi.config_name`` matches training.
- **Ray shows only one node:** Check firewall rules, ``RLINF_COMM_NET_DEVICES``, and that the head IP is reachable from other nodes.
- **Step-count errors:** Ensure ``max_steps_per_rollout_epoch`` is divisible by ``num_action_chunks``.
- **ZED / Robotiq:** See :doc:`../../examples/embodied/franka_zed_robotiq`.
