# Milestone 4C.2T.2 — Offline, nonblocking controller-target telemetry design

**Status:** Standalone C++17 SPSC queue prototype and synthetic tests implemented. **Not integrated, built, loaded or deployed on the NUC.** The deployed 1 kHz `JointImpedanceIKController` and Robotiq bridge are unchanged. This is not a 7D training-action exporter.

## Architecture

```text
Controller 1 kHz update (proposed future safety-reviewed integration)
  compute gates and IK target -> rate-limit q_desired_ -> torque update
  snapshot values after reference update (no heap, lock, I/O, ROS publish)
  fixed-size SPSC try_push(record) -> on full DROP NEWEST, count drop
                                            |
                                            v
                              non-RT single drain thread
                       validate/serialize/session metadata/disk
```

File: `toolkits/realworld_check/fr3_rt_telemetry_queue.hpp`. It contains no ROS imports, hardware control APIs, allocator calls, mutexes, logging, or filesystem writes. `SpscQueue<T, Capacity>` requires a power-of-two capacity, a trivially copyable record, one producer and exactly one consumer. Atomic release/acquire publication protects sample ownership. The full-queue branch increments an atomic drop counter rather than waiting. The queue never overwrites an unread record. Its runtime memory allocation is bounded to the fixed `std::array` constructed before either thread starts.

`ArmTickSnapshot` reserves fields for sample sequence, controller tick sequence, NUC monotonic timestamps (activation, input receive, sampled state, IK solve, target state/command), six normalized SpaceMouse axes, seven measured joints, seven IK target joints, seven q_desired_ values **before and after** the RT update, seven velocity limits, step duration and timeout, real gate decisions, and cumulative drops read before attempting enqueue. Values are captured as a **snapshot**, never a pointer to mutable RT state. If the target is invalid, its numeric array must be ignored and the valid bit unset; do not mistake zero-filled unused fields for valid data.

## Data semantics and loss accounting

- The RT producer assigns a monotonically increasing *attempted sample* `seq`, including samples rejected by a full buffer. It also records `controller_tick_seq` so later deliberate decimation is distinct from queue loss. Drop-newest means gaps in recorded sequences can signal loss; first and trailing drops also require cumulative counters.
- The draining worker must be the sole consumer, must write the count of attempted/saved/dropped samples in session metadata, and must flush/close on a controlled shutdown. Disk I/O, JSONL serialization, clock-domain calibration and ROS diagnostics belong exclusively outside the RT callback.
- A single-producer/single-consumer queue is NOT appropriate for merging arm and Robotiq callback threads. Gripper action-goal telemetry needs a separate non-RT queue/source, carrying goal UUID, send/accept/result times, requested target and measured response, with a shared documented NUC clock domain.
- The existing 4C.2T.1 Python validator (`fr3_controller_target_telemetry_proposal_v1`) expects positive timestamps and uses `null` for invalid IK targets. This C++ transport snapshot allows zero for unavailable timestamps; the future *non-RT serializer* must explicitly translate unknown/invalid fields or revise/version the JSONL schema. **No serializer is implemented here**, and none may invent timestamps or valid IK targets.
- Do not replace samples with robot-state broadcaster `last_desired_pose` or `desired_joint_states`: neither was proven equivalent to internal IK target and post-limit `q_desired_`. These are proposed capture fields, not observed values.

## RT integration review gates (not authorized yet)

1. Verify the exact deployed controller API and placement: capture both `q_desired_` before and after rate limiting, the current IK target, measured q, actual gate results, and the monotonic tick stamp **without changing command execution**. The snapshot must not invoke IK, ROS calls or additional command calculations.
2. Preallocate the queue and consumer thread at controller initialization. Avoid lifecycle races and require deterministic stopping: producer stopped before the queue is destroyed; consumer drains before join; never block in `update()`.
3. Establish the NUC clock basis for all arm and gripper events, record clock resets/activation changes, and separately calibrate workstation/camera clocks. Do not compare raw ROS wall-time stamps with NUC monotonic nanoseconds.
4. Choose capture rate and queue capacity from measured sustained throughput. A 1 kHz recording rate is a candidate, **not** a measured guarantee. Size buffers for worst-case consumer stalls; document allowable drop threshold and where to escalate telemetry degradation without modifying control behavior.
5. Measure worst-case/percentile overhead, control-loop jitter, memory use, CPU affinity/scheduling, contention, sustained disk stalls, shutdown behavior and drop rates on representative *non-actuating* staging infrastructure. Run sanitizers/stress tests and review with the lab. Lock-free atomics alone do not establish deterministic real-time scheduling.
6. Require controller code review, lab safety sign-off, staged deployment/rollback, and a separately supervised physical validation before enabling telemetry on an active robot. **Never restart or patch the running controller as part of this offline milestone.**

## Offline synthetic verification

Source: `tests/unit_tests/fr3_rt_telemetry_queue_native.cpp`
Pytest wrapper: `tests/unit_tests/test_fr3_rt_telemetry_queue.py`

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD" python -m pytest -q tests/unit_tests/test_fr3_rt_telemetry_queue.py
```

Tests compile C++17 with `g++` in a temporary directory; no ROS installation, robot hardware, user datasets or elevated permissions required. The standalone checks cover FIFO/overflow/drop accounting, correct immutable snapshot copy, and concurrent monotonic ordering with 200,000 attempts. In an isolated Linux environment, these three scenarios passed; AddressSanitizer/UndefinedBehaviorSanitizer checks also passed and an additional 20 concurrent runs passed. **These are software correctness checks, not NUC timing measurements or safety verification.** Workstation test output is still pending.

## Milestone acceptance boundary

**4C.2T.2 offline prototype:** implementation complete; **3/3 workstation pytest scenarios passed in 0.58 s on November 7, 2026** (Python 3.12.3). **4C.2T.3 / live instrumentation:** not approved or implemented. **4C.2 production action contract:** blocked on accurate controller-reference telemetry, frame/tool calibration, explicit action interval, action-source provenance, cross-domain/host clock mapping and safety-reviewed collection. Existing stationary raw episodes remain non-training data.
