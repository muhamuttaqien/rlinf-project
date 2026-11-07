"""Pure unit tests of the isolated ROS 2 shadow bridge. No rclpy imported."""

from dataclasses import replace

import pytest

from toolkits.fr3_policy_ros2.rlinf_fr3_policy_bridge.gate import (
    OWNER, SYNTHETIC_PROFILE, ShadowGate, ShadowRejected,
    make_synthetic_request,
)
from toolkits.fr3_policy_ros2.rlinf_fr3_policy_bridge.node import (
    assert_isolated_environment,
)


def gate_fixture():
    g = ShadowGate(SYNTHETIC_PROFILE, start_ns=1_000_000_000,
                   lease_ns=500_000_000)
    req = make_synthetic_request(sequence=1, observed_ns=1_000_000_000,
                                 issued_ns=1_010_000_000)
    return g, req


def test_valid_synthetic_preview_and_no_robot_actuation():
    g, req = gate_fixture()
    g.heartbeat(OWNER, now_ns=1_020_000_000)
    result = g.accept(req, now_ns=1_030_000_000)
    assert result["shadow_axes"][0] == pytest.approx(0.001/0.007)
    assert result["shadow_gripper_event"] == "open_preview"
    assert result["robot_commands_sent"] == 0
    assert result["real_controller_authority"] is False


def test_missing_heartbeat_rejected():
    g, req = gate_fixture()
    with pytest.raises(ShadowRejected, match="heartbeat"):
        g.accept(req, now_ns=1_030_000_000)
    assert g.state == "fault_latched"


def test_heartbeat_timeout_latches():
    g, req = gate_fixture()
    g.heartbeat(OWNER, now_ns=1_001_000_000)
    with pytest.raises(ShadowRejected, match="Heartbeat"):
        g.accept(req, now_ns=1_080_000_000)
    assert g.state == "fault_latched"


def test_wrong_owner_latches():
    g, _ = gate_fixture()
    with pytest.raises(ShadowRejected, match="owner"):
        g.heartbeat("any-other-publisher", now_ns=1_010_000_000)
    assert g.state == "fault_latched"


@pytest.mark.parametrize("bad", [
    {"origin": "stackcube_prediction"},
    {"schema": "unknown"},
    {"frame": "camera"},
    {"rotation_frame": "camera"},
    {"clock_domain": "workstation_clock"},
    {"owner": "bad"},
])
def test_incompatible_contracts_blocked(bad):
    g, req = gate_fixture()
    g.heartbeat(OWNER, now_ns=1_020_000_000)
    req.update(bad)
    with pytest.raises(ShadowRejected, match="contract"):
        g.accept(req, now_ns=1_030_000_000)


@pytest.mark.parametrize("bad_values", [
    [0.0]*6, [float("nan")] + [0.0]*6,
    [0.006, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.025, 0, 0],
    [0, 0, 0, 0, 0, 0, 0.2],
])
def test_bad_7d_vectors_blocked(bad_values):
    g, req = gate_fixture()
    g.heartbeat(OWNER, now_ns=1_020_000_000)
    req["values"] = bad_values
    with pytest.raises(ShadowRejected):
        g.accept(req, now_ns=1_030_000_000)
    assert g.state == "fault_latched"


def test_binary_close_preview_only():
    g, req = gate_fixture()
    req["values"][6] = 0.7929
    g.heartbeat(OWNER, now_ns=1_020_000_000)
    assert g.accept(req, now_ns=1_030_000_000)["shadow_gripper_event"] == "close_preview"


def test_replay_fault_latched():
    g, req = gate_fixture()
    g.heartbeat(OWNER, now_ns=1_020_000_000)
    g.accept(req, now_ns=1_030_000_000)
    with pytest.raises(ShadowRejected, match="replayed"):
        g.accept(req, now_ns=1_040_000_000)
    assert g.state == "fault_latched"


def test_stale_command_rejected():
    g, req = gate_fixture()
    g.heartbeat(OWNER, now_ns=1_150_000_000)
    with pytest.raises(ShadowRejected, match="Stale"):
        g.accept(req, now_ns=1_180_000_000)


def test_clock_regression_fault():
    g, req = gate_fixture()
    g.heartbeat(OWNER, now_ns=1_020_000_000)
    with pytest.raises(ShadowRejected, match="Non-monotonic"):
        g.check_timeout(now_ns=1_010_000_000)


def test_lease_expiry_fault():
    g, _ = gate_fixture()
    with pytest.raises(ShadowRejected, match="lease"):
        g.check_timeout(now_ns=1_510_000_000)


@pytest.mark.parametrize("env", [
    {}, {"RLINF_FR3_SHADOW_ONLY": "1", "ROS_DOMAIN_ID": "70"},
    {"RLINF_FR3_SHADOW_ONLY": "1", "ROS_DOMAIN_ID": "71"},
    {"RLINF_FR3_SHADOW_ONLY": "1", "ROS_DOMAIN_ID": "nonsense"},
])
def test_live_domain_or_missing_shadow_optin_rejected(env):
    with pytest.raises(PermissionError):
        assert_isolated_environment(env)


def test_isolated_domain_passes_preflight():
    assert assert_isolated_environment({
        "RLINF_FR3_SHADOW_ONLY": "1", "ROS_DOMAIN_ID": "220",
    }) == 220


def test_profile_limits_are_not_implicit():
    with pytest.raises(ValueError, match="positive"):
        replace(SYNTHETIC_PROFILE, translation_step_m=0)


def test_no_automatic_rearm_after_latched_fault():
    g, req = gate_fixture()
    with pytest.raises(ShadowRejected):
        g.accept(req, now_ns=1_030_000_000)
    with pytest.raises(ShadowRejected, match="fault latched"):
        g.heartbeat(OWNER, now_ns=1_040_000_000)
