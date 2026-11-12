"""Offline request-boundary tests. No ROS or robot access."""
import time
import pytest
from toolkits.realworld_check.fr3_pi05_ros_executor import validate_chunk


def test_accepts_valid_policy_chunk():
    now = time.monotonic_ns()
    chunk = [[0.5, -0.5, 0, 0, 0, 0, 1.04] for _ in range(10)]
    assert len(validate_chunk(chunk, observation_ns=now-100_000_000,
                              issued_ns=now-10_000_000, now_ns=now)) == 10


@pytest.mark.parametrize("bad", [
    [[0.] * 7] * 9,
    [[1.01, 0, 0, 0, 0, 0, 0]] * 10,
    [[0, 0, 0, 0, 0, 0, 1.5]] * 10,
    [[float("nan"), 0, 0, 0, 0, 0, 0]] * 10,
])
def test_rejects_bad_requests(bad):
    now = time.monotonic_ns()
    with pytest.raises(ValueError):
        validate_chunk(bad, observation_ns=now-10_000_000,
                       issued_ns=now-1_000_000, now_ns=now)


def test_rejects_old_observation():
    now = time.monotonic_ns()
    with pytest.raises(ValueError, match="Stale"):
        validate_chunk([[0.] * 7] * 10,
                       observation_ns=now-2_000_000_000,
                       issued_ns=now-1_000_000, now_ns=now)
