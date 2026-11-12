"""Check diagnostic staging against the actual committed FR3 controller patch.

The reference patch holds the controller revision inspected on the NUC.
This is an offline source-anchor compatibility test, NOT a C++ compile,
verification of the NUC's local source, or physical execution approval.
"""

from pathlib import Path

from toolkits.realworld_check.fr3_controller_diagnostic_stage import (
    HEADER_REL, SOURCE_REL, instrument,
)

PATCH = (
    Path(__file__).resolve().parents[2]
    / "docs/experiments/openpi_recap_franka/fr3_startup_fix/franka_arm_controllers.patch"
)


def new_file_from_patch(path: Path) -> str:
    """Reconstruct new-side source fragments from unified-diff hunks."""
    lines = PATCH.read_text(encoding="utf-8").splitlines()
    desired = "b/" + str(path)
    collecting = False
    in_hunk = False
    fragments = []
    for line in lines:
        if line.startswith("+++ "):
            collecting = (line[4:] == desired)
            in_hunk = False
            continue
        if not collecting:
            continue
        if line.startswith("@@"):
            in_hunk = True
            continue
        if line.startswith("--- "):
            in_hunk = False
            continue
        if in_hunk and line[:1] in ("+", " "):
            fragments.append(line[1:])
    assert fragments, f"No new-side reference patch content for {path}"
    return "\n".join(fragments)


def test_diagnostics_match_inspected_controller_patch():
    header = new_file_from_patch(HEADER_REL)
    source = new_file_from_patch(SOURCE_REL)
    new_header, new_source = instrument(header, source)

    # All anchors have to match the controller revision in the committed patch.
    assert "diag_rx_" in new_header
    assert "diag_ik_valid_" in new_header
    assert "FR3_DIAG_V1 rx=%llu" in new_source

    # The same normalized request gains and motor-control write survive.
    for unchanged in (
        "0.007 * linear.x()",
        "0.03 * a[3]",
        "command_interfaces_[command_indices_[i]].set_value(torque[i]);",
        "const double step = velocity_limits_[i] * dt;",
        "q_desired_[i] += std::clamp(target.q[i] - q_desired_[i], -step, step);",
        "target.valid = ik_solver_->CartToJnt(q_init_, pose, q_result_) >= 0;",
    ):
        assert unchanged in source
        assert unchanged in new_source

    # The source patch adds observation only, never a new ROS motor publisher.
    assert "create_publisher" not in (new_source.replace(source, ""))
    assert "diag_ref_advanced_ticks_.fetch_add" in new_source
    assert "diag_effort_write_ticks_.fetch_add" in new_source
