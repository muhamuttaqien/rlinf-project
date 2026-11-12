"""Pure numerical tests: offline SpaceMouse request normalization."""
import numpy as np
import pytest

from toolkits.realworld_check.fr3_request_norm_stats import compute_from_arrays


def test_32d_quantiles_and_constant_channels_are_finite():
    state = np.zeros((32, 19), dtype=np.float64)
    state[:, 0] = np.linspace(-0.2, 0.2, 32)
    state[:, 6] = 1.0
    actions = np.zeros((32, 7), dtype=np.float64)
    actions[:, 0] = np.linspace(-0.5, 0.5, 32)
    actions[16:, -1] = 1
    stats = compute_from_arrays(state, actions)
    for key in ("state", "actions"):
        assert all(len(stats[key][field]) == 32
                   for field in ("mean", "std", "q01", "q99"))
        assert all(hi > lo for lo, hi in
                   zip(stats[key]["q01"], stats[key]["q99"]))
        assert all(np.isfinite(stats[key]["std"]))
    assert stats["actions"]["mean"][7:] == [0] * 25
    assert stats["actions"]["std"][7:] == [1] * 25
    assert stats["actions"]["q01"][7:] == [0] * 25
    assert stats["actions"]["q99"][7:] == [1] * 25


def test_fails_on_nonbinary_gripper_or_wrong_dimensions():
    states = np.zeros((4, 19))
    actions = np.zeros((4, 7))
    actions[:, 6] = 0.5
    with pytest.raises(ValueError, match="normalized"):
        compute_from_arrays(states, actions)
    with pytest.raises(ValueError, match="aligned"):
        compute_from_arrays(states[:3], np.zeros((4, 7)))


def test_list_encoded_lerobot_arrow_columns_are_detected(tmp_path):
    """LeRobot array features can have physical Parquet leaf names 'element'."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "sample.parquet"
    table = pa.table({
        "state": pa.array([[float(i)] * 19 for i in range(3)]),
        "actions": pa.array([[0.0] * 6 + [1.0] for _ in range(3)]),
    })
    pq.write_table(table, path)
    parquet = pq.ParquetFile(path)
    assert {"state", "actions"}.issubset(set(parquet.schema_arrow.names))
    batches = list(parquet.iter_batches(columns=["state", "actions"]))
    batch = batches[0]
    s = np.asarray(batch.column(batch.schema.get_field_index("state")).to_pylist())
    a = np.asarray(batch.column(batch.schema.get_field_index("actions")).to_pylist())
    assert s.shape == (3, 19)
    assert a.shape == (3, 7)
