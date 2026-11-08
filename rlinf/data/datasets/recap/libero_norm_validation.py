"""Validate LIBERO normalization layouts before expensive distributed training.

LIBERO observations have 8 state values and 7 action values. Some OpenPI
checkpoints store the statistics in 32-wide tensors padded with zeros.
The OpenPI Normalize transform slices the statistics to the raw data width.
"""
from __future__ import annotations

import numpy as np


def validate_libero_norm_stats(norm_stats) -> tuple[int, int]:
    """Accept native (8, 7) or zero-padded (32, 32) LIBERO statistics.

    Reject mixed/partial padding, nonzero padded channels, missing statistics,
    and malformed arrays. This must be called on the data-config stats, not
    just a checkpoint filename.
    """
    widths = {}
    for key, real_width in (("state", 8), ("actions", 7)):
        item = norm_stats.get(key) if norm_stats else None
        if item is None:
            raise ValueError(f"LIBERO norm stats missing required '{key}' entry")
        fields = {}
        for stat_name in ("mean", "std", "q01", "q99"):
            val = getattr(item, stat_name, None)
            if val is None:
                raise ValueError(f"LIBERO norm stats '{key}.{stat_name}' is missing")
            arr = np.asarray(val)
            if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.number):
                raise ValueError(f"LIBERO norm stats '{key}.{stat_name}' must be a numeric 1-D array")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"LIBERO norm stats '{key}.{stat_name}' contains NaN or Inf")
            fields[stat_name] = arr

        width = len(fields["mean"])
        if any(len(arr) != width for arr in fields.values()):
            raise ValueError(f"LIBERO norm stats '{key}' has inconsistent field widths")
        if width not in (real_width, 32):
            raise ValueError(
                f"LIBERO normalization mismatch for {key}: width={width}; "
                f"expected {real_width} or zero-padded 32"
            )
        if width == 32 and any(
            not np.allclose(arr[real_width:], 0.0, rtol=0, atol=1e-6)
            for arr in fields.values()
        ):
            raise ValueError(
                f"LIBERO norm stats '{key}' has nonzero values in padded "
                f"dimensions {real_width}:32"
            )
        widths[key] = width

    widths_pair = (widths["state"], widths["actions"])
    if widths_pair not in ((8, 7), (32, 32)):
        raise ValueError(
            f"LIBERO normalization mixed-width layout {widths_pair}; "
            "expected (8, 7) or (32, 32)"
        )
    return widths_pair
