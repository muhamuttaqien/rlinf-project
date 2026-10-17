#!/usr/bin/env python3
"""Summarize per-episode RLinf evaluation diagnostics without requiring RLinf."""
import csv
import json
from collections import Counter
from pathlib import Path

out = Path("logs/evaluation-red-cube3-failures-seed")
files = sorted(out.glob("failure_episodes_worker_*.csv"))
if not files:
    raise SystemExit(f"No per-episode CSV found in {out}. Run evaluation first.")
rows = []
for path in files:
    with path.open(newline="") as fp:
        rows.extend(csv.DictReader(fp))
flags = ["failure_missed_red_grasp", "failure_wrong_cube", "failure_red_dropped", "failure_failed_lift"]
n = len(rows)
counts = {k: sum(float(row.get(k, 0)) > 0.5 for row in rows) for k in flags}
counts["success_at_end"] = sum(float(row.get("success_at_end", 0)) > 0.5 for row in rows)
counts["success_once"] = sum(float(row.get("success_once", 0)) > 0.5 for row in rows)
counts["ever_wrong_grasp"] = sum(float(row.get("ever_wrong_grasp", 0)) > 0.5 for row in rows)
result = {"total_episodes": n, "counts": counts,
          "rates": {k: round(v / n, 4) for k, v in counts.items()}}
(out / "failure_summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print(f"Saved: {out / 'failure_summary.json'}")
