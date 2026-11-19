# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Serve the ORIGINAL Matplotlib FR3 episode visualizer through a web browser.

Unlike the independently styled FR3 episode dashboard, this launcher reuses
fr3_visualize.py from the recorded dataset folder verbatim, via Matplotlib's
WebAgg backend. Figure colors, 3x2 grid, plots, labels, buttons, slider,
keyboard shortcuts, and playback behavior therefore come from the *actual*
local visualizer.

Launch on the lab workstation:
    python -m toolkits.realworld_check.fr3_webagg_visualizer \
      --root /data1/Muttaqien/fr3_peg_100 --port 8767

From Windows, use an SSH tunnel (a separate terminal):
    ssh -N -L 8767:127.0.0.1:8767 muttaqien@10.17.94.130

Then visit http://localhost:8767/

SECURITY: WebAgg is bound ONLY to 127.0.0.1. Never expose raw robot recordings
via a public network interface or an unauthenticated reverse proxy.
This viewer reads recorded data. It does not connect to ROS or robot hardware.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/data1/Muttaqien/fr3_peg_100"))
    parser.add_argument("--script", type=Path, default=None,
                        help="Local fr3_visualize.py (default: <root>/fr3_visualize.py)")
    parser.add_argument("--episode", type=str, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be between 1024 and 65535")
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("Playback speed must be positive and finite")

    root = args.root.expanduser().resolve(strict=True)
    if not root.is_dir():
        parser.error(f"Dataset root is not a directory: {root}")
    script = (args.script if args.script is not None
              else root / "fr3_visualize.py").expanduser().resolve(strict=True)
    if not script.is_file():
        parser.error(f"Cannot find local visualizer: {script}")
    if args.script is None and script.parent != root:
        parser.error("Default visualizer must be inside the dataset root")

    import matplotlib

    matplotlib.use("WebAgg", force=True)
    matplotlib.rcParams["webagg.address"] = "127.0.0.1"
    matplotlib.rcParams["webagg.port"] = args.port
    matplotlib.rcParams["webagg.port_retries"] = 1  # Fail rather than surprise port.
    matplotlib.rcParams["webagg.open_in_browser"] = False

    spec = importlib.util.spec_from_file_location(
        "fr3_local_matplotlib_visualizer", script
    )
    if spec is None or spec.loader is None:
        parser.error(f"Cannot import visualizer: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for attr in ("Viewer", "episode_paths", "choose_index"):
        if not hasattr(module, attr):
            parser.error(f"Local visualizer is missing {attr}: {script}")

    paths = module.episode_paths(root)
    if not paths:
        parser.error(f"No recorded episodes found in {root}")
    start = module.choose_index(paths, args.episode)
    print(f"Original FR3 visualizer (WebAgg): http://127.0.0.1:{args.port}/",
          flush=True)
    print(f"Loaded {len(paths)} episodes from {root}; read-only", flush=True)
    print("Browser access requires SSH port forwarding; Ctrl+C to stop.",
          flush=True)

    # The unmodified viewer builds its Matplotlib figure, widgets, timer, and
    # event callbacks. WebAgg sends THAT figure to the browser via WebSockets.
    module.Viewer(root, paths, start, True, speed=args.speed)


if __name__ == "__main__":
    main()
