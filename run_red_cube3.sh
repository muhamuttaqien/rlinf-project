#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
python - <<'PY'
import gymnasium as gym
import mani_skill.envs
import custom_envs.pick_red_cube3
from pathlib import Path
import re

cfg = Path('examples/embodiment/config/maniskill_sac_flow_red_cube3.yaml')
env = gym.make('PickRedCube3-v1', num_envs=1, obs_mode='state',
               control_mode='pd_ee_delta_pos', sim_backend='gpu',
               reward_mode='dense', render_mode='rgb_array')
try:
    obs, info = env.reset(seed=42)
    dim = int(obs.shape[-1])
    assert env.action_space.shape[-1] == 4, env.action_space
    assert len(env.unwrapped.scene.actors) > 0
    assert hasattr(env.unwrapped, 'green_cube') and hasattr(env.unwrapped, 'blue_cube')
    print(f'PickRedCube3-v1 ready | state dimension: {dim} | action dimension: 4')
    action = env.action_space.sample()
    env.step(action)
finally:
    env.close()
text = cfg.read_text()
assert re.search(r'(?m)^\s*obs_dim: \d+\s*$', text)
text = re.sub(r'(?m)^(\s*obs_dim: )\d+(\s*)$', lambda m: m.group(1) + str(dim) + m.group(2), text)
cfg.write_text(text)
print('SAC-Flow observation dimension synchronized.')
PY
exec bash examples/embodiment/run_embodiment.sh maniskill_sac_flow_red_cube3
