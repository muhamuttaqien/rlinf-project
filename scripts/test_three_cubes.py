import gymnasium as gym
import numpy as np
import mani_skill.envs

import custom_envs.pick_red_cube3  # Registers custom task

env = gym.make(
    "PickRedCube3-v1",
    obs_mode="state",
    control_mode="pd_ee_delta_pos",
    num_envs=1,
    render_mode="rgb_array",
    reward_mode="dense",
)

obs, info = env.reset(seed=42)

print("Environment:", env.spec.id)
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)

image = env.render()
if hasattr(image, "cpu"):
    image = image.cpu().numpy()
if image.ndim == 4:
    image = image[0]

from PIL import Image
Image.fromarray(np.asarray(image).astype(np.uint8)).save(
    "three_cubes.png"
)
print("Saved: three_cubes.png")

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print("Reward:", reward)
print("Success:", info["success"])

env.close()
