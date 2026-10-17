#!/usr/bin/env python3
"""Keyboard teleoperation for the custom ManiSkill PickRedCube3-v1 task.

Run from the rlinf-project root:
    PYTHONPATH="$PWD" python scripts/control_franka.py

Click the OpenCV window, then press a key. This is incremental, step-by-step
teleoperation (not held-key velocity control), so commands are predictable.
"""

import argparse

import sapien
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env

import cv2
import gymnasium as gym
import numpy as np
import torch
import mani_skill.envs  # Registers built-in tasks
from custom_envs.pick_red_cube3 import PickRedCube3Env


@register_env("PickRedCube3Teleop-v1", max_episode_steps=100)
class PickRedCube3TeleopEnv(PickRedCube3Env):
    """Teleop-only environment with a camera mounted on Panda hand."""

    @property
    def _default_sensor_configs(self):
        original = super()._default_sensor_configs
        cameras = list(original) if isinstance(original, (list, tuple)) else [original]
        cameras.append(
            CameraConfig(
                uid="wrist_camera",
                # Local transform relative to panda_hand (tweak if desired).
                pose=sapien.Pose(
                    p=[0.0465, -0.0200, 0.0360],
                    q=[0.0, 0.70710678, 0.0, 0.70710678],
                ),
                width=640,
                height=480,
                fov=np.pi / 2,
                near=0.01,
                far=10.0,
                mount=self.agent.robot.links_map["panda_hand"],
            )
        )
        return cameras

TITLE = "ManiSkill - Franka Panda - Red Target (WASD / IJKL)"

# Commands are normalized to the [-1, 1] action range. The ManiSkill
# controller determines the physical motion per control step.
TRANSLATION = 0.45
ROTATION = 0.35

# 7D pd_ee_delta_pose = xyz translation, xyz axis-angle rotation, gripper.
# Rotation is expressed in end-effector coordinates in this controller.
KEY_ACTIONS = {
    ord('w'): (0, +TRANSLATION), ord('s'): (0, -TRANSLATION),
    ord('a'): (1, +TRANSLATION), ord('d'): (1, -TRANSLATION),
    ord('r'): (2, +TRANSLATION), ord('f'): (2, -TRANSLATION),
    ord('i'): (3, +ROTATION), ord('k'): (3, -ROTATION),
    ord('j'): (4, +ROTATION), ord('l'): (4, -ROTATION),
    ord('u'): (5, +ROTATION), ord('o'): (5, -ROTATION),
    # Local gripper-axis twist is Z-axis rotation for pd_ee_delta_pose.
    ord('z'): (5, +ROTATION), ord('x'): (5, -ROTATION),
}


def to_numpy_image(frame):
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise RuntimeError(f"Unexpected render frame shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255 if frame.max() <= 1.0 else frame, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR if frame.shape[-1] == 4 else cv2.COLOR_RGB2BGR)


def get_wrist_frame(obs):
    """Read the RGB observation produced by the mounted wrist camera."""
    return to_numpy_image(obs["sensor_data"]["wrist_camera"]["rgb"])


def add_wrist_inset(main, wrist):
    """Place a wrist-camera preview in the bottom-left corner."""
    h, w = main.shape[:2]
    inset_width = min(480, max(220, w // 3))
    inset_height = int(inset_width * wrist.shape[0] / wrist.shape[1])
    inset = cv2.resize(wrist, (inset_width, inset_height), interpolation=cv2.INTER_AREA)
    x = 18
    y = h - inset_height - 18
    if y < 34 or x + inset_width + 5 >= w:
        return main
    cv2.rectangle(main, (x - 3, y - 29), (x + inset_width + 3, y + inset_height + 3), (15, 15, 15), -1)
    main[y:y + inset_height, x:x + inset_width] = inset
    cv2.putText(main, "WRIST CAMERA", (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return main


def show_controls():
    print("""
Franka Panda / ManiSkill 3-cube keyboard teleoperation
  W/S: +X/-X         A/D: +Y/-Y        R/F: +Z/-Z
  I/K: +roll/-roll   J/L: +pitch/-pitch
  U/O: +yaw/-yaw     Z/X: local gripper Z-axis twist (+/-)
  SPACE: open/close gripper
  Q: reset robot and cubes     ESC: quit
  C: toggle wrist-camera inset

Click the video window before pressing keys. Each press makes one action step.
The red cube is the target; green and blue are distractors.
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    args = parser.parse_args()

    env = gym.make(
        'PickRedCube3Teleop-v1',
        num_envs=1,
        obs_mode='rgb',
        control_mode='pd_ee_delta_pose',
        reward_mode='dense',
        render_mode='rgb_array',
        sim_backend='gpu',
        # env.render() uses ManiSkill's human-render camera, not sensor_configs.
        # Render natively at the requested resolution instead of upscaling
        # a low-resolution frame in the OpenCV window.
        human_render_camera_configs={
            'width': args.width,
            'height': args.height,
        },
    )
    try:
        obs, info = env.reset(seed=args.seed)
        action_dim = int(env.action_space.shape[-1])
        if action_dim != 7:
            raise RuntimeError(f"Expected 7D pd_ee_delta_pose action, got {env.action_space.shape}")
        cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TITLE, args.width, args.height)
        show_controls()
        first_frame = to_numpy_image(env.render())
        print(f'[RENDER] Native resolution: {first_frame.shape[1]}x{first_frame.shape[0]}')
        closed = False
        show_wrist = True
        steps = 0
        while True:
            img = to_numpy_image(env.render())
            cv2.putText(img, f"steps={steps}  gripper={'CLOSED' if closed else 'OPEN'}  [Q] reset [ESC] quit", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
            if show_wrist:
                img = add_wrist_inset(img, get_wrist_frame(obs))
            cv2.imshow(TITLE, img)
            key = cv2.waitKeyEx(30)
            if key in (27,):
                break
            if key == -1:
                # No need to advance physics while no key is pressed.
                continue
            key = key & 0xff
            if key == ord('c'):
                show_wrist = not show_wrist
                print('[WRIST]', 'ON' if show_wrist else 'OFF')
                continue
            if key == ord('q'):
                obs, info = env.reset()
                steps = 0
                closed = False
                print('[RESET]')
                continue
            if key == ord(' '):
                closed = not closed
                print('[GRIPPER]', 'CLOSED' if closed else 'OPEN')
            elif key not in KEY_ACTIONS:
                continue
            action = np.zeros((1, 7), dtype=np.float32)
            if key in KEY_ACTIONS:
                axis, value = KEY_ACTIONS[key]
                action[0, axis] = value
            # For Panda in ManiSkill: +1 generally closes, -1 opens.
            action[0, -1] = 1.0 if closed else -1.0
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if steps % 10 == 0:
                print(f'[STEP {steps}] reward={reward.detach().cpu().reshape(-1)[0].item():.3f}')
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
