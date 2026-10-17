#!/usr/bin/env python3
"""ManiSkill UR5e keyboard teleoperation for a registered custom UR5e agent.

Run from the rlinf-project root:
    PYTHONPATH="$PWD" python scripts/control_ur5e.py --robot-module custom_robots.ur5e

IMPORTANT: Standard ManiSkill PickCube/your Panda-based PickRedCube3-v1 does not
ship a UR5e robot. You must first register a custom ManiSkill UR5e agent with
an appropriate gripper and 'pd_ee_delta_pose' controller. The 3-cube task may
also need adaptations to robot-specific grasp/reward/initialization logic.
"""
import argparse
import importlib

import cv2
import gymnasium as gym
import numpy as np
import torch
import sapien
import mani_skill.envs
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env

from custom_envs.pick_red_cube3 import PickRedCube3Env

TITLE = 'ManiSkill - UR5e - Three Cubes'
TRANSLATION = 0.45
ROTATION = 0.35
KEY_ACTIONS = {
    ord('w'): (0, TRANSLATION), ord('s'): (0, -TRANSLATION),
    ord('a'): (1, TRANSLATION), ord('d'): (1, -TRANSLATION),
    ord('r'): (2, TRANSLATION), ord('f'): (2, -TRANSLATION),
    ord('i'): (3, ROTATION), ord('k'): (3, -ROTATION),
    ord('j'): (4, ROTATION), ord('l'): (4, -ROTATION),
    ord('u'): (5, ROTATION), ord('o'): (5, -ROTATION),
    ord('z'): (5, ROTATION), ord('x'): (5, -ROTATION),
}

# Configure before gym.make(): BaseEnv builds its sensors during construction.
WRIST_LINK = 'tool0'
WRIST_POS = [0.045, -0.025, 0.040]
WRIST_QUAT = [0.0, 0.70710678, 0.0, 0.70710678]

@register_env('PickRedCube3UR5eTeleop-v1', max_episode_steps=100)
class UR5eThreeCubeTeleopEnv(PickRedCube3Env):
    """Teleoperation variant; requires a separately registered UR5e agent."""
    SUPPORTED_ROBOTS = list(getattr(PickRedCube3Env, 'SUPPORTED_ROBOTS', [])) + ['ur5e']

    @property
    def _default_sensor_configs(self):
        original = super()._default_sensor_configs
        cameras = list(original) if isinstance(original, (list, tuple)) else [original]
        links = self.agent.robot.links_map
        if WRIST_LINK not in links:
            raise RuntimeError(
                f'Wrist link {WRIST_LINK!r} not found. '
                f'Available UR5e links: {list(links)}. '
                'Pass --wrist-link with the correct tool/gripper link.'
            )
        cameras.append(CameraConfig(
            uid='wrist_camera',
            pose=sapien.Pose(p=WRIST_POS, q=WRIST_QUAT),
            width=640, height=480, fov=np.pi / 2,
            near=0.01, far=10.0, mount=links[WRIST_LINK],
        ))
        return cameras


def to_bgr(frame):
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255 if frame.max() <= 1 else frame, 0, 255).astype(np.uint8)
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise RuntimeError(f'Unexpected camera image shape: {frame.shape}')
    return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR if frame.shape[-1] == 4 else cv2.COLOR_RGB2BGR)


def add_wrist_inset(scene, wrist):
    h, w = scene.shape[:2]
    iw = min(420, max(180, w // 4))
    ih = int(iw * wrist.shape[0] / wrist.shape[1])
    x, y = 18, h - ih - 18
    if y < 35 or x + iw >= w:
        return scene
    scene[y-28:y+ih+3, x-3:x+iw+3] = (15, 15, 15)
    scene[y:y+ih, x:x+iw] = cv2.resize(wrist, (iw, ih), interpolation=cv2.INTER_AREA)
    cv2.putText(scene, 'UR5e WRIST CAMERA', (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return scene


def scalar(x):
    return float(x.detach().cpu().reshape(-1)[0].item()) if isinstance(x, torch.Tensor) else float(np.asarray(x).reshape(-1)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--robot-module', help='Importable module registering a ManiSkill UR5e agent (required unless already registered)')
    ap.add_argument('--robot-uid', default='ur5e', help='Registered ManiSkill robot UID (default: ur5e)')
    ap.add_argument('--wrist-link', default='tool0', help='UR5e link carrying the wrist camera')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--gripper-close-sign', type=int, choices=(-1, 1), default=1,
                    help='Normalized command that closes the gripper; depends on custom UR5e agent')
    args = ap.parse_args()
    global WRIST_LINK
    WRIST_LINK = args.wrist_link
    if args.robot_module:
        importlib.import_module(args.robot_module)

    try:
        env = gym.make(
            'PickRedCube3UR5eTeleop-v1', robot_uids=args.robot_uid,
            num_envs=1, obs_mode='rgb', control_mode='pd_ee_delta_pose',
            reward_mode='dense', render_mode='rgb_array', sim_backend='gpu',
            human_render_camera_configs={'width': args.width, 'height': args.height},
        )
    except Exception as exc:
        raise RuntimeError(
            'Could not instantiate the UR5e three-cube environment. '
            'ManiSkill does not provide a stock UR5e agent for this task. '
            'Register a UR5e agent implementing pd_ee_delta_pose, a gripper, '
            'and update Panda-specific task logic if necessary. '
            'See --robot-module and --wrist-link.'
        ) from exc

    try:
        obs, _ = env.reset(seed=args.seed)
        dim = int(env.action_space.shape[-1])
        if dim != 7:
            raise RuntimeError(f'Expected 7D Cartesian pose + gripper actions, got {dim}')
        cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TITLE, args.width, args.height)
        print('UR5e: W/S X, A/D Y, R/F Z; I/K roll, J/L pitch, U/O yaw; Z/X twist; SPACE gripper; C wrist; Q reset; ESC quit')
        print('Each keypress advances one simulation step; click the image window first.')
        closed, show_wrist, steps = False, True, 0
        while True:
            scene = to_bgr(env.render())
            cv2.putText(scene, f'UR5e  steps={steps}  gripper={"CLOSED" if closed else "OPEN"}',
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
            if show_wrist:
                scene = add_wrist_inset(scene, to_bgr(obs['sensor_data']['wrist_camera']['rgb']))
            cv2.imshow(TITLE, scene)
            k = cv2.waitKeyEx(30)
            if k == 27:
                break
            if k < 0:
                continue
            k &= 0xff
            if k == ord('c'):
                show_wrist = not show_wrist
                continue
            if k == ord('q'):
                obs, _ = env.reset()
                closed, steps = False, 0
                continue
            if k == ord(' '):
                closed = not closed
            elif k not in KEY_ACTIONS:
                continue
            action = np.zeros((1, dim), dtype=np.float32)
            if k in KEY_ACTIONS:
                axis, value = KEY_ACTIONS[k]
                action[0, axis] = value
            action[0, -1] = args.gripper_close_sign if closed else -args.gripper_close_sign
            obs, reward, _, _, _ = env.step(action)
            steps += 1
            if steps % 10 == 0:
                print(f'[STEP {steps}] reward={scalar(reward):.3f}')
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
