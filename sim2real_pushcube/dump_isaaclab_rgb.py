"""Dump the IsaacLab PushCube env POLICY-camera rgb to a PNG, with the cube
fixed at [0,0,0.02] and the goal at [0.2,0,0], for side-by-side comparison
with ManiSkill's base_camera image to check for a vertical/horizontal flip.

Run (from the IsaacLab repo root):
    ./isaaclab.sh -p sim2real_pushcube/dump_isaaclab_rgb.py --enable_cameras

Then compare isaaclab_rgb.png against maniskill_rgb.png (from dump_maniskill_rgb.py).
If the blue cube centroid is at the SAME row in both -> no vertical flip.
If the rows sum to the image height -> vertical flip. (Same idea for cols / horizontal.)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
simulation_app = app.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pushcube_isaaclab_env import PushCubeIsaacLabEnv  # noqa: E402


def blue_centroid(rgb):
    mask = (
        (np.abs(rgb[..., 0].astype(int) - 12) < 60)
        & (np.abs(rgb[..., 1].astype(int) - 42) < 60)
        & (np.abs(rgb[..., 2].astype(int) - 160) < 60)
    )
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return ys.mean(), xs.mean(), mask.sum()


def main():
    env = PushCubeIsaacLabEnv(num_envs=1, device="cuda", obs_mode="rgb")
    env.reset()

    dev = env.device
    env_ids = torch.arange(1, device=dev)
    ident = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=dev)  # xyzw identity
    origin = env.scene.env_origins[:1]

    # fix cube at [0,0,0.02] and goal at [0.2,0,0] (env-local -> world = origin + local)
    cube_pose = torch.cat([origin + torch.tensor([0.0, 0.0, 0.02], device=dev), ident], dim=1)
    env.cube.write_root_pose_to_sim_index(root_pose=cube_pose, env_ids=env_ids)
    env.cube.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), device=dev), env_ids=env_ids
    )
    goal_world = torch.cat([origin + torch.tensor([0.2, 0.0, 1e-3], device=dev), ident], dim=1)
    for disc in env.goal_discs:
        disc.write_root_pose_to_sim_index(root_pose=goal_world, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.step()
    env.scene.update(env.sim_dt)

    rgb = env.camera.data.output["rgb"].torch[0].cpu().numpy()
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    Image.fromarray(rgb).save("isaaclab_rgb.png")
    print(f"[isaac] saved isaaclab_rgb.png  shape={rgb.shape}", flush=True)
    c = blue_centroid(rgb)
    if c:
        row, col, n = c
        print(f"[isaac] blue cube centroid: row={row:.1f}/{rgb.shape[0]}, "
              f"col={col:.1f}/{rgb.shape[1]}  ({n} px)", flush=True)
    else:
        print("[isaac] blue cube NOT detected (check the PNG visually)", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
