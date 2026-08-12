"""Dump ManiSkill PushCube-v1 base_camera (the POLICY camera, 128x128) rgb to a
PNG, with the cube fixed at [0,0,0.02] and the goal at [0.2,0,0], for side-by-
side comparison with the IsaacLab policy-camera image to check for a
vertical/horizontal flip.

Run from the ManiSkill repo root (where `mani_skill` is importable):
    python dump_maniskill_rgb.py

Then compare maniskill_rgb.png against isaaclab_rgb.png.
If the blue cube centroid is at the SAME row in both -> no vertical flip.
If the rows sum to the image height -> vertical flip. (Cols / horizontal likewise.)
"""
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401  (registers PushCube-v1)
import sapien
from PIL import Image


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
    env = gym.make(
        "PushCube-v1", num_envs=1, obs_mode="rgb",
        render_mode="rgb_array", sim_backend="physx_cuda",
    )
    env.reset(seed=0)
    # fix cube + goal to known positions for a comparable frame
    env.unwrapped.obj.set_pose(sapien.Pose(p=[0.0, 0.0, 0.02]))
    env.unwrapped.goal_region.set_pose(sapien.Pose(p=[0.2, 0.0, 1e-3]))
    obs = None
    action = torch.zeros((1, env.action_space.shape[-1]))
    for _ in range(3):
        obs, *_ = env.step(action)
    rgb = obs["rgb"][0]
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    Image.fromarray(rgb).save("maniskill_rgb.png")
    print(f"[maniskill] saved maniskill_rgb.png  shape={rgb.shape}", flush=True)
    c = blue_centroid(rgb)
    if c:
        row, col, n = c
        print(f"[maniskill] blue cube centroid: row={row:.1f}/{rgb.shape[0]}, "
              f"col={col:.1f}/{rgb.shape[1]}  ({n} px)", flush=True)
    else:
        print("[maniskill] blue cube NOT detected (check the PNG visually)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
