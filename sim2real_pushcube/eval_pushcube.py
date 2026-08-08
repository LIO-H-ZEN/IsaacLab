"""Evaluate a ManiSkill PushCube-v1 RGB-PPO checkpoint inside IsaacLab.

Runs the ManiSkill-trained policy (loaded from a ``.pt`` state_dict saved by
ppo_rgb.py) on the IsaacLab PushCube env in ``pushcube_isaaclab_env`` and prints
the success rate over a number of episodes.

Usage (on a machine that can run Isaac Sim):
    ./isaaclab.sh -p sim2real_pushcube/eval_pushcube.py \
        --ckpt /path/to/runs/PushCube-v1__ppo_rgb__1__<ts>/final_ckpt.pt \
        --num_envs 8 --num_eval_episodes 50 \
        --headless --enable_cameras

Notes:
- ``--enable_cameras`` is required (the policy needs the 128x128 rgb observation).
- ``--headless`` + ``--enable_cameras`` selects the headless kit WITH RTX rendering.
- Success is sticky per episode (cube reaches the goal at any step during the
  50-step episode). Reported as successes / episodes.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# --- argparse + AppLauncher MUST come before any isaaclab.* import ---
parser = argparse.ArgumentParser(description="Evaluate ManiSkill PushCube ckpt in IsaacLab")
parser.add_argument("--ckpt", type=str, required=True, help="path to a ppo .pt checkpoint (state_dict)")
parser.add_argument("--obs_mode", choices=["state", "rgb"], default="rgb",
                    help="policy obs mode: 'state' (ppo.py MLP, 35-dim state) or 'rgb' (ppo_rgb.py NatureCNN)")
parser.add_argument("--num_envs", type=int, default=8, help="number of parallel envs")
parser.add_argument("--num_eval_episodes", type=int, default=50, help="total episodes to evaluate")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--video", action="store_true", help="record env-0 rgb frames to a video file")
parser.add_argument("--video_path", type=str, default="pushcube_eval.mp4", help="output video path")
parser.add_argument("--video_episodes", type=int, default=3, help="number of env-0 episodes to record")
parser.add_argument(
    "--video_camera", choices=["policy", "render"], default="render",
    help="which camera to record: 'render' matches the ManiSkill demo (front-right 45deg, 512x512); "
         "'policy' is the 128x128 camera the policy sees",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- now safe to import isaaclab / torch / local modules ---
import os
import sys

import torch

# make sibling modules (mani_skill_agent, pushcube_isaaclab_env) importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mani_skill_agent import build_agent, build_state_agent
from pushcube_isaaclab_env import PushCubeIsaacLabEnv


def write_frames_to_video(frames, path, fps=20):
    """Write a list of (H, W, 3) uint8 numpy frames to a video file.

    Tries imageio (mp4) -> torchvision (mp4) -> PIL (animated gif) so it works
    regardless of which video backend is installed in the container.
    """
    import numpy as np

    arr = np.stack(frames)  # (T, H, W, 3) uint8

    # 1) imageio mp4
    try:
        import imageio

        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
        for f in arr:
            writer.append_data(f)
        writer.close()
        print(f"[video] saved {len(arr)} frames to {path} (imageio mp4)", flush=True)
        return
    except Exception as e:
        print(f"[video] imageio unavailable/failed ({e}); trying torchvision", flush=True)

    # 2) torchvision mp4
    try:
        import torch as _t
        from torchvision.io import write_video

        write_video(path, _t.from_numpy(arr).permute(0, 3, 1, 2), fps=fps)
        print(f"[video] saved {len(arr)} frames to {path} (torchvision mp4)", flush=True)
        return
    except Exception as e:
        print(f"[video] torchvision unavailable/failed ({e}); falling back to GIF", flush=True)

    # 3) PIL animated gif (always works if Pillow is installed)
    try:
        from PIL import Image

        gif_path = path[:-4] + ".gif" if path.lower().endswith(".mp4") else path
        imgs = [Image.fromarray(f) for f in arr]
        imgs[0].save(
            gif_path, save_all=True, append_images=imgs[1:],
            duration=int(1000 / fps), loop=0,
        )
        print(f"[video] saved {len(arr)} frames to {gif_path} (PIL gif)", flush=True)
        return
    except Exception as e:
        print(f"[video] all video writers failed: {e}", flush=True)


def main():
    torch.manual_seed(args_cli.seed)
    device = args_cli.device

    obs_mode = args_cli.obs_mode
    print(f"[eval] building env (obs_mode={obs_mode})...", flush=True)
    env = PushCubeIsaacLabEnv(num_envs=args_cli.num_envs, device=device, obs_mode=obs_mode,
                              video=args_cli.video)
    print("[eval] loading policy checkpoint...", flush=True)
    if obs_mode == "state":
        agent = build_state_agent(
            ckpt_path=args_cli.ckpt, device=torch.device(device), state_dim=35, action_dim=8,
        )
    else:
        agent = build_agent(
            ckpt_path=args_cli.ckpt,
            num_envs=args_cli.num_envs,
            device=torch.device(device),
            rgb_shape=(128, 128, 3),
            state_dim=25,
            action_dim=8,
        )
    print(f"[eval] env ready ({args_cli.num_envs} envs, obs_mode={obs_mode}), "
          f"policy loaded from {args_cli.ckpt}", flush=True)

    obs = env.reset()
    episodes_done = 0
    successes = 0
    target = args_cli.num_eval_episodes
    # generous safety bound so the loop can't run forever if episodes never end
    safety_max_steps = target * (env.max_steps + 5) + 100
    step = 0

    # video recording: collect env-0 rgb frames for the first few episodes
    video_frames = []
    video_budget = args_cli.video_episodes * env.max_steps if args_cli.video else 0
    video_cam = env.camera if args_cli.video_camera == "policy" else env.camera_render
    if args_cli.video and len(video_frames) < video_budget:
        video_frames.append(video_cam.data.output["rgb"][0].cpu().contiguous().numpy())

    while episodes_done < target and step < safety_max_steps:
        with torch.no_grad():
            agent_input = obs["state"] if obs_mode == "state" else obs
            action = agent.get_action(agent_input, deterministic=True)  # (N,8)
        obs, done, info = env.step(action)
        step += 1
        if args_cli.video and len(video_frames) < video_budget:
            video_frames.append(video_cam.data.output["rgb"][0].cpu().contiguous().numpy())

        reset_ids = done.nonzero(as_tuple=True)[0]
        if len(reset_ids) > 0:
            # record sticky success for episodes that just ended, THEN reset (reset clears ever_success)
            successes += int(env.ever_success[reset_ids].sum().item())
            episodes_done += len(reset_ids)
            obs = env.reset(reset_ids)

        if step % 5 == 0:
            rate = successes / max(episodes_done, 1)
            print(
                f"[step {step}] episodes={episodes_done}/{target} "
                f"successes={successes} running_rate={rate:.3f}",
                flush=True,
            )

    print("\n" + "=" * 60)
    if episodes_done > 0:
        print(f"Eval result: {successes}/{episodes_done} = {successes / episodes_done:.3f} success rate")
    else:
        print("Eval result: 0 episodes completed (check that episodes terminate).")
    print("=" * 60)

    if args_cli.video and video_frames:
        write_frames_to_video(video_frames, args_cli.video_path)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
