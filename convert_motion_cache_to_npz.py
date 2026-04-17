#!/usr/bin/env python3
"""Convert motion cache (.npy arrays) to individual .npz files for MultiMotionLoader.

The motion cache stores pre-processed data as flat numpy arrays. This script
reconstructs full body kinematics (all body positions, quaternions, velocities)
via MuJoCo forward kinematics and saves each clip as an individual .npz file
compatible with holosoma's MotionLoader/MultiMotionLoader.

Usage:
    python convert_motion_cache_to_npz.py \
        --cache-dir path/to/omomo_amass_kimodo_v001_full.train.motion_cache \
        --robot-xml src/holosoma/holosoma/data/robots/t1/t1_29dof.xml \
        --output-dir motions_t1_29dof_npz \
        [--max-clips 100] [--workers 8]
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import mujoco
import numpy as np


def load_cache(cache_dir: str) -> dict:
    """Load motion cache arrays using memory-mapped files."""
    cache = {}
    for name in [
        "joint_pos", "joint_vel", "root_pos", "root_quat",
        "root_lin_vel_w", "root_ang_vel_w",
        "motion_start_idx", "motion_num_frames", "clip_lengths",
        "metadata_json",
    ]:
        path = os.path.join(cache_dir, f"{name}.npy")
        if name == "metadata_json":
            cache[name] = json.loads(np.load(path, allow_pickle=True).item())
        elif name in ("motion_start_idx", "motion_num_frames", "clip_lengths"):
            cache[name] = np.load(path)
        else:
            cache[name] = np.load(path, mmap_mode="r")
    return cache


def compute_body_kinematics(
    model_path: str,
    root_pos: np.ndarray,      # (T, 3)
    root_quat: np.ndarray,     # (T, 4)  wxyz
    joint_pos: np.ndarray,     # (T, 29)
    joint_vel: np.ndarray,     # (T, 29)
    root_lin_vel: np.ndarray,  # (T, 3)
    root_ang_vel: np.ndarray,  # (T, 3)
) -> dict:
    """Run MuJoCo FK per frame to get all body positions, quats, velocities."""
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    num_frames = root_pos.shape[0]
    # Skip body 0 (world) — bodies 1..nbody-1 are the robot
    num_bodies = m.nbody - 1

    body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float64)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float64)  # wxyz
    body_lin_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float64)
    body_ang_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float64)

    for t in range(num_frames):
        # Set root pose: qpos[0:3] = xyz, qpos[3:7] = wxyz quat
        d.qpos[0:3] = root_pos[t]
        d.qpos[3:7] = root_quat[t]  # wxyz
        d.qpos[7:36] = joint_pos[t]

        # Set velocities for body velocity computation
        d.qvel[0:3] = root_lin_vel[t]
        d.qvel[3:6] = root_ang_vel[t]
        d.qvel[6:35] = joint_vel[t]

        mujoco.mj_kinematics(m, d)
        mujoco.mj_comVel(m, d)

        # Extract body data (skip world body at index 0)
        body_pos_w[t] = d.xpos[1:]
        body_quat_w[t] = d.xquat[1:]  # wxyz
        # cvel is (nbody, 6): [ang_vel(3), lin_vel(3)]
        body_ang_vel_w[t] = d.cvel[1:, :3]
        body_lin_vel_w[t] = d.cvel[1:, 3:]

    return {
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }


def process_clip(args: tuple) -> str | None:
    """Process a single clip: FK + save NPZ. Returns output path or None on error."""
    (clip_idx, start, nframes, cache_dir, model_path, output_dir,
     joint_names, body_names, fps) = args

    try:
        # Load slice from mmap
        joint_pos = np.array(np.load(os.path.join(cache_dir, "joint_pos.npy"), mmap_mode="r")[start:start + nframes])
        joint_vel = np.array(np.load(os.path.join(cache_dir, "joint_vel.npy"), mmap_mode="r")[start:start + nframes])
        root_pos = np.array(np.load(os.path.join(cache_dir, "root_pos.npy"), mmap_mode="r")[start:start + nframes])
        root_quat = np.array(np.load(os.path.join(cache_dir, "root_quat.npy"), mmap_mode="r")[start:start + nframes])
        root_lin_vel = np.array(np.load(os.path.join(cache_dir, "root_lin_vel_w.npy"), mmap_mode="r")[start:start + nframes])
        root_ang_vel = np.array(np.load(os.path.join(cache_dir, "root_ang_vel_w.npy"), mmap_mode="r")[start:start + nframes])

        # FK
        body_kin = compute_body_kinematics(
            model_path, root_pos, root_quat, joint_pos, joint_vel,
            root_lin_vel, root_ang_vel,
        )

        # Build holosoma NPZ format
        # joint_pos with root DOFs prepended: [x,y,z, qw,qx,qy,qz, j0..j28]
        jp_full = np.concatenate([root_pos, root_quat, joint_pos], axis=1)  # (T, 36)
        # joint_vel with root vel prepended: [vx,vy,vz, wx,wy,wz, jv0..jv28]
        jv_full = np.concatenate([root_lin_vel, root_ang_vel, joint_vel], axis=1)  # (T, 35)

        out_path = os.path.join(output_dir, f"clip_{clip_idx:05d}.npz")
        np.savez_compressed(
            out_path,
            fps=np.array([fps]),
            joint_pos=jp_full,
            joint_vel=jv_full,
            body_pos_w=body_kin["body_pos_w"],
            body_quat_w=body_kin["body_quat_w"],
            body_lin_vel_w=body_kin["body_lin_vel_w"],
            body_ang_vel_w=body_kin["body_ang_vel_w"],
            body_names=np.array(body_names),
            joint_names=np.array(joint_names),
            qpos=jp_full,  # alias for viser_playlist_player compatibility
        )
        return out_path
    except Exception as e:
        print(f"  ERROR clip {clip_idx}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert motion cache to individual NPZ files")
    parser.add_argument("--cache-dir", required=True, help="Path to motion cache directory")
    parser.add_argument("--robot-xml", required=True, help="Path to MuJoCo robot XML")
    parser.add_argument("--output-dir", required=True, help="Output directory for NPZ files")
    parser.add_argument("--max-clips", type=int, default=0, help="Max clips to convert (0=all)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--fps", type=int, default=30, help="FPS (default: 30)")
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata
    meta = json.loads(np.load(os.path.join(cache_dir, "metadata_json.npy"), allow_pickle=True).item())
    joint_names = meta["joint_names"]
    body_names = meta["body_names"]

    # Remap body names: motion cache uses left/right_inspire_hand,
    # but the MuJoCo XML uses left/right_hand_link
    body_names_remapped = []
    for bn in body_names:
        if bn == "left_inspire_hand":
            body_names_remapped.append("left_hand_link")
        elif bn == "right_inspire_hand":
            body_names_remapped.append("right_hand_link")
        else:
            body_names_remapped.append(bn)
    body_names = body_names_remapped

    start_idx = np.load(os.path.join(cache_dir, "motion_start_idx.npy"))
    num_frames = np.load(os.path.join(cache_dir, "motion_num_frames.npy"))
    num_clips = len(start_idx)

    if args.max_clips > 0:
        num_clips = min(num_clips, args.max_clips)

    print(f"Converting {num_clips} clips from {cache_dir}")
    print(f"Robot XML: {args.robot_xml}")
    print(f"Output: {output_dir}")
    print(f"Joint names ({len(joint_names)}): {joint_names}")
    print(f"Body names ({len(body_names)}): {body_names}")
    print()

    # Skip clips with 0 frames
    tasks = []
    for i in range(num_clips):
        nf = int(num_frames[i])
        if nf < 2:
            continue
        tasks.append((
            i, int(start_idx[i]), nf, cache_dir, os.path.abspath(args.robot_xml),
            output_dir, joint_names, body_names, args.fps,
        ))

    print(f"Processing {len(tasks)} clips (skipped {num_clips - len(tasks)} with <2 frames)")

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            results = []
            for j, result in enumerate(pool.imap_unordered(process_clip, tasks)):
                if (j + 1) % 500 == 0 or j + 1 == len(tasks):
                    print(f"  {j+1}/{len(tasks)} clips done")
                results.append(result)
    else:
        results = []
        for j, task in enumerate(tasks):
            result = process_clip(task)
            results.append(result)
            if (j + 1) % 500 == 0 or j + 1 == len(tasks):
                print(f"  {j+1}/{len(tasks)} clips done")

    success = sum(1 for r in results if r is not None)
    print(f"\nDone: {success}/{len(tasks)} clips converted to {output_dir}")


if __name__ == "__main__":
    main()
