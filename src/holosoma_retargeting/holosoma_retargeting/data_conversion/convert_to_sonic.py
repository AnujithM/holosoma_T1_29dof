#!/usr/bin/env python3
"""Convert retargeted .npz files to SONIC/GR00T CSV format for whole-body tracking deployment.

Usage:
  # Single file
  python convert_to_sonic.py --input path/to/retargeted.npz --output-dir sonic_output/

  # Batch: all .npz files in a directory tree
  python convert_to_sonic.py --input-dir demo_results_parallel/g1/object_interaction/omomo/ \
                             --output-dir sonic_output/ --target-fps 50
"""

import argparse
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

# 14 tracked body names matching holosoma WBT config (config_values/wbt/g1/command.py)
TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

# Corresponding IsaacLab body indexes (into g1_29dof body_names list)
ISAACLAB_BODY_INDEXES = np.array([0, 2, 4, 6, 9, 11, 13, 17, 19, 21, 24, 26, 28, 31])

N_JOINTS = 29
N_BODIES = len(TRACKED_BODY_NAMES)  # 14

# SONIC/IsaacSim uses an interleaved left/right/center joint ordering (USD convention)
# which differs from the URDF/MuJoCo sequential ordering used by the retargeter.
# The SONIC visualizer applies isaaclab_to_mujoco to convert CSV→MuJoCo order,
# so we must output in the interleaved order.
# This is the inverse of the visualizer's isaaclab_to_mujoco mapping.
MUJOCO_TO_SONIC = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]


def get_tracked_body_ids(model):
    """Resolve tracked body names to MuJoCo body IDs."""
    body_ids = []
    for name in TRACKED_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            raise ValueError(f"Body '{name}' not found in MuJoCo model")
        body_ids.append(bid)
    return body_ids


def run_fk(model, data, qpos_all, tracked_body_ids):
    """Run forward kinematics for all frames, extract tracked body positions & quaternions."""
    T = qpos_all.shape[0]
    n_tracked = len(tracked_body_ids)

    body_pos = np.zeros((T, n_tracked, 3))
    body_quat = np.zeros((T, n_tracked, 4))  # wxyz (MuJoCo convention)

    for t in range(T):
        data.qpos[: model.nq] = qpos_all[t, : model.nq]
        mujoco.mj_kinematics(model, data)

        for i, bid in enumerate(tracked_body_ids):
            body_pos[t, i] = data.xpos[bid]
            body_quat[t, i] = data.xquat[bid]

    return body_pos, body_quat


def resample_positions(data, src_times, dst_times):
    """Linearly interpolate array data from src_times to dst_times."""
    orig_shape = data.shape
    flat = data.reshape(len(src_times), -1)
    f = interp1d(src_times, flat, axis=0, kind="linear")
    result = f(dst_times)
    return result.reshape(len(dst_times), *orig_shape[1:])


def resample_quaternions(quats, src_times, dst_times):
    """SLERP quaternion data. quats: (T, N, 4) in wxyz."""
    _, N, _ = quats.shape
    T_dst = len(dst_times)
    result = np.zeros((T_dst, N, 4))

    for n in range(N):
        q_xyzw = quats[:, n, [1, 2, 3, 0]]
        rots = Rotation.from_quat(q_xyzw)
        slerp = Slerp(src_times, rots)
        interp_rots = slerp(dst_times)
        q_interp = interp_rots.as_quat()  # xyzw
        result[:, n] = q_interp[:, [3, 0, 1, 2]]  # back to wxyz

    return result


def compute_angular_velocity(quats, dt):
    """Compute angular velocity from quaternion time series.
    quats: (T, N, 4) in wxyz. Returns: (T, N, 3) angular velocities in world frame.
    """
    T, N, _ = quats.shape
    ang_vel = np.zeros((T, N, 3))

    for n in range(N):
        q_xyzw = quats[:, n, [1, 2, 3, 0]]
        rots = Rotation.from_quat(q_xyzw)

        for t in range(T):
            if t == 0:
                drot = rots[1] * rots[0].inv()
                dt_used = dt
            elif t == T - 1:
                drot = rots[t] * rots[t - 1].inv()
                dt_used = dt
            else:
                drot = rots[t + 1] * rots[t - 1].inv()
                dt_used = 2.0 * dt

            ang_vel[t, n] = drot.as_rotvec() / dt_used

    return ang_vel


def save_csv(filepath, data, headers):
    """Save 2D array as CSV with headers."""
    with open(filepath, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in data:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


def convert_single(npz_path, output_dir, model, data, tracked_body_ids, target_fps=50, default_fps=30):
    """Convert one retargeted .npz file to SONIC CSV folder."""
    npz = np.load(npz_path)
    qpos_all = npz["qpos"]  # (T, 43) — [root(7), joints(29), object(7)]
    src_fps = float(npz["fps"]) if "fps" in npz else default_fps
    T_src = qpos_all.shape[0]

    if T_src < 2:
        print(f"  Skipping {npz_path}: only {T_src} frame(s)")
        return 0

    # Check for NaN
    if np.any(np.isnan(qpos_all[:, : model.nq])):
        print(f"  Skipping {npz_path}: contains NaN")
        return 0

    # Extract 29 joint positions (skip root freejoint qpos[0:7])
    joint_pos_src = qpos_all[:, 7:36]  # (T, 29)

    # Robot-only qpos for FK (drop object columns)
    robot_qpos = qpos_all[:, : model.nq]  # (T, nq)

    # --- Forward kinematics at source FPS ---
    body_pos_src, body_quat_src = run_fk(model, data, robot_qpos, tracked_body_ids)

    # --- Resample to target FPS ---
    src_times = np.arange(T_src) / src_fps
    duration = (T_src - 1) / src_fps
    T_dst = int(round(duration * target_fps)) + 1
    dst_times = np.linspace(0.0, duration, T_dst)

    if target_fps == src_fps:
        joint_pos = joint_pos_src
        body_pos = body_pos_src
        body_quat = body_quat_src
    else:
        joint_pos = resample_positions(joint_pos_src, src_times, dst_times)
        body_pos = resample_positions(body_pos_src, src_times, dst_times)
        body_quat = resample_quaternions(body_quat_src, src_times, dst_times)

    # --- Compute velocities at target FPS via finite differences ---
    dst_dt = 1.0 / target_fps
    joint_vel = np.gradient(joint_pos, dst_dt, axis=0)
    body_lin_vel = np.gradient(body_pos, dst_dt, axis=0)
    body_ang_vel = compute_angular_velocity(body_quat, dst_dt)

    # --- Reorder joints from MuJoCo/URDF order to SONIC interleaved order ---
    joint_pos = joint_pos[:, MUJOCO_TO_SONIC]
    joint_vel = joint_vel[:, MUJOCO_TO_SONIC]

    # --- Flatten body arrays for CSV ---
    body_pos_flat = body_pos.reshape(T_dst, -1)  # (T, 42)
    body_quat_flat = body_quat.reshape(T_dst, -1)  # (T, 56)
    body_lin_vel_flat = body_lin_vel.reshape(T_dst, -1)  # (T, 42)
    body_ang_vel_flat = body_ang_vel.reshape(T_dst, -1)  # (T, 42)

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)

    save_csv(
        os.path.join(output_dir, "joint_pos.csv"),
        joint_pos,
        [f"joint_{i}" for i in range(N_JOINTS)],
    )
    save_csv(
        os.path.join(output_dir, "joint_vel.csv"),
        joint_vel,
        [f"joint_vel_{i}" for i in range(N_JOINTS)],
    )
    save_csv(
        os.path.join(output_dir, "body_pos.csv"),
        body_pos_flat,
        [f"body_{i // 3}_{'xyz'[i % 3]}" for i in range(N_BODIES * 3)],
    )
    save_csv(
        os.path.join(output_dir, "body_quat.csv"),
        body_quat_flat,
        [f"body_{i // 4}_{'wxyz'[i % 4]}" for i in range(N_BODIES * 4)],
    )
    save_csv(
        os.path.join(output_dir, "body_lin_vel.csv"),
        body_lin_vel_flat,
        [f"body_{i // 3}_vel_{'xyz'[i % 3]}" for i in range(N_BODIES * 3)],
    )
    save_csv(
        os.path.join(output_dir, "body_ang_vel.csv"),
        body_ang_vel_flat,
        [f"body_{i // 3}_angvel_{'xyz'[i % 3]}" for i in range(N_BODIES * 3)],
    )

    # Metadata
    name = Path(npz_path).stem
    with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
        f.write(f"Metadata for: {name}\n")
        f.write("=" * 30 + "\n\n")
        f.write("Body part indexes:\n")
        f.write(f"{ISAACLAB_BODY_INDEXES}\n\n")
        f.write(f"Total timesteps: {T_dst}\n\n")
        f.write("Data arrays summary:\n")
        f.write(f"  joint_pos: ({T_dst}, {N_JOINTS}) (float32)\n")
        f.write(f"  joint_vel: ({T_dst}, {N_JOINTS}) (float32)\n")
        f.write(f"  body_pos_w: ({T_dst}, {N_BODIES}, 3) (float32)\n")
        f.write(f"  body_quat_w: ({T_dst}, {N_BODIES}, 4) (float32)\n")
        f.write(f"  body_lin_vel_w: ({T_dst}, {N_BODIES}, 3) (float32)\n")
        f.write(f"  body_ang_vel_w: ({T_dst}, {N_BODIES}, 3) (float32)\n")

    # Info
    with open(os.path.join(output_dir, "info.txt"), "w") as f:
        f.write(f"Motion Information: {name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Source file: {npz_path}\n")
        f.write(f"Source FPS: {src_fps}\n")
        f.write(f"Target FPS: {target_fps}\n")
        f.write(f"Source frames: {T_src}\n")
        f.write(f"Target frames: {T_dst}\n")
        f.write(f"Duration: {duration:.3f}s\n\n")
        f.write("Tracked bodies:\n")
        for i, bname in enumerate(TRACKED_BODY_NAMES):
            f.write(f"  body_{i}: {bname} (IsaacLab idx {ISAACLAB_BODY_INDEXES[i]})\n")
        f.write("\n")
        for arr_name, arr in [
            ("joint_pos", joint_pos),
            ("joint_vel", joint_vel),
            ("body_pos", body_pos),
            ("body_quat", body_quat),
            ("body_lin_vel", body_lin_vel),
            ("body_ang_vel", body_ang_vel),
        ]:
            f.write(f"{arr_name}:\n")
            f.write(f"  Shape: {arr.shape}\n")
            f.write(f"  Range: [{arr.min():.3f}, {arr.max():.3f}]\n\n")

    return T_dst


def main():
    parser = argparse.ArgumentParser(description="Convert retargeted .npz to SONIC CSV format")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Single .npz file to convert")
    group.add_argument("--input-dir", type=str, help="Directory of .npz files (recursive)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output base directory")
    parser.add_argument("--target-fps", type=int, default=50, help="Target FPS (default: 50)")
    parser.add_argument(
        "--model-xml",
        type=str,
        default=None,
        help="Path to g1_29dof.xml (auto-detected if not specified)",
    )
    parser.add_argument(
        "--default-fps",
        type=int,
        default=30,
        help="Default source FPS if not stored in .npz (default: 30)",
    )
    args = parser.parse_args()

    # Auto-detect model XML
    if args.model_xml is None:
        script_dir = Path(__file__).resolve().parent.parent
        args.model_xml = str(script_dir / "models" / "g1" / "g1_29dof.xml")
    if not os.path.exists(args.model_xml):
        print(f"Error: model XML not found: {args.model_xml}")
        sys.exit(1)

    # Load MuJoCo model once
    model = mujoco.MjModel.from_xml_path(args.model_xml)
    data = mujoco.MjData(model)
    tracked_body_ids = get_tracked_body_ids(model)
    print(f"Loaded model: {args.model_xml} (nq={model.nq}, nbody={model.nbody})")
    print(f"Tracking {N_BODIES} bodies: {', '.join(TRACKED_BODY_NAMES)}")
    print(f"MuJoCo body IDs: {tracked_body_ids}")
    print(f"Target FPS: {args.target_fps}")

    # Collect input files
    if args.input:
        npz_files = [args.input]
    else:
        npz_files = sorted(str(p) for p in Path(args.input_dir).rglob("*.npz"))

    print(f"\nFound {len(npz_files)} .npz file(s)")

    success = 0
    failed = 0
    for i, npz_path in enumerate(npz_files):
        rel = os.path.relpath(npz_path, args.input_dir if args.input_dir else os.path.dirname(npz_path))
        out_name = Path(rel).stem
        out_dir = os.path.join(args.output_dir, out_name)

        print(f"[{i + 1}/{len(npz_files)}] {rel} -> {out_name}/")
        try:
            n_frames = convert_single(npz_path, out_dir, model, data, tracked_body_ids, args.target_fps, args.default_fps)
            if n_frames > 0:
                print(f"  -> {n_frames} frames at {args.target_fps}Hz")
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\nDone: {success} converted, {failed} failed out of {len(npz_files)} total")


if __name__ == "__main__":
    main()
