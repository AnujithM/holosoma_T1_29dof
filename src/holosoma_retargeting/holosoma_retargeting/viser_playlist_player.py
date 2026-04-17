#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]


# SMPL-H skeleton: parent index for each of the 52 joints (-1 = root)
# 0:Pelvis 1:L_Hip 2:L_Knee 3:L_Ankle 4:L_Toe 5:R_Hip 6:R_Knee 7:R_Ankle 8:R_Toe
# 9:Torso 10:Spine 11:Chest 12:Neck 13:Head 14:L_Thorax 15:L_Shoulder 16:L_Elbow
# 17:L_Wrist 18-32:L_Hand fingers 33:R_Thorax 34:R_Shoulder 35:R_Elbow 36:R_Wrist
# 37-51:R_Hand fingers
SMPLH_PARENT = [
    -1,  # 0  Pelvis
     0,  # 1  L_Hip
     1,  # 2  L_Knee
     2,  # 3  L_Ankle
     3,  # 4  L_Toe
     0,  # 5  R_Hip
     5,  # 6  R_Knee
     6,  # 7  R_Ankle
     7,  # 8  R_Toe
     0,  # 9  Torso
     9,  # 10 Spine
    10,  # 11 Chest
    11,  # 12 Neck
    12,  # 13 Head
    11,  # 14 L_Thorax
    14,  # 15 L_Shoulder
    15,  # 16 L_Elbow
    16,  # 17 L_Wrist
    17,  # 18 L_Index1
    18,  # 19 L_Index2
    19,  # 20 L_Index3
    17,  # 21 L_Middle1
    21,  # 22 L_Middle2
    22,  # 23 L_Middle3
    17,  # 24 L_Pinky1
    24,  # 25 L_Pinky2
    25,  # 26 L_Pinky3
    17,  # 27 L_Ring1
    27,  # 28 L_Ring2
    28,  # 29 L_Ring3
    17,  # 30 L_Thumb1
    30,  # 31 L_Thumb2
    31,  # 32 L_Thumb3
    11,  # 33 R_Thorax
    33,  # 34 R_Shoulder
    34,  # 35 R_Elbow
    35,  # 36 R_Wrist
    36,  # 37 R_Index1
    37,  # 38 R_Index2
    38,  # 39 R_Index3
    36,  # 40 R_Middle1
    40,  # 41 R_Middle2
    41,  # 42 R_Middle3
    36,  # 43 R_Pinky1
    43,  # 44 R_Pinky2
    44,  # 45 R_Pinky3
    36,  # 46 R_Ring1
    46,  # 47 R_Ring2
    47,  # 48 R_Ring3
    36,  # 49 R_Thumb1
    49,  # 50 R_Thumb2
    50,  # 51 R_Thumb3
]

# Bone pairs (child, parent) for skeleton line segments
SMPLH_BONES = [(i, SMPLH_PARENT[i]) for i in range(len(SMPLH_PARENT)) if SMPLH_PARENT[i] >= 0]


def load_npz(npz_path: str) -> tuple[np.ndarray, int, np.ndarray | None]:
    data = np.load(npz_path, allow_pickle=True)
    qpos = data["qpos"]
    fps = int(data["fps"]) if "fps" in data else 30
    human_joints = data["human_joints"] if "human_joints" in data else None
    return qpos, fps, human_joints


def is_tty_stdin() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def discover_object_urdfs(models_dir: str) -> dict[str, str]:
    """Discover available object URDFs: dirs containing <name>/<name>.urdf."""
    urdfs: dict[str, str] = {}
    models_path = Path(models_dir)
    if not models_path.is_dir():
        return urdfs
    for obj_dir in sorted(models_path.iterdir()):
        if not obj_dir.is_dir():
            continue
        name = obj_dir.name
        urdf_path = obj_dir / f"{name}.urdf"
        if urdf_path.is_file():
            urdfs[name] = str(urdf_path)
    return urdfs


def detect_object_from_filename(filename: str, known_objects: list[str]) -> str | None:
    """Detect object name from npz filename (longest match first)."""
    name_lower = filename.lower()
    for obj in sorted(known_objects, key=len, reverse=True):
        if obj in name_lower:
            return obj
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Play multiple retargeted npz files in one Viser localhost session")
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="Glob pattern for npz files, e.g. demo_results_parallel/g1/object_interaction/omomo/*_original.npz",
    )
    parser.add_argument("--robot-urdf", type=str, required=True, help="Robot URDF path")
    parser.add_argument(
        "--object-models-dir",
        type=str,
        default="models",
        help="Directory containing object model subdirectories (each with <name>/<name>.urdf)",
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--fps", type=int, default=30, help="Fallback FPS")
    parser.add_argument("--show-meshes", action="store_true", help="Show meshes")
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {args.pattern}")

    # Discover object URDFs
    available_objects = discover_object_urdfs(args.object_models_dir)
    object_names = sorted(available_objects.keys())
    print(f"[playlist] Discovered {len(available_objects)} objects: {', '.join(object_names)}")

    # Build per-object file index: object_name -> list of file indices
    object_file_indices: dict[str, list[int]] = {name: [] for name in object_names}
    for i, fpath in enumerate(files):
        detected = detect_object_from_filename(Path(fpath).name, object_names)
        if detected:
            object_file_indices[detected].append(i)
    for name, indices in object_file_indices.items():
        print(f"  {name}: {len(indices)} files")

    server = viser.ViserServer()

    # Robot
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    robot_urdf_y = yourdfpy.URDF.load(args.robot_urdf, load_meshes=True, build_scene_graph=True)
    vr = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")

    # Pre-load all object URDFs (each under its own scene root)
    object_visuals: dict[str, tuple[ViserUrdf, viser.SceneNodeHandle]] = {}
    for obj_name in object_names:
        print(f"[playlist] Loading object mesh: {obj_name}")
        obj_frame = server.scene.add_frame(f"/object_{obj_name}", show_axes=False)
        obj_urdf_y = yourdfpy.URDF.load(
            available_objects[obj_name], load_meshes=True, build_scene_graph=True
        )
        obj_viser = ViserUrdf(server, urdf_or_path=obj_urdf_y, root_node_name=f"/object_{obj_name}")
        obj_viser.show_visual = False
        object_visuals[obj_name] = (obj_viser, obj_frame)

    server.scene.add_grid("/grid", width=8.0, height=8.0, position=(0.0, 0.0, 0.0))

    joint_limits = vr.get_actuated_joint_limits()
    robot_dof = len(joint_limits)

    vr.show_visual = bool(args.show_meshes)

    state_lock = threading.Lock()
    update_flag = {"frame": False, "seq": False, "object": False}

    clip_cache: dict[str, tuple[np.ndarray, int, np.ndarray | None]] = {}

    # Human skeleton visualization handles
    human_points_handle = None
    human_lines_handle = None

    state: dict = {
        "seq_idx": 0,
        "qpos": None,
        "fps": int(args.fps),
        "playing": False,
        "frame_f": 0.0,
        "loop": bool(args.loop),
        "prev_robot_q": None,
        "prev_obj_q": None,
        "current_object": None,  # str | None — currently active object name
    }

    def _norm_quat(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        n = float(np.linalg.norm(q))
        return q if n == 0.0 else q / n

    def _continuous(prev_q: np.ndarray | None, curr_q: np.ndarray) -> np.ndarray:
        q = _norm_quat(curr_q)
        if prev_q is None:
            return q
        return -q if float(np.dot(prev_q, q)) < 0.0 else q

    def _get_clip(idx: int) -> tuple[np.ndarray, int, np.ndarray | None]:
        path = files[idx]
        if path not in clip_cache:
            clip_cache[path] = load_npz(path)
        return clip_cache[path]

    def _switch_object(new_obj: str | None) -> None:
        """Show only the selected object, hide all others."""
        old_obj = state["current_object"]
        if old_obj == new_obj:
            return
        if old_obj and old_obj in object_visuals:
            object_visuals[old_obj][0].show_visual = False
        if new_obj and new_obj in object_visuals:
            object_visuals[new_obj][0].show_visual = bool(show_meshes_cb.value)
        state["current_object"] = new_obj
        state["prev_obj_q"] = None

    def _draw_human_skeleton(joints_3d: np.ndarray) -> None:
        """Draw SMPL-H skeleton as point cloud + line segments, offset to the side."""
        nonlocal human_points_handle, human_lines_handle
        offset = np.array([float(human_offset_slider.value), 0.0, 0.0])
        pts = joints_3d + offset  # (52, 3)

        # Joint spheres
        colors_pts = np.full((pts.shape[0], 3), 60, dtype=np.uint8)
        colors_pts[:, 2] = 255  # blue joints
        human_points_handle = server.scene.add_point_cloud(
            "/human_skeleton/joints", pts.astype(np.float32), colors_pts,
            point_size=0.025, point_shape="circle",
            visible=bool(show_human_cb.value),
        )
        # Bones — viser expects shape (N, 2, 3)
        bone_pts = []
        for child, parent in SMPLH_BONES:
            bone_pts.append([pts[child], pts[parent]])
        bone_pts = np.array(bone_pts, dtype=np.float32)  # (N, 2, 3)
        colors_lines = np.full((bone_pts.shape[0], 2, 3), 100, dtype=np.uint8)
        colors_lines[:, :, 2] = 220  # blue-ish bones
        human_lines_handle = server.scene.add_line_segments(
            "/human_skeleton/bones", bone_pts, colors_lines,
            line_width=2.0,
            visible=bool(show_human_cb.value),
        )

    def _apply_frame(frame_q: np.ndarray) -> None:
        joints = frame_q[7 : 7 + robot_dof]
        if joints.shape[0] != robot_dof:
            if joints.shape[0] > robot_dof:
                joints = joints[:robot_dof]
            else:
                joints = np.pad(joints, (0, robot_dof - joints.shape[0]))

        vr.update_cfg(joints)

        robot_root.position = frame_q[0:3]
        robot_q = _continuous(state["prev_robot_q"], frame_q[3:7])
        state["prev_robot_q"] = robot_q
        robot_root.wxyz = robot_q

        cur_obj = state["current_object"]
        if cur_obj and cur_obj in object_visuals and frame_q.shape[0] >= (7 + robot_dof + 7):
            _, obj_frame = object_visuals[cur_obj]
            obj_frame.position = frame_q[-7:-4]
            obj_q = _continuous(state["prev_obj_q"], frame_q[-4:])
            state["prev_obj_q"] = obj_q
            obj_frame.wxyz = obj_q

        # Draw human skeleton for current frame
        human_joints = state.get("human_joints")
        frame_idx = int(state["frame_f"])
        if human_joints is not None and frame_idx < human_joints.shape[0]:
            _draw_human_skeleton(human_joints[frame_idx])

    def _set_sequence(idx: int) -> None:
        idx = int(np.clip(idx, 0, len(files) - 1))
        qpos, fps, human_joints = _get_clip(idx)

        with state_lock:
            state["seq_idx"] = idx
            state["qpos"] = qpos
            state["human_joints"] = human_joints
            state["fps"] = int(fps)
            state["playing"] = False
            state["frame_f"] = 0.0
            state["prev_robot_q"] = None
            state["prev_obj_q"] = None

        # Auto-detect object from filename
        if auto_detect_cb.value:
            detected = detect_object_from_filename(Path(files[idx]).name, object_names)
            if detected and detected in object_visuals:
                _switch_object(detected)
                update_flag["object"] = True
                object_dropdown.value = detected
                update_flag["object"] = False

        obj_label = state["current_object"] or "none"
        title_md.content = (
            f"**{Path(files[idx]).name}**\n\n"
            f"Frames: {qpos.shape[0]} | FPS: {fps} | Sequence: {idx + 1}/{len(files)} | Object: {obj_label}"
        )

        update_flag["frame"] = True
        new_max = max(0, int(qpos.shape[0]) - 1)
        frame_slider.max = new_max
        frame_slider.value = 0
        update_flag["frame"] = False

        if use_clip_fps.value:
            fps_in.value = int(fps)

        _apply_frame(qpos[0])

    # --- GUI ---
    with server.gui.add_folder("Sequence"):
        title_md = server.gui.add_markdown("Loading...")
        seq_slider = server.gui.add_slider("Sequence", min=0, max=len(files) - 1, step=1, initial_value=0)
        prev_seq_btn = server.gui.add_button("Previous sequence")
        next_seq_btn = server.gui.add_button("Next sequence")

    with server.gui.add_folder("Object"):
        dropdown_options = ["(none)"] + object_names
        object_dropdown = server.gui.add_dropdown("Object", options=dropdown_options, initial_value="(none)")
        auto_detect_cb = server.gui.add_checkbox("Auto-detect from filename", initial_value=True)

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=0, step=1, initial_value=0)
        play_btn = server.gui.add_button("Play / Pause")
        use_clip_fps = server.gui.add_checkbox("Use clip FPS", initial_value=True)
        fps_in = server.gui.add_number("FPS", initial_value=int(args.fps), min=1, max=240, step=1)
        loop_cb = server.gui.add_checkbox("Loop", initial_value=bool(args.loop))

    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", initial_value=bool(args.show_meshes))
        show_human_cb = server.gui.add_checkbox("Show human skeleton", initial_value=True)
        human_offset_slider = server.gui.add_slider("Human X offset", min=-5.0, max=5.0, step=0.1, initial_value=0.0)

    @show_meshes_cb.on_update
    def _(_evt) -> None:
        vr.show_visual = bool(show_meshes_cb.value)
        cur_obj = state["current_object"]
        if cur_obj and cur_obj in object_visuals:
            object_visuals[cur_obj][0].show_visual = bool(show_meshes_cb.value)

    @show_human_cb.on_update
    def _(_evt) -> None:
        vis = bool(show_human_cb.value)
        if human_points_handle is not None:
            human_points_handle.visible = vis
        if human_lines_handle is not None:
            human_lines_handle.visible = vis

    @human_offset_slider.on_update
    def _(_evt) -> None:
        # Re-draw skeleton at new offset
        with state_lock:
            human_joints = state.get("human_joints")
            frame_f = state["frame_f"]
        if human_joints is not None:
            i = int(np.clip(int(frame_f), 0, int(human_joints.shape[0]) - 1))
            _draw_human_skeleton(human_joints[i])

    @object_dropdown.on_update
    def _(_evt) -> None:
        if update_flag["object"]:
            return
        val = object_dropdown.value
        new_obj = None if val == "(none)" else val
        _switch_object(new_obj)
        # Jump to the first sequence of the selected object
        if new_obj and new_obj in object_file_indices and object_file_indices[new_obj]:
            first_idx = object_file_indices[new_obj][0]
            update_flag["seq"] = True
            seq_slider.value = first_idx
            update_flag["seq"] = False
            _set_sequence(first_idx)
        elif new_obj is None:
            # (none) selected — just keep current sequence
            pass

    @seq_slider.on_update
    def _(_evt) -> None:
        if update_flag["seq"]:
            return
        raw_val = seq_slider.value
        if raw_val != raw_val:  # NaN guard
            return
        _set_sequence(int(raw_val))

    @prev_seq_btn.on_click
    def _(_evt) -> None:
        with state_lock:
            nxt = (int(state["seq_idx"]) - 1) % len(files)
        update_flag["seq"] = True
        seq_slider.value = nxt
        update_flag["seq"] = False
        _set_sequence(nxt)

    @next_seq_btn.on_click
    def _(_evt) -> None:
        with state_lock:
            nxt = (int(state["seq_idx"]) + 1) % len(files)
        update_flag["seq"] = True
        seq_slider.value = nxt
        update_flag["seq"] = False
        _set_sequence(nxt)

    @play_btn.on_click
    def _(_evt) -> None:
        with state_lock:
            state["playing"] = not bool(state["playing"])

    @loop_cb.on_update
    def _(_evt) -> None:
        with state_lock:
            state["loop"] = bool(loop_cb.value)

    @frame_slider.on_update
    def _(_evt) -> None:
        if update_flag["frame"]:
            return
        with state_lock:
            qpos = state["qpos"]
            if qpos is None:
                return
            state["playing"] = False
            raw_val = frame_slider.value
            if raw_val != raw_val:  # NaN guard
                return
            i = int(np.clip(int(raw_val), 0, int(qpos.shape[0]) - 1))
            state["frame_f"] = float(i)
            state["prev_robot_q"] = None
            state["prev_obj_q"] = None
            frame_q = qpos[i]
        _apply_frame(frame_q)

    _set_sequence(0)

    def playback_loop() -> None:
        while True:
            with state_lock:
                qpos = state["qpos"]
                playing = bool(state["playing"])
                clip_fps = int(state["fps"])
                frame_f = float(state["frame_f"])
                do_loop = bool(state["loop"])

            if qpos is None or qpos.shape[0] <= 1 or not playing:
                time.sleep(0.02)
                continue

            fps_val = clip_fps if use_clip_fps.value else max(1, int(fps_in.value))
            dt = 1.0 / float(max(1, fps_val))

            frame_f = frame_f + 1.0
            n_frames = int(qpos.shape[0])

            if frame_f >= n_frames:
                if do_loop:
                    frame_f = 0.0
                else:
                    frame_f = float(n_frames - 1)
                    with state_lock:
                        state["playing"] = False

            i = int(np.clip(int(frame_f), 0, n_frames - 1))
            frame_q = qpos[i]

            with state_lock:
                state["frame_f"] = float(i)

            _apply_frame(frame_q)

            update_flag["frame"] = True
            frame_slider.value = i
            update_flag["frame"] = False

            time.sleep(dt)

    def terminal_keys_loop() -> None:
        if not is_tty_stdin():
            return
        if os.name != "posix":
            return

        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        print("[playlist] Keyboard shortcuts: n/right=next, p/left=prev, space=play/pause, q=quit")

        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch in {"q", "\x03"}:  # q or Ctrl+C
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    print("\n[playlist] Exiting...")
                    os._exit(0)
                if ch == " ":
                    with state_lock:
                        state["playing"] = not bool(state["playing"])
                    continue
                if ch in {"n", "N"}:
                    with state_lock:
                        nxt = (int(state["seq_idx"]) + 1) % len(files)
                    update_flag["seq"] = True
                    seq_slider.value = nxt
                    update_flag["seq"] = False
                    _set_sequence(nxt)
                    continue
                if ch in {"p", "P"}:
                    with state_lock:
                        nxt = (int(state["seq_idx"]) - 1) % len(files)
                    update_flag["seq"] = True
                    seq_slider.value = nxt
                    update_flag["seq"] = False
                    _set_sequence(nxt)
                    continue

                # Arrow keys: ESC [ C (right), ESC [ D (left)
                if ch == "\x1b":
                    ch2 = sys.stdin.read(1)
                    ch3 = sys.stdin.read(1)
                    if ch2 == "[" and ch3 == "C":
                        with state_lock:
                            nxt = (int(state["seq_idx"]) + 1) % len(files)
                        update_flag["seq"] = True
                        seq_slider.value = nxt
                        update_flag["seq"] = False
                        _set_sequence(nxt)
                    elif ch2 == "[" and ch3 == "D":
                        with state_lock:
                            nxt = (int(state["seq_idx"]) - 1) % len(files)
                        update_flag["seq"] = True
                        seq_slider.value = nxt
                        update_flag["seq"] = False
                        _set_sequence(nxt)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    threading.Thread(target=playback_loop, daemon=True).start()
    threading.Thread(target=terminal_keys_loop, daemon=True).start()

    print(f"[playlist] Loaded {len(files)} files, {len(object_visuals)} objects")
    print("[playlist] Open the Viser URL printed above")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
