#!/usr/bin/env python3
"""Convert local Minerva T1 motion refs into Holosoma WBT NPZ clips."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml


BODY_NAME_ALIASES = {
    "left_inspire_hand": "left_hand_link",
    "right_inspire_hand": "right_hand_link",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def _component_roots(manifest_path: Path, metadata_path: Path | None) -> dict[str, Path]:
    candidates = []
    if metadata_path is not None:
        candidates.append(metadata_path.expanduser().resolve())
    candidates.append(manifest_path.parent.parent / "metadata.json")

    for candidate in candidates:
        if not candidate.is_file():
            continue
        metadata = _load_json(candidate)
        roots: dict[str, Path] = {}
        for source in metadata.get("sources", []):
            if not isinstance(source, dict):
                continue
            name = source.get("name")
            root_path = source.get("root_path")
            if isinstance(name, str) and isinstance(root_path, str):
                roots[name] = Path(root_path).expanduser().resolve()
        if roots:
            return roots
    return {}


def _resolve_clip_path(root_path: Path, entry_file: str, component_roots: dict[str, Path]) -> Path:
    direct = root_path / entry_file
    if direct.is_file():
        return direct

    parts = Path(entry_file).parts
    if len(parts) >= 3 and parts[0].startswith("clips_"):
        component_root = component_roots.get(parts[1])
        if component_root is not None:
            candidate = component_root.joinpath(*parts[2:])
            if candidate.is_file():
                return candidate

    if len(parts) >= 2:
        component_root = component_roots.get(parts[0])
        if component_root is not None:
            candidate = component_root.joinpath(*parts[1:])
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(f"Could not resolve clip path for {entry_file!r}; tried {direct}")


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "clip"


def _convert_clip(clip: dict[str, Any]) -> dict[str, np.ndarray]:
    joint_pos = np.asarray(clip["joint_pos"], dtype=np.float32)
    joint_vel = np.asarray(clip["joint_vel"], dtype=np.float32)
    body_pos = np.asarray(clip["body_pos"], dtype=np.float32)
    body_quat = np.asarray(clip["body_quat"], dtype=np.float32)
    body_lin_vel = np.asarray(clip["body_lin_vel"], dtype=np.float32)
    body_ang_vel = np.asarray(clip["body_ang_vel"], dtype=np.float32)

    root_pos = body_pos[:, 0]
    root_quat = body_quat[:, 0]
    root_lin_vel = body_lin_vel[:, 0]
    root_ang_vel = body_ang_vel[:, 0]

    body_names = [BODY_NAME_ALIASES.get(name, name) for name in clip["body_names"]]
    joint_names = list(clip["joint_names"])

    joint_pos_full = np.concatenate([root_pos, root_quat, joint_pos], axis=1)
    joint_vel_full = np.concatenate([root_lin_vel, root_ang_vel, joint_vel], axis=1)

    return {
        "fps": np.asarray([float(clip.get("fps", 30.0))], dtype=np.float32),
        "joint_pos": joint_pos_full,
        "joint_vel": joint_vel_full,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
        "body_names": np.asarray(body_names),
        "joint_names": np.asarray(joint_names),
        "qpos": joint_pos_full,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="motions_t1_29dof_npz", type=Path)
    parser.add_argument("--metadata", default=None, type=Path)
    parser.add_argument("--max-clips", default=0, type=int)
    parser.add_argument("--progress-every", default=500, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = _load_yaml(manifest_path)
    split = args.split.lower()
    entries = [
        entry
        for entry in manifest.get("clips", [])
        if isinstance(entry, dict) and str(entry.get("split", "")).lower() == split
    ]
    if args.max_clips > 0:
        entries = entries[: args.max_clips]
    if not entries:
        raise ValueError(f"No clips found for split={args.split!r} in {manifest_path}")

    root_path = Path(str(manifest.get("root_path", "."))).expanduser()
    if not root_path.is_absolute():
        root_path = (manifest_path.parent / root_path).resolve()
    component_roots = _component_roots(manifest_path, args.metadata)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Manifest: {manifest_path}")
    print(f"[INFO] Split: {split} ({len(entries)} clips)")
    print(f"[INFO] Output dir: {output_dir}")

    converted = 0
    skipped = 0
    for index, entry in enumerate(entries):
        clip_id = str(entry.get("clip_id") or entry.get("source_id") or index)
        out_path = output_dir / f"clip_{index:05d}_{_safe_name(clip_id)}.npz"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        clip_path = _resolve_clip_path(root_path, str(entry["file"]), component_roots)
        clip = _load_json(clip_path)
        arrays = _convert_clip(clip)
        np.savez_compressed(out_path, **arrays)
        converted += 1

        done = index + 1
        if args.progress_every > 0 and (done % args.progress_every == 0 or done == len(entries)):
            print(f"[INFO] Processed {done}/{len(entries)} clips (converted={converted}, skipped={skipped})", flush=True)

    print(f"[INFO] Done. converted={converted}, skipped={skipped}, output_dir={output_dir}")


if __name__ == "__main__":
    main()
