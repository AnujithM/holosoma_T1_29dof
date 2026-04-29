#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-logs/WholeBodyTracking/20260413_200959-t1_29dof_wbt_manager-locomotion/model_47000.pt}"
EXPORT_MOTION_FILE="${EXPORT_MOTION_FILE:-motions_t1_29dof_npz/clip_00000_sub5_suitcase_022.npz}"

unset PYTHONPATH
unset LD_LIBRARY_PATH
unset ISAACGYM_ROOT
export OMNI_KIT_ACCEPT_EULA=1

exec /home/mrahme/miniconda3/envs/env_isaaclab/bin/python -m holosoma.eval_agent \
  --checkpoint="${CHECKPOINT}" \
  --training.num-envs=1 \
  --training.headless=True \
  --training.export-onnx=True \
  --training.max-eval-steps=1 \
  --command.setup_terms.motion_command.params.motion_config.motion_dir="" \
  --command.setup_terms.motion_command.params.motion_config.motion_file="${EXPORT_MOTION_FILE}"
