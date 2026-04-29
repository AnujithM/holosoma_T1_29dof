#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-logs/WholeBodyTracking/20260413_200959-t1_29dof_wbt_manager-locomotion/exported/model_47000.onnx}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  cat >&2 <<EOF
ONNX model not found: ${MODEL_PATH}

Export it first:
  bash scripts/export_t1_47000_wbt.sh
EOF
  exit 1
fi

unset PYTHONPATH
unset LD_LIBRARY_PATH
unset ISAACGYM_ROOT
export PYTHONPATH="/home/mrahme/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/cmeel.prefix/lib/python3.11/site-packages"

exec /home/mrahme/miniconda3/envs/env_isaaclab/bin/python -m holosoma_inference.run_policy \
  inference:t1-29dof-wbt \
  --task.model-path "${MODEL_PATH}" \
  --task.no-use-joystick \
  --task.use-sim-time \
  --task.rl-rate 50 \
  --task.interface lo
