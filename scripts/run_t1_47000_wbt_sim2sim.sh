#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/mrahme/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-env_isaaclab}"
SESSION_NAME="${SESSION_NAME:-holosoma_t1_wbt_47000}"
POLICY_DELAY="${POLICY_DELAY:-3}"
MODEL_PATH="${1:-logs/WholeBodyTracking/20260413_200959-t1_29dof_wbt_manager-locomotion/exported/model_47000.onnx}"

build_cmd() {
  local script_path="$1"
  shift || true

  printf "cd %q && source %q && conda activate %q && bash %q" \
    "${ROOT_DIR}" "${CONDA_SH}" "${CONDA_ENV_NAME}" "${script_path}"
  for arg in "$@"; do
    printf " %q" "${arg}"
  done
}

SIM_CMD="$(build_cmd "scripts/run_t1_wbt_sim.sh")"
POLICY_CMD="$(build_cmd "scripts/run_t1_47000_wbt_policy.sh" "${MODEL_PATH}")"

if [[ ! -f "${ROOT_DIR}/${MODEL_PATH}" && ! -f "${MODEL_PATH}" ]]; then
  echo "ONNX model not found: ${MODEL_PATH}" >&2
  echo "Export it first: bash scripts/export_t1_47000_wbt.sh" >&2
  exit 1
fi

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}" >&2
    echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
    exit 1
  fi

  tmux new-session -d -s "${SESSION_NAME}" -n sim "${SIM_CMD}"
  tmux new-window -t "${SESSION_NAME}" -n policy "sleep ${POLICY_DELAY}; ${POLICY_CMD}"
  tmux select-window -t "${SESSION_NAME}:policy"

  echo "Started tmux session: ${SESSION_NAME}"
  echo "Policy controls: Enter, ], m, o"
  exec tmux attach -t "${SESSION_NAME}"
fi

if [[ -n "${DISPLAY:-}" ]] && command -v gnome-terminal >/dev/null 2>&1; then
  gnome-terminal --title "T1 WBT MuJoCo Sim" -- bash -lc "${SIM_CMD}; exec bash"
  sleep "${POLICY_DELAY}"
  gnome-terminal --title "T1 WBT Policy 47000" -- bash -lc "${POLICY_CMD}; exec bash"
  echo "Started two gnome-terminal windows."
  echo "Use the policy terminal controls: Enter, ], m, o"
  exit 0
fi

echo "No tmux or supported GUI terminal found. Run these in two terminals:" >&2
echo >&2
echo "Terminal 1:" >&2
echo "  ${SIM_CMD}" >&2
echo >&2
echo "Terminal 2:" >&2
echo "  ${POLICY_CMD}" >&2
exit 1
