#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH
unset LD_LIBRARY_PATH
unset ISAACGYM_ROOT

exec /home/mrahme/miniconda3/envs/env_isaaclab/bin/python -m holosoma.run_sim \
  robot:t1-29dof-waist-wrist \
  simulator:mujoco \
  --simulator.config.bridge.enabled=True \
  --simulator.config.bridge.interface=lo \
  "$@"
