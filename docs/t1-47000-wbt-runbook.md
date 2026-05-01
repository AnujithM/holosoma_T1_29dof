# T1 47000 WBT Runbook

This repo is installed into `env_isaaclab` for Holosoma WBT evaluation.

## Checkpoint

Use the Holosoma checkpoint:

```text
logs/WholeBodyTracking/20260413_200959-t1_29dof_wbt_manager-locomotion/model_47000.pt
```

The checkpoint metadata says:

```text
iter: 47000
env: holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager
robot: t1_29dof
algo: PPO
motion_dir: motions_t1_29dof_npz
```

## Dataset

The checkpoint expects Holosoma-format per-clip NPZ files in:

```text
motions_t1_29dof_npz
```

That directory has been rebuilt from the local v002 release:

```text
/home/mrahme/Minerva/data/releases/t1_29dof/omomo_amass_kimodo_v002
```

It currently contains `12,171` train clips and is about `6.1G`. The first and last converted clips were validated with Holosoma's `MotionLoader`.

To rebuild it:

```bash
conda run -n env_isaaclab python scripts/build_holosoma_t1_wbt_npz.py \
  --manifest /home/mrahme/Minerva/data/releases/t1_29dof/omomo_amass_kimodo_v002/manifests/omomo_amass_kimodo_v002_full.yaml \
  --split train \
  --output-dir motions_t1_29dof_npz
```

## Export ONNX

Holosoma sim2sim needs an ONNX policy, not the raw `.pt` checkpoint. Export from `model_47000.pt`:

```bash
conda activate env_isaaclab
bash scripts/export_t1_47000_wbt.sh
```

The export wrapper intentionally overrides the checkpoint's full `motion_dir` with one converted v002 NPZ clip. This keeps ONNX export from loading the whole 12,171-clip training corpus into VRAM.

Expected output:

```text
logs/WholeBodyTracking/20260413_200959-t1_29dof_wbt_manager-locomotion/exported/model_47000.onnx
```

## Sim2sim

Easiest launcher:

```bash
conda activate env_isaaclab
bash scripts/run_t1_47000_wbt_sim2sim.sh
```

That script starts the MuJoCo sim and the policy in separate `tmux` panes if `tmux` is available. If not, it falls back to separate `gnome-terminal` windows. The underlying manual commands are below.

Terminal 1, start MuJoCo sim with the bridge enabled:

```bash
conda activate env_isaaclab
bash scripts/run_t1_wbt_sim.sh
```

This uses Holosoma's `robot:t1-29dof-waist-wrist` robot preset, which is the repo's T1 29-DoF config name for direct MuJoCo simulation.

Terminal 2, run the exported policy:

```bash
conda activate env_isaaclab
bash scripts/run_t1_47000_wbt_policy.sh
```

If the policy reports that it is waiting for robot state, the MuJoCo bridge is not publishing yet. Start the sim terminal first and wait for the robot window/bridge to come up.

The policy wrapper also prepends the `pin` package's cmeel site path so `import pinocchio` resolves to the robotics Pinocchio bindings, not the unrelated PyPI nose plugin named `pinocchio`.

Policy terminal controls:

```text
Enter  initialize stiff hold mode
]      start policy
m      start WBT motion clip
o      stop policy and return to stiff hold
i      default pose
```

MuJoCo window controls:

```text
7/8        decrease/increase elastic band length
9          toggle elastic band enable/disable
Backspace  reset sim
```

The ONNXRuntime GPU discovery warning and the `requests` dependency warning are non-fatal in the tested setup.
