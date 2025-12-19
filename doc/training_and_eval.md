# Training and Eval Instructions

This guide covers headless training at scale, a live training visualization, and policy evaluation with video output.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .[train]
```

## Headless training (many envs)

Use `--num-envs` for parallel training and `--headless` to set `MUJOCO_GL=egl`:

```bash
python -m neoskidrl.scripts.train --config config/train.yml --num-envs 256 --total-steps 2000000 --headless
```

Outputs (model + config) are saved under `runs/train/<run_name>/`.

For large runs (1000+ envs), increase system resources and expect slower per-step performance.

## Live training visualization (camera + lidar)

This runs SAC in chunks and opens a live window for the camera and lidar after each chunk:

```bash
pip install -e .[train,viz]
python -m neoskidrl.scripts.visual_train --config config/static_goal.yml --total-steps 300000 --chunk-steps 20000 --rollout-steps 400
```

The static goal is configured in `config/static_goal.yml`.

## Eval mode (scenarios + video)

Evaluate a trained policy on a deterministic scenario and record MP4:

```bash
pip install -e .[train,video]
python -m neoskidrl.scripts.eval --model runs/train/<run_name>/model.zip --config config/eval.yml --scenario medium
```

If you are running on a headless machine:

```bash
python -m neoskidrl.scripts.eval --model runs/train/<run_name>/model.zip --config config/eval.yml --scenario medium --headless
```

Videos and metrics are written to `runs/eval_videos/<scenario>/`.
Scenario presets are defined under `scenarios` in `config/eval.yml`.
