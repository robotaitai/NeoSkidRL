# Training and Eval Instructions

This guide covers headless training at scale, a live training visualization, and policy evaluation with video output.
It also explains where metrics are written and what they mean.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .[train]
```

## Headless training (many envs)

Use `--num-envs` for parallel training and `--headless` to set `MUJOCO_GL=egl`:

```bash
python -m neoskidrl.scripts.train --algo sac --config config/train.yml --num-envs 256 --total-steps 2000000 --headless
```

Outputs (model + config) are saved under `runs/train/<run_name>/`.

### Automatic eval during training

Periodic evaluation is configured in `config/train.yml` under `train.eval`.
It runs in a separate env (no rendering in training envs), saves checkpoints, and writes:

- Checkpoints: `runs/checkpoints/ckpt_<timesteps>.zip`
- Metrics: `runs/eval/<scenario>/ckpt_<timesteps>/metrics.json`
- Videos (optional): `runs/eval_videos/<scenario>/ckpt_<timesteps>/*.mp4`

To disable eval or video, set:

```yaml
train:
  eval:
    enabled: false
    video:
      enabled: false
```

### PPO training

```bash
python -m neoskidrl.scripts.train --algo ppo --config config/train.yml --num-envs 256 --total-steps 2000000 --headless
```

For large runs (1000+ envs), increase system resources and expect slower per-step performance.

## Live training visualization (camera + lidar)

This runs SAC in chunks and opens a live window for the camera and lidar after each chunk.
If you want PPO visuals later, we can add a PPO variant of the visual training script.

```bash
pip install -e .[train,viz]
python -m neoskidrl.scripts.visual_train --config config/static_goal.yml --total-steps 300000 --chunk-steps 20000 --rollout-steps 400
```

The static goal is configured in `config/static_goal.yml`.

## Eval mode (scenarios + video)

Evaluate a trained policy on a deterministic scenario and record MP4:

```bash
pip install -e .[train,video]
python -m neoskidrl.scripts.eval --algo sac --model runs/train/<run_name>/model.zip --config config/eval.yml --scenario medium
```

If you are running on a headless machine:

```bash
python -m neoskidrl.scripts.eval --algo sac --model runs/train/<run_name>/model.zip --config config/eval.yml --scenario medium --headless
```

Videos and metrics are written to `runs/eval_videos/<scenario>/`.
Scenario presets are defined under `scenarios` in `config/eval.yml`.

## Camera selection

Rendered videos use the camera defined by `sensors.cameras.render_camera` in your config.
Default is `front` (on-vehicle). You can override per-run with:

```bash
python -m neoskidrl.scripts.eval --model runs/latest --config config/eval.yml --scenario easy --camera track
```

## MuJoCo viewer (no video)

To see the full MuJoCo scene with your latest policy:

```bash
python -m neoskidrl.scripts.view_policy --model runs/latest --config config/eval.yml --scenario medium --bev --follow
```

This opens the native MuJoCo viewer and runs the policy live.
Use `--show-ui` if you want the left/right MuJoCo panels visible.

### PPO evaluation

```bash
pip install -e .[train,video]
python -m neoskidrl.scripts.eval --algo ppo --model runs/train/<run_name>/model.zip --config config/eval.yml --scenario medium
```

## Metrics output (what you get per policy)

Each evaluation run writes `metrics.json` under `runs/eval_videos/<scenario>/`.
It includes per-episode and summary metrics:

- success rate: fraction of episodes that ended in success
- collision rate: fraction of episodes that ended in collision
- stuck rate: fraction of episodes that terminated due to being stuck
- time-to-goal: `steps * (dt * frame_skip)` in seconds
- steps: number of environment steps in the episode
- path length: total distance traveled in the XY plane
- smoothness: sum of `|Δaction|` across the episode

You can compare these across SAC vs PPO by running `eval` with `--algo sac` and `--algo ppo` and reading the metrics files.

### Metrics reader

To summarize multiple runs in a table:

```bash
python -m neoskidrl.scripts.read_metrics runs/eval_videos
```

It scans for `metrics.json` and prints a simple table grouped by scenario and algo.
