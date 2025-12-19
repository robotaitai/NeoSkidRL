# NeoSkidRL

MuJoCo + Gymnasium skid-steer navigation with procedural obstacles.

## Quick start

Create a venv and install deps:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Smoke test (random actions):

```bash
python -m neoskidrl.scripts.smoke_test
```

Headless SAC training (optional dependency):

```bash
pip install -e .[train]
python -m neoskidrl.scripts.train --algo sac --config config/train.yml --num-envs 64 --headless
```

Headless PPO training (optional dependency):

```bash
pip install -e .[train]
python -m neoskidrl.scripts.train --algo ppo --config config/train.yml --num-envs 64 --headless
```

Evaluation with video (optional dependency):

```bash
pip install -e .[train,video]
python -m neoskidrl.scripts.eval --algo sac --model runs/train/<run_name>/model.zip --config config/eval.yml --scenario easy
```

Evaluation with PPO:

```bash
pip install -e .[train,video]
python -m neoskidrl.scripts.eval --algo ppo --model runs/train/<run_name>/model.zip --config config/eval.yml --scenario easy
```

Metrics summary:

```bash
python -m neoskidrl.scripts.read_metrics runs/eval_videos
```

## Environment

- Gymnasium env: `neoskidrl.envs.NeoSkidNavEnv`
- Observations: 2D lidar normalized ranges, goal relative pose `(dx, dy, dyaw)`, speed `(v, wz)`
- Rendering: `render_mode="rgb_array"` or `render_mode="depth_array"` for visualization (not used in policy)
- Actions (configurable):
  - `control.action_space: v_w` -> normalized `(v, w)` mapped to wheel velocities
  - `control.action_space: wheel_velocities` -> normalized `[w_fl, w_fr, w_rl, w_rr]`
- Procedural obstacles: random mix of boxes/cylinders per episode
- Termination: success, collision, stuck, or timeout (~10s by default)

Config overview:
- `config/train.yml`: training defaults (randomized)
- `config/eval.yml`: eval scenarios + fixed seeds
- `config/default.yml`: legacy defaults
- Reward terms are computed in code and aggregated via `reward.enabled_terms` + `reward.weights`.

Reproducibility: `train` saves the exact config + args under `runs/train/<run_name>/`.

Periodic eval during training is configured in `config/train.yml` under `train.eval`.
Outputs go to `runs/checkpoints`, `runs/eval`, and `runs/eval_videos` by default.

Visual training checkpoints:
- `runs/latest.zip` is updated after every chunk and on exit.
- `runs/checkpoints/ckpt_<steps>.zip` is saved every chunk.

Example eval on the latest checkpoint:

```bash
python -m neoskidrl.scripts.eval --model runs/latest --config config/eval.yml --scenario easy --episodes 3 --headless
```

## Docs

- Visual demo: `doc/visual_demo.md`
- Training and eval: `doc/training_and_eval.md`

## Testing

```bash
pip install -e .[test]
pytest -q
```
