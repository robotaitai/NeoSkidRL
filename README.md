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

Minimal SAC training (optional dependency):

```bash
pip install -e .[train]
python -m neoskidrl.scripts.train_sac
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

Config lives in `config/default.yml` (also packaged under `src/neoskidrl/config/default.yml`).
Reward terms are computed in code and aggregated via `reward.enabled_terms` + `reward.weights` in the config.

## Docs

- Visual demo: `doc/visual_demo.md`
- Training and eval: `doc/training_and_eval.md`

## Testing

```bash
pip install -e .[test]
pytest -q
```
