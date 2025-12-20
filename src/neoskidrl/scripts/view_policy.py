from __future__ import annotations

import argparse
import time
from pathlib import Path

from neoskidrl.config import load_config, merge_config
from neoskidrl.scripts.eval import _normalize_model_path


def _resolve_base_path(eval_path: str, base_config: str) -> Path:
    base_path = Path(base_config)
    if not base_path.is_absolute():
        base_path = Path(eval_path).parent / base_path
    return base_path


def _resolve_env_config(config_path: str, scenario: str | None) -> dict:
    cfg = load_config(config_path)
    if "scenarios" in cfg and "base_config" in cfg:
        if scenario is None:
            raise ValueError("--scenario is required when using an eval config.")
        base_path = _resolve_base_path(config_path, cfg.get("base_config", "train.yml"))
        base_cfg = load_config(base_path)
        scenario_cfg = cfg.get("scenarios", {}).get(scenario)
        if scenario_cfg is None:
            raise ValueError(f"Unknown scenario '{scenario}'.")
        return merge_config(base_cfg, scenario_cfg)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="View a policy in the MuJoCo viewer (no video).")
    parser.add_argument("--model", required=True, help="Path to SB3 model (SAC/PPO).")
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    parser.add_argument("--config", default="config/eval.yml", help="Env or eval config.")
    parser.add_argument("--scenario", default="easy", help="Scenario (when using eval config).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional sleep per step.")
    args = parser.parse_args()

    try:
        import mujoco.viewer
    except Exception as exc:  # pragma: no cover - requires viewer
        raise RuntimeError("mujoco.viewer is not available (no GUI).") from exc

    try:
        from stable_baselines3 import PPO, SAC
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("stable-baselines3 not installed. Use `pip install -e .[train]`.") from exc

    from neoskidrl.envs import NeoSkidNavEnv

    cfg = _resolve_env_config(args.config, args.scenario)
    env = NeoSkidNavEnv(config=cfg, render_mode=None)
    obs, _info = env.reset(seed=args.seed)

    model_path = _normalize_model_path(args.model)
    if args.algo == "sac":
        model = SAC.load(model_path)
    else:
        model = PPO.load(model_path)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for _ in range(args.steps):
            if not viewer.is_running():
                break
            action, _state = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, _info = env.step(action)
            if term or trunc:
                obs, _info = env.reset()
            viewer.sync()
            if args.sleep > 0.0:
                time.sleep(args.sleep)

    env.close()


if __name__ == "__main__":
    main()
