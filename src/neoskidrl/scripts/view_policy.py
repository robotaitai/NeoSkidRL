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
    parser.add_argument("--show-ui", action="store_true", help="Show MuJoCo UI panels.")
    parser.add_argument("--bev", action="store_true", help="Start in bird's-eye view.")
    parser.add_argument("--bev-distance", type=float, default=6.0)
    parser.add_argument("--bev-elevation", type=float, default=-90.0)
    parser.add_argument("--bev-azimuth", type=float, default=90.0)
    parser.add_argument("--follow", action="store_true", help="Keep camera centered on the robot.")
    args = parser.parse_args()

    try:
        import mujoco
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

    show_ui = bool(args.show_ui)
    with mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=show_ui, show_right_ui=show_ui) as viewer:
        if args.bev:
            base_xy = _info.get("base_xy") if _info is not None else None
            if base_xy is not None:
                viewer.cam.lookat[:] = [float(base_xy[0]), float(base_xy[1]), 0.0]
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance = float(args.bev_distance)
            viewer.cam.elevation = float(args.bev_elevation)
            viewer.cam.azimuth = float(args.bev_azimuth)
        for _ in range(args.steps):
            if not viewer.is_running():
                break
            action, _state = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, _info = env.step(action)
            if args.follow and _info is not None and "base_xy" in _info:
                viewer.cam.lookat[:] = [float(_info["base_xy"][0]), float(_info["base_xy"][1]), 0.0]
            if term or trunc:
                obs, _info = env.reset()
            viewer.sync()
            if args.sleep > 0.0:
                time.sleep(args.sleep)

    env.close()


if __name__ == "__main__":
    main()
