from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import yaml


def _make_env(config_path: str, seed: int | None, rank: int):
    def _init():
        from neoskidrl.envs import NeoSkidNavEnv

        env = NeoSkidNavEnv(config_path=config_path, render_mode=None)
        if seed is not None:
            env.reset(seed=seed + rank)
        return env

    return _init


def _resolve_run_dir(run_dir: str | None, run_name: str | None) -> Path:
    if run_dir:
        return Path(run_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or stamp
    return Path("runs/train") / name


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC on NeoSkidRL (headless-friendly).")
    parser.add_argument("--config", default="config/train.yml", help="Path to config YAML.")
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", default=None, help="Output directory for logs, model, and config.")
    parser.add_argument("--run-name", default=None, help="Folder name under runs/train/ if --run-dir is not set.")
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    parser.add_argument("--headless", action="store_true", help="Set MUJOCO_GL=egl for offscreen rendering.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.headless and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("stable-baselines3 not installed. Use `pip install -e .[train]`.") from exc

    run_dir = _resolve_run_dir(args.run_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    from neoskidrl.config import load_config

    cfg = load_config(args.config)
    (run_dir / "config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "args.yml").write_text(yaml.safe_dump(vars(args), sort_keys=False))

    env_fns = [_make_env(args.config, args.seed, i) for i in range(args.num_envs)]
    if args.num_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    env = VecMonitor(env)

    if args.algo == "sac":
        model = SAC(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
            learning_rate=3e-4,
            buffer_size=200_000,
            batch_size=256,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            device=args.device,
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            device=args.device,
        )

    model.learn(total_timesteps=args.total_steps)
    model.save(str(run_dir / "model.zip"))


if __name__ == "__main__":
    main()
