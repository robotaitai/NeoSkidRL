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
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-dir", default=None, help="Output directory for logs, model, and config.")
    parser.add_argument("--run-name", default=None, help="Folder name under runs/train/ if --run-dir is not set.")
    parser.add_argument("--algo", choices=["sac", "ppo"], default=None)
    parser.add_argument("--headless", action="store_true", help="Force headless rendering.")
    parser.add_argument("--no-headless", action="store_true", help="Disable headless rendering.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from neoskidrl.config import load_config

    cfg = load_config(args.config)
    train_cfg = cfg.get("train", {})

    algo = (args.algo or train_cfg.get("algo", "sac")).lower()
    total_steps = int(args.total_steps or train_cfg.get("total_timesteps", 300_000))
    num_envs = int(args.num_envs or train_cfg.get("num_envs", 1))
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 0))

    headless = bool(train_cfg.get("headless", False))
    if args.headless:
        headless = True
    if args.no_headless:
        headless = False

    if headless and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("stable-baselines3 not installed. Use `pip install -e .[train]`.") from exc

    run_dir = _resolve_run_dir(args.run_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "args.yml").write_text(yaml.safe_dump(vars(args), sort_keys=False))

    env_fns = [_make_env(args.config, seed, i) for i in range(num_envs)]
    if num_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    env = VecMonitor(env)

    if algo == "sac":
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

    callback = None
    eval_cfg = train_cfg.get("eval", {})
    if eval_cfg.get("enabled", False):
        from neoskidrl.train.callbacks import PeriodicEvalCallback

        eval_config_path = eval_cfg.get("eval_config", "config/eval.yml")
        video_cfg = eval_cfg.get("video", {})
        callback = PeriodicEvalCallback(
            eval_config_path=eval_config_path,
            scenario=str(eval_cfg.get("scenario", "easy")),
            eval_freq_steps=int(eval_cfg.get("eval_freq_steps", 10_000)),
            episodes=eval_cfg.get("episodes"),
            seeds=eval_cfg.get("seeds"),
            video_cfg=video_cfg,
            output_dir=str(eval_cfg.get("output_dir", "runs/eval")),
            video_dir=str(video_cfg.get("video_dir", "runs/eval_videos")),
            checkpoint_dir=str(eval_cfg.get("checkpoints", {}).get("dir", "runs/checkpoints")),
            headless=bool(eval_cfg.get("headless", True)),
            deterministic=bool(eval_cfg.get("deterministic", True)),
            algo=algo,
        )

    model.learn(total_timesteps=total_steps, callback=callback)
    model.save(str(run_dir / "model.zip"))


if __name__ == "__main__":
    main()
