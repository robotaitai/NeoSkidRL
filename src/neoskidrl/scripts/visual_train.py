from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from neoskidrl.utils import generate_run_name


def _make_env(config_path: str, render_mode: str | None = None):
    def _init():
        from neoskidrl.envs import NeoSkidNavEnv

        return NeoSkidNavEnv(config_path=config_path, render_mode=render_mode)

    return _init


def _lidar_points(obs: np.ndarray, rays: int, lidar_range: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    ranges = obs[:rays] * lidar_range
    angles = np.linspace(-np.pi, np.pi, rays, endpoint=False)
    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)
    dx, dy, _dyaw = obs[rays:rays + 3]
    return xs, ys, float(dx), float(dy)


def _run_live_rollout(env, model, steps: int, fps: float, seed: int | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib not installed. Use `pip install -e .[viz]`.") from exc

    obs, _info = env.reset(seed=seed)
    frame = env.render()
    if frame is None:
        raise RuntimeError(
            "Renderer not available (no OpenGL context). If headless, set MUJOCO_GL=egl or osmesa."
        )

    rays = int(env.rays)
    lidar_range = float(env.lidar_range)
    xs, ys, dx, dy = _lidar_points(obs, rays, lidar_range)

    plt.ion()
    fig, (ax_cam, ax_lidar) = plt.subplots(1, 2, figsize=(10, 5))
    cam_im = ax_cam.imshow(frame)
    ax_cam.set_title("Camera (RGB)")
    ax_cam.axis("off")

    lidar_scatter = ax_lidar.scatter(xs, ys, s=6, c="#1f77b4", alpha=0.8)
    goal_scatter = ax_lidar.scatter([dx], [dy], s=60, c="#d62728", marker="x")
    ax_lidar.scatter([0.0], [0.0], s=30, c="#111111")
    ax_lidar.set_xlim(-lidar_range, lidar_range)
    ax_lidar.set_ylim(-lidar_range, lidar_range)
    ax_lidar.set_aspect("equal", adjustable="box")
    ax_lidar.set_title("Lidar (robot frame)")
    ax_lidar.grid(True, alpha=0.2)

    dt = 0.0 if fps <= 0 else 1.0 / fps
    for _step in range(steps):
        if not plt.fignum_exists(fig.number):
            break
        action, _state = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, _info = env.step(action)
        frame = env.render()
        if frame is not None:
            cam_im.set_data(frame)
        xs, ys, dx, dy = _lidar_points(obs, rays, lidar_range)
        lidar_scatter.set_offsets(np.c_[xs, ys])
        goal_scatter.set_offsets(np.array([[dx, dy]], dtype=np.float32))
        fig.canvas.draw_idle()
        plt.pause(0.001)
        if dt > 0:
            time.sleep(dt)
        if term or trunc:
            obs, _info = env.reset()


def _save_checkpoint(model, checkpoint_dir: Path, latest_path: Path, steps_done: int) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"ckpt_{steps_done:06d}"
    ckpt_path = checkpoint_dir / ckpt_name
    model.save(str(ckpt_path))
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(latest_path))
    return ckpt_path


def run_training_chunks(
    config_path: str,
    total_steps: int,
    chunk_steps: int,
    rollout_steps: int,
    seed: int,
    run_name: str,
    logdir: str,
    checkpoint_dir: str,
    latest_path: str,
    model_out: str,
    fps: float,
    headless: bool,
    device: str,
    enable_viz: bool,
    num_envs: int = 1,
    vec_env: str | None = None,
    batch_size: int = 256,
    buffer_size: int = 200_000,
    learning_rate: float = 3e-4,
    ent_coef: str | float = "auto",
    target_entropy: str | float = "auto",
    resume: str | None = None,
    finetune_from: str | None = None,
    reset_timesteps: bool = False,
    eval_every_steps: int = 20000,
    eval_episodes: int = 10,
    eval_enabled: bool = True,
):
    if headless and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import CallbackList
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("stable-baselines3 not installed. Use `pip install -e .[train]`.") from exc

    from neoskidrl.envs import NeoSkidNavEnv
    from neoskidrl.train.callbacks import EpisodeJSONLLogger
    try:
        from neoskidrl.logging.rich_dashboard import RichDashboardLogger
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("rich not installed. Use `pip install -e .[train]`.") from exc

    from neoskidrl.train.callbacks_rich import SB3RichCallback

    print(f"\n{'='*60}")
    print(f"🚀 Starting training run: {run_name}")
    print(f"{'='*60}\n")
    
    print(f"Creating {num_envs} parallel training environments...")
    if vec_env is None:
        vec_env = "subproc" if num_envs >= 4 else "dummy"
    env_fns = [_make_env(config_path=config_path, render_mode=None) for _ in range(num_envs)]
    if vec_env == "subproc" and num_envs > 1:
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv(env_fns)
    train_env = VecMonitor(train_env)
    train_env.seed(seed)
    train_env.reset()

    # Build SAC model with configurable entropy
    sac_kwargs = {
        "policy": "MlpPolicy",
        "env": train_env,
        "verbose": 0,
        "tensorboard_log": logdir,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "device": device,
    }
    
    # Add entropy coefficient (auto or fixed value)
    if ent_coef != "auto":
        sac_kwargs["ent_coef"] = float(ent_coef)
    else:
        sac_kwargs["ent_coef"] = "auto"
    
    # Add target entropy if specified
    if target_entropy != "auto":
        sac_kwargs["target_entropy"] = float(target_entropy)
    
    print(f"SAC entropy config: ent_coef={ent_coef}, target_entropy={target_entropy}")
    
    load_path = resume or finetune_from
    if load_path:
        model = SAC.load(load_path, env=train_env, device=device)
        model.tensorboard_log = logdir
        print(f"✅ Loaded model from: {load_path}")
    else:
        model = SAC(**sac_kwargs)

    viz_env = None
    if enable_viz:
        viz_env = NeoSkidNavEnv(config_path=config_path, render_mode="rgb_array")

    # Setup episode logger for reward dashboard
    episode_logger = EpisodeJSONLLogger(
        output_path="runs/metrics/episodes.jsonl",
        run_id=run_name,  # Use the generated run name
        algo="SAC",
        seed=seed,
        verbose=1,
    )

    rich_logger = RichDashboardLogger(run_name=run_name, total_steps=total_steps, chunk_steps=chunk_steps)
    rich_callback = SB3RichCallback(
        logger=rich_logger,
        total_steps=total_steps,
        chunk_steps=chunk_steps,
        eval_every_steps=eval_every_steps,
        eval_episodes=eval_episodes,
        eval_enabled=eval_enabled,
        config_path=config_path,
    )
    callbacks = CallbackList([episode_logger, rich_callback])
    
    print(f"\n📊 Logging:")
    print(f"  Run ID: {run_name}")
    print(f"  Tensorboard: {logdir}")
    print(f"  Episodes: runs/metrics/episodes.jsonl")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Latest: {latest_path}\n")

    steps_done = 0
    latest = Path(latest_path)
    checkpoint_dir = Path(checkpoint_dir)
    first_learn = True
    try:
        while steps_done < total_steps:
            chunk = min(chunk_steps, total_steps - steps_done)
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=(reset_timesteps and first_learn),
                progress_bar=False,
                log_interval=10,
                callback=callbacks,
            )
            first_learn = False
            steps_done += chunk
            _save_checkpoint(model, checkpoint_dir, latest, steps_done)
            if enable_viz and viz_env is not None:
                _run_live_rollout(viz_env, model, rollout_steps, fps, seed)
    except KeyboardInterrupt:
        _save_checkpoint(model, checkpoint_dir, latest, steps_done)
        raise
    finally:
        _save_checkpoint(model, checkpoint_dir, latest, steps_done)
        train_env.close()
        if viz_env is not None:
            viz_env.close()
        if rich_logger is not None:
            rich_logger.close()

    model.save(model_out)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC and periodically visualize progress.")
    parser.add_argument("--config", default="config/default.yml", help="Path to config YAML.")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--chunk-steps", type=int, default=10_000, help="Steps per chunk (default: 10k for more frequent checkpoints)")
    parser.add_argument("--rollout-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None, help="Run name (auto-generated if not provided).")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument(
        "--vec-env",
        choices=["dummy", "subproc"],
        default=None,
        help="Vec env type (default: subproc if num_envs>=4).",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for SAC training.")
    parser.add_argument("--buffer-size", type=int, default=200_000, help="Replay buffer size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for SAC.")
    parser.add_argument("--ent-coef", default="auto", help="Entropy coefficient ('auto' or float, e.g. 0.1).")
    parser.add_argument("--target-entropy", default="auto", help="Target entropy ('auto' or float, e.g. -1.0).")
    parser.add_argument("--resume", default=None, help="Path to a saved SAC .zip to resume training (same run).")
    parser.add_argument("--finetune-from", default=None, help="Path to a saved SAC .zip to start from (new run).")
    parser.add_argument("--reset-timesteps", action="store_true", help="Reset timestep counter (useful for finetune).")
    parser.add_argument("--eval-every-steps", type=int, default=20000, help="Run eval every N steps.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of eval episodes.")
    parser.add_argument("--no-eval", action="store_true", help="Disable periodic evaluation.")
    parser.add_argument("--no-viz", action="store_true", help="Disable matplotlib rollout visualization.")
    parser.add_argument("--viz", action="store_true", help="Enable matplotlib rollout visualization.")
    parser.add_argument("--logdir", default="runs/tb")
    parser.add_argument("--model-out", default="runs/final")
    parser.add_argument("--checkpoint-dir", default="runs/checkpoints")
    parser.add_argument("--latest-path", default="runs/latest")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--headless", action="store_true", help="Set MUJOCO_GL=egl for offscreen rendering.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    
    if args.run_name is None and args.resume:
        args.run_name = Path(args.resume).stem.replace(".zip", "")

    # Generate run name if not provided
    if args.run_name is None:
        run_name = generate_run_name(prefix="sac")
    else:
        run_name = args.run_name
    
    enable_viz = args.viz and not args.headless
    if args.no_viz:
        enable_viz = False

    # Update paths to include run name
    logdir = f"{args.logdir}/{run_name}"
    checkpoint_dir = f"{args.checkpoint_dir}/{run_name}"
    latest_path = f"{args.latest_path}/{run_name}"
    model_out = f"{args.model_out}/{run_name}"

    run_training_chunks(
        config_path=args.config,
        total_steps=args.total_steps,
        chunk_steps=args.chunk_steps,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        run_name=run_name,
        logdir=logdir,
        checkpoint_dir=checkpoint_dir,
        latest_path=latest_path,
        model_out=model_out,
        fps=args.fps,
        headless=args.headless,
        device=args.device,
        enable_viz=enable_viz,
        num_envs=args.num_envs,
        vec_env=args.vec_env,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        target_entropy=args.target_entropy,
        resume=args.resume,
        finetune_from=args.finetune_from,
        reset_timesteps=args.reset_timesteps,
        eval_every_steps=args.eval_every_steps,
        eval_episodes=args.eval_episodes,
        eval_enabled=not args.no_eval,
    )


if __name__ == "__main__":
    main()
