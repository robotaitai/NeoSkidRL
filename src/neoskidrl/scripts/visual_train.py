from __future__ import annotations

import argparse
import os
import time

import numpy as np


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC and periodically visualize progress.")
    parser.add_argument("--config", default="config/default.yml", help="Path to config YAML.")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--chunk-steps", type=int, default=20_000)
    parser.add_argument("--rollout-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", default="runs/tb")
    parser.add_argument("--model-out", default="runs/skidnav_sac.zip")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--headless", action="store_true", help="Set MUJOCO_GL=egl for offscreen rendering.")
    args = parser.parse_args()

    if args.headless and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("stable-baselines3 not installed. Use `pip install -e .[train]`.") from exc

    from neoskidrl.envs import NeoSkidNavEnv

    train_env = DummyVecEnv([lambda: NeoSkidNavEnv(config_path=args.config, render_mode=None)])
    train_env = VecMonitor(train_env)

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=args.logdir,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
    )

    viz_env = NeoSkidNavEnv(config_path=args.config, render_mode="rgb_array")

    steps_done = 0
    while steps_done < args.total_steps:
        chunk = min(args.chunk_steps, args.total_steps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_done += chunk
        _run_live_rollout(viz_env, model, args.rollout_steps, args.fps, args.seed)

    model.save(args.model_out)


if __name__ == "__main__":
    main()
