from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from neoskidrl.envs import NeoSkidNavEnv


def _simple_goal_policy(obs: np.ndarray, env: NeoSkidNavEnv) -> np.ndarray:
    rays = int(env.rays)
    dx, dy, dyaw = obs[rays:rays + 3]
    v = np.clip(0.6 * dx, -1.0, 1.0)
    w = np.clip(2.0 * dy + 0.5 * dyaw, -1.0, 1.0)

    if env.action_space_mode == "v_w":
        return np.array([v, w], dtype=np.float32)

    v_lin = v * float(env.v_max)
    w_ang = w * float(env.w_max)
    v_l = v_lin - w_ang * (float(env.track) / 2.0)
    v_r = v_lin + w_ang * (float(env.track) / 2.0)
    w_l = np.clip(v_l / float(env.wheel_r), -env.wheel_vel_max, env.wheel_vel_max)
    w_r = np.clip(v_r / float(env.wheel_r), -env.wheel_vel_max, env.wheel_vel_max)
    return np.array([w_l, w_r, w_l, w_r], dtype=np.float32) / float(env.wheel_vel_max)


def _write_ppm(path: Path, frame: np.ndarray) -> None:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    height, width, channels = frame.shape
    if channels != 3:
        raise ValueError(f"Expected RGB frame, got shape {frame.shape}")
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(frame.tobytes())


def _write_pgm(path: Path, depth: np.ndarray) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(np.max(depth))
    if max_val < 1e-6:
        max_val = 1.0
    scaled = np.clip(depth / max_val, 0.0, 1.0)
    img = (255.0 * (1.0 - scaled)).astype(np.uint8)
    height, width = img.shape
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(img.tobytes())


def _lidar_points(obs: np.ndarray, rays: int, lidar_range: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    ranges = obs[:rays] * lidar_range
    angles = np.linspace(-np.pi, np.pi, rays, endpoint=False)
    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)
    dx, dy, _dyaw = obs[rays:rays + 3]
    return xs, ys, float(dx), float(dy)


def _load_policy(policy: str, model_path: str | None):
    if policy == "random":
        return None
    if policy == "heuristic":
        return None
    if policy == "sac":
        if model_path is None:
            raise ValueError("--model is required for policy=sac")
        try:
            from stable_baselines3 import SAC
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("stable-baselines3 not installed. Use `pip install -e .[train]`.") from exc
        return SAC.load(model_path)
    raise ValueError(f"Unknown policy: {policy}")


def _choose_action(policy: str, model, obs: np.ndarray, env: NeoSkidNavEnv) -> np.ndarray:
    if policy == "random":
        return env.action_space.sample()
    if policy == "heuristic":
        return _simple_goal_policy(obs, env)
    action, _state = model.predict(obs, deterministic=True)
    return np.asarray(action, dtype=np.float32)


def _run_live(env: NeoSkidNavEnv, steps: int, policy: str, model, fps: float, seed: int | None) -> None:
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
        action = _choose_action(policy, model, obs, env)
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


def _run_frames(
    env: NeoSkidNavEnv,
    steps: int,
    outdir: Path,
    policy: str,
    model,
    render: str,
    seed: int | None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    obs, _info = env.reset(seed=seed)

    frame = env.render()
    if frame is None:
        raise RuntimeError(
            "Renderer not available (no OpenGL context). If headless, set MUJOCO_GL=egl or osmesa."
        )
    if render == "rgb":
        _write_ppm(outdir / "frame_0000.ppm", frame)
    else:
        _write_pgm(outdir / "frame_0000.pgm", frame)

    for step in range(1, steps + 1):
        action = _choose_action(policy, model, obs, env)
        obs, _r, term, trunc, _info = env.step(action)
        frame = env.render()
        if render == "rgb":
            _write_ppm(outdir / f"frame_{step:04d}.ppm", frame)
        else:
            _write_pgm(outdir / f"frame_{step:04d}.pgm", frame)
        if term or trunc:
            obs, _info = env.reset()


def main() -> None:
    parser = argparse.ArgumentParser(description="NeoSkidRL visual demo (camera + lidar)")
    parser.add_argument("--config", default="config/default.yml", help="Path to config YAML.")
    parser.add_argument("--mode", choices=["live", "frames"], default="live")
    parser.add_argument("--policy", choices=["heuristic", "random", "sac"], default="heuristic")
    parser.add_argument("--model", default=None, help="Path to SB3 model for policy=sac.")
    parser.add_argument("--render", choices=["rgb", "depth"], default="rgb")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default="runs/demo_frames")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    render_mode = "rgb_array"
    if args.mode == "frames" and args.render == "depth":
        render_mode = "depth_array"

    env = NeoSkidNavEnv(config_path=args.config, render_mode=render_mode)
    model = _load_policy(args.policy, args.model)
    if args.mode == "live":
        _run_live(env, args.steps, args.policy, model, args.fps, args.seed)
    else:
        _run_frames(env, args.steps, Path(args.outdir), args.policy, model, args.render, args.seed)


if __name__ == "__main__":
    main()
