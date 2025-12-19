from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np

from neoskidrl.config import load_config, merge_config
from neoskidrl.metrics import action_delta_l1, path_length_increment


def _resolve_base_path(eval_path: str, base_config: str) -> Path:
    base_path = Path(base_config)
    if not base_path.is_absolute():
        base_path = Path(eval_path).parent / base_path
    return base_path


def _parse_seeds(seed_arg: str | None, eval_cfg: dict, episodes: int | None) -> list[int]:
    if seed_arg:
        seeds = [int(s) for s in seed_arg.split(",") if s.strip()]
    else:
        seeds = [int(s) for s in eval_cfg.get("seeds", [0])]
    if episodes is not None:
        if len(seeds) < episodes:
            base = seeds[-1] if seeds else 0
            seeds = seeds + [base + i + 1 for i in range(episodes - len(seeds))]
        else:
            seeds = seeds[:episodes]
    return seeds


def _normalize_model_path(model_path: str) -> str:
    return model_path[:-4] if model_path.endswith(".zip") else model_path


def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    frame = np.asarray(frame)
    if frame.max() <= 1.0:
        frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _build_eval_config(eval_config_path: str, scenario: str) -> tuple[dict, dict]:
    eval_cfg = load_config(eval_config_path)
    base_path = _resolve_base_path(eval_config_path, eval_cfg.get("base_config", "train.yml"))
    base_cfg = load_config(base_path)
    scenario_cfg = eval_cfg.get("scenarios", {}).get(scenario)
    if scenario_cfg is None:
        raise ValueError(f"Unknown scenario '{scenario}'.")
    cfg = merge_config(base_cfg, scenario_cfg)
    return cfg, eval_cfg


def run_eval(
    model_path: str,
    eval_config_path: str = "config/eval.yml",
    scenario: str = "easy",
    seeds: list[int] | None = None,
    episodes: int | None = None,
    output_dir: str = "runs/eval",
    video_dir: str = "runs/eval_videos",
    record_video: bool = True,
    headless: bool = False,
    deterministic: bool = True,
    algo: str = "sac",
    run_id: str | None = None,
    camera: str | None = None,
) -> dict:
    model_path = _normalize_model_path(model_path)
    if headless and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    try:
        from stable_baselines3 import PPO, SAC
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("stable-baselines3 not installed. Use `pip install -e .[train]`.") from exc

    imageio = None
    if record_video:
        try:
            import imageio.v2 as imageio
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("imageio not installed. Use `pip install -e .[video]`.") from exc

    cfg, eval_cfg = _build_eval_config(eval_config_path, scenario)
    if seeds is None:
        seeds = _parse_seeds(None, eval_cfg, episodes)
    else:
        if episodes is not None:
            seeds = seeds[:episodes]
    if episodes is not None and len(seeds) < episodes:
        base = seeds[-1] if seeds else 0
        seeds = seeds + [base + i + 1 for i in range(episodes - len(seeds))]

    video_fps = int(eval_cfg.get("eval", {}).get("video_fps", 30))
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path(output_dir) / scenario / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    video_out_dir = Path(video_dir) / scenario / run_id
    if record_video:
        video_out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out_dir / "eval.yml").write_text(yaml.safe_dump(eval_cfg, sort_keys=False))

    from neoskidrl.envs import NeoSkidNavEnv

    if algo == "sac":
        model = SAC.load(model_path)
    else:
        model = PPO.load(model_path)

    render_mode = "rgb_array" if record_video else None
    step_dt = float(cfg["env"]["dt"]) * float(cfg["env"].get("frame_skip", 1))
    results = []
    for seed in seeds:
        env = NeoSkidNavEnv(config=cfg, render_mode=render_mode)
        if camera is not None:
            env.render_camera = camera
        obs, _info = env.reset(seed=seed)

        writer = None
        video_path = None
        if record_video:
            video_path = video_out_dir / f"{scenario}_seed{seed}.mp4"
            writer = imageio.get_writer(str(video_path), fps=video_fps)
            frame = env.render()
            if frame is not None:
                writer.append_data(_ensure_uint8(frame))

        done = False
        ep_return = 0.0
        ep_len = 0
        path_length = 0.0
        smoothness = 0.0
        prev_action = None
        prev_pos = _info.get("base_xy") if _info is not None else None
        last_info = {}
        while not done:
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, reward, term, trunc, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            last_info = info
            pos = info.get("base_xy") if info is not None else None
            if pos is not None:
                pos = np.asarray(pos, dtype=np.float32)
            path_length += path_length_increment(prev_pos, pos)
            smoothness += action_delta_l1(prev_action, np.asarray(action, dtype=np.float32))
            prev_pos = pos
            prev_action = np.asarray(action, dtype=np.float32)
            if writer is not None:
                frame = env.render()
                if frame is not None:
                    writer.append_data(_ensure_uint8(frame))
            done = term or trunc

        if writer is not None:
            writer.close()
        env.close()
        results.append(
            {
                "seed": seed,
                "return": ep_return,
                "steps": ep_len,
                "time_s": ep_len * step_dt,
                "path_length": path_length,
                "smoothness": smoothness,
                "success": bool(last_info.get("success", False)),
                "collision": bool(last_info.get("collision", False)),
                "stuck": bool(last_info.get("stuck", False)),
                "video": str(video_path) if video_path is not None else None,
            }
        )

    success_rate = sum(1 for r in results if r["success"]) / max(1, len(results))
    collision_rate = sum(1 for r in results if r["collision"]) / max(1, len(results))
    stuck_rate = sum(1 for r in results if r["stuck"]) / max(1, len(results))

    summary = {
        "scenario": scenario,
        "algo": algo,
        "run_id": run_id,
        "episodes": len(results),
        "avg_return": sum(r["return"] for r in results) / max(1, len(results)),
        "avg_steps": sum(r["steps"] for r in results) / max(1, len(results)),
        "avg_time_s": sum(r["time_s"] for r in results) / max(1, len(results)),
        "avg_path_length": sum(r["path_length"] for r in results) / max(1, len(results)),
        "avg_smoothness": sum(r["smoothness"] for r in results) / max(1, len(results)),
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "stuck_rate": stuck_rate,
        "results": results,
    }

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a policy and record MP4 videos.")
    parser.add_argument("--model", required=True, help="Path to SB3 model.")
    parser.add_argument("--config", default="config/eval.yml", help="Path to eval config YAML.")
    parser.add_argument("--scenario", default="easy", help="Scenario preset from eval config.")
    parser.add_argument("--seeds", default=None, help="Comma-separated list of seeds (override config).")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes to run.")
    parser.add_argument("--output-dir", default="runs/eval")
    parser.add_argument("--video-dir", default="runs/eval_videos")
    parser.add_argument("--headless", action="store_true", help="Set MUJOCO_GL=egl for offscreen rendering.")
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions.")
    parser.add_argument("--no-video", action="store_true", help="Disable MP4 recording.")
    parser.add_argument("--run-id", default=None, help="Optional run id for output folders.")
    parser.add_argument("--camera", default=None, help="Camera name from the MJCF (default: track).")
    args = parser.parse_args()

    seeds = None
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    summary = run_eval(
        model_path=args.model,
        eval_config_path=args.config,
        scenario=args.scenario,
        seeds=seeds,
        episodes=args.episodes,
        output_dir=args.output_dir,
        video_dir=args.video_dir,
        record_video=not args.no_video,
        headless=args.headless,
        deterministic=not args.stochastic,
        algo=args.algo,
        run_id=args.run_id,
        camera=args.camera,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
