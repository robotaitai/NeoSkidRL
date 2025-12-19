from __future__ import annotations

import argparse
import json
import os
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a policy and record MP4 videos.")
    parser.add_argument("--model", required=True, help="Path to SB3 model (SAC).")
    parser.add_argument("--config", default="config/eval.yml", help="Path to eval config YAML.")
    parser.add_argument("--scenario", default="easy", help="Scenario preset from eval config.")
    parser.add_argument("--seeds", default=None, help="Comma-separated list of seeds (override config).")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes to run.")
    parser.add_argument("--video-dir", default="runs/eval_videos")
    parser.add_argument("--headless", action="store_true", help="Set MUJOCO_GL=egl for offscreen rendering.")
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions.")
    args = parser.parse_args()

    if args.headless and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    try:
        from stable_baselines3 import PPO, SAC
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("stable-baselines3 not installed. Use `pip install -e .[train]`.") from exc

    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("imageio not installed. Use `pip install -e .[video]`.") from exc

    eval_cfg = load_config(args.config)
    base_path = _resolve_base_path(args.config, eval_cfg.get("base_config", "train.yml"))
    base_cfg = load_config(base_path)
    scenario_cfg = eval_cfg.get("scenarios", {}).get(args.scenario)
    if scenario_cfg is None:
        raise ValueError(f"Unknown scenario '{args.scenario}'.")
    cfg = merge_config(base_cfg, scenario_cfg)

    video_fps = int(eval_cfg.get("eval", {}).get("video_fps", 30))
    seeds = _parse_seeds(args.seeds, eval_cfg, args.episodes)

    out_dir = Path(args.video_dir) / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out_dir / "eval.yml").write_text(yaml.safe_dump(eval_cfg, sort_keys=False))

    from neoskidrl.envs import NeoSkidNavEnv

    if args.algo == "sac":
        model = SAC.load(args.model)
    else:
        model = PPO.load(args.model)
    deterministic = not args.stochastic

    step_dt = float(cfg["env"]["dt"]) * float(cfg["env"].get("frame_skip", 1))
    results = []
    for seed in seeds:
        env = NeoSkidNavEnv(config=cfg, render_mode="rgb_array")
        obs, _info = env.reset(seed=seed)
        video_path = out_dir / f"{args.scenario}_seed{seed}.mp4"
        writer = imageio.get_writer(str(video_path), fps=video_fps)

        frame = env.render()
        if frame is not None:
            writer.append_data(frame)

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
            frame = env.render()
            if frame is not None:
                writer.append_data(frame)
            done = term or trunc

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
                "video": str(video_path),
            }
        )

    success_rate = sum(1 for r in results if r["success"]) / max(1, len(results))
    collision_rate = sum(1 for r in results if r["collision"]) / max(1, len(results))
    stuck_rate = sum(1 for r in results if r["stuck"]) / max(1, len(results))

    summary = {
        "scenario": args.scenario,
        "algo": args.algo,
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
