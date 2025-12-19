import os

import pytest


def test_run_eval_headless_metrics(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3 import SAC

    os.environ.setdefault("MUJOCO_GL", "egl")

    from neoskidrl.envs import NeoSkidNavEnv
    from neoskidrl.scripts.eval import run_eval

    env = NeoSkidNavEnv(config_path="config/train.yml", render_mode=None)
    model = SAC("MlpPolicy", env, verbose=0, device="cpu")
    model.learn(total_timesteps=64)
    model_path = tmp_path / "model.zip"
    model.save(str(model_path))
    env.close()

    out_dir = tmp_path / "eval"
    video_dir = tmp_path / "videos"
    summary = run_eval(
        model_path=str(model_path),
        eval_config_path="config/eval.yml",
        scenario="easy",
        episodes=1,
        output_dir=str(out_dir),
        video_dir=str(video_dir),
        record_video=False,
        headless=True,
        deterministic=True,
        algo="sac",
    )

    assert summary["episodes"] == 1
    assert "success_rate" in summary
    assert "avg_time_s" in summary
