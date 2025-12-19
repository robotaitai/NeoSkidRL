import os
from pathlib import Path

import pytest


def test_eval_callback_saves_checkpoint(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3 import SAC

    os.environ.setdefault("MUJOCO_GL", "egl")

    from neoskidrl.envs import NeoSkidNavEnv
    from neoskidrl.train.callbacks import PeriodicEvalCallback

    env = NeoSkidNavEnv(config_path="config/train.yml", render_mode=None)
    model = SAC("MlpPolicy", env, verbose=0, device="cpu")

    ckpt_dir = tmp_path / "checkpoints"
    eval_out = tmp_path / "eval"
    video_dir = tmp_path / "videos"

    callback = PeriodicEvalCallback(
        eval_config_path="config/eval.yml",
        scenario="easy",
        eval_freq_steps=32,
        episodes=1,
        seeds=[0],
        video_cfg={"enabled": False},
        output_dir=str(eval_out),
        video_dir=str(video_dir),
        checkpoint_dir=str(ckpt_dir),
        headless=True,
        deterministic=True,
        algo="sac",
    )

    model.learn(total_timesteps=64, callback=callback)
    env.close()

    ckpts = list(Path(ckpt_dir).glob("ckpt_*.zip"))
    assert ckpts, "No checkpoints saved."
    metrics = list(Path(eval_out).rglob("metrics.json"))
    assert metrics, "No eval metrics written."
