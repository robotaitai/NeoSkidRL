import os
from pathlib import Path

import pytest


def test_visual_train_saves_checkpoints():
    pytest.importorskip("stable_baselines3")
    os.environ.setdefault("MUJOCO_GL", "egl")

    from neoskidrl.scripts.visual_train import run_training_chunks

    run_training_chunks(
        config_path="config/static_goal.yml",
        total_steps=128,
        chunk_steps=64,
        rollout_steps=0,
        seed=123,
        logdir="runs/tb",
        checkpoint_dir="runs/checkpoints",
        latest_path="runs/latest",
        model_out="runs/final",
        fps=0.0,
        headless=True,
        device="cpu",
        enable_viz=False,
    )

    ckpts = list(Path("runs/checkpoints").glob("ckpt_*.zip"))
    assert ckpts, "No checkpoint files created."
    assert Path("runs/latest.zip").exists()
