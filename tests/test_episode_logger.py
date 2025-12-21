"""Tests for EpisodeJSONLLogger callback."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_episode_logger_writes_jsonl():
    """Test that episode logger writes valid JSONL with correct structure."""
    pytest.importorskip("stable_baselines3")
    
    from stable_baselines3.common.vec_env import DummyVecEnv
    from neoskidrl.envs import NeoSkidNavEnv
    from neoskidrl.train.callbacks import EpisodeJSONLLogger
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "episodes.jsonl"
        
        # Create a simple env
        env = DummyVecEnv([lambda: NeoSkidNavEnv(render_mode=None)])
        
        # Create logger
        logger = EpisodeJSONLLogger(
            output_path=output_path,
            run_id="test_run",
            algo="TEST",
            seed=42,
        )
        
        # Initialize logger
        logger.training_env = env
        logger._on_training_start()
        
        # Simulate a few steps of an episode
        obs = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 50:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            # Simulate callback step
            logger.num_timesteps = step_count
            logger.locals = {
                "infos": info,
                "dones": done,
                "rewards": reward,
            }
            logger._on_step()
            
            step_count += 1
        
        env.close()
        
        # Check that file was created and has valid content
        assert output_path.exists(), "JSONL file should be created"
        
        with output_path.open("r") as f:
            lines = f.readlines()
        
        # Should have at least one episode logged
        assert len(lines) >= 1, "Should have logged at least one episode"
        
        # Parse first line
        record = json.loads(lines[0])
        
        # Check required fields
        assert "timestamp" in record
        assert "run_id" in record
        assert record["run_id"] == "test_run"
        assert "algo" in record
        assert record["algo"] == "TEST"
        assert "seed" in record
        assert record["seed"] == 42
        assert "episode_idx" in record
        assert "ep_len" in record
        assert "ep_return" in record
        assert "success" in record
        assert "collision" in record
        assert "stuck" in record
        assert "reward_terms_sum" in record
        assert "reward_contrib_sum" in record
        assert "reward_contrib_abs_sum" in record
        
        # Check reward_terms_sum is a dict
        assert isinstance(record["reward_terms_sum"], dict)
        assert isinstance(record["reward_contrib_sum"], dict)
        assert isinstance(record["reward_contrib_abs_sum"], dict)


def test_episode_logger_multiple_episodes():
    """Test that logger handles multiple episodes correctly."""
    pytest.importorskip("stable_baselines3")
    
    from stable_baselines3.common.vec_env import DummyVecEnv
    from neoskidrl.envs import NeoSkidNavEnv
    from neoskidrl.train.callbacks import EpisodeJSONLLogger
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "episodes.jsonl"
        
        env = DummyVecEnv([lambda: NeoSkidNavEnv(render_mode=None)])
        
        logger = EpisodeJSONLLogger(
            output_path=output_path,
            run_id="multi_test",
            algo="TEST",
            seed=0,
        )
        
        logger.training_env = env
        logger._on_training_start()
        
        # Run multiple short episodes
        num_episodes = 3
        episodes_logged = 0
        
        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 20:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                
                logger.num_timesteps = ep * 20 + step_count
                logger.locals = {
                    "infos": info,
                    "dones": done,
                    "rewards": reward,
                }
                logger._on_step()
                
                if done[0]:
                    episodes_logged += 1
                
                step_count += 1
        
        env.close()
        
        # Check file content
        with output_path.open("r") as f:
            lines = f.readlines()
        
        # Should have logged episodes
        assert len(lines) >= 1, f"Should have logged at least 1 episode, got {len(lines)}"
        
        # Check episode indices are incrementing
        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["episode_idx"] == i
