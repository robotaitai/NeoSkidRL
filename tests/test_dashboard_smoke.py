"""Smoke test for reward dashboard - verify it can load with fake data."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def _require_dashboard_deps():
    pytest.importorskip("streamlit")
    pytest.importorskip("streamlit_autorefresh")
    pytest.importorskip("altair")
    pytest.importorskip("pandas")


def test_dashboard_with_fake_data():
    """Test that dashboard utilities work with minimal fake episode data."""
    # This test doesn't launch Streamlit, just tests the data processing functions
    _require_dashboard_deps()
    
    from neoskidrl.ui.reward_dashboard import (
        load_episodes_jsonl,
        compute_metrics,
        expand_reward_terms,
        get_reward_weights,
        set_reward_weights,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake episodes.jsonl
        episodes_path = Path(tmpdir) / "episodes.jsonl"
        
        fake_episodes = [
            {
                "timestamp": 1234567890.0 + i,
                "run_id": "test_run",
                "algo": "SAC",
                "seed": 0,
                "episode_idx": i,
                "ep_len": 100 + i * 10,
                "ep_return": -5.0 + i * 2.0,
                "success": i % 3 == 0,
                "collision": i % 5 == 0,
                "stuck": i % 7 == 0,
                "reward_terms_sum": {
                    "progress": 2.0 + i * 0.5,
                    "time": -100.0 - i * 10,
                    "smooth": -2.5 - i * 0.1,
                    "collision": -10.0 if i % 5 == 0 else 0.0,
                    "goal_bonus": 10.0 if i % 3 == 0 else 0.0,
                },
                "timesteps": i * 100,
            }
            for i in range(20)
        ]
        
        with episodes_path.open("w") as f:
            for ep in fake_episodes:
                f.write(json.dumps(ep) + "\n")
        
        # Test loading
        df = load_episodes_jsonl(episodes_path)
        assert len(df) == 20
        assert "ep_return" in df.columns
        
        # Test metrics computation
        metrics = compute_metrics(df)
        assert metrics["total_episodes"] == 20
        assert 0 <= metrics["success_rate"] <= 100
        assert 0 <= metrics["collision_rate"] <= 100
        
        # Test expanding reward terms
        df_expanded = expand_reward_terms(df)
        assert "term_progress" in df_expanded.columns
        assert "term_time" in df_expanded.columns
        
        # Test config weight handling
        config = {
            "reward": {
                "w_progress": 1.0,
                "w_time": -0.01,
                "w_smooth": -0.1,
            }
        }
        
        weights = get_reward_weights(config)
        assert "progress" in weights
        assert weights["progress"] == 1.0
        
        # Test setting weights
        new_weights = {"progress": 2.0, "time": -0.02}
        updated_config = set_reward_weights(config, new_weights)
        assert updated_config["reward"]["w_progress"] == 2.0
        
        print("✓ Dashboard smoke test passed!")


if __name__ == "__main__":
    test_dashboard_with_fake_data()
