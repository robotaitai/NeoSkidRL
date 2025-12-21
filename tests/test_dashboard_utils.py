"""Tests for reward dashboard utility functions."""
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


def test_load_episodes_jsonl():
    """Test loading episodes from JSONL file."""
    _require_dashboard_deps()
    
    from neoskidrl.ui.reward_dashboard import load_episodes_jsonl
    
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "test_episodes.jsonl"
        
        # Write fake episode data
        episodes = [
            {
                "timestamp": 1234567890.0,
                "run_id": "test_run_1",
                "algo": "SAC",
                "seed": 0,
                "episode_idx": 0,
                "ep_len": 100,
                "ep_return": -5.5,
                "success": False,
                "collision": False,
                "stuck": False,
                "reward_terms_sum": {"progress": 2.0, "time": -100.0, "smooth": -2.5},
            },
            {
                "timestamp": 1234567900.0,
                "run_id": "test_run_1",
                "algo": "SAC",
                "seed": 0,
                "episode_idx": 1,
                "ep_len": 50,
                "ep_return": 10.0,
                "success": True,
                "collision": False,
                "stuck": False,
                "reward_terms_sum": {"progress": 5.0, "time": -50.0, "smooth": -1.0, "goal_bonus": 1.0},
            },
        ]
        
        with jsonl_path.open("w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")
        
        # Load and verify
        df = load_episodes_jsonl(jsonl_path)
        
        assert len(df) == 2
        assert "ep_return" in df.columns
        assert "success" in df.columns
        assert df.iloc[0]["ep_return"] == -5.5
        assert df.iloc[1]["success"] is True


def test_load_episodes_with_filter():
    """Test filtering episodes by run_id."""
    _require_dashboard_deps()
    
    from neoskidrl.ui.reward_dashboard import load_episodes_jsonl
    
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "test_episodes.jsonl"
        
        episodes = [
            {"run_id": "run_a", "episode_idx": 0, "ep_return": 1.0},
            {"run_id": "run_b", "episode_idx": 1, "ep_return": 2.0},
            {"run_id": "run_a", "episode_idx": 2, "ep_return": 3.0},
        ]
        
        with jsonl_path.open("w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")
        
        # Filter for run_a
        df = load_episodes_jsonl(jsonl_path, run_id_filter="run_a")
        
        assert len(df) == 2
        assert all(df["run_id"] == "run_a")


def test_compute_metrics():
    """Test metrics computation from episode dataframe."""
    _require_dashboard_deps()
    import pandas as pd
    
    from neoskidrl.ui.reward_dashboard import compute_metrics
    
    # Create fake episode data
    data = {
        "episode_idx": [0, 1, 2, 3, 4],
        "ep_return": [1.0, 2.0, 3.0, 4.0, 5.0],
        "ep_len": [100, 100, 100, 100, 100],
        "success": [False, True, False, True, True],
        "collision": [True, False, False, False, False],
        "stuck": [False, False, True, False, False],
    }
    
    df = pd.DataFrame(data)
    metrics = compute_metrics(df)
    
    assert metrics["total_episodes"] == 5
    assert metrics["success_rate"] == 60.0  # 3/5 * 100
    assert metrics["collision_rate"] == 20.0  # 1/5 * 100
    assert metrics["stuck_rate"] == 20.0  # 1/5 * 100
    assert metrics["avg_return"] == 3.0  # (1+2+3+4+5)/5
    assert metrics["avg_ep_len"] == 100.0


def test_get_reward_weights_legacy_format():
    """Test extracting reward weights from legacy config format."""
    _require_dashboard_deps()
    from neoskidrl.ui.reward_dashboard import get_reward_weights
    
    config = {
        "reward": {
            "w_progress": 1.0,
            "w_time": -0.01,
            "w_smooth": -0.1,
            "w_collision": -10.0,
            "w_goal_bonus": 10.0,
        }
    }
    
    weights = get_reward_weights(config)
    
    assert weights["progress"] == 1.0
    assert weights["time"] == -0.01
    assert weights["smooth"] == -0.1
    assert weights["collision"] == -10.0
    assert weights["goal_bonus"] == 10.0


def test_get_reward_weights_new_format():
    """Test extracting reward weights from new config format."""
    _require_dashboard_deps()
    from neoskidrl.ui.reward_dashboard import get_reward_weights
    
    config = {
        "reward": {
            "weights": {
                "progress": 2.0,
                "time": -0.02,
                "smooth": -0.2,
                "collision": -20.0,
                "goal_bonus": 20.0,
            }
        }
    }
    
    weights = get_reward_weights(config)
    
    assert weights["progress"] == 2.0
    assert weights["time"] == -0.02


def test_set_reward_weights():
    """Test updating reward weights in config."""
    _require_dashboard_deps()
    from neoskidrl.ui.reward_dashboard import set_reward_weights
    
    config = {
        "reward": {
            "w_progress": 1.0,
            "w_time": -0.01,
        }
    }
    
    new_weights = {
        "progress": 3.0,
        "time": -0.03,
        "smooth": -0.5,
    }
    
    updated_config = set_reward_weights(config, new_weights)
    
    assert updated_config["reward"]["w_progress"] == 3.0
    assert updated_config["reward"]["w_time"] == -0.03
    assert updated_config["reward"]["w_smooth"] == -0.5


def test_expand_reward_terms():
    """Test expanding reward_terms_sum into separate columns."""
    _require_dashboard_deps()
    import pandas as pd
    
    from neoskidrl.ui.reward_dashboard import expand_reward_terms
    
    data = {
        "episode_idx": [0, 1],
        "ep_return": [1.0, 2.0],
        "reward_terms_sum": [
            {"progress": 1.0, "time": -10.0},
            {"progress": 2.0, "time": -20.0},
        ],
    }
    
    df = pd.DataFrame(data)
    expanded = expand_reward_terms(df)
    
    assert "term_progress" in expanded.columns
    assert "term_time" in expanded.columns
    assert expanded.iloc[0]["term_progress"] == 1.0
    assert expanded.iloc[1]["term_time"] == -20.0
