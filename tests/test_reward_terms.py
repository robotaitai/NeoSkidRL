import numpy as np

from neoskidrl.rewards.skidnav_reward import aggregate_reward, compute_reward_terms


def test_reward_terms_keys_and_values():
    action = np.array([0.1, -0.2], dtype=np.float32)
    terms = compute_reward_terms(
        dist=1.0,
        prev_dist=1.5,
        action=action,
        collided=False,
        success=True,
    )

    expected_keys = {"progress", "time", "smooth", "heading", "collision", "goal_bonus"}
    assert expected_keys.issubset(set(terms.keys()))
    assert all(np.isfinite(list(terms.values())))

    heading_terms = compute_reward_terms(
        dist=1.0,
        prev_dist=1.5,
        action=action,
        collided=False,
        success=False,
        goal_angle=0.2,
        prev_goal_angle=0.6,
    )
    assert heading_terms["heading"] > 0.0

    cfg = {
        "reward": {
            "enabled_terms": ["progress", "time", "smooth", "collision", "goal_bonus"],
            "weights": {
                "progress": 1.0,
                "time": -0.1,
                "smooth": -0.01,
                "collision": -5.0,
                "goal_bonus": 10.0,
            },
        }
    }
    total = aggregate_reward(terms, cfg)
    expected = 0.5 - 0.1 - 0.01 * np.linalg.norm(action) + 10.0
    assert np.isclose(total, expected)
