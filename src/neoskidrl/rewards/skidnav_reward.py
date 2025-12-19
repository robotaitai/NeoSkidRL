from __future__ import annotations

from typing import Dict

import numpy as np


def compute_reward_terms(
    dist: float,
    prev_dist: float | None,
    action: np.ndarray,
    collided: bool,
    success: bool,
) -> Dict[str, float]:
    if prev_dist is None:
        prev_dist = dist
    progress = float(prev_dist - dist)
    smooth = float(np.linalg.norm(action))
    return {
        "progress": progress,
        "time": 1.0,
        "smooth": smooth,
        "collision": 1.0 if collided else 0.0,
        "goal_bonus": 1.0 if success else 0.0,
    }


def _weights_from_legacy(cfg: dict) -> Dict[str, float]:
    return {
        "progress": float(cfg.get("w_progress", 0.0)),
        "time": float(cfg.get("w_time", 0.0)),
        "smooth": float(cfg.get("w_smooth", 0.0)),
        "collision": float(cfg.get("w_collision", 0.0)),
        "goal_bonus": float(cfg.get("w_goal_bonus", 0.0)),
    }


def aggregate_reward(terms: Dict[str, float], cfg: dict) -> float:
    reward_cfg = cfg.get("reward", {})
    weights = reward_cfg.get("weights")
    enabled_terms = reward_cfg.get("enabled_terms")
    if weights is None:
        weights = _weights_from_legacy(reward_cfg)
        if enabled_terms is None:
            enabled_terms = [key for key, val in weights.items() if val != 0.0]

    if enabled_terms is None:
        enabled_terms = list(weights.keys())

    total = 0.0
    for name in enabled_terms:
        total += float(weights.get(name, 0.0)) * float(terms.get(name, 0.0))
    return float(total)
