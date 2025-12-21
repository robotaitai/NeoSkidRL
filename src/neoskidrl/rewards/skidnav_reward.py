from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def compute_reward_terms(
    dist: float,
    prev_dist: float | None,
    action: np.ndarray,
    collided: bool,
    success: bool,
    stuck: bool = False,
    min_lidar: float | None = None,
    prev_action: np.ndarray | None = None,
    goal_angle: float | None = None,
    prev_goal_angle: float | None = None,
) -> Dict[str, float]:
    """
    Compute reward terms for skid-steer navigation.
    
    Args:
        dist: Current distance to goal
        prev_dist: Previous distance to goal (None on first step)
        action: Current applied action
        collided: Whether collision occurred
        success: Whether goal was reached
        stuck: Whether robot is stuck
        min_lidar: Minimum lidar reading (for clearance reward)
        prev_action: Previous applied action (for smoothness)
        goal_angle: Current goal angle in robot frame (radians)
        prev_goal_angle: Previous goal angle in robot frame (radians)
    """
    if prev_dist is None:
        prev_dist = dist
    
    # Progress: distance moved toward goal (positive when approaching)
    progress = float(prev_dist - dist)
    
    # Time: penalty per timestep (encourages faster completion)
    time = 1.0
    
    # Smoothness: penalize action changes (encourages smooth motion)
    if prev_action is not None:
        smooth = float(np.linalg.norm(action - prev_action))
    else:
        smooth = float(np.linalg.norm(action))

    # Heading: reward turning toward the goal (positive when angle error shrinks)
    heading = 0.0
    if goal_angle is not None and prev_goal_angle is not None:
        heading = float(abs(prev_goal_angle) - abs(goal_angle))
    
    # Collision: binary indicator
    collision = 1.0 if collided else 0.0
    
    # Goal bonus: sparse reward for success
    goal_bonus = 1.0 if success else 0.0
    
    # Stuck: penalty for getting stuck
    stuck_penalty = 1.0 if stuck else 0.0
    
    # Clearance: penalize being too close to obstacles
    clearance = 0.0
    if min_lidar is not None:
        d_safe = 0.5  # Safe distance in meters
        if min_lidar < d_safe:
            clearance = max(0.0, (d_safe - min_lidar) / d_safe)
    
    return {
        "progress": progress,
        "time": time,
        "smooth": smooth,
        "heading": heading,
        "collision": collision,
        "goal_bonus": goal_bonus,
        "stuck": stuck_penalty,
        "clearance": clearance,
    }


def _weights_from_legacy(cfg: dict) -> Dict[str, float]:
    return {
        "progress": float(cfg.get("w_progress", 0.0)),
        "time": float(cfg.get("w_time", 0.0)),
        "smooth": float(cfg.get("w_smooth", 0.0)),
        "heading": float(cfg.get("w_heading", 0.0)),
        "collision": float(cfg.get("w_collision", 0.0)),
        "goal_bonus": float(cfg.get("w_goal_bonus", 0.0)),
        "stuck": float(cfg.get("w_stuck", 0.0)),
        "clearance": float(cfg.get("w_clearance", 0.0)),
    }

def _resolve_reward_weights(cfg: dict) -> Tuple[Dict[str, float], list[str]]:
    reward_cfg = cfg.get("reward", {})
    weights = reward_cfg.get("weights")
    enabled_terms = reward_cfg.get("enabled_terms")
    if weights is None:
        weights = _weights_from_legacy(reward_cfg)
        if enabled_terms is None:
            enabled_terms = [key for key, val in weights.items() if val != 0.0]

    if enabled_terms is None:
        enabled_terms = list(weights.keys())
    return weights, enabled_terms


def compute_reward_contributions(terms: Dict[str, float], cfg: dict) -> Dict[str, float]:
    weights, enabled_terms = _resolve_reward_weights(cfg)
    enabled = set(enabled_terms)
    contributions = {}
    for name, value in terms.items():
        weight = float(weights.get(name, 0.0)) if name in enabled else 0.0
        contributions[name] = float(weight) * float(value)
    return contributions


def aggregate_reward(terms: Dict[str, float], cfg: dict) -> float:
    weights, enabled_terms = _resolve_reward_weights(cfg)

    total = 0.0
    for name in enabled_terms:
        total += float(weights.get(name, 0.0)) * float(terms.get(name, 0.0))
    return float(total)
