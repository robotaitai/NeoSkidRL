from __future__ import annotations

import numpy as np


def action_delta_l1(prev_action: np.ndarray | None, action: np.ndarray) -> float:
    if prev_action is None:
        return 0.0
    return float(np.sum(np.abs(action - prev_action)))


def path_length_increment(prev_pos: np.ndarray | None, pos: np.ndarray | None) -> float:
    if prev_pos is None or pos is None:
        return 0.0
    return float(np.linalg.norm(pos - prev_pos))
