from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict


class RLRunLogger(ABC):
    """Algorithm-agnostic interface for run logging."""

    @abstractmethod
    def on_step(self, step_metrics: dict) -> None:
        pass

    @abstractmethod
    def on_episode_end(self, ep_summary: dict) -> None:
        pass

    @abstractmethod
    def on_eval(self, eval_summary: dict) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


@dataclass
class RollingWindow:
    maxlen: int
    episodes: Deque[dict] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.episodes = deque(maxlen=self.maxlen)

    def push(self, ep_summary: dict) -> None:
        self.episodes.append(ep_summary)

    def count(self) -> int:
        return len(self.episodes)

    def mean(self, key: str) -> float | None:
        values = [float(ep[key]) for ep in self.episodes if ep.get(key) is not None]
        if not values:
            return None
        return sum(values) / len(values)

    def rate(self, key: str) -> float | None:
        values = [1.0 if ep.get(key) else 0.0 for ep in self.episodes if key in ep]
        if not values:
            return None
        return sum(values) / len(values)

    def dict_mean(self, key: str) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for ep in self.episodes:
            data = ep.get(key)
            if not isinstance(data, dict):
                continue
            for name, value in data.items():
                totals[name] = totals.get(name, 0.0) + float(value)
                counts[name] = counts.get(name, 0) + 1
        means = {}
        for name, total in totals.items():
            denom = counts.get(name, 0)
            means[name] = total / denom if denom else 0.0
        return means

