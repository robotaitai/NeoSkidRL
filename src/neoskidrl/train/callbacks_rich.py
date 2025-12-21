from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from neoskidrl.envs import NeoSkidNavEnv
from neoskidrl.logging.run_logger import RLRunLogger


@dataclass
class _EpisodeAccumulator:
    ep_return: float = 0.0
    ep_len: int = 0
    sum_reward_terms: dict | None = None
    sum_reward_contrib: dict | None = None
    sum_reward_contrib_abs: dict | None = None
    min_dist: float | None = None
    final_dist: float | None = None
    sum_abs_v: float = 0.0
    sum_abs_wz: float = 0.0
    sum_action_delta: float = 0.0
    sum_action_sat: float = 0.0
    action_count: int = 0
    pos_ok_count: int = 0
    yaw_ok_count: int = 0
    stop_ok_count: int = 0
    pos_yaw_count: int = 0
    pos_stop_count: int = 0
    yaw_stop_count: int = 0
    success: bool = False
    collision: bool = False
    stuck: bool = False
    timeout: bool = False


class SB3RichCallback(BaseCallback):
    def __init__(
        self,
        logger: RLRunLogger,
        total_steps: int,
        chunk_steps: int,
        eval_every_steps: int = 20000,
        eval_episodes: int = 10,
        eval_enabled: bool = True,
        eval_seed: int = 0,
        config_path: str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.run_logger = logger
        self.total_steps = total_steps
        self.chunk_steps = chunk_steps
        self.eval_every_steps = eval_every_steps
        self.eval_episodes = eval_episodes
        self.eval_enabled = eval_enabled
        self.eval_seed = eval_seed
        self.config_path = config_path
        self._episode_idx = 0
        self._accumulators: list[_EpisodeAccumulator] = []
        self._prev_actions: list[np.ndarray | None] = []
        self._start_time = None
        self._last_time = None
        self._last_steps = 0
        self._next_eval = eval_every_steps
        self._eval_env = None

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs if isinstance(self.training_env, VecEnv) else 1
        self._accumulators = [_EpisodeAccumulator() for _ in range(n_envs)]
        self._prev_actions = [None for _ in range(n_envs)]
        now = time.time()
        self._start_time = now
        self._last_time = now
        self._last_steps = 0

    def _on_step(self) -> bool:
        now = time.time()
        if self._last_time is None:
            self._last_time = now
        elapsed = max(1e-6, now - self._last_time)
        steps_delta = self.num_timesteps - self._last_steps
        fps = steps_delta / elapsed if elapsed > 0 else 0.0
        self._last_time = now
        self._last_steps = self.num_timesteps

        algo_metrics = self._extract_algo_metrics()
        chunk_end = 0
        if self.chunk_steps > 0:
            chunk_end = ((self.num_timesteps // self.chunk_steps) + 1) * self.chunk_steps

        self.run_logger.on_step(
            {
                "timesteps": int(self.num_timesteps),
                "fps": float(fps),
                "wall_time_s": float(now - (self._start_time or now)),
                "total_steps": int(self.total_steps),
                "chunk_end": int(chunk_end),
                "algo_metrics": algo_metrics,
            }
        )

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        actions = self.locals.get("actions", None)

        if actions is not None:
            actions = np.asarray(actions)

        for env_idx, (info, done, reward) in enumerate(zip(infos, dones, rewards)):
            acc = self._accumulators[env_idx]
            acc.ep_return += float(reward)
            acc.ep_len += 1

            if "reward_terms" in info:
                if acc.sum_reward_terms is None:
                    acc.sum_reward_terms = {k: 0.0 for k in info["reward_terms"].keys()}
                for key, val in info["reward_terms"].items():
                    acc.sum_reward_terms[key] += float(val)
            if "reward_contrib" in info:
                if acc.sum_reward_contrib is None:
                    acc.sum_reward_contrib = {k: 0.0 for k in info["reward_contrib"].keys()}
                    acc.sum_reward_contrib_abs = {k: 0.0 for k in info["reward_contrib"].keys()}
                for key, val in info["reward_contrib"].items():
                    val_f = float(val)
                    acc.sum_reward_contrib[key] += val_f
                    acc.sum_reward_contrib_abs[key] += abs(val_f)

            if "dist" in info:
                dist = float(info["dist"])
                acc.final_dist = dist
                if acc.min_dist is None:
                    acc.min_dist = dist
                else:
                    acc.min_dist = min(acc.min_dist, dist)
            if "speed_v" in info:
                acc.sum_abs_v += abs(float(info["speed_v"]))
            if "speed_wz" in info:
                acc.sum_abs_wz += abs(float(info["speed_wz"]))
            pos_ok = bool(info.get("pos_ok", False))
            yaw_ok = bool(info.get("yaw_ok", False))
            stop_ok = bool(info.get("stop_ok", False))
            if pos_ok:
                acc.pos_ok_count += 1
            if yaw_ok:
                acc.yaw_ok_count += 1
            if stop_ok:
                acc.stop_ok_count += 1
            if pos_ok and yaw_ok:
                acc.pos_yaw_count += 1
            if pos_ok and stop_ok:
                acc.pos_stop_count += 1
            if yaw_ok and stop_ok:
                acc.yaw_stop_count += 1

            if actions is not None and env_idx < actions.shape[0]:
                action = actions[env_idx]
                prev = self._prev_actions[env_idx]
                if prev is not None:
                    acc.sum_action_delta += float(np.linalg.norm(action - prev, ord=1))
                sat = float(np.mean(np.abs(action) >= 0.95))
                acc.sum_action_sat += sat
                acc.action_count += 1
                self._prev_actions[env_idx] = action.copy()

            if "success" in info:
                acc.success = acc.success or bool(info["success"])
            if "collision" in info:
                acc.collision = acc.collision or bool(info["collision"])
            if "stuck" in info:
                acc.stuck = acc.stuck or bool(info["stuck"])
            if "timeout" in info:
                acc.timeout = acc.timeout or bool(info["timeout"])

            if done:
                self._episode_idx += 1
                ep_len = max(1, acc.ep_len)
                ep_summary = {
                    "episode_idx": self._episode_idx,
                    "ep_len": acc.ep_len,
                    "ep_return": acc.ep_return,
                    "success": acc.success,
                    "collision": acc.collision,
                    "stuck": acc.stuck,
                    "timeout": acc.timeout,
                    "final_dist": acc.final_dist,
                    "min_dist": acc.min_dist,
                    "mean_abs_v": acc.sum_abs_v / ep_len,
                    "mean_abs_wz": acc.sum_abs_wz / ep_len,
                    "action_delta_mean": acc.sum_action_delta / max(1, acc.action_count),
                    "action_saturation_pct": (acc.sum_action_sat / max(1, acc.action_count)) * 100.0,
                    "pos_ok_rate": acc.pos_ok_count / ep_len,
                    "yaw_ok_rate": acc.yaw_ok_count / ep_len,
                    "stop_ok_rate": acc.stop_ok_count / ep_len,
                    "pos_and_yaw_rate": acc.pos_yaw_count / ep_len,
                    "pos_and_stop_rate": acc.pos_stop_count / ep_len,
                    "yaw_and_stop_rate": acc.yaw_stop_count / ep_len,
                    "reward_terms_sum": acc.sum_reward_terms or {},
                    "reward_contrib_sum": acc.sum_reward_contrib or {},
                    "reward_contrib_abs_sum": acc.sum_reward_contrib_abs or {},
                }
                self.run_logger.on_episode_end(ep_summary)
                self._accumulators[env_idx] = _EpisodeAccumulator()
                self._prev_actions[env_idx] = None

        if self.eval_enabled and self.eval_every_steps > 0 and self.num_timesteps >= self._next_eval:
            eval_summary = self._run_eval()
            if eval_summary:
                self.run_logger.on_eval(eval_summary)
            self._next_eval += self.eval_every_steps

        return True

    def _extract_algo_metrics(self) -> dict:
        metrics: dict = {}
        ent_coef_mode = None
        ent_coef_value = None
        logger = getattr(self.model, "logger", None)
        name_to_value = getattr(logger, "name_to_value", None) if logger else None
        if isinstance(name_to_value, dict):
            if "train/actor_loss" in name_to_value:
                metrics["actor_loss"] = float(name_to_value["train/actor_loss"])
            if "train/critic_loss" in name_to_value:
                metrics["critic_loss"] = float(name_to_value["train/critic_loss"])
            if "train/ent_coef" in name_to_value:
                ent_coef_value = float(name_to_value["train/ent_coef"])
        log_ent_coef = getattr(self.model, "log_ent_coef", None)
        if log_ent_coef is not None:
            ent_coef_mode = "auto"
            try:
                ent_coef_value = float(np.exp(log_ent_coef).item())
            except Exception:
                pass
        else:
            ent_coef_mode = "fixed"
            if ent_coef_value is None and hasattr(self.model, "ent_coef"):
                try:
                    ent_coef_value = float(self.model.ent_coef)
                except Exception:
                    pass

        if ent_coef_mode is not None:
            metrics["ent_coef_mode"] = ent_coef_mode
        if ent_coef_value is not None:
            metrics["ent_coef_value"] = ent_coef_value
        target_entropy = getattr(self.model, "target_entropy", None)
        if target_entropy is not None:
            try:
                metrics["target_entropy"] = float(target_entropy)
            except Exception:
                pass
        return metrics

    def _run_eval(self) -> dict | None:
        if self.config_path is None:
            return None
        if self._eval_env is None:
            self._eval_env = NeoSkidNavEnv(config_path=self.config_path, render_mode=None)

        results = []
        for i in range(self.eval_episodes):
            seed = self.eval_seed + i
            obs, info = self._eval_env.reset(seed=seed)
            done = False
            ep_return = 0.0
            final_dist = None
            success = False
            collision = False
            stuck = False
            timeout = False
            while not done:
                action, _state = self.model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = self._eval_env.step(action)
                ep_return += float(reward)
                final_dist = float(info.get("dist", final_dist or 0.0))
                success = success or bool(info.get("success", False))
                collision = collision or bool(info.get("collision", False))
                stuck = stuck or bool(info.get("stuck", False))
                timeout = timeout or bool(info.get("timeout", False))
                done = term or trunc
            results.append(
                {
                    "return": ep_return,
                    "final_dist": final_dist,
                    "success": success,
                    "collision": collision,
                    "stuck": stuck,
                    "timeout": timeout,
                }
            )

        if not results:
            return None

        def _mean(values: list[float]) -> float:
            return sum(values) / max(1, len(values))

        success_rate = _mean([1.0 if r["success"] else 0.0 for r in results])
        timeout_rate = _mean([1.0 if r["timeout"] else 0.0 for r in results])
        collision_rate = _mean([1.0 if r["collision"] else 0.0 for r in results])
        stuck_rate = _mean([1.0 if r["stuck"] else 0.0 for r in results])
        return_mean = _mean([float(r["return"]) for r in results])
        final_dist_mean = _mean([float(r["final_dist"] or 0.0) for r in results])

        return {
            "episodes": len(results),
            "success_rate": success_rate,
            "timeout_rate": timeout_rate,
            "collision_rate": collision_rate,
            "stuck_rate": stuck_rate,
            "return_mean": return_mean,
            "final_dist_mean": final_dist_mean,
        }

    def _on_training_end(self) -> None:
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None
