from __future__ import annotations

import json
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from neoskidrl.scripts.eval import run_eval


def _should_record_video(eval_calls: int, video_cfg: dict) -> bool:
    if not video_cfg.get("enabled", False):
        return False
    first_n = int(video_cfg.get("record_first_n_evals", 0))
    if eval_calls < first_n:
        return True
    every_n = int(video_cfg.get("then_every_n_evals", 0))
    if every_n <= 0:
        return False
    return ((eval_calls - first_n) % every_n) == 0


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_config_path: str,
        scenario: str,
        eval_freq_steps: int,
        episodes: int | None,
        seeds: list[int] | None,
        video_cfg: dict,
        output_dir: str,
        video_dir: str,
        checkpoint_dir: str,
        headless: bool,
        deterministic: bool,
        algo: str,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_config_path = eval_config_path
        self.scenario = scenario
        self.eval_freq_steps = int(eval_freq_steps)
        self.episodes = episodes
        self.seeds = seeds
        self.video_cfg = video_cfg
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.checkpoint_dir = checkpoint_dir
        self.headless = headless
        self.deterministic = deterministic
        self.algo = algo
        self._next_eval = self.eval_freq_steps
        self._eval_calls = 0

    def _on_step(self) -> bool:
        if self.eval_freq_steps <= 0:
            return True
        if self.num_timesteps < self._next_eval:
            return True

        ckpt_dir = Path(self.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = f"ckpt_{self.num_timesteps:06d}"
        ckpt_path = ckpt_dir / f"{ckpt_name}.zip"
        self.model.save(str(ckpt_path))

        record_video = _should_record_video(self._eval_calls, self.video_cfg)
        run_eval(
            model_path=str(ckpt_path),
            eval_config_path=self.eval_config_path,
            scenario=self.scenario,
            seeds=self.seeds,
            episodes=self.episodes,
            output_dir=self.output_dir,
            video_dir=self.video_dir,
            record_video=record_video,
            headless=self.headless,
            deterministic=self.deterministic,
            algo=self.algo,
            run_id=ckpt_name,
        )

        self._eval_calls += 1
        self._next_eval += self.eval_freq_steps
        return True

    def update_video_policy(self, video_cfg: dict) -> None:
        self.video_cfg = video_cfg


class EpisodeJSONLLogger(BaseCallback):
    """
    Logs episode-level metrics to a JSONL file for the reward dashboard.
    
    Writes one line per completed episode with:
    - timestamp, run_id, algo, seed, episode_idx
    - ep_len, ep_return, success, collision, stuck
    - reward_terms_sum: dict of per-term episode sums
    """
    
    def __init__(
        self,
        output_path: str | Path,
        run_id: str = "default",
        algo: str = "SAC",
        seed: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.output_path = Path(output_path)
        self.run_id = run_id
        self.algo = algo
        self.seed = seed
        
        # Episode tracking per environment
        self._episode_idx = 0
        self._ep_returns = None
        self._ep_lengths = None
        self._ep_reward_terms = None
        self._ep_reward_contrib = None
        self._ep_reward_contrib_abs = None
        self._ep_success = None
        self._ep_collision = None
        self._ep_stuck = None
        self._ep_timeout = None
        self._ep_last_dist = None
        self._ep_sum_abs_v = None
        self._ep_sum_abs_wz = None
        
    def _on_training_start(self) -> None:
        """Initialize tracking arrays for each parallel environment."""
        n_envs = self.training_env.num_envs if isinstance(self.training_env, VecEnv) else 1
        self._ep_returns = [0.0] * n_envs
        self._ep_lengths = [0] * n_envs
        self._ep_reward_terms = [None] * n_envs
        self._ep_reward_contrib = [None] * n_envs
        self._ep_reward_contrib_abs = [None] * n_envs
        self._ep_success = [False] * n_envs
        self._ep_collision = [False] * n_envs
        self._ep_stuck = [False] * n_envs
        self._ep_timeout = [False] * n_envs
        self._ep_last_dist = [None] * n_envs
        self._ep_sum_abs_v = [0.0] * n_envs
        self._ep_sum_abs_wz = [0.0] * n_envs
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _on_step(self) -> bool:
        """Accumulate episode data and write on episode completion."""
        # Handle both vectorized and non-vectorized environments
        if isinstance(self.training_env, VecEnv):
            # Get info from all environments
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            rewards = self.locals.get("rewards", [])
            
            for env_idx, (info, done, reward) in enumerate(zip(infos, dones, rewards)):
                # Accumulate episode data
                self._ep_returns[env_idx] += float(reward)
                self._ep_lengths[env_idx] += 1
                
                # Accumulate reward terms
                if "reward_terms" in info:
                    if self._ep_reward_terms[env_idx] is None:
                        self._ep_reward_terms[env_idx] = {k: 0.0 for k in info["reward_terms"].keys()}
                    for key, val in info["reward_terms"].items():
                        self._ep_reward_terms[env_idx][key] += float(val)
                if "reward_contrib" in info:
                    if self._ep_reward_contrib[env_idx] is None:
                        self._ep_reward_contrib[env_idx] = {k: 0.0 for k in info["reward_contrib"].keys()}
                    if self._ep_reward_contrib_abs[env_idx] is None:
                        self._ep_reward_contrib_abs[env_idx] = {k: 0.0 for k in info["reward_contrib"].keys()}
                    for key, val in info["reward_contrib"].items():
                        value = float(val)
                        self._ep_reward_contrib[env_idx][key] += value
                        self._ep_reward_contrib_abs[env_idx][key] += abs(value)
                
                # Track episode outcomes
                if "success" in info:
                    self._ep_success[env_idx] = self._ep_success[env_idx] or bool(info["success"])
                if "collision" in info:
                    self._ep_collision[env_idx] = self._ep_collision[env_idx] or bool(info["collision"])
                if "stuck" in info:
                    self._ep_stuck[env_idx] = self._ep_stuck[env_idx] or bool(info["stuck"])
                if "timeout" in info:
                    self._ep_timeout[env_idx] = self._ep_timeout[env_idx] or bool(info["timeout"])
                if "dist" in info:
                    self._ep_last_dist[env_idx] = float(info["dist"])
                if "speed_v" in info:
                    self._ep_sum_abs_v[env_idx] += abs(float(info["speed_v"]))
                if "speed_wz" in info:
                    self._ep_sum_abs_wz[env_idx] += abs(float(info["speed_wz"]))
                
                # Write episode data when done
                if done:
                    self._write_episode(
                        ep_len=self._ep_lengths[env_idx],
                        ep_return=self._ep_returns[env_idx],
                        success=self._ep_success[env_idx],
                        collision=self._ep_collision[env_idx],
                        stuck=self._ep_stuck[env_idx],
                        timeout=self._ep_timeout[env_idx],
                        final_dist=self._ep_last_dist[env_idx],
                        mean_abs_v=(self._ep_sum_abs_v[env_idx] / max(1, self._ep_lengths[env_idx])),
                        mean_abs_wz=(self._ep_sum_abs_wz[env_idx] / max(1, self._ep_lengths[env_idx])),
                        reward_terms_sum=self._ep_reward_terms[env_idx] or {},
                        reward_contrib_sum=self._ep_reward_contrib[env_idx] or {},
                        reward_contrib_abs_sum=self._ep_reward_contrib_abs[env_idx] or {},
                    )
                    
                    # Reset tracking for this environment
                    self._ep_returns[env_idx] = 0.0
                    self._ep_lengths[env_idx] = 0
                    self._ep_reward_terms[env_idx] = None
                    self._ep_reward_contrib[env_idx] = None
                    self._ep_reward_contrib_abs[env_idx] = None
                    self._ep_success[env_idx] = False
                    self._ep_collision[env_idx] = False
                    self._ep_stuck[env_idx] = False
                    self._ep_timeout[env_idx] = False
                    self._ep_last_dist[env_idx] = None
                    self._ep_sum_abs_v[env_idx] = 0.0
                    self._ep_sum_abs_wz[env_idx] = 0.0
        
        return True
    
    def _write_episode(
        self,
        ep_len: int,
        ep_return: float,
        success: bool,
        collision: bool,
        stuck: bool,
        timeout: bool,
        final_dist: float | None,
        mean_abs_v: float,
        mean_abs_wz: float,
        reward_terms_sum: dict,
        reward_contrib_sum: dict,
        reward_contrib_abs_sum: dict,
    ) -> None:
        """Write a single episode record to the JSONL file."""
        sum_progress = float(reward_terms_sum.get("progress", 0.0))
        sum_time = float(reward_terms_sum.get("time", 0.0))
        record = {
            "timestamp": time.time(),
            "run_id": self.run_id,
            "algo": self.algo,
            "seed": self.seed,
            "episode_idx": self._episode_idx,
            "ep_len": ep_len,
            "ep_return": float(ep_return),
            "success": success,
            "collision": collision,
            "stuck": stuck,
            "timeout": timeout,
            "final_dist": float(final_dist) if final_dist is not None else None,
            "mean_abs_v": float(mean_abs_v),
            "mean_abs_wz": float(mean_abs_wz),
            "sum_progress": sum_progress,
            "sum_time": sum_time,
            "reward_terms_sum": {k: float(v) for k, v in reward_terms_sum.items()},
            "reward_contrib_sum": {k: float(v) for k, v in reward_contrib_sum.items()},
            "reward_contrib_abs_sum": {k: float(v) for k, v in reward_contrib_abs_sum.items()},
            "timesteps": self.num_timesteps,
        }
        
        # Append to JSONL file
        with self.output_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        
        self._episode_idx += 1
        
        if self.verbose > 0:
            if self._episode_idx == 0:
                print(
                    "[RewardDebug] ep0 "
                    f"sum_progress={sum_progress:.3f} sum_time={sum_time:.3f} "
                    f"collision={collision} stuck={stuck} timeout={timeout} "
                    f"final_dist={record['final_dist']} mean|v|={mean_abs_v:.3f} "
                    f"mean|w|={mean_abs_wz:.3f}"
                )
            elif self._episode_idx % 10 == 0:
                print(f"[EpisodeLogger] Logged episode {self._episode_idx}: return={ep_return:.2f}, success={success}")
