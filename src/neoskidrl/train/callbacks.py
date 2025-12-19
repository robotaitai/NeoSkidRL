from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

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
