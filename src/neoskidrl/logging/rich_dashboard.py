from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, Iterable

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from neoskidrl.logging.run_logger import RLRunLogger, RollingWindow


def _fmt(value: float | None, digits: int = 3, default: str = "n/a") -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return default
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 1, default: str = "n/a") -> str:
    if value is None:
        return default
    return f"{value * 100:.{digits}f}%"


def _bar(share: float, width: int = 10) -> str:
    share = max(0.0, min(1.0, share))
    filled = int(round(share * width))
    return "[" + ("=" * filled) + ("-" * (width - filled)) + "]"


class RichDashboardLogger(RLRunLogger):
    def __init__(
        self,
        run_name: str,
        total_steps: int | None = None,
        chunk_steps: int | None = None,
        update_hz: float = 2.0,
        event_every: int = 20,
        alert_cooldown_sec: float = 30.0,
        critic_loss_threshold: float = 10.0,
        console: Console | None = None,
    ):
        self.run_name = run_name
        self.total_steps = total_steps
        self.chunk_steps = chunk_steps
        self.update_interval = 1.0 / max(0.1, update_hz)
        self.event_every = max(1, event_every)
        self.alert_cooldown_sec = alert_cooldown_sec
        self.critic_loss_threshold = critic_loss_threshold
        self.console = console or Console()

        self._window_20 = RollingWindow(20)
        self._window_100 = RollingWindow(100)
        self._episode_idx = 0
        self._events = deque(maxlen=12)
        self._last_step_metrics: dict = {}
        self._last_eval_summary: dict | None = None
        self._last_render = 0.0
        self._last_alert_at: Dict[str, float] = {}
        self._prev_final_dist_100: float | None = None

        self._live = Live(self._render(), console=self.console, refresh_per_second=4)
        self._live.start()

    def on_step(self, step_metrics: dict) -> None:
        self._last_step_metrics = step_metrics
        self._maybe_alert_step(step_metrics)
        self._maybe_render()

    def on_episode_end(self, ep_summary: dict) -> None:
        self._episode_idx += 1
        self._window_20.push(ep_summary)
        self._window_100.push(ep_summary)
        self._maybe_emit_event(ep_summary)
        self._maybe_alert_episode(ep_summary)
        self._maybe_render(force=True)

    def on_eval(self, eval_summary: dict) -> None:
        self._last_eval_summary = eval_summary
        msg = (
            f"EVAL success={_fmt_pct(eval_summary.get('success_rate'))} "
            f"return={_fmt(eval_summary.get('return_mean'), 2)} "
            f"final_dist={_fmt(eval_summary.get('final_dist_mean'), 2)}"
        )
        self._events.append(msg)
        self._maybe_render(force=True)

    def close(self) -> None:
        if self._live:
            self._live.stop()

    def _maybe_render(self, force: bool = False) -> None:
        now = time.time()
        if force or (now - self._last_render) >= self.update_interval:
            self._live.update(self._render())
            self._last_render = now

    def _window_stats(self, window: RollingWindow) -> dict:
        return {
            "success_rate": window.rate("success"),
            "final_dist_mean": window.mean("final_dist"),
            "min_dist_mean": window.mean("min_dist"),
            "mean_abs_v": window.mean("mean_abs_v"),
            "mean_abs_wz": window.mean("mean_abs_wz"),
            "action_saturation_pct": window.mean("action_saturation_pct"),
            "pos_ok_rate": window.mean("pos_ok_rate"),
            "yaw_ok_rate": window.mean("yaw_ok_rate"),
            "stop_ok_rate": window.mean("stop_ok_rate"),
            "pos_and_yaw_rate": window.mean("pos_and_yaw_rate"),
            "pos_and_stop_rate": window.mean("pos_and_stop_rate"),
            "yaw_and_stop_rate": window.mean("yaw_and_stop_rate"),
            "timeout_rate": window.rate("timeout"),
            "stuck_rate": window.rate("stuck"),
            "collision_rate": window.rate("collision"),
            "ep_len_mean": window.mean("ep_len"),
            "return_mean": window.mean("ep_return"),
            "reward_contrib_mean": window.dict_mean("reward_contrib_sum"),
            "reward_contrib_abs_mean": window.dict_mean("reward_contrib_abs_sum"),
        }

    def _maybe_emit_event(self, ep_summary: dict) -> None:
        success = bool(ep_summary.get("success", False))
        collision = bool(ep_summary.get("collision", False))
        stuck = bool(ep_summary.get("stuck", False))
        timeout = bool(ep_summary.get("timeout", False))
        should_print = success or collision or stuck or timeout or (self._episode_idx % self.event_every == 0)
        if not should_print:
            return
        tag = "OK"
        if collision:
            tag = "COLLISION"
        elif stuck:
            tag = "STUCK"
        elif timeout:
            tag = "TIMEOUT"
        msg = (
            f"{tag} ep={self._episode_idx} return={_fmt(ep_summary.get('ep_return'), 2)} "
            f"dist_final={_fmt(ep_summary.get('final_dist'), 2)} steps={ep_summary.get('ep_len')}"
        )
        self._events.append(msg)

    def _maybe_alert_step(self, step_metrics: dict) -> None:
        algo = step_metrics.get("algo_metrics") or {}
        critic_loss = algo.get("critic_loss")
        if critic_loss is not None:
            if isinstance(critic_loss, float) and (math.isnan(critic_loss) or math.isinf(critic_loss)):
                self._emit_alert("critic_nan", "critic_loss is NaN/Inf (diverging?)")
            elif float(critic_loss) > self.critic_loss_threshold:
                self._emit_alert("critic_spike", f"critic_loss spike: {critic_loss:.3f}")

    def _maybe_alert_episode(self, _ep_summary: dict) -> None:
        stats_100 = self._window_stats(self._window_100)
        timesteps = int(self._last_step_metrics.get("timesteps", 0))
        final_dist_mean = stats_100.get("final_dist_mean")
        success_rate = stats_100.get("success_rate")
        mean_abs_v = stats_100.get("mean_abs_v")
        timeout_rate = stats_100.get("timeout_rate")

        if (
            timesteps > 100_000
            and success_rate is not None
            and success_rate == 0.0
            and final_dist_mean is not None
        ):
            if self._prev_final_dist_100 is not None and final_dist_mean >= self._prev_final_dist_100 - 1e-3:
                self._emit_alert("no_learning", "no learning: success_rate_100=0 and final_dist not improving")
            self._prev_final_dist_100 = final_dist_mean

        if mean_abs_v is not None and mean_abs_v < 0.05:
            self._emit_alert("timid_policy", "timid policy: mean|v|_100 < 0.05")

        if timeout_rate is not None and timeout_rate > 0.98 and final_dist_mean is not None:
            if self._prev_final_dist_100 is not None and final_dist_mean >= self._prev_final_dist_100 - 1e-3:
                self._emit_alert("timeouts", "timeouts ~100% and final_dist not decreasing")

    def _emit_alert(self, key: str, message: str) -> None:
        now = time.time()
        last = self._last_alert_at.get(key, 0.0)
        if now - last < self.alert_cooldown_sec:
            return
        self._last_alert_at[key] = now
        self._events.append(f"ALERT: {message}")
        self.console.print(f"[bold red]ALERT:[/bold red] {message}")

    def _render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="top", size=12),
            Layout(name="mid", size=12),
            Layout(name="bottom", size=8),
        )
        layout["top"].split_row(Layout(name="kpi"), Layout(name="perf"))
        layout["mid"].split_row(Layout(name="reward"), Layout(name="outcomes"))

        layout["kpi"].update(self._render_kpi_panel())
        layout["perf"].update(self._render_perf_panel())
        layout["reward"].update(self._render_reward_panel())
        layout["outcomes"].update(self._render_outcomes_panel())
        layout["bottom"].update(self._render_events_panel())
        return layout

    def _render_kpi_panel(self) -> Panel:
        stats_20 = self._window_stats(self._window_20)
        stats_100 = self._window_stats(self._window_100)
        table = Table(title=f"KPIs (run: {self.run_name})", box=box.SIMPLE_HEAVY)
        table.add_column("Metric")
        table.add_column("W20")
        table.add_column("W100")
        table.add_row("success_rate", _fmt_pct(stats_20.get("success_rate")), _fmt_pct(stats_100.get("success_rate")))
        table.add_row("pos_ok_rate", _fmt_pct(stats_20.get("pos_ok_rate")), _fmt_pct(stats_100.get("pos_ok_rate")))
        table.add_row("yaw_ok_rate", _fmt_pct(stats_20.get("yaw_ok_rate")), _fmt_pct(stats_100.get("yaw_ok_rate")))
        table.add_row("stop_ok_rate", _fmt_pct(stats_20.get("stop_ok_rate")), _fmt_pct(stats_100.get("stop_ok_rate")))
        table.add_row("pos&yaw_rate", _fmt_pct(stats_20.get("pos_and_yaw_rate")), _fmt_pct(stats_100.get("pos_and_yaw_rate")))
        table.add_row("pos&stop_rate", _fmt_pct(stats_20.get("pos_and_stop_rate")), _fmt_pct(stats_100.get("pos_and_stop_rate")))
        table.add_row("yaw&stop_rate", _fmt_pct(stats_20.get("yaw_and_stop_rate")), _fmt_pct(stats_100.get("yaw_and_stop_rate")))
        table.add_row("final_dist_mean", _fmt(stats_20.get("final_dist_mean"), 2), _fmt(stats_100.get("final_dist_mean"), 2))
        table.add_row("min_dist_mean", _fmt(stats_20.get("min_dist_mean"), 2), _fmt(stats_100.get("min_dist_mean"), 2))
        table.add_row("mean|v|", _fmt(stats_20.get("mean_abs_v"), 3), _fmt(stats_100.get("mean_abs_v"), 3))
        table.add_row("mean|w|", _fmt(stats_20.get("mean_abs_wz"), 3), _fmt(stats_100.get("mean_abs_wz"), 3))
        table.add_row(
            "action_sat_pct",
            _fmt(stats_20.get("action_saturation_pct"), 1),
            _fmt(stats_100.get("action_saturation_pct"), 1),
        )
        return Panel(table, title="Stop-this-run KPIs", border_style="cyan")

    def _render_reward_panel(self) -> Panel:
        stats_100 = self._window_stats(self._window_100)
        contrib_mean = stats_100.get("reward_contrib_mean") or {}
        contrib_abs_mean = stats_100.get("reward_contrib_abs_mean") or {k: abs(v) for k, v in contrib_mean.items()}
        abs_total = sum(contrib_abs_mean.values()) if contrib_abs_mean else 0.0

        ordered_terms = [
            "progress",
            "heading",
            "velocity",
            "time",
            "stuck",
            "collision",
            "goal_bonus",
        ]
        table = Table(title="Reward Breakdown (W100)", box=box.SIMPLE)
        table.add_column("Term")
        table.add_column("Mean")
        table.add_column("Share")

        for term in ordered_terms:
            if term not in contrib_mean and term not in contrib_abs_mean:
                continue
            mean_val = contrib_mean.get(term, 0.0)
            share = (contrib_abs_mean.get(term, 0.0) / abs_total) if abs_total > 0 else 0.0
            table.add_row(
                term,
                _fmt(mean_val, 3),
                f"{share * 100:.0f}% {_bar(share, width=8)}",
            )

        table.add_row("total_return", _fmt(stats_100.get("return_mean"), 2), "")
        return Panel(table, title="Reward Terms", border_style="magenta")

    def _render_outcomes_panel(self) -> Panel:
        stats_100 = self._window_stats(self._window_100)
        table = Table(title="Episode Outcomes (W100)", box=box.SIMPLE)
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("timeout_rate", _fmt_pct(stats_100.get("timeout_rate")))
        table.add_row("stuck_rate", _fmt_pct(stats_100.get("stuck_rate")))
        table.add_row("collision_rate", _fmt_pct(stats_100.get("collision_rate")))
        table.add_row("ep_len_mean", _fmt(stats_100.get("ep_len_mean"), 1))
        return Panel(table, title="Outcomes", border_style="yellow")

    def _render_perf_panel(self) -> Panel:
        step = self._last_step_metrics
        timesteps = int(step.get("timesteps", 0))
        fps = float(step.get("fps", 0.0)) if step.get("fps") is not None else 0.0
        wall_time = float(step.get("wall_time_s", 0.0)) if step.get("wall_time_s") is not None else 0.0
        total_steps = int(step.get("total_steps", self.total_steps or 0))
        chunk_end = int(step.get("chunk_end", 0))

        eta_total = None
        eta_chunk = None
        if fps > 0:
            if total_steps > 0:
                eta_total = max(0.0, (total_steps - timesteps) / fps)
            if chunk_end > 0:
                eta_chunk = max(0.0, (chunk_end - timesteps) / fps)

        table = Table(title="Perf", box=box.SIMPLE)
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("timesteps", f"{timesteps:,}")
        table.add_row("fps", _fmt(fps, 1))
        table.add_row("wall_time_s", _fmt(wall_time, 1))
        table.add_row("eta_chunk_s", _fmt(eta_chunk, 1))
        table.add_row("eta_total_s", _fmt(eta_total, 1))

        algo = step.get("algo_metrics") or {}
        if algo:
            table.add_row("ent_mode", algo.get("ent_coef_mode", "n/a"))
            table.add_row("alpha", _fmt(algo.get("ent_coef_value"), 4))
            table.add_row("target_entropy", _fmt(algo.get("target_entropy"), 3))
            table.add_row("actor_loss", _fmt(algo.get("actor_loss"), 4))
            table.add_row("critic_loss", _fmt(algo.get("critic_loss"), 4))

        if self._last_eval_summary:
            table.add_row("eval_success", _fmt_pct(self._last_eval_summary.get("success_rate")))
            table.add_row("eval_return", _fmt(self._last_eval_summary.get("return_mean"), 2))

        return Panel(table, title="Performance", border_style="green")

    def _render_events_panel(self) -> Panel:
        if not self._events:
            body = Text("No events yet.")
        else:
            body = Text("\n".join(self._events))
        return Panel(body, title="Events", border_style="white")
