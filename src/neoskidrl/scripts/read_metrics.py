from __future__ import annotations

import argparse
import json
from pathlib import Path


def _collect_metrics_paths(paths: list[Path]) -> list[Path]:
    metrics = []
    for p in paths:
        if p.is_dir():
            metrics.extend(sorted(p.rglob("metrics.json")))
        elif p.is_file():
            if p.name == "metrics.json":
                metrics.append(p)
        else:
            raise FileNotFoundError(f"Path not found: {p}")
    return metrics


def _load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text())
    data["path"] = str(path)
    return data


def _format_float(val: float, width: int = 8, prec: int = 3) -> str:
    return f"{val:{width}.{prec}f}"


def _print_table(rows: list[dict]) -> None:
    headers = [
        "scenario",
        "algo",
        "episodes",
        "success",
        "collision",
        "stuck",
        "avg_steps",
        "avg_time_s",
        "avg_path",
        "avg_smooth",
        "avg_return",
    ]
    print(" ".join(h.rjust(10) for h in headers))
    for r in rows:
        line = [
            f"{r.get('scenario','-'):>10}",
            f"{r.get('algo','-'):>10}",
            f"{int(r.get('episodes',0)):>10}",
            _format_float(float(r.get("success_rate", 0.0))),
            _format_float(float(r.get("collision_rate", 0.0))),
            _format_float(float(r.get("stuck_rate", 0.0))),
            _format_float(float(r.get("avg_steps", 0.0))),
            _format_float(float(r.get("avg_time_s", 0.0))),
            _format_float(float(r.get("avg_path_length", 0.0))),
            _format_float(float(r.get("avg_smoothness", 0.0))),
            _format_float(float(r.get("avg_return", 0.0))),
        ]
        print(" ".join(line))


def main() -> None:
    parser = argparse.ArgumentParser(description="Read eval metrics.json files and print a summary table.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["runs/eval_videos"],
        help="Files or directories to search (default: runs/eval_videos).",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a table.")
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths]
    metrics_paths = _collect_metrics_paths(paths)
    if not metrics_paths:
        raise SystemExit("No metrics.json files found.")

    rows = [_load_metrics(p) for p in metrics_paths]
    rows.sort(key=lambda r: (r.get("scenario", ""), r.get("algo", ""), r.get("path", "")))

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        _print_table(rows)


if __name__ == "__main__":
    main()
