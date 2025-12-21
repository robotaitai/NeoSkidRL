from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import yaml

def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        repo_root = Path(__file__).resolve().parents[2]
        default_path = repo_root / "config" / "default.yml"
        if default_path.exists():
            return yaml.safe_load(default_path.read_text())
        raise FileNotFoundError("Default config not found. Pass --config explicitly.")
    p = Path(path)
    return yaml.safe_load(p.read_text())


def merge_config(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
