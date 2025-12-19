from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import yaml
import importlib.resources as ir

def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        # packaged default
        with ir.as_file(ir.files("neoskidrl.config").joinpath("default.yml")) as p:
            return yaml.safe_load(p.read_text())
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
