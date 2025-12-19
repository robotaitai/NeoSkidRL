from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml
import importlib.resources as ir

def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        # packaged default
        with ir.as_file(ir.files("neoskidrl.config").joinpath("default.yml")) as p:
            return yaml.safe_load(p.read_text())
    p = Path(path)
    return yaml.safe_load(p.read_text())
