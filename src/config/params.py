from pathlib import Path

import yaml

PARAMS_PATH = Path("config/params.yaml")


def load_params() -> dict:
    """Load experiment parameters."""

    with PARAMS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
