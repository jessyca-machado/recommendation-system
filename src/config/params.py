from pathlib import Path
from typing import Any

import yaml

PARAMS_PATH = Path("config/params.yaml")


def load_params() -> dict[str, Any]:
    """Carrega os parâmetros do experimento a partir do arquivo YAML.

    Returns:
        dict[str, Any]: Configuração do pipeline de treinamento.
    """

    with PARAMS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
