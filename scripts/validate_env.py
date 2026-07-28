"""Valida se o ambiente está pronto para executar o pipeline."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from src.config.params import load_params
from src.config.settings import settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    problems = collect_problems()

    if problems:
        print("❌ Ambiente inválido\n")

        for problem in problems:
            print(f" - {problem}")

        sys.exit(1)

    print("✅ Ambiente validado com sucesso.")


def check_params() -> list[str]:
    problems: list[str] = []

    try:
        params = load_params()
    except Exception as exc:
        return [f"Não foi possível carregar params.yaml: {exc}"]

    experiment = params.get("experiment", {})

    random_seed = experiment.get("random_seed")
    if not isinstance(random_seed, int) or random_seed < 0:
        problems.append("experiment.random_seed deve ser um inteiro não negativo.")

    top_k = experiment.get("top_k")
    if not isinstance(top_k, int) or top_k <= 0:
        problems.append("experiment.top_k deve ser maior que zero.")

    return problems


def collect_problems() -> list[str]:
    problems: list[str] = []

    problems.extend(check_tools())
    problems.extend(check_files())
    problems.extend(check_directories())
    problems.extend(check_settings())
    problems.extend(check_params())

    return problems


def check_tools() -> list[str]:
    problems: list[str] = []

    required_tools = [
        "git",
        "dvc",
    ]

    for tool in required_tools:
        if shutil.which(tool) is None:
            problems.append(f"Ferramenta '{tool}' não encontrada no PATH.")

    return problems


def check_files() -> list[str]:
    problems: list[str] = []

    required_files = [
        PROJECT_ROOT / "config" / "params.yaml",
        PROJECT_ROOT / "dvc.yaml",
        PROJECT_ROOT / "pyproject.toml",
    ]

    for file in required_files:
        if not file.exists():
            problems.append(f"Arquivo obrigatório não encontrado: {file}")

    return problems


def check_directories() -> list[str]:
    problems: list[str] = []

    required_dirs = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "scripts",
    ]

    for directory in required_dirs:
        if not directory.exists():
            problems.append(f"Diretório obrigatório não encontrado: {directory}")

    return problems


def check_settings() -> list[str]:
    problems: list[str] = []

    if not settings.mlflow_tracking_uri:
        problems.append("MLFLOW_TRACKING_URI não configurado.")
    else:
        try:
            from mlflow.tracking import MlflowClient

            MlflowClient().search_experiments()
        except Exception as exc:
            problems.append(f"Não foi possível conectar ao MLflow: {exc}")

    raw_parent = settings.raw_data_dir.parent

    if not raw_parent.exists():
        problems.append(f"Pasta de dados brutos não encontrada: {raw_parent}")

    if not settings.raw_data_dir.exists():
        problems.append(f"Arquivo de entrada não encontrado: {settings.raw_data_dir}")

    return problems


if __name__ == "__main__":
    main()
