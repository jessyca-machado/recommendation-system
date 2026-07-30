import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configurações da aplicação."""

    # ========= MLflow =========
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "recommendation-system"
    mlflow_registered_model_name: str = "recommendation-model"

    # ========= Diretórios =========
    root_dir: Path = ROOT_DIR

    config_dir: Path = ROOT_DIR / "config"

    data_dir: Path = ROOT_DIR / "data"
    raw_data_dir: Path = ROOT_DIR / "data" / "raw"
    processed_data_dir: Path = ROOT_DIR / "data" / "processed"

    artifacts_dir: Path = ROOT_DIR / "artifacts"
    metrics_dir: Path = ROOT_DIR / "artifacts" / "metrics"
    models_dir: Path = ROOT_DIR / "artifacts" / "models"

    # ========= Arquivos =========
    raw_events_path: Path = ROOT_DIR / "data" / "raw" / "events.csv"

    train_path: Path = ROOT_DIR / "data" / "processed" / "train.parquet"
    test_path: Path = ROOT_DIR / "data" / "processed" / "test.parquet"

    # ========= Runtime =========
    openblas_num_threads: int = 1
    omp_num_threads: int = 1
    mkl_num_threads: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


def configure_runtime(settings: Settings) -> None:
    """
    Configura variáveis de ambiente antes do carregamento
    de bibliotecas numéricas (NumPy, SciPy, Implicit etc.).
    """

    os.environ["OPENBLAS_NUM_THREADS"] = str(settings.openblas_num_threads)
    os.environ["OMP_NUM_THREADS"] = str(settings.omp_num_threads)
    os.environ["MKL_NUM_THREADS"] = str(settings.mkl_num_threads)
