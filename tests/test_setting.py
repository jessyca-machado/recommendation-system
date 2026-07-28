from src.config.settings import Settings


def test_settings_load_defaults():
    config = Settings(_env_file=None)

    assert config.mlflow_experiment_name == "recommendation-system"
    assert config.openblas_num_threads == 1
    assert config.omp_num_threads == 1
    assert config.mkl_num_threads == 1
