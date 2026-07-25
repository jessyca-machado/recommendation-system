import os

from src.config.settings import settings


def test_runtime_user():
    """
    Valida que o processo não roda como root.
    """
    assert os.getuid() != 0


def test_artifacts_directories_are_writable():
    """
    Valida que os diretórios configurados possuem permissão de escrita.
    """

    paths = [
        settings.processed_data_dir,
        settings.metrics_dir,
        settings.models_dir,
    ]

    for path in paths:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        test_file = path / "test_permission.txt"

        test_file.write_text("ok")

        assert test_file.exists()

        test_file.unlink()


def test_can_write_temp_file(tmp_path):
    """
    Valida que o ambiente permite criação de arquivos.
    """

    file = tmp_path / "test.txt"

    file.write_text("ok")

    assert file.exists()


def test_settings_paths_are_configured():
    """
    Valida que os caminhos principais estão configurados no Settings.
    """

    paths = [
        settings.processed_data_dir,
        settings.metrics_dir,
        settings.models_dir,
    ]

    for path in paths:
        assert path is not None
