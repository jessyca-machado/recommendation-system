import subprocess
from pathlib import Path

from remote_adapter import (
    RemoteDefinition,
    build_remote_plan,
)


def run(command: str) -> None:
    """Executa um comando do shell com verificação de erro.

    Args:
        command: Comando shell a ser executado.
    """

    subprocess.run(
        command,
        shell=True,
        check=True,
    )


def setup_dvc_init() -> None:
    """Inicializa o DVC no repositório, quando ainda não estiver ativo."""

    if Path(".dvc").exists():
        print("DVC já inicializado.")
        return

    run("dvc init")


def setup_dataset() -> None:
    """Versiona o dataset principal com o DVC, se ainda não estiver versionado."""

    dvc_file = Path("data/raw/events.csv.dvc")

    if dvc_file.exists():
        print("Dataset já versionado pelo DVC.")
        return

    run("dvc add data/raw/events.csv")


def remote_exists(name: str) -> bool:
    """Verifica se um remoto DVC já está configurado.

    Args:
        name: Nome do remoto DVC.

    Returns:
        bool: True se o remoto existir, False caso contrário.
    """

    result = subprocess.run(
        ["dvc", "remote", "list"],
        capture_output=True,
        text=True,
        check=True,
    )

    return name in result.stdout


def main() -> None:
    """Executa a configuração completa do DVC e dos remotos."""

    setup_dvc_init()

    setup_dataset()

    remote = RemoteDefinition(
        name="local-cache",
        uri="./artifacts/dvc-cache",
    )

    plan = build_remote_plan(
        remote,
        Path.cwd(),
    )

    if not remote_exists(plan.name):
        for command in plan.dvc_commands:
            run(command)
    else:
        print(f"Remote {plan.name} já configurado.")

    print("DVC configurado com sucesso.")


if __name__ == "__main__":
    main()
