import subprocess
from pathlib import Path

from remote_adapter import (
    RemoteDefinition,
    build_remote_plan,
)


def run(command: str) -> None:
    subprocess.run(
        command,
        shell=True,
        check=True,
    )


def setup_dvc_init() -> None:
    if Path(".dvc").exists():
        print("DVC já inicializado.")
        return

    run("dvc init")


def setup_dataset() -> None:
    dvc_file = Path("data/raw/events.csv.dvc")

    if dvc_file.exists():
        print("Dataset já versionado pelo DVC.")
        return

    run("dvc add data/raw/events.csv")


def remote_exists(name: str) -> bool:
    result = subprocess.run(
        ["dvc", "remote", "list"],
        capture_output=True,
        text=True,
        check=True,
    )

    return name in result.stdout


def main() -> None:
    # 1. Inicializa DVC
    setup_dvc_init()

    # 2. Versiona dataset
    setup_dataset()

    # 3. Configura remote
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
