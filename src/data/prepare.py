"""Prepara os datasets de treino e teste a partir do dataset Retail Rocket.

Uso:
    python src/data/prepare.py
"""
import logging
from typing import Any

import pandas as pd

from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Carrega e pré-processa o dataset Retail Rocket.

    Returns:
        pd.DataFrame: Dados brutos do dataset com as colunas originais.
    """
    df = pd.read_csv(settings.raw_events_path)

    logger.info("Dataset carregado: %d linhas", len(df))

    return df


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide os dados em conjuntos de treino e teste por timestamp.

    Args:
        df: DataFrame com as interações brutas.
        test_size: Fração do conjunto usada para teste.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames de treino e teste.
    """
    df = df.copy()

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    cutoff = df["datetime"].quantile(1 - test_size)

    train = df[df["datetime"] < cutoff]
    test = df[df["datetime"] >= cutoff]

    train_users = set(train["visitorid"])
    train_items = set(train["itemid"])

    test = test[test["visitorid"].isin(train_users)]

    test = test[test["itemid"].isin(train_items)]

    return train, test


def save_item_catalog(train: pd.DataFrame) -> None:
    """Salva o catálogo de itens utilizados na inferência.

    Args:
        train: DataFrame de treino já com índices mapeados.
    """
    item_catalog = train[["item_idx", "itemid"]].drop_duplicates().sort_values("item_idx")

    item_catalog.to_parquet(
        settings.processed_data_dir / "item_catalog.parquet",
        index=False,
    )

    logger.info(
        "Item catalog salvo: %d itens",
        len(item_catalog),
    )


def save_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Salva dados de treino e teste em formato parquet.

    Args:
        train: DataFrame de treino preparado.
        test: DataFrame de teste preparado.
    """
    settings.processed_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_parquet(
        settings.train_path,
        index=False,
    )

    test.to_parquet(
        settings.test_path,
        index=False,
    )

    save_item_catalog(train)


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os dados de treino e teste a partir dos arquivos parquet.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames de treino e teste.
    """
    train = pd.read_parquet(
        settings.train_path,
    )

    test = pd.read_parquet(
        settings.test_path,
    )

    logger.info("Dataset carregado: %d linhas", len(train))
    logger.info("Dataset carregado: %d linhas", len(test))

    return train, test


def build_mappings(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[Any, int], dict[Any, int]]:
    """Cria mapeamentos de usuários e itens para índices numéricos.

    Args:
        df: DataFrame com as colunas de usuário e item.

    Returns:
        tuple[pd.DataFrame, dict[Any, int], dict[Any, int]]: DataFrame com índices e
            dicionários de mapeamento.
    """
    user_ids = df["visitorid"].unique()
    item_ids = df["itemid"].unique()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {i: j for j, i in enumerate(item_ids)}

    df = df.copy()

    df["user_idx"] = df["visitorid"].map(user2idx)
    df["item_idx"] = df["itemid"].map(item2idx)

    return df, user2idx, item2idx


def apply_mappings(
    df: pd.DataFrame,
    user2idx: dict[Any, int],
    item2idx: dict[Any, int],
) -> pd.DataFrame:
    """Aplica mapeamentos de usuários e itens aos dados de teste.

    Args:
        df: DataFrame com colunas de usuário e item originais.
        user2idx: Mapeamento de usuário para índice.
        item2idx: Mapeamento de item para índice.

    Returns:
        pd.DataFrame: DataFrame com colunas de índice numéricas.
    """
    df = df.copy()

    df["user_idx"] = df["visitorid"].map(user2idx)
    df["item_idx"] = df["itemid"].map(item2idx)

    before = len(df)

    df = df.dropna(
        subset=[
            "user_idx",
            "item_idx",
        ]
    )

    after = len(df)

    logger.info(
        "Removed %s interactions with unseen users/items",
        before - after,
    )

    df["user_idx"] = df["user_idx"].astype(int)
    df["item_idx"] = df["item_idx"].astype(int)

    return df


def main() -> None:
    """Executa o pipeline de preparação dos dados."""
    df = load_data()

    train, test = split_train_test(df)

    train, user2idx, item2idx = build_mappings(train)
    test = apply_mappings(test, user2idx, item2idx)

    save_datasets(
        train=train,
        test=test,
    )

    train, test = load_train_test()


if __name__ == "__main__":
    main()
