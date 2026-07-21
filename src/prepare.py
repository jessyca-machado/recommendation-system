"""Prepara os datasets de treino e teste a partir do dataset Retail Rocket.

Uso:
    python src/prepare.py
"""
import logging
import pandas as pd

from src.config import RAW_DATA_PATH, TRAIN_PATH, TEST_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Carrega e pré-processa o dataset Retail Rocket.

    Returns:
        DataFrame com os dados do dataset
    """
    df = pd.read_csv(RAW_DATA_PATH)

    logger.info("Dataset carregado: %d linhas", len(df))

    return df


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide o dataset em treino e teste com base na coluna de timestamp.

    Returns:
        train: DataFrame com os dados de treino
        test: DataFrame com os dados de teste
    """
    df = df.copy()

    df['datetime'] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    cutoff = df["datetime"].quantile(1 - test_size)

    train = df[df["datetime"] < cutoff]
    test = df[df["datetime"] >= cutoff]

    train_users = set(train["visitorid"])
    train_items = set(train["itemid"])

    test = test[
        test["visitorid"].isin(train_users)
    ]

    test = test[
        test["itemid"].isin(train_items)
    ]

    return train, test


def save_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Salva os datasets de treino e teste em arquivos parquet.

    Returns:
        None
    """
    train.to_parquet(TRAIN_PATH)
    test.to_parquet(TEST_PATH)


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os datasets de treino e teste a partir dos arquivos parquet.

    Returns:
        train: DataFrame com os dados de treino
        test: DataFrame com os dados de teste
    """
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    logger.info("Dataset carregado: %d linhas", len(train))
    logger.info("Dataset carregado: %d linhas", len(test))

    return train, test


def build_mappings(df):

    user_ids = df["visitorid"].unique()
    item_ids = df["itemid"].unique()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {i: j for j, i in enumerate(item_ids)}

    df = df.copy()

    df["user_idx"] = df["visitorid"].map(user2idx)
    df["item_idx"] = df["itemid"].map(item2idx)

    return df, user2idx, item2idx


def apply_mappings(
    df,
    user2idx,
    item2idx
):

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


if __name__ == "__main__":
    df = load_data()

    train, test = split_train_test(df)

    train, user2idx, item2idx = build_mappings(train)
    test = apply_mappings(
        test,
        user2idx,
        item2idx
    )

    save_datasets(
        train=train,
        test=test,
    )

    train, test = load_train_test()
