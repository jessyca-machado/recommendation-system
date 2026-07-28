import numpy as np
import pandas as pd


def build_candidates_ncf(
    test: pd.DataFrame,
    train: pd.DataFrame,
    n_items: int,
    num_eval_negatives: int = 99,
    seed: int = 42,
    positive_strategy: str = "random_one",
) -> pd.DataFrame:
    """
    Protocolo NCF: 1 positivo por usuário + N negativos amostrados.

    positive_strategy:
        - "random_one": escolhe 1 interação do test por usuário (aleatória)
        - "last": escolhe a última interação do test por usuário (requer coluna 'timestamp')

    Returns:
        user_idx, item_idx, label (1/0)
    """
    rng = np.random.default_rng(seed)

    if positive_strategy == "random_one":
        positives = (
            test.groupby("user_idx", group_keys=False)
            .sample(n=1, random_state=seed)
            .reset_index(drop=True)
        )[["user_idx", "item_idx"]].copy()

    elif positive_strategy == "last":
        if "timestamp" not in test.columns:
            raise ValueError("positive_strategy='last' requer coluna 'timestamp' no test")
        positives = (
            test.sort_values(["user_idx", "timestamp"]).groupby("user_idx", as_index=False).tail(1)
        )[["user_idx", "item_idx"]].copy()
    else:
        raise ValueError(f"positive_strategy inválida: {positive_strategy}")

    positives["label"] = 1

    seen = (
        pd.concat([train[["user_idx", "item_idx"]], test[["user_idx", "item_idx"]]])
        .groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )

    rows = []
    for u, pos in positives[["user_idx", "item_idx"]].itertuples(index=False):
        rows.append((u, pos, 1))
        user_seen = seen.get(u, set())

        negs = set()
        while len(negs) < num_eval_negatives:
            j = int(rng.integers(0, n_items))
            if j == pos or j in user_seen or j in negs:
                continue
            negs.add(j)

        for j in negs:
            rows.append((u, j, 0))

    return pd.DataFrame(rows, columns=["user_idx", "item_idx", "label"])
