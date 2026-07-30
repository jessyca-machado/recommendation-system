import pandas as pd


def recommend(
    model,
    candidates: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    return model.recommend(
        candidates=candidates,
        k=k,
    )


def recommend_for_user(
    model,
    item_catalog: pd.DataFrame,
    user_idx: int,
    k: int = 10,
) -> pd.DataFrame:
    candidates = pd.DataFrame(
        {
            "user_idx": [user_idx] * len(item_catalog),
            "item_idx": item_catalog["item_idx"],
        }
    )

    recommendations = model.recommend(
        candidates=candidates,
        k=k,
    )

    recommendations = recommendations.merge(
        item_catalog,
        on="item_idx",
        how="left",
    )

    return recommendations
