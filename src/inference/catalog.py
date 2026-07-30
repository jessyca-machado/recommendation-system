from functools import lru_cache

import pandas as pd

from src.config.settings import settings


@lru_cache(maxsize=1)
def load_item_catalog() -> pd.DataFrame:
    return pd.read_parquet(settings.processed_data_dir / "item_catalog.parquet")
