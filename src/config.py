from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "events.csv"

PROCESSED_DIR = ROOT_DIR / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
