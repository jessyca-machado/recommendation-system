import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from src.config.params import load_params
from src.models.factory import build_models


params = load_params()

models = build_models(
    params["models"]
)


for name, cfg in models.items():
    print(
        name,
        type(cfg["model"]).__name__,
        cfg["input"]
    )
