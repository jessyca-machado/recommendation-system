from src.config.params import load_params
from src.models.factory import build_models

from src.experiments.runner import run_experiments


params = load_params()

models = build_models(params["models"])

print(models.keys())
