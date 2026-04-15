from shared.models import TEST_MODELS
from tools.canal.dsl import experiment

EXPERIMENTS = [*[experiment(name, analysis="breaks") for name in TEST_MODELS.keys()]]
