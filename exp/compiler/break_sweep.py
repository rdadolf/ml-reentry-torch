from shared.models import ALL
from tools.canal.dsl import experiment

EXPERIMENTS = [*[experiment(name, analysis="breaks") for name in ALL.keys()]]
