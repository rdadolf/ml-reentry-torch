from tools.canal.dsl import experiment

EXPERIMENTS = [
    experiment("pointwise_chain", analysis="codegen"),
    experiment("custom_pointwise_chain", analysis="codegen"),
]
