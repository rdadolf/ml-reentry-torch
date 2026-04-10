from tools.canal.dsl import experiment

EXPERIMENTS = [
    experiment("toy_llama", analysis="fx"),
    experiment("toy_llama", analysis="ir"),
    experiment("toy_llama", analysis="codegen"),
]
