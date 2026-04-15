from tools.canal.dsl import experiment

EXPERIMENTS = [
    # Mostly the same. Triton dispatch wrapper call stands out a bit.
    experiment("silu_ffn", analysis="fx", device="cuda"),
    experiment("custom_silu_ffn", analysis="fx", device="cuda"),
    experiment("custom_py_silu_ffn", analysis="fx", device="cuda"),
    experiment("custom_triton_silu_ffn", analysis="fx", device="cuda"),
    # The silu_ffn v custom_triton_silu_ffn kernels are fun to look at
    experiment("silu_ffn", analysis="codegen", device="cuda"),
    experiment("custom_silu_ffn", analysis="codegen", device="cuda"),
    experiment("custom_py_silu_ffn", analysis="codegen", device="cuda"),
    experiment("custom_triton_silu_ffn", analysis="codegen", device="cuda"),
]
