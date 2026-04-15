// CUDA implementations for the reentry namespace custom ops.
// Mirrors ops.cpp — same ops, same stable API, GPU kernels.

#include <cmath>
#include <type_traits>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

namespace {

// ── identity ───────────────────────────────────────────────────────

torch::stable::Tensor identity_cuda(const torch::stable::Tensor& x) {
    return torch::stable::clone(x);
}

// ── silu_and_mul ───────────────────────────────────────────────────

template <typename TGate, typename TUp>
__global__ void silu_and_mul_kernel(
        const TGate* gate,
        const TUp* up,
        std::common_type_t<TGate, TUp>* output,
        int64_t n) {
    using TOut = std::common_type_t<TGate, TUp>;
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        TOut g = static_cast<TOut>(gate[i]);
        TOut u = static_cast<TOut>(up[i]);
        output[i] = g / (TOut(1.0) + exp(-g)) * u;
    }
}

torch::stable::Tensor silu_and_mul_cuda(
        const torch::stable::Tensor& gate,
        const torch::stable::Tensor& up) {
    using enum torch::headeronly::ScalarType;

    STD_TORCH_CHECK(gate.sizes().equals(up.sizes()));

    auto gate_t = gate.scalar_type();
    auto up_t = up.scalar_type();
    auto out_t = (gate_t == Float && up_t == Float) ? Float : Double;

    torch::stable::Tensor gate_contig = torch::stable::contiguous(gate);
    torch::stable::Tensor up_contig = torch::stable::contiguous(up);
    torch::stable::Tensor output = torch::stable::new_empty(gate_contig, gate.sizes(), out_t);

    int64_t n = gate.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    if (gate_t == Float && up_t == Float) {
        silu_and_mul_kernel<float, float><<<blocks, threads>>>(
            gate_contig.const_data_ptr<float>(),
            up_contig.const_data_ptr<float>(),
            output.mutable_data_ptr<float>(),
            n);
    } else if (gate_t == Float && up_t == Double) {
        silu_and_mul_kernel<float, double><<<blocks, threads>>>(
            gate_contig.const_data_ptr<float>(),
            up_contig.const_data_ptr<double>(),
            output.mutable_data_ptr<double>(),
            n);
    } else if (gate_t == Double && up_t == Float) {
        silu_and_mul_kernel<double, float><<<blocks, threads>>>(
            gate_contig.const_data_ptr<double>(),
            up_contig.const_data_ptr<float>(),
            output.mutable_data_ptr<double>(),
            n);
    } else if (gate_t == Double && up_t == Double) {
        silu_and_mul_kernel<double, double><<<blocks, threads>>>(
            gate_contig.const_data_ptr<double>(),
            up_contig.const_data_ptr<double>(),
            output.mutable_data_ptr<double>(),
            n);
    } else {
        STD_TORCH_CHECK(false, "Unsupported dtype combination: gate=", gate_t, ", up=", up_t);
    }

    return output;
}

STABLE_TORCH_LIBRARY_IMPL(reentry, CUDA, m) {
    m.impl("identity", TORCH_BOX(&identity_cuda));
    m.impl("silu_and_mul", TORCH_BOX(&silu_and_mul_cuda));
}

} // namespace
