// Custom ops for the reentry namespace (torch.ops.reentry.*).
// Built via torch.utils.cpp_extension.load() — see register.py.

#include <cmath> // std::exp
#include <type_traits> // std::common_type_t

#include <Python.h>

#include <torch/csrc/stable/library.h> // Registration macros
#include <torch/csrc/stable/ops.h> // Basic Tensor operations
#include <torch/csrc/stable/tensor.h> // Tensor type
#include <torch/headeronly/core/ScalarType.h> // Other basic types
#include <torch/headeronly/macros/Macros.h> // Check macros

extern "C" {
    // Null Python module. Required by Python to trigger the rest of this file on `import`.
    PyObject* PyInit_reentry_ops(void)
    {
        static struct PyModuleDef module_def = {
            .m_base = PyModuleDef_HEAD_INIT,
            .m_name = "reentry_ops",
            .m_size = -1,
        };
        return PyModule_Create(&module_def);
    }
}

namespace {

// ── identity ───────────────────────────────────────────────────────
// Pass-through: returns a contiguous copy of the input.
// Exists to validate the registration/build/test infrastructure
// before implementing real ops.

torch::stable::Tensor identity_cpu(const torch::stable::Tensor& x) {
    return torch::stable::clone(x);
}

// ── silu_and_mul ───────────────────────────────────────────────────
// Fused SiLU/multiply.

template <typename TGate, typename TUp>
void silu_and_mul_impl(
        torch::stable::Tensor& output,
        const torch::stable::Tensor& gate,
        const torch::stable::Tensor& up) {
    using TOut = std::common_type_t<TGate, TUp>;
    
    const TGate* gate_data = gate.const_data_ptr<TGate>();
    const TUp* up_data = up.const_data_ptr<TUp>();
    TOut* output_data = output.mutable_data_ptr<TOut>();

    for (int64_t i = 0; i < gate.numel(); ++i) {
        TOut gate_val = static_cast<TOut>(gate_data[i]);
        TOut up_val = static_cast<TOut>(up_data[i]);
        output_data[i] = gate_val / (TOut(1.0) + std::exp(-gate_val)) * up_val;
    }
}

torch::stable::Tensor silu_and_mul_cpu(const torch::stable::Tensor& gate, const torch::stable::Tensor& up) {
    using enum torch::headeronly::ScalarType;
    using enum torch::headeronly::DeviceType;

    STD_TORCH_CHECK(gate.sizes().equals(up.sizes()));

    auto gate_t = gate.scalar_type();
    auto up_t = up.scalar_type();
    auto out_t = (gate_t == Float && up_t == Float) ? Float : Double;

    torch::stable::Tensor gate_contig = torch::stable::contiguous(gate);
    torch::stable::Tensor up_contig = torch::stable::contiguous(up);
    torch::stable::Tensor output = torch::stable::new_empty(gate_contig, gate.sizes(), out_t);

    if (gate_t == Float && up_t == Float) {
        silu_and_mul_impl<float, float>(output, gate_contig, up_contig);
    } else if (gate_t == Float && up_t == Double) {
        silu_and_mul_impl<float, double>(output, gate_contig, up_contig);
    } else if (gate_t == Double && up_t == Float) {
        silu_and_mul_impl<double, float>(output, gate_contig, up_contig);
    } else if (gate_t == Double && up_t == Double) {
        silu_and_mul_impl<double, double>(output, gate_contig, up_contig);
    } else {
        STD_TORCH_CHECK(false, "Unsupported dtype combination: gate=", gate_t, ", up=", up_t);
    }

    return output;
}

STABLE_TORCH_LIBRARY(reentry, m) {
    m.def("identity(Tensor x) -> Tensor");
    m.def("silu_and_mul(Tensor gate, Tensor up) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(reentry, CPU, m) {
    m.impl("identity", TORCH_BOX(&identity_cpu));
    m.impl("silu_and_mul", TORCH_BOX(&silu_and_mul_cpu));
}

} // namespace