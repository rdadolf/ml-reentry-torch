# Custom Op Fusion Tests

## Pointwise Chain

Simple elementwise operator chain:
```
        x = F.relu(x)
        x = x + 1
        x = x * 2
        x = F.relu(x)
        x = x - 0.5
        return x
```
vs. the same chain with a custom identity pass-through op inserted:
```
        from shared.custom_ops.register import identity

        x = F.relu(x)
        x = x + 1
        x = identity(x)  # fusion barrier
        x = x * 2
        x = F.relu(x)
        x = x - 0.5
        return x
```

Op fusion on CPU is limited, but Inductor does fuse elementwise operations, and `reentry.identity` is opaque to it. So we'd expect a fully-fused chain in the base example and two separate fused kernels in the custom test. And that's what we get:

```
=== pointwise_chain_codegen [pointwise_chain => codegen] on: cpu
subgraph: model__0_inference_0.0
  generated code: 107 lines
  compiled kernels: 1
  kernel names: cpp_fused_add_mul_relu_sub_0
```
versus
```
=== custom_pointwise_chain_codegen [custom_pointwise_chain => codegen] on: cpu
subgraph: model__1_inference_1.1
  generated code: 136 lines
  compiled kernels: 2
  kernel names: cpp_fused_add_identity_relu_0, cpp_fused_mul_relu_sub_1
```

The names are a bit misleading here. It looks like torch just picks all the ops and sorts a list alphabetically without duplicates. So while the op chain is `relu-add-(identity)-mul-relu-sub`, that gets smashed down to just `fused_add_mul_relu_sub`. The code is all of it:

```
extern "C"  void  kernel(const float* in_ptr0, float* out_ptr0)
...
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = static_cast<float>(2.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = at::vec::clamp_min(tmp7, decltype(tmp7)(0));
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    tmp11.store(out_ptr0 + static_cast<int64_t>(x0));
...
```

For the custom-insert, we have two kernels `fused_add_identity_relu` and `fused_mul_relu_sub`. The latter is exactly what you'd expect and looks just like the above example starting at `tmp5`. The former is a little more funny because it includes the `identity`---except that it really doesn't:

```
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
...
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<int64_t>(x0));
...
```
which is the exact prefix of the base kernel up to `tmp4`---there's no identity. Not that we expected one. It's just that the fused kernel name happens to include it for some reason. Instead, our custom kernel is called outside the fused block as we anticipated:

```
        cpp_fused_add_identity_relu_0(arg0_1, buf0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.relu, aten.add, reentry.identity]
        buf1 = torch.ops.reentry.identity.default(buf0)
...
        cpp_fused_mul_relu_sub_1(buf3)
        return (buf3, )
```