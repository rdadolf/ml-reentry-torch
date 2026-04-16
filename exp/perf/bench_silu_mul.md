# Dispatch wall-clock performance tests

## Raw results:

```
  [ 1/33]  native       | eager            | (128, 4096) |     18.5 ±    1.8 us (11 blocks)
  [ 2/33]  native       | compiled         | (128, 4096) |     88.5 ±   12.0 us (22 blocks) (!)
  [ 3/33]  native       | compiled+graphs  | (128, 4096) |    123.9 ±   17.9 us (16 blocks) (!)
  [ 4/33]  cpp_cuda     | eager            | (128, 4096) |     30.8 ±    0.8 us (7 blocks)
  [ 5/33]  cpp_cuda     | compiled         | (128, 4096) |    112.2 ±    2.8 us (18 blocks)
  [ 6/33]  custom_op    | eager            | (128, 4096) |     54.1 ±    1.6 us (37 blocks)
  [ 7/33]  custom_op    | compiled         | (128, 4096) |    136.9 ±   14.1 us (15 blocks) (!)
  [ 8/33]  custom_op    | compiled+graphs  | (128, 4096) |    114.9 ±    3.4 us (18 blocks)
  [ 9/33]  triton       | eager            | (128, 4096) |     84.1 ±    2.7 us (24 blocks)
  [10/33]  triton       | compiled         | (128, 4096) |     89.2 ±    2.4 us (22 blocks)
  [11/33]  triton       | compiled+graphs  | (128, 4096) |    114.9 ±    8.5 us (17 blocks)
  [12/33]  native       | eager            | (512, 4096) |     23.7 ±    1.3 us (81 blocks)
  [13/33]  native       | compiled         | (512, 4096) |     86.5 ±    2.8 us (23 blocks)
  [14/33]  native       | compiled+graphs  | (512, 4096) |    113.7 ±    3.4 us (18 blocks)
  [15/33]  cpp_cuda     | eager            | (512, 4096) |     31.5 ±    1.1 us (7 blocks)
  [16/33]  cpp_cuda     | compiled         | (512, 4096) |    118.4 ±   15.5 us (17 blocks) (!)
  [17/33]  custom_op    | eager            | (512, 4096) |     54.8 ±    1.4 us (36 blocks)
  [18/33]  custom_op    | compiled         | (512, 4096) |    138.8 ±   14.9 us (14 blocks) (!)
  [19/33]  custom_op    | compiled+graphs  | (512, 4096) |    123.3 ±    6.0 us (16 blocks)
  [20/33]  triton       | eager            | (512, 4096) |     83.8 ±    3.5 us (24 blocks)
  [21/33]  triton       | compiled         | (512, 4096) |     87.6 ±    1.7 us (23 blocks)
  [22/33]  triton       | compiled+graphs  | (512, 4096) |    114.7 ±    6.0 us (17 blocks)
  [23/33]  native       | eager            | (1024, 8192) |    351.0 ±    7.3 us (6 blocks)
  [24/33]  native       | compiled         | (1024, 8192) |    221.3 ±    1.1 us (10 blocks)
  [25/33]  native       | compiled+graphs  | (1024, 8192) |    506.9 ±    8.9 us (40 blocks)
  [26/33]  cpp_cuda     | eager            | (1024, 8192) |    210.0 ±    2.0 us (10 blocks)
  [27/33]  cpp_cuda     | compiled         | (1024, 8192) |    215.4 ±    5.8 us (10 blocks)
  [28/33]  custom_op    | eager            | (1024, 8192) |    560.3 ±    7.0 us (36 blocks)
  [29/33]  custom_op    | compiled         | (1024, 8192) |    575.4 ±   10.1 us (35 blocks)
  [30/33]  custom_op    | compiled+graphs  | (1024, 8192) |    870.7 ±    7.1 us (24 blocks)
  [31/33]  triton       | eager            | (1024, 8192) |    212.3 ±    3.8 us (10 blocks)
  [32/33]  triton       | compiled         | (1024, 8192) |    220.5 ±    0.7 us (10 blocks)
  [33/33]  triton       | compiled+graphs  | (1024, 8192) |    505.4 ±    3.7 us (40 blocks)
```

## `Compare` output

### (128, 4096)

| impl | eager | compiled | compiled+graphs |
|------|------:|---------:|----------------:|
| native | 20 | 90 (! 14%) | 100 (! 14%) |
| cpp_cuda | 31 | 110 | — |
| custom_op | 54 | 100 (! 10%) | 110 |
| triton | 84 | 89 | 100 |

### (512, 4096)

| impl | eager | compiled | compiled+graphs |
|------|------:|---------:|----------------:|
| native | 24 | 86 | 110 |
| cpp_cuda | 30 | 100 (! 13%) | — |
| custom_op | 55 | 100 (! 11%) | 120 |
| triton | 84 | 88 | 110 |

### (1024, 8192)

| impl | eager | compiled | compiled+graphs |
|------|------:|---------:|----------------:|
| native | 350 | 220 | 510 |
| cpp_cuda | 210 | 220 | — |
| custom_op | 560 | 580 | 870 |
| triton | 210 | 220 | 505 |

Times are in microseconds (us).
(! XX%) Measurement has high variance, where XX is the IQR / median * 100.

## Observations

### The op is memory-bandwidth bound

silu_and_mul reads 2 tensors and writes 1, all float32. Total memory traffic is
12N bytes for ~6 FLOPs/element, giving an arithmetic intensity of 0.5 FLOPs/byte.
The RTX 4070's compute/bandwidth ratio is ~57 FLOPs/byte, so this op is firmly
bandwidth-bound at all sizes tested. Theoretical kernel times range from ~12 us
at (128, 4096) to ~200 us at (1024, 8192).

### Small tensors are dominated by dispatch/framework overhead

At (128, 4096) and (512, 4096), the theoretical kernel time is 12-50 us, but
all compiled variants measure 85-140 us. The gap is torch.compile's per-call
dispatch overhead (guard checks, wrapper logic). This overhead is roughly
constant, which is why the small and medium sizes show similar compiled times.
Eager native avoids this overhead entirely (18-24 us).

### compiled+graphs is slower than compiled for isolated ops

`mode="reduce-overhead"` uses CUDAGraphTreeManager, which copies inputs into
static graph buffers and copies outputs back on every replay. For (1024, 8192),
that's 3 x 8M x 4 bytes = 96 MB of memcpy per call (~190 us at 504 GB/s).
This explains the ~280 us gap between compiled (220 us) and compiled+graphs
(510 us) at that size. CUDA graphs are designed for long kernel sequences
(e.g., full transformer layers) where eliminating per-kernel dispatch overhead
outweighs the copy cost. For a single op, the copies dominate.

### cpp_cuda + CUDA graphs produces empty graphs

The C++ extension dispatches via `STABLE_TORCH_LIBRARY_IMPL`, which launches
CUDA kernels below the level where graph capture intercepts operations. The
captured graph is empty (confirmed by torch warning). This combination is
excluded from the results.

### @custom_op registration overhead is visible in eager

custom_op (Python `@custom_op` decorator) shows ~54 us eager vs native's ~20 us
and cpp_cuda's ~31 us. The ~23 us gap between custom_op and cpp_cuda is
consistent with the additional Python-level dispatch overhead introduced by the
`@custom_op` decorator path, relevant to the regression reported in
pytorch#139500. At large sizes (1024, 8192), custom_op eager (560 us) is
significantly slower than cpp_cuda eager (210 us) — the Python dispatch overhead
grows because `@custom_op` decomposes into multiple separate PyTorch ops
(`sigmoid`, `mul`, `mul`) rather than executing a single fused kernel.

### Methodology notes

- `torch.set_num_threads(1)` is set before benchmarking to match Timer's
  default, preventing dynamo guard invalidation from thread-count changes
  between warmup and measurement.
- `torch._dynamo.reset()` is called before each compiled measurement to avoid
  cache pollution across compiled wrappers sharing the same underlying function.
- Warmup uses 5 iterations (10 for compiled+graphs) to ensure compilation and
  graph capture complete before the timed region.