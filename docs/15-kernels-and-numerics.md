# 15 — Kernels, numerics and threading

Settles **D2** (W4A8 or W4A16), **D4** (accumulation precision) and **D6**
(threading model) from [13](13-open-decisions.md). The three travel together
because each constrains the others.

## D2 — W4A32, not W4A8

Two kernel families for int4 weights:

| | W4A8 | **W4A32** (chosen) |
|---|---|---|
| instruction | `vpdpbusd` (AVX-VNNI), u8×s8→s32 | `vfmadd` fp32 |
| MAC throughput | ~4× | reference |
| additional error | yes, dynamic activation quantisation | **none** |
| cost | per-token scale, requantise on every layer | unpack and multiply |

The reflex is to pick W4A8 for throughput. **Here the reflex is wrong.**

From [08](08-performance-ceiling.md): per token it is ~2.17 GB of reads against
7.1 GFLOP of compute. At 60 GB/s and ~1–2 TOPS int8, that is **~33 ms of memory
against 4–7 ms of compute** — compute utilisation around **20%**.

MAC throughput is not the bottleneck. There is plenty of slack, and spending it
to buy precision is the right trade: with no activation quantisation, M1's
token-exact oracle becomes much easier to hit, and there is no second source of
error to debug when it fails.

**Decision:** W4A32 in M1. Migrate to W4A8 only if measurement shows compute to
be the limit — and then with the number in hand, not by reflex.

Practical consequence: the kernel unpacks int4 → fp32 and multiplies with FMA.
The unpacking is part of the compute and fits inside that 20%; the half/half
layout in [14](14-container-format.md) exists so that it costs one `and`, one
`shift` and a conversion, with no permutation.

## D4 — accumulation precision and order

M1's exit criterion is **32/32 tokens identical to the oracle**. That does not
happen by accident; it comes from deliberate choices.

| operation | decision |
|---|---|
| matmul accumulator | `float` (fp32) |
| int4 unpacking | to `float`, scale applied per group at the end |
| rmsnorm | sum of squares in `float`, fixed order |
| router softmax | `float`, with max subtraction |
| RoPE | `float`, sine/cosine table precomputed at init |
| attention (scores, softmax) | `float` |
| residual | `float` |

Nothing in `double`: the `transformers` oracle runs in fp32 and matching it is
the goal, not beating it.

### The rule that decides reproducibility

> **The reduction order is fixed and independent of the thread count.**

Floating-point addition is not associative. If a matmul's reduction depends on
how the work was split, the result changes with `--threads` — and the determinism
test in [03](03-architecture.md) fails by construction, not by bug.

Concretely: **each thread owns whole output rows**. The reduction along `I`
happens entirely inside one thread, always in the same order (group 0, 1, 2, …),
with one scalar accumulator per row. Threads never sum each other's partials.

That costs: nothing on large matmuls (there are plenty of rows to distribute) and
something on short vectors, where some threads sit idle. Buying reproducibility
with a little idleness is the right trade when reproducibility is an exit
criterion.

The same applies to the sum inside a SIMD vector: the lane combination order is
fixed and documented in the kernel, not left to the compiler. No `-ffast-math`.

### Where MLA can diverge

MLA attention admits two algebraically equivalent formulations: unabsorb the
compressed KV into a full `[nope|rope]` and do normal attention, or absorb `kv_b`
into the query projection and operate in the compressed space. **They do not
produce the same bits.**

The oracle uses the first. M1 implements the first. The second is an M5
optimisation, and when it lands it will have to be validated as an acceptable
output change or rejected — not introduced silently.

## D6 — threading

**Our own pool, not OpenMP.** OpenMP would work, but it drags in `libgomp` and
the promise in [10](10-implementation.md) is static linking with minimal runtime.
The pool is ~200 lines: N threads, one barrier, one index task queue.

```
main:       for each layer (47, serial)
              publish task {func, arguments, total_rows}
              barrier
            (the layer chain is serial by nature; there is nothing to overlap
             within a single sequence)
worker i:   grabs row ranges by atomic index
            processes the whole range, local accumulator
            barrier
```

Partitioning **by output row**, dynamic granularity (ranges taken from an atomic
counter) to tolerate the imbalance between P-cores and E-cores. Never by expert:
with top-4 of 64, partitioning by expert would leave most threads idle.

Affinity as per [09](09-resource-limits.md): P-cores by default, with
`--affinity all` for measurement. Pinning happens at thread creation.

Barrier: a short spin followed by blocking. At 22 tok/s that is ~47 barriers per
token, ~1000 per second — pure spinning would burn the E-cores for nothing.

## Implementation order in M1

1. reference `matmul_f32`, no SIMD, no threads — the executable definition
2. `matmul_int4_g` likewise, naive, validated against `dequant_grouped` in Python
3. the token-exact oracle passing with both naive versions
4. only then SIMD, and the oracle has to keep passing
5. only then threads, and the oracle has to keep passing **with any
   `--threads`**

Each step keeps the previous one as an executable reference. When SIMD diverges —
and it will — comparing against the naive version locates the error in minutes,
instead of turning the oracle into a test of everything against nothing.
