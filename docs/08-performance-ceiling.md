# 08 — Performance ceiling

What the upper bound is, in tok/s, for TootegaEngine running GLM-4.7-Flash on
this machine. A roofline analysis from the model's real config.

## Correcting an earlier error

The first survey estimated **1.12 GB active per token**. That was wrong: the
script counted only the routed experts' FFN and **omitted attention and
`lm_head`**. In this model `hidden` is small (2048) while the MLA head dimensions
are large, so attention weighs far more than usual.

The correct number is **3.573B active parameters per token**.

## Active parameters, exact

From the config: `L=47`, `hidden=2048`, `heads=20`, `E=64`, `topk=4`, `shared=1`,
`moe_inter=1536`, `dense_inter=10240`, `first_dense=1`,
MLA `q_lora=768 kv_lora=512 qk_nope=192 qk_rope=64 v_head=256`.

| component | active/token | % |
|---|---|---|
| routed experts (top-4 of 64) | 1.736B | 48.6% |
| **attention (all layers)** | **1.023B** | **28.6%** |
| shared expert | 0.434B | 12.1% |
| **`lm_head`** | **0.317B** | **8.9%** |
| dense layer (MLP) | 0.063B | 1.8% |
| **total** | **3.573B** | |

Stored: 29.9B by the calculation, against 31.2B from the repo's bytes — the
difference is embeddings and norms that are not modelled. It closes.

In int4 with f32 scales per group of 64 (0.5625 B/weight): **2.01 GB of weights
read per token**.

## KV: the term that grows

MLA compresses the state to `kv_lora + qk_rope = 576` values per token per layer.
In fp16:

```
47 layers × 576 values × 2 B = 52.9 KB per token of context
```

In decode, attention reads the **whole** KV on every token. With context `N`,
that is `52.9 KB × N` of traffic per generated token.

## Roofline

`time_per_token = (2.01 GB + 52.9 KB × N) / bandwidth`

| context N | KV read | total/token | CPU @60 GB/s | 39% GPU split |
|---|---|---|---|---|
| 1,024 | 0.05 GB | 2.06 GB | **29.1 tok/s** | **44.0 tok/s** |
| 4,096 | 0.22 GB | 2.23 GB | **26.9** | **40.6** |
| 16,384 | 0.87 GB | 2.88 GB | **20.9** | **31.5** |
| 65,536 | 3.47 GB | 5.48 GB | **11.0** | **16.5** |
| 131,072 | 6.93 GB | 8.94 GB | **6.7** | **10.1** |

The CPU column does not depend on container size — only on the **active** weight
per token, which is still 2.01 GB. The "split" column does: it was 45% of the
layers in VRAM when a 15.6 GB container was assumed; with the measured 19.3 GB
and ~18.1 GB of layer weights, the same ~7 GB of usable VRAM covers **39%**
([04](04-target-model.md)).

The "split" column assumes 39% of the layers resident in VRAM computing at
448 GB/s and 61% on the CPU at 60 GB/s — the per-layer partition described in
[03-architecture.md](03-architecture.md). It is an **[estimate]**, contingent on
the hypothesis M2 validates.

## What the roofline does not include

These numbers assume **100% bandwidth efficiency, zero compute and zero kernel
overhead**. None of the three holds in practice:

- **Bandwidth efficiency.** Well-written memory-bound engines reach 60–80% of
  peak bandwidth. The 60 GB/s is already an "effective" estimate over 76.8 GB/s
  theoretical, so some discount is baked in — but not enough.
- **Compute is not negligible.** 2 × 3.573B = **7.1 GFLOP per token**. The 14900K
  with AVX-VNNI delivers ~1–2 TOPS int8, i.e. **4–7 ms**, against ~33 ms of
  memory at short context. That is ~20% of the time, and it needs to **overlap**
  with the read, not add to it. If the kernel does not overlap, the ceiling drops
  proportionally.
- **The chain is serial.** 47 layers in sequence; none parallelises with the next
  in single-sequence decode.

## Honest range

Applying 60–80% to the roofline:

| scenario | roofline | **realistic** |
|---|---|---|
| CPU-only, short context (≤4k) | 27–29 | **17–22 tok/s** |
| CPU+GPU split, short context | 41–44 | **25–32 tok/s** |
| CPU-only, agent context (32k) | ~15 | **10–12 tok/s** |
| split, agent context (32k) | ~24 | **15–19 tok/s** |

Baseline to beat: **0.93 tok/s** (streaming engine + GLM-5.2, measured). The gain
is between **11× and 34×**, depending on context and on whether the split pays.

## Two design consequences

**KV dominates from ~16k on.** In an agentic CLI, long context is the normal
case, not the exception: system prompt + tool schemas + files read + history
reach tens of thousands of tokens quickly. At 64k, KV is **63% of the traffic per
token**. Optimising KV reads matters as much as optimising weight reads.

MLA is already the big win here — 52.9 KB/token against what an equivalent GQA
would cost. That is not an accident of the chosen model, it is a reason for
choosing it.

**Prefix reuse is worth more than any kernel.** A turn that reuses 100k tokens of
cached KV avoids recomputing the entire prefill. No matmul optimisation comes
close to that gain — which makes the prompt-ordering rule in
[03-architecture.md](03-architecture.md) a first-order performance decision, not
hygiene.

## Reproduce

`roofline.py` computes the table from the config downloaded from the Hub. Rerun
it after any change of target model.
