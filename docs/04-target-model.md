# 04 — Target model: GLM-4.7-Flash

`zai-org/GLM-4.7-Flash` — 31.2B total, ~3.57B active per token, a 62.4 GB BF16
source, **19.3 GB in the int4 container** (measured; the 15.6 GB estimate that
circulated earlier ignored scales, int8 MTP and embeddings — see
[Quantisation format](#quantisation-format)).

## Why this one

We surveyed 89 MoE models published in the last 400 days and ranked them by real
*fit* on this machine. The candidates that survived:

| model | date | int4 | SWE-bench | context | fits in 64 GB |
|---|---|---|---|---|---|
| **GLM-4.7-Flash** | 2026-01-19 | **19.3 GB** | **59.2** | **202,752** | yes, comfortably |
| Ling-2.6-flash | 2026-04-28 | 53.7 GB† | 61.2 | 131,072 | yes, tight |
| DeepSeek-V4-Flash | 2026-04-22 | 79.8 GB† | — | 1,048,576 | **no** |
| Qwen3-30B-A3B-2507 | 2025-07-28 | 15.3 GB† | — | 262,144 | yes |

† estimated at 0.5 B/weight. Only the GLM was converted and measured; the others
would grow in the same proportion (~+12% for scales, plus whatever the precision
policy preserves) if they went through the same converter. The relative
comparison does not change — Ling is still ~2.8× the GLM.

Ling wins by 2 points on SWE-bench while being nearly **3× larger**. And for
agentic use the KV enters the budget alongside the weights:

```
GLM-4.7-Flash   19.3 (weights) + 11 (KV at 200k) = 30.3 GB   34 GB left
Ling-2.6-flash  53.7 (weights) + 4.8 (KV at 131k) = 58.5 GB   5 GB left
```

Five GB of headroom is the margin at which we watched that engine **refuse to
start** for fear of OOM. That is not margin, it is a trap.

Qwen3-30B-A3B is similar in size and is good, but it uses GQA without
`kv_lora_rank` — irrelevant for TootegaEngine, which will not have the streaming
engine's hard-coded restriction, but relevant for M1, where we want to compare
against an engine that already works.

An honest caveat: the benchmark numbers come from the vendor comparing itself
against competitors. Apply the usual discount. Ling-2.6-flash stays as plan B if
the real quality disappoints.

## Specification

```
num_hidden_layers        47      (1 dense + 46 MoE)
hidden_size              2048
num_attention_heads      20
n_routed_experts         64
num_experts_per_tok      4
n_shared_experts         1
moe_intermediate_size    1536
intermediate_size        10240   (dense layer)
first_k_dense_replace    1
vocab_size               154880
max_position_embeddings  202752
torch_dtype              bfloat16
```

**MLA attention:**

```
q_lora_rank              768
kv_lora_rank             512
qk_nope_head_dim         192
qk_rope_head_dim         64
v_head_dim               256
```

**Routing:**

```
n_group                  1
topk_group               1
norm_topk_prob           true
routed_scaling_factor    1.8
```

**Speculation:** `num_nextn_predict_layers = 1` — there is an MTP head. Testable,
and the resident regime is exactly where prior experience says speculation
finally pays (near full residency, not under streaming).

**No DSA:** there is no `index_topk`. Dense attention; GLM-5.2's lightning
indexer does not apply. Less code to write.

## What the engine has to implement

| piece | detail |
|---|---|
| MLA attention | with q-LoRA (`q_a_proj` → `q_b_proj`), KV compressed to 576 values/token/layer |
| Partial RoPE | 64 of the 256 dimensions per head are rotated; 192 are not |
| MoE router | top-4 of 64, `norm_topk_prob`, scale 1.8, **no** group limit |
| Shared expert | 1 per MoE layer, always active |
| Dense layer | layer 0 is a plain MLP with `intermediate_size=10240` |
| MTP | 1 head, optional in M1 |
| Tokenizer | `tokenizer.json` in HF format; mandatory |

## Quantisation format

Target: **int4 with f32 scales per group of 64** along the input dimension —
0.5625 effective bytes per weight. It is the format the ceiling calculation
assumes.

**Known blocker:** the source is BF16, and the streaming engine's converter
(`c/tools/convert_fp8_to_int4.py`) was written for GLM-5.2's FP8 checkpoint. The
model is compatible with that *engine*, but the quantisation path needs to accept
BF16. That is code to write, not a download — and it is the first M0 item on the
[roadmap](05-roadmap.md).

Precision per tensor, following the lesson learned (an int4 MTP head drops draft
acceptance to 0–4%). Counts **read from the index of the container already
written**, not from `--dry-run`:

| tensor | precision | tensors | output |
|---|---|---|---|
| experts, shared, dense, attention | int4 g64 | 9,161 | 16.452 GB |
| MTP layer (`model.layers.47.*`) + `lm_head` | int8 g64 | 203 | 1.358 GB |
| router (`gate`), `e_score_correction_bias`, norms | fp32 | 337 | 0.247 GB |
| `embed_tokens` (of the model **and** of the MTP) | bf16 | 2 | 1.269 GB |
| **total** | | **9,703** | **19.326 GB** |

File on disk: 19.327 GB. The 1.5 MB delta is the JSON index; the 64 B alignment
padding disappears in the rounding.

**It is 19.3 GB, not the 15.6 estimated earlier.** The difference: scales
(+0.0625 B/weight), int8 MTP and preserved embeddings. Of those, ~18.1 GB are
resident — both `embed_tokens` stay file-backed, read one row per token.

> An earlier table here said 212 tensors / 1.69 GB in int8 and a single
> `embed_tokens`, closing at 19.03 GB. It came from a `--dry-run` predating the
> `POLICY` ordering fix — the MTP norms were caught by the int8 rule before
> reaching the fp32 rule, and the MTP has its own `embed_tokens`.

### Measured cost of quantisation

From the converter, relative error of a **matmul's output** (the metric that
predicts quality; per-weight error in isolation does not). Whole model, pure
`absmax/qmax` scale (`--clip-grid 1`):

| | per weight | output | worst output | n |
|---|---|---|---|---|
| int4 g32 † | — | 10.44% | — | — |
| **int4 g64** | 11.47% | **10.81%** | 14.13% | 9,161 |
| int4 g128 † | — | 12.76% | — | — |
| int8 g64 | 0.65% | **0.62%** | 1.17% | 203 |

† g32 and g128 come from earlier partial conversions; only g64 ran on the whole
model.

**Nothing is broken — this is the round-to-nearest floor.** With step
`Δ = absmax/qmax` and uniform error inside the step, `σ_e = Δ/√12`, and since
`per_output = ‖E‖_F/‖W‖_F`, the ratio is exactly `σ_e/σ_w`. The sanity check is
the ratio between the classes: **10.81 / 0.62 = 17.4**, against a step ratio of
`127/7 = 18.1`. It matches. If that ever diverges much, it is a bug, not
quantisation.

Three conclusions that change the ablation plan:

**The error does not shrink with matrix size.** In `y = Wx`, noise and signal
both grow with `√cols`, so the ratio stays `σ_e/σ_w` — independent of how many
terms the sum has. There is no cancellation.

**Group size is a weak lever.** g64 → g32 buys 1.3 percentage points and costs
+11% of file size (0.5625 → 0.625 B/weight). A bad trade, and that is why
`--group` stopped being described as "the main knob".

**The optimal scale is not `absmax`.** `absmax/qmax` optimises the group's worst
weight, not the group. Shrinking the scale clips the extremes and tightens the
step for every other weight. `tools/convert` does a per-group MSE search over a
grid of fractions of `absmax` (`--clip-grid`, default 13 points between 0.70 and
1.00). Measured on 24 real tensors from shard 4, 512 rows each, with the same `x`
per tensor:

| `--clip-grid` | int4 g64 output | gain | int8 g64 output | gain |
|---|---|---|---|---|
| 1 (pure absmax) | 10.89% | — | 0.60% | — |
| 5 | 9.76% | −10.4% | 0.60% | 0% |
| **13 (default)** | **9.63%** | **−11.6%** | 0.60% | 0% |
| 25 | 9.58% | −12.1% | 0.60% | 0.8% |

For free: **the format does not change** (still one f32 scale per group) and the
kernel cannot tell the difference. In int8 the gain is zero — with 255 levels
`absmax` is already essentially optimal, and the search stays there only to avoid
opening a special case.

### Measured on the whole model — no longer an extrapolation

The table above came from 24 tensors of one shard. The full conversion has now
run, and the probe covers **every** quantised tensor:

| container | scale | int4 g64 output | int8 g64 output | n (int4) |
|---|---|---|---|---|
| v3 | pure `absmax/qmax` | **10.81%** | 0.62% | 9,161 |
| **v4** | **13-point clip grid** | **9.52%** | **0.62%** | 9,165 |

**−11.9% on the whole model**, against the −11.6% predicted from 24 tensors. The
extrapolation held. int8 did not move by a thousandth, exactly as the table
above says it would not — with 255 levels `absmax` is already optimal.

Worst tensor: **12.25%** (v4) against 14.13% (v3). Per-weight error 9.92%.

**The sanity ratio changes with the grid, and that is expected.** With pure
`absmax` the int4/int8 ratio was 10.81/0.62 = 17.4, next to the step ratio of
`127/7 = 18.1`. With the grid it is 9.52/0.62 = **15.4**, because only the int4
side improved. A ratio far below that is a bug; 15–18 is the healthy band.

**Cost: 171 s** for 9,703 tensors, 8 threads — against the **11.3 min** the
single-threaded 1-point conversion took. The search does 13 quantisation passes
per block, but they run on the pool, partitioned by output row, so the container
does not depend on `--threads` and the conversion got **4× faster** while doing
13× the arithmetic.

Side effect: with a scale smaller than `absmax/7`, code `-8` becomes reachable.
Before, `rint` never produced it and int4 used **15 of the 16 levels** that the
dequantiser already knew how to read. Verified by unpacking a 1-shard test
container and matching the round-trip against the source: 16/16 levels occupied,
against 15/16 before.

### What the table above does *not* measure

The probe uses `x ~ N(0,1)` iid. Real LLM activations are not that: some channels
carry 10–100× the median magnitude, and the real output error is dominated by the
columns of `W` aligned with those channels. **The numbers above are optimistic.**
That is AWQ's entire premise.

Closing that hole is still worth more than any scale tuning:

| path | expected gain | cost |
|---|---|---|
| **Hadamard rotation** before quantising | attacks the outlier at the root | the kernel must apply the rotation to the activations |
| **GPTQ** (Hessian-based compensation) | 2–4× on output error | calibration data + hours of conversion; format and kernel **identical** |
| int8 on the sensitive tensor | ~18× | 2× the bytes of that tensor |

Whether ~9.6% is acceptable is not something this section answers — only
perplexity or the token-exact oracle answer that, and that is what M0's exit
criterion demands.

### Measured perplexity — M0's exit criterion

`Tests/ablate`, fixed set (the `docs/*.md` files themselves), **10 windows
of 384 tokens, 3,830 predictions**, **paired** comparison window by window. The
int4 side is measured with `tootega.exe` — the artefact that actually runs — and
BF16 is read from the original safetensors through an independent path.

```
nll delta (int4 - BF16):     +0.0513 +- 0.0076
95% interval:                [+0.0365, +0.0661]
perplexity:                  1.0526x  (+5.26%)
per window: +0.028 +0.027 +0.066 +0.078 +0.017 +0.047 +0.051 +0.093 +0.045 +0.061
```

**int4 g64 costs +5.3% perplexity.** All ten windows came out positive and the
interval does not cross zero: it is a measurement, not an impression.

And it is about **half** what the per-matmul output error would suggest — which
confirms the caveat above that per-tensor error does not predict quality.

> **How I nearly published the wrong number.** The first two attempts were
> unpaired: 191 predictions gave **−4.2%** — int4 "better" than BF16, which is
> impossible — and 381 gave **+2.5%**. The sign flipped when the sample doubled.
>
> The cause: perplexity varies from **223 to 473 across windows** of this corpus,
> and that variance swallows a 5% effect. The paired comparison cancels it,
> because both configurations see exactly the same text and what is measured is
> the per-window difference — whose deviation is twenty times smaller.
>
> Comparing two global means would have produced a publishable, wrong number.

### Ablation — the current table

`Tests/ablate` (C, reusing `src/`), the same 10 windows. Each group is restored
to the original BF16 **one at a time**, on top of the int4 baseline, and the
delta is **paired window by window** — which is the whole point:

| group restored to BF16 | nll delta | verdict |
|---|---|---|
| **experts** | **−0.0225 ± 0.0076** | **settled** |
| attention | −0.0007 ± 0.0019 | noise |
| shared experts | +0.0216 ± 0.0325 | noise |
| `lm_head` | −0.0008 ± 0.0007 | noise |
| dense layer 0 | +0.0168 ± 0.0097 | noise |

**Only the experts have an effect above the noise**, and that is the structural
answer: they are 9,024 tensors and 93% of the container. **Attention in int4
costs nothing measurable** — `--attn int8` remains available and buys 0.49 GB of
file for a delta whose interval sits astride zero.

That is why the converter's default is `--attn int4` and why container v4 has it.

> ### Retracted: the first ablation, and why it was wrong
>
> A previous version of this section concluded the exact opposite — *"all the
> degradation comes from attention"*, attention `−0.0661` recovering 100.3% —
> and the converter carried `--attn int8` as its default because of it. Both
> the table and the conclusion are **withdrawn**. Two defects, and each is a
> lesson worth more than the number:
>
> - **The forward underneath was wrong.** The measurement ran with RoPE in the
>   `rotate_half` variant instead of `interleave` ([17](17-diagnosis.md)). An
>   ablation on top of a broken forward measures the break, distributed over
>   whatever group happens to touch it most — and attention is what touches
>   position.
> - **It compared global means, not paired windows.** The same defect that had
>   already produced a perplexity measurement **with the sign inverted**, three
>   sections above. Knowing about a trap does not protect you from it; only the
>   tool refusing the unpaired path does.
>
> The retracted numbers are kept here on purpose. A document that quietly
> replaces a wrong table teaches nothing, and the next person to run an ablation
> on this project needs to know that this one has already been wrong once, in a
> way that looked entirely convincing.

### The container as it is written today

Container **v4**, `--attn int4 --clip-grid 13`. It is published, so this table
describes a file anyone can fetch:
[`HermesX21/GLM-4.7-Flash-tootega-int4`](https://huggingface.co/HermesX21/GLM-4.7-Flash-tootega-int4).

| precision | tensors | bytes |
|---|---|---|
| int4 g64 | 9,165 | 16.46 GB |
| int8 g64 | 198 | 1.33 GB |
| f32 | 338 | 252.2 MB |
| bf16 | 2 | 1.27 GB |
| **total** | **9,703** | **19.32 GB** |

Resident 18.05 GB — `embed_tokens` stays file-backed, read one row per token.

> **What was nearly changed after measuring, and was not.** An earlier container
> moved `kv_a_proj_with_mqa` from f32 to int8 — 47 tensors the ablation never
> evaluated, because there they were already exact. The rule that settled it:
> **publish what was measured**, not what looks safer. It is f32 today, which is
> both what the converter writes and what the numbers on this page describe.

## Agentic use

`chat_template.jinja` with tool support (`tools`, `tool_call`, `tool_calls`),
202,752 tokens of context, SWE-bench Verified 59.2 and τ²-Bench 79.5 in the
repo's own table — which compares against Qwen3-30B-A3B-Thinking (22.0) and
GPT-OSS-20B (34.0) on the same test.

For a sense of scale: 19.3 GB of weights sustaining 59.2 on SWE-bench. It is the
quality-per-byte ratio that makes the project viable.
