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

Confirmed a second time by the native converter, which measures the probe with a
generator of its own: over the first 51 int4 tensors, output error **10.87% with
`--clip-grid 1` against 9.67% with the default 13** — −11.0%, against the −11.6%
the table above predicted on a different sample.

> **The container on disk does not have this yet.** It was written before the
> change, with a pure `absmax/qmax` scale, and its error is the one in the first
> table: **10.81%**. Its `meta` does not even have the `clip_grid` key. The
> **~9.6%** on the whole model is still an **extrapolation**, and it only holds
> after reconverting. The search does run 13 quantisation passes per block, but
> `tools/convert` now spreads them over the P-cores, partitioned by output row —
> so the conversion does not cost 13× the 11.3 min that the 1-point grid took.

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

### Ablation — and the precision table was wrong

`tools/ablate.exe` (C, reusing `src/`), the same 10 windows. Each group is
restored to the original BF16 **one at a time**, on top of the int4 baseline:

| restored group | tensors | nll delta | of the degradation |
|---|---|---|---|
| **attention** (`self_attn.*`) | 193 | **−0.0661** | **100.3%** |
| shared experts | 141 | −0.0132 | 20.0% |
| dense layer 0 | 3 | −0.0019 | 2.8% |
| `lm_head` | 1 | +0.0002 | −0.3% |
| routed experts | 9,024 | +0.0221 | −33.6% |

**All the degradation comes from attention.** Restoring 193 tensors — 2% of the
count, ~520 MB — recovers 100.3% of what quantisation cost: nll 5.8309 against
5.8311 for full BF16. The 9,024 expert tensors, which are 16.45 GB and 93% of the
container, are **not a source of loss**.

That inverts the hypothesis this section assumed. The reason is structural, not
accidental: each token activates 4 experts of 64, so one expert's error enters
diluted and uncorrelated; attention runs **in full, across all 47 layers, for
every token**, and its error compounds with depth.

**Consequence for the converter: attention in int8 g64, experts stay int4.**
Estimated cost: the ~520 MB of attention become ~1.04 GB, the container goes from
19.33 to ~19.85 GB — **+2.7% in size for ~100% of the quality back**.

Two honest caveats, because the numbers ask for them:

- The 95% interval on the total cost is `[+0.0366, +0.0952]` — wide, and the
  experts' `+0.0221` falls inside it. **The tool does not yet report the paired
  interval per group**, so "experts in int4 improve the model" is not
  demonstrated; what is demonstrated is that they **do not degrade** it.
- Per-window degradation ranges from `−0.000` to `+0.152`. The effect is real
  (9 of 10 windows positive) but heterogeneous, and a larger corpus would tighten
  the interval.

### The prediction was tested — and held

Container reconverted with `tools/convert.exe` (attention in int8, everything
else unchanged), same 10 windows, same tool:

| container | attention | nll delta vs BF16 | perplexity | size |
|---|---|---|---|---|
| v1 | int4 g64 | **+0.0659 ± 0.0149** | +6.81% | 19.33 GB |
| **v2** | **int8 g64** | **−0.0019 ± 0.0078** | **−0.19%** | 19.65 GB |

The v2 95% interval is `[−0.0171, +0.0134]` — it **crosses zero**. It is not that
quantisation got good: it is that **it can no longer be distinguished from BF16
with this measurement**. For 1.7% in size.

Final v2 counts: int4 8,973 (15.91 GB), int8 438 (2.44 GB), f32 290 (25.7 MB),
bf16 2 (1.27 GB). Resident 18.38 GB.

> **What I nearly changed after measuring.** v2 also moved
> `kv_a_proj_with_mqa` from f32 to int8 — 47 tensors the ablation never
> evaluated, because there they were already exact. I went as far as fixing that
> in the converter.
>
> But the container **that was measured** is the one with that tensor in int8,
> and it measured a loss indistinguishable from zero. Reverting would produce an
> *untested* configuration, 167 MB larger, to buy a guarantee the measurement
> already gives.
>
> The rule stands: **publish what was measured**, not what looks safer.

## Agentic use

`chat_template.jinja` with tool support (`tools`, `tool_call`, `tool_calls`),
202,752 tokens of context, SWE-bench Verified 59.2 and τ²-Bench 79.5 in the
repo's own table — which compares against Qwen3-30B-A3B-Thinking (22.0) and
GPT-OSS-20B (34.0) on the same test.

For a sense of scale: 19.3 GB of weights sustaining 59.2 on SWE-bench. It is the
quality-per-byte ratio that makes the project viable.
