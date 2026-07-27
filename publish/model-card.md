---
license: mit
base_model: zai-org/GLM-4.7-Flash
base_model_relation: quantized
language:
  - en
  - zh
pipeline_tag: text-generation
# Sem widget: nenhuma biblioteca do Hub le este formato, entao a caixa de
# Inference API so renderiza um erro para quem chegar aqui.
inference: false
tags:
  - tootega
  - int4
  - moe
  - cpu-inference
  - quantized
---

# GLM-4.7-Flash — int4 g64, `.tootega` container

GLM-4.7-Flash quantised to **int4 with f32 scales per group of 64**, in the
container format the [TootegaEngine](https://github.com/HermesSilva/TootegaInference)
reads. **19.3 GB**, resident in RAM, decoding on **CPU only**.

> **This file does not load in `transformers`, vLLM, llama.cpp or anything else.**
> `.tootega` is not GGUF, not safetensors and not a variant of either — it is the
> format of one engine, specified in
> [`docs/14`](https://github.com/HermesSilva/TootegaInference/blob/master/docs/14-container-format.md).
> If you were looking for a quantisation that runs on your usual stack, this is
> not it. What is here is the artefact that engine consumes, published so that
> nobody has to download 62.4 GB and convert it again.

## Files

| file | size | what it is |
|---|---|---|
| `model.tootega` | 19.3 GB | the container: weights, scales, index and the full `config.json` inside |
| `tokenizer.json` | 20 MB | HuggingFace tokenizer, copied unchanged from the base model |
| `chat_template.jinja` | 3 KB | reference only — the engine reproduces this template in C, without Jinja |
| `config.json` | 1 KB | the architecture, also embedded in the container |

The engine reads the architecture **from the container**, never from a file
alongside it. `config.json` is here so the metadata is visible on this page.

## How to run

```powershell
git clone https://github.com/HermesSilva/TootegaInference
cd TootegaInference
.\build.ps1                       # MSVC, found via vswhere; Linux uses the Makefile
.\tootega.exe serve model.tootega --tokenizer tokenizer.json
```

The web chat comes up at `http://127.0.0.1:8080`. Also available:

```powershell
.\tootega.exe chat     model.tootega --tokenizer tokenizer.json
.\tootega.exe generate model.tootega --tokenizer tokenizer.json --prompt "..." -n 64
.\tootega.exe inspect  model.tootega        # header, index and budget
.\tootega.exe plan     model.tootega --ram-gb 32
```

`inspect` opens and validates the file: magic, version, the 64-byte alignment of
every blob and the relationship between `shape`, `group` and each blob's size. If
it opens, the container is intact.

## Measured, on the machine below

Reference machine: **i9-14900K** (8 P-cores), **64 GB DDR5-4800 dual channel**,
RAID0 NVMe. **No GPU is used** — a CUDA backend was built and measured on a
separate branch, and rejected: the granularity was wrong, and no GPU code lands
on master until it shows a gain in the complete decode.

```
memory bandwidth     69.4 GB/s     measured at runtime, not from the spec sheet
traffic per token    2.38 GB       tensors a decode actually touches
roofline ceiling     29.1 tok/s    bandwidth / traffic

warm decode          ~50 ms/token  = ~19.8 tok/s
mean decode          17.12 tok/s   includes the cold first prompt (59% of ceiling)
cold decode          84 ms/token   = 1.7x worse than warm
mean ttft            2.51 s
prefill              10.4 tok/s
RSS                  16.69 GB peak
KV                   108.3 KB per position; 221.8 MB for 2048
```

Matmul in isolation reaches 62 GB/s — 90% of bandwidth. The gap to the ceiling is
in attention and thread barriers, not in the matmuls.

The 2.38 GB of traffic per token is what makes the number: only 4 of 64 experts
run per layer, and the whole of attention is int4 here.

### Quality

Generation against the original BF16 weights — twelve prompts, up to 96 tokens,
greedy, this container and the previous one measured **in the same run** against
a BF16 reference generated in that same run:

| container | `match@ref` | `rep4` | tok/s |
|---|---|---|---|
| BF16 (reference) | 96.0 | 7.3% | 3.34 |
| previous (plain scale, int8 attention) | 32.4 | 5.3% | 8.43 |
| **this one** | **42.7** | 6.8% | **10.98** |

`match@ref` counts the initial tokens identical to the BF16 generation before
diverging; `rep4` is the fraction of repeated 4-grams, where above ~15% means
degeneration. **The scale search bought a third more matching tokens**, and this
comparison is paired by construction — same prompts, same reference, one run.

| property | result | measured on |
|---|---|---|
| teacher-forced oracle, independent implementation | **32/32** identical tokens, 1e-4% relative error | **this container** |
| generation `match@ref` against BF16 | **42.7** against the previous container's 32.4 | **this container** |
| perplexity against BF16, paired, 10 windows of 384 tokens | **+2.86%**, interval `[-2.8%, +8.8%]` — **inconclusive** | **this container** |
| determinism | **byte-identical** logits from 1 to 20 threads | **this container** |
| perplexity, 384 positions of prose, BF16 weights | engine **8.63** against `transformers`' **8.89** | earlier container |
| tokenizer against HuggingFace `tokenizers` | **36/36** cases, exact round-trip | independent of container |

Determinism is not a nice property here, it is the one that makes every other
number reproducible: the reduction order is fixed and independent of how the work
is split, so `--threads` changes the time and nothing else. Verified by dumping
19.8 MB of f32 logits at 1 thread and at 20 and comparing the files.

**The perplexity row says inconclusive and means it.** The interval crosses zero
because one window of the corpus has a perplexity near 4,600 — a table-heavy
stretch where nothing is predictable — and it alone is four times any other
window's difference. Dropping it would give `+5.4%` with an interval clear of
zero, and that number is deliberately **not** the one published: choosing which
window to discard after seeing the result is how a measurement becomes an
opinion. The corpus is the thing to fix, not the arithmetic.

So: generation says this container is closer to the original weights than the
previous one, and perplexity on this corpus cannot tell them apart. Both are
stated because both were measured.

The paired comparison is the one that counts: comparing two configurations by the
global mean swallows the effect in the variance between windows, and that mistake
once produced a perplexity measurement **with the sign inverted**.

## Precision policy, per tensor

Not everything is int4, and the split comes from a paired ablation, not from
taste:

| group | precision | why |
|---|---|---|
| experts, attention | int4 g64 | 9,165 tensors, 16.5 GB — this is where the model's size is |
| `lm_head`, MTP layer | int8 g64 | 198 tensors, 1.3 GB; int4 on the MTP head drops draft acceptance to 0–4% |
| norms, router, `e_score_correction_bias` | f32 | the router decides *which* experts run: an error swaps the expert, not the last bit |
| `kv_a_proj_with_mqa` | f32 | feeds the compressed KV of every position |
| `embed_tokens` | bf16 | read one row per token; never resident |

The ablation found only one group with an effect above the noise: **the experts**
(−0.0225 ± 0.0076 nll when restored to BF16). Attention gives −0.0007 ± 0.0019 —
noise. An earlier version of this table said the opposite; that measurement had
the RoPE wrong and compared global means instead of paired windows.

## Quantisation detail

Symmetric, per group of 64 along the input dimension, no zero-point, one f32
scale per group. Nibbles packed **half/half** inside the group: `byte[j]` carries
weight `j` in the low nibble and `j + 32` in the high one, which unpacks with
AVX2 into two contiguous vectors with no activation shuffle.

The scale is **not** `absmax/qmax`. That choice optimises the group's worst
weight instead of the group; shrinking it clips the extremes and tightens the
step for all the others. The converter searches 13 fractions of `absmax` between
0.70 and 1.00 and keeps the one with the lowest squared error.

Measured over **every** quantised tensor of this container, relative error of a
matmul's output — the metric that predicts quality; per-weight error in isolation
does not:

| scale | int4 g64 | worst int4 | int8 g64 | n (int4) |
|---|---|---|---|---|
| pure `absmax/qmax` | 10.81% | 14.13% | 0.62% | 9,161 |
| **13-point clip grid** | **9.52%** | **12.25%** | 0.62% | 9,165 |

**−11.9%**, with no change to the format — still one f32 scale per group — and no
change to the kernel, which cannot tell the difference. int8 does not move: with
255 levels `absmax` is already optimal, and the search stays there only to avoid
opening a special case.

Side effect that pays for itself: with a scale below `absmax/7` the code `-8`
becomes reachable. With the exact scale, round-to-nearest never emits it and int4
uses **15 of the 16 levels** the dequantiser already knows how to read.

Sanity check on the numbers: 9.52/0.62 = 15.4, against a step ratio of
`127/7 = 18.1`. It sits below 18 precisely because only int4 gains from the
search. A ratio far outside 15–18 would be a bug, not quantisation.

## Provenance and licence

Derived from [`zai-org/GLM-4.7-Flash`](https://huggingface.co/zai-org/GLM-4.7-Flash),
**MIT**. The weights are theirs; the quantisation, the container format and the
engine are [TootegaEngine](https://github.com/HermesSilva/TootegaInference), also MIT.
`tokenizer.json` and `chat_template.jinja` are copied unchanged from the base
model.

Reproducing this file from the original checkpoint:

```powershell
python download_model.py                          # 62.4 GB in BF16, resumable
.\tools\convert.exe models\glm47flash-bf16 --out model.tootega --clip-grid 13
.\tools\convert.exe --verify model.tootega
```

The conversion takes **171 s** on 8 threads and streams one tensor at a time —
the 62.4 GB source and the 19.3 GB output are never both in memory. `--dry-run`
prints the plan and the estimated size without writing anything.

## What is not here

- **Batching and slots.** One request at a time; a second client waits in the TCP
  backlog, silently.
- **MTP speculation** and **persistent state**: the tensors are in the container,
  the paths are not implemented.
- **GPU.** Master is pure CPU by decision, not by omission.
- **A guarantee that int4 is good enough for your case.** +5.26% perplexity is a
  measurement, not a verdict. Measure on your own task.
