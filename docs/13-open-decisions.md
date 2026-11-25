# 13 — Open decisions

What is still undecided, in order of urgency. Each item carries a recommendation
and the criterion that would settle it.

## Status

| # | decision | state |
|---|---|---|
| D1 | nibble packing | **SETTLED** — half/half, see [14](14-container-format.md) |
| D2 | W4A8 or W4A16 | **SETTLED** — W4A32, see [15](15-kernels-and-numerics.md) |
| D3 | container format | **SETTLED** — [14](14-container-format.md) |
| D4 | accumulation precision | **SETTLED** — [15](15-kernels-and-numerics.md) |
| D5 | tokenizer | **SETTLED** — [16](16-tokenizer.md) |
| D6 | threading model | **SETTLED** — our own pool, [15](15-kernels-and-numerics.md) |
| D7 | engine ↔ service protocol | open, M3 |
| D8 | paged KV, details | open, M3 |
| D9 | sampling | **SETTLED** — greedy, temperature, top-k, top-p; `src/sample.c` |
| D10 | licence and repo structure | **SETTLED** — MIT, see `LICENSE` |
| **D11** | **absorbing the compressed KV into the query** | **SETTLED** — absorbed is the default, 32/32 held |
| **D12** | **online softmax (single pass)** | **SETTLED** — it exists, but only above L3 |
| **D13** | **KV in bf16** | **REJECTED** — 29/32 on the oracle; implemented, not the default |
| **D14** | **chat history: does reasoning stay or go** | open — it stays, and the reason is the cache |
| **D15** | **retrieval in `src/`** | **SETTLED** — built and verified; **off by default**, see D17 |
| **D16** | **batched prefill** | open — the kernel exists and nothing calls it; 3–4× measured |
| **D17** | **the content belongs in the weights** | open — the prompt goes unmodified; the fine-tune has to carry the doctrine |
| **D18** | **`lm_head` with a shortlist** | **MEASURED** — exact version buys ~4% of the traffic and is not worth a format; the statistical one stays an option |
| **D19** | **experts as a base plus a delta** | **REJECTED** — the experts of a layer are unrelated, measured |
| **D20** | **prefix KV on disk** | **SETTLED** — on by default in `serve`, identical reply; [12](12-persistent-state.md) |

The text below preserves the reasoning behind each decision.

---

## SETTLED — found by measuring, not by planning

### D11. MLA absorption, pulled forward from M5 to M1

M1 implements the **unabsorbed** formulation because it is the oracle's
([15](15-kernels-and-numerics.md)). It requires caching **expanded** K and V:
20 heads × (256 + 256) values per token per layer.

```
expanded KV     47 × 4,096 × 10,240 × 4 B = 7.9 GB   <- read IN FULL every token
compressed KV   47 × 4,096 ×    576 × 4 B = 443 MB
```

The first number is the problem, and it is not about storage — it is about
**traffic**. On every decode token, attention reads all the KV of all 47 layers.
At 4k of context that is **7.9 GB per token against 2.17 GB of weights**: KV
becomes 78% of the traffic, and the ceiling in
[08](08-performance-ceiling.md) — computed over weights alone — stops holding.
At 60 GB/s, 7.9 GB per token caps you at **7.5 tok/s before any compute**,
against the ≥10 criterion.

Measured on this machine, 8 P-cores, AVX2, averaged over 128 positions around
each point:

| context | expanded | absorbed | gain |
|---|---|---|---|
| 128 | 13.63 tok/s | 13.63 tok/s | — |
| 1,024 | 6.15 | **12.48** | 2.0× |
| 2,048 | 3.64 | **9.98** | 2.7× |
| 3,072 | — | **8.31** | |
| 4,096 | — | **7.48** | |
| **average over 4,096** | — | **10.09 tok/s** | |

The slope fell from 6.5 µs to 1.5 µs per context position — 4.3× flatter. What is
left is no longer KV traffic: it is the 8.4 GFLOP/token that attention costs at
4k, at ~137 effective GFLOP/s. One layer's compressed cache (9.4 MB at 4k) fits
in the 36 MB L3 and is reused by all 20 heads, which is why traffic did not
explode with per-head reading.

**Against M1's criterion** (`≥ 10 tok/s at 4k`): the average over 4,096 positions
gives **10.09**; the instantaneous rate *with* 4k of context already gives
**7.48**. On the strict reading — decoding with the context full — the criterion
is not met yet. The paths, in order of return: a single pass with online softmax
(cuts half the KV reads), KV in bf16 (halves L3 pressure) and M2.

**Recommendation: pull absorption forward into M1.** Folding `kv_b` into the
query projection makes attention operate in the compressed space: traffic drops
18×, and the extra cost is ~4.5 GFLOP/token of compute — which is available,
because compute utilisation sits at ~20%
([15](15-kernels-and-numerics.md), D2).

**The caveat from [15](15-kernels-and-numerics.md) still holds**: the two
formulations are algebraically equivalent and **do not produce the same bits**.
So absorption landed like this, and not silently:

1. both implementations coexist, selectable (`--attn absorbed|expanded`);
2. the unabsorbed one remains the oracle's reference;
3. the absorbed one would only be accepted with **32/32 identical tokens**
   against the same `ref.json`.

**Result: 32/32, including position 31, whose margin is 0.0007.** Absorption is
authorised and is the default.

### The identity

```
q_nope . k_nope[t]   =  q_nope^T (W_k c[t])   =  (W_k^T q_nope)^T c[t]
sum_t w[t] v[t]      =  sum_t w[t] (W_v c[t]) =  W_v (sum_t w[t] c[t])
```

You project the **query** once — a fixed cost, independent of context — instead
of expanding **every position** of the cache.

### The trap the first version fell into

`W_k^T` suggests materialising `kv_b` in fp32 so you can read columns. Done that
way, absorption made things **worse**: 5.6 tok/s against 12.3 for the expanded
version. The reason is that dense `kv_b` is 18.4 MB per layer, **863 MB read per
token** — against 108 MB for the packed int4. Absorption bought 18× of KV traffic
and gave back 8× of weight traffic.

The fix is noticing that the transpose **does not require columns**: since `W_k`
is `[nope][rank]` row-major, `W_k^T q` is an accumulation of rows
(`for r: acc += q[r] * W_k[r][:]`), and rows are exactly what the container
delivers cheaply. No copy, no extra traffic.

> The lesson is not about MLA: it is that "optimising" by trading work for memory
> needs the traffic arithmetic done **first**, in an engine whose declared
> bottleneck is memory bandwidth ([08](08-performance-ceiling.md)).

---

### D12. Online softmax — correct, and slower where it matters today

Attention reads the cache twice: once for the scores, once to accumulate the
output. Online softmax keeps a running max and sum and does it all in a single
pass:

```
m' = max(m, s)      acc' = acc·exp(m − m') + exp(s − m')·c[t]
```

Algebraically it is the same max-subtracted softmax — the common factor comes out
of the sum — and it holds **32/32**. Measured at 4k:

| | tok/s (average over 4,096) |
|---|---|
| two passes | **10.09** |
| single pass | 9.19 |

**Nine per cent slower.** The reason: at 4k, one layer's cache is 9.4 MB and fits
in the 36 MB L3 — which is still shared by the 20 heads — so the second pass
costs almost nothing, while the single pass pays a branch per position and
rescales 512 floats on every new maximum.

Above L3 the arithmetic flips, and then the second pass comes from DRAM. So both
exist and the choice is **by size**: single pass only when the layer's cache
exceeds 12 MB (`TT_L3_BUDGET`, conservative against the real 36 MB because L3 is
also serving the weights streaming through).

> Worth keeping as a warning: "fewer reads" is not the same as "faster" when the
> reads you cut were already coming from cache.

---

### D13. KV in bf16 — REJECTED by its own criterion

It would halve the cache (108.3 KB → 54.2 KB per token) and relieve L3 pressure,
which is exactly the resource D12 showed to be decisive.

**It failed: 29/32.** Three positions flipped their token — 15, 18 and 31 — all
with margins between 0.012 and 0.066. bf16 has 7 mantissa bits against 23; the
error enters each `c[t]` and comes out amplified by the sum over 4,096 positions.

It is implemented and available (`--kv bf16`), with the default at `f32`. It is
not "close enough": by the criterion that authorised absorption, it is an
**output change**, and adopting it would require deciding that losing 3 tokens in
32 is worth the memory — a decision you do not take from inside an optimisation.

### int8 g64 in the cache — the "better" path also fails

The obvious way out was per-group quantisation with a scale: it errs 0.59%
against bf16's ~1.5% ([04](04-target-model.md)) and takes **less** space than it
— 576 int8 plus 9 scales give 612 B per token per layer, against bf16's 1,152 B
and f32's 2,304 B. Implemented (`--kv int8`), the cache dropped from 108.3 KB to
**28.8 KB per token**, 3.8×.

It failed **identically: 29/32.** And the time gain was 2% at 1,024 positions
(15.62 against 15.32 tok/s) — dequantisation in the loop eats the traffic saving
while the cache still fits in L3.

| | per token | 32/32 | tok/s at 1,024 |
|---|---|---|---|
| f32 | 108.3 KB | **32/32** | 15.32 |
| bf16 | 54.2 KB | 29/32 | ~15.4 |
| int8 g64 | **28.8 KB** | 29/32 | 15.62 |

### What that teaches about the criterion itself

Two quantisers of very different quality — 1.5% and 0.59% — fail with the same
score and similar logit deltas (0.02 to 0.06 on logits of ~90, five hundredths of
a per cent). The reason is not their quality: it is that **this prompt's margins
are small**, 0.0007 at position 31.

In other words: 32/32 detects an *output change*, which is what it promises, but
it **does not distinguish "different" from "worse"**. To decide whether int8 KV
actually degrades, we are missing the measurement M0 planned and never ran:
**perplexity on a fixed set**. Until that exists, the correct default is the one
that passes the test that does exist — `f32` — and the other two stay available
and documented.

Where they will matter: agent context. At 32k, the f32 cache is 3.5 GB and leaves
L3; in int8 it is 0.9 GB. Then the arithmetic flips and the decision comes back —
with perplexity in hand.

---

## SETTLED — would have frozen at 19 GB if the conversion had run

### D1. Nibble packing order

The converter already wrote int4 as **even/odd**: `byte[i]` holds weight
`2i` in the low nibble and `2i+1` in the high one.

When unpacking 32 bytes with AVX2, `v & 0x0F` delivers weights `0, 2, 4, …, 62`
and `v >> 4` delivers `1, 3, 5, …, 63` — an **interleaved order**. To multiply
against the activations you have to interleave those too.

The industry convention (llama.cpp `Q4_0` and derivatives) is **half/half**: for
a group of 64, `byte[i] = w[i] | (w[i+32] << 4)`. Unpacking delivers `w[0..31]`
and `w[32..63]` already contiguous, with no permutation at all.

**Recommendation: switch to half/half before converting.**

The cost of not switching is not large — permuting the activation is `O(cols)`
once per matmul against the matmul's `O(rows × cols)`, i.e. amortised. But the
cost of switching **now** is zero, and later it is reconverting 19 GB. There is no
reason to pick the less common convention.

**Resolution:** edit `quant_grouped`/`dequant_grouped` and run `--limit 1`.

### D2. W4A8 or W4A16 — what the kernel consumes

Two families, and the choice decides precision, kernel, and whether the container
needs extra metadata.

| | W4A8 (quantises the activation) | W4A16/A32 (dequantises the weight) |
|---|---|---|
| instruction | `vpdpbusd` (AVX-VNNI), u8×s8→s32 | `vfmadd` fp32 |
| throughput | ~4× more MACs/cycle | reference |
| extra error | yes, dynamic activation quantisation | no |
| complexity | per-token scale, requantise on every layer | unpack and multiply |

The reflex is to pick W4A8 for throughput. **Here the reflex is wrong**, and the
reason is in [08](08-performance-ceiling.md): decode is **memory-bound**. It is
~33 ms of memory reads against 4–7 ms of compute per token — **compute
utilisation around 20%**. MAC throughput is not the bottleneck; there is plenty
of slack.

**Recommendation: W4A32 in M1** — unpack int4 to fp32 and use FMA. Simpler, no
additional activation error, and the token-exact oracle becomes easier to hit.
Migrate to W4A8 only if measurement shows compute to be the limit, and then with
a number.

**Resolution:** it is a kernel decision, not a container one — the on-disk format
serves both. It does not block conversion, but it does block M1.

---

## OPEN — where the doctrine has to live

### D17. The prompt goes unmodified; the weights have to carry the work

D15 built retrieval and it works: the passages go in, the answers come out right,
and the numbers are in [22](22-finetune-and-retrieval.md). **It is off by
default anyway**, and the reason is not performance.

Editing a user's question before the model sees it adulterates the prompt. What
arrives at the model is no longer what the person wrote, the reply is partly a
function of a ranking they never saw, and a chat that silently rewrites its own
input is a different product from an engine that answers. That is the call, and
it is a product call rather than a measurement — so the measurement does not
override it.

**What it costs, stated once and not softened.** The v5 container was trained
with RAFT, which trades memorisation for reading on purpose. Served with an
unmodified prompt it scores **0 of 11** on the hand-reviewed closed-book set,
against 26 of 29 with a passage. It says *Arena* names a football stadium.

So the container has to change. Retrieval stays in the engine — built, tested,
reachable with `--index` — and stops being the answer.

**What failed, and what it does not rule out.** Continued pre-training on the raw
text failed: holdout perplexity fell 45%, t = −40.8, 61 of 61 windows improved,
and the model asserted a creator the first essay spends sections denying. That is
`train_lora.py`, kept for the record.

It is tempting to read that as "memorisation does not work at this corpus size".
It does not say that. CPT fits the *distribution of the text* — it teaches how
the work sounds — and the run measured exactly that: register acquired, content
not. **Closed-book question-answer training is a different objective**, mapping a
question onto a fact, and it was never tried here.

**And the dataset is already shaped for it.** `finetune/data/sft_final.jsonl`
holds 868 pairs with `question`, `answer` and `context` as *separate fields* —
none of them carries a passage baked into a prompt string. Training closed-book
is building the prompt from `question` alone. The answers were written with the
passage present, so they are grounded; what changes is only whether the model
sees the passage while learning to produce them.

**The open risk, which is real.** 868 pairs over 726 sections is 1.2 questions
per section. A RAFT model generalises to unasked questions because it reads; a
closed-book model generalises only as far as its training questions cover the
space. Thin coverage will show up as confident answers on the questions it saw
and drift on the ones it did not — which is the failure mode that looks most like
success. If the first run is thin, the fix is more questions per section from the
pipeline that already generates them, not a bigger adapter.

**What would settle it:** the same closed-book set that scored 0 of 11, scored
again on the retrained container, with the prompt unmodified.

## OPEN — the fast kernel nothing calls

### D16. Batched prefill

`tt_forward` takes **one token per call**, and the prefill loop in `server.c` and
in `cmd_chat` runs it over the prompt. Prefill therefore *is* the decode loop: on
this machine it measures **13.9 tok/s against decode's 11.9**, which is the same
number, because both pay a full read of the active weights for every single
token.

That was invisible while prompts were short. Retrieval made them long — 1,300
tokens for three passages, of which ~1,150 are the retrieved text — and the cost
became the dominant term:

| prompt | prefill |
|---|---|
| bare question, no index | < 1 s |
| question + 1 passage | 29–36 s |
| question + 3 passages | 93–97 s |
| the shape training used (4 × 2000 chars) | ~215 s |

**The batched matmul already exists**, in `kernels.c`, checked bit-for-bit
against the loop of singles by `Tests/mmbatch.exe` — which is the condition for
using it, since a batched prefill that changes bits changes tokens and takes the
32/32 criterion of [15](15-kernels-and-numerics.md) with it. Measured on the v5
container:

| batch | speed-up | GFLOP/s |
|---|---|---|
| 1 | 1.00× | 138 |
| 4 | 3.2–3.7× | 438–455 |
| 32 | 2.9–4.1× | 392–414 |

Nothing on the prefill path calls it.

**Recommendation: a `tt_forward_batch` taking a block of tokens**, with attention
over the block and the MoE grouping tokens by expert. The ceiling is not the
matmul — at ~410 GFLOP/s and ~9.5 GFLOP per token, 1,300 tokens of prompt is
~30 s of compute, so ~3× is what there is to win with this kernel. Going below
that is a different problem, and it is worth noting that **prefill is the regime
the GPU branch was rejected for not helping**: docs/18 measured `run_layer` at
decode granularity, where the transfer never amortises. A compute-bound batch of
a thousand tokens is the opposite case and was never measured.

**What would settle it:** 32/32 against `ref32.json` with the batched path on,
and the prefill of a three-passage prompt under 35 s.

**What has landed since, and it is not the batch.** "Both pay a full read of the
active weights for every single token" is no longer exact. A prompt position
with more prompt ahead of it predicts an id the caller is already holding, and
it was running the vocabulary projection to do so — `lm_head`, 337 MB of the
2.38 GB a token reads. `tt_prefill` now runs the projection on the last position
only (`tt_forward_prefill` for the others). The logits dump is unchanged, same
SHA-256, and the generated text is byte-identical on a 22-token and a 641-token
prompt.

Measured, interleaved, 641-token prompt, same sitting:

| pair | before | after | |
|---|---|---|---|
| 1 | 12.6 tok/s | 13.1 | +4% |
| 2 | 14.7 | 16.7 | +14% |
| 3 | 13.3 | 14.6 | +10% |
| mean | 13.5 | 14.8 | **+9%** |

The traffic arithmetic promised up to 14% and three pairs do not say 14%: the
machine was noisy that day — decode, whose code did not change, moved between
12.5 and 16.2 tok/s across the same runs. Direction established, magnitude
between 4% and 14%.

`--verbose` still runs the projection on every prompt position, on purpose: the
prompt perplexity it prints needs those logits. And `tt_prefill` is now the one
entry point every prefill goes through — which is where the batch will land
without any caller changing.

## MEASURED — two ways of reading fewer bytes, tried on the container first

Both would change the container, so neither got engine code before
`Tests/analyze` said whether the bytes were there. It is the rule of
[04](04-target-model.md) applied before the fact instead of after.

### D18. `lm_head` with a shortlist

The projection reads 154,880 rows to keep a handful. Split every int8 code into
its top `b` bits and the rest; a coarse pass over the top bits, plus a bound on
what the rest can add, rules rows out, and only the survivors are rescored
exactly — same kernel, same token.

The bound that needs no assumption about the data is
`c · Σ_g |scale_g| · Σ_{j∈g} |x_j|`. Rows left to rescore, of 154,880, over 24
real hidden states:

| top bits | rigorous, mean | rigorous, worst | 6-sigma band (not exact) |
|---|---|---|---|
| 4 | 154,838 | 154,880 | 114 |
| 5 | 29,220 | 154,878 | 15 |
| 6 | 1,126 | 23,623 | 3 |

**The exact version does not pay.** Its best point is 5 bits: 241 MB against
337, about 4% of the decode traffic, for a new container layout and a second
kernel. The reason is in the last column of the tool's output: the real error is
**2% of the rigorous bound**. A guarantee has to allow every low bit to line up
with the activation, across 2,048 dimensions; in the data they cancel.

The statistical band is where the bytes are — 114 rows at 4 bits, roughly 6–7%
of the traffic — and it is **not exact**: the token still comes from an exact
rescoring, but the true argmax can fall outside the list. That is an output
change by the criterion of [15](15-kernels-and-numerics.md), and adopting it is
not a decision an optimisation gets to take. It stays measured and unbuilt.

The tool checks itself on every row of every sample: the bound was never
violated and the true argmax was never outside the rigorous list.

### D19. Experts as a common base plus a small difference

A token reads 4 of 64 experts per layer. Were the experts of a layer close to
each other, the layer could store their mean once and each expert as a narrow
delta.

They are not. `residual = Σ_e |W_e − mean|² / Σ_e |W_e|²`, 256 rows per matrix,
every MoE layer:

| | gate | up | down | 64 unrelated matrices |
|---|---|---|---|---|
| residual | 0.9828 | 0.9844 | 0.9844 | 0.9844 |

A delta would need 0.012 fewer bits per weight than the 4 it has, and the base
would have to be read on top: **+25%** of the routed-expert traffic, not less.
The only trace of structure is `gate_proj` in the last layers (0.980 at layer
46), far too little to use.

The comparison is entry by entry, so it is blind to experts that are equal up to
a permutation of their hidden units. What it rules out is the cheap encoding,
not the possibility that two experts compute similar things.

### What `Tests/route` adds to that picture

The same token does not take the same path. Ten prompts, each in a fresh
context, 645 tokens; every occurrence of an id compared with its occurrences in
the other prompts (1,919 pairs) against pairs of different ids (185,246):

| router of layer | experts in common, same id | different ids | identical set, same id |
|---|---|---|---|
| 1 | 2.39 of 4 | 0.35 | 11% |
| 2 | 2.97 | 0.33 | 19% |
| 20 | 0.97 | 0.59 | 3% |
| 45 | 2.38 | 2.07 | 7% |

The id decides the first layers, the context decides the middle, and the last
layers are concentrated regardless of the token — two arbitrary tokens already
share two of their four experts there. That last line is the one with a
consequence: it is where a batched prefill shares the most expert reads. One run,
ten Portuguese prompts, and the same-id pairs are mostly punctuation and function
words; it is a first look, not a distribution.

## SETTLED — a measurement that added a subsystem

### D15. Retrieval in `src/`

It arrived with the domain container of [22](22-finetune-and-retrieval.md), and
the number that created it is stark: of answers reviewed by hand, **26 of 29 were
doctrinally correct with the source passage in context, and 0 of 11 without it**.
The trained model reads well and remembers badly — asked with no passage, it said
a technical term of the work it was tuned on names a football stadium.

So the engine could not serve that container alone. It had to retrieve.

**BM25, ours, in C11.** The reason is not retrieval quality — embeddings would
likely retrieve better. It is that BM25 is an inverted index and two divisions,
which ports without dependencies and stays auditable in one reading session; an
embedding retriever would put a second neural model inside `src/` and end the
property this project exists to have.

Three pieces, all landed:

| piece | what |
|---|---|
| `tools/index` | the corpus becomes a `.ttindex`; the `.ttpack` of `tools/finetune` was the precedent for the format |
| `src/retrieve.c` | the tokeniser, the mmapped reader and the search |
| `src/server.c` | `--index`, and every user turn rewritten in the shape the model was trained on |

**The exit criterion was that the container answer correctly through `tootega
serve`, with no passages pasted by hand.** It does. The two questions that
produced the complaint, before and after, same container:

| question | without the index | with it |
|---|---|---|
| *o que significa arena* | "material fino, resultante da decomposição de rochas" | "o plano da vida vivida, o campo de ação" |
| *o que é Mente Universal* | "a consciência primordial que permeia o universo" | "a Totalidade — não possui vontade nem deliberação" |

The reference is `finetune/scripts/retrieve.py`, the code that chose the passages
every training pair was written against, and the C reproduces it exactly: same
726 sections, same 13,465 terms, same 157,638 tokens, and on twelve queries the
**same top five with the same scores**. That equality is the point and not a
nicety — a retriever that ranks differently shows the model something other than
what it was taught to expect.

**Three things this cost, each recorded because none announced itself:**

*The index must contain **all** sections.* Six essays were held out of training,
correctly — that is what proved generalisation. Holding them out of the *index*
would make the chat answer "the work does not treat that" about subjects it
treats in whole essays. 726 sections, not the 666 the curation used.

*The fold could not be hand-written.* The first version covered Latin-1 and Latin
Extended-A, which is everything Portuguese needs, and silently cut every
transliterated citation the books make in half: `puruṣa` indexed as `puru`,
`upaniṣad` as `upani`. Four terms of vocabulary apart from the reference, and the
only symptom would have been that asking about the Upanishads retrieved nothing.
The fold is now generated by `tools/ucd` from the Unicode database — lowercase,
canonical decomposition, marks dropped — and `--check` verifies it.

*Reasoning had to default off when an index is loaded.* Every RAFT training pair
closed `</think>` immediately. Opening it puts the model in a state the fine-tune
never covered, and the first request through the HTTP endpoint came back as an
English scratchpad deliberating about a Portuguese question. D14 had already
measured that reasoning costs the prefix cache; here it also costs the answer.

### D3. Container format specification

The converter **has already defined** the format de facto: magic `TOOTEGA\x01`,
128-byte header, blobs aligned to 64, JSON index at the end, one entry per tensor
with `kind`/`shape`/`data`/`scales`/`group`.

What is missing is writing that up as a specification for the C reader — the
engine has to match the writer byte for byte. As long as it is only Python code,
it is a tacit convention, and a tacit convention between two programs in
different languages is where alignment bugs are born.

**Resolution:** a short document with the exact binary layout, written **before**
the loader.

### D4. Accumulation precision per operation

M1's exit criterion is 32/32 tokens identical to the `transformers` oracle. That
is only reachable if the numeric choices are deliberate:

| operation | pending |
|---|---|
| matmul accumulator | fp32 (likely) or int32 |
| rmsnorm | fp32; fixed-order sum? |
| router softmax | fp32, with max subtraction |
| RoPE | fp32; precomputed table or `sinf`/`cosf` |
| MLA attention | where to unabsorb the compressed KV |
| parallel reduction | **fixed order** or the result varies with the thread count |

The last one is the one that bites: floating-point addition is not associative.
If the reduction depends on how many threads ran, the result changes with
`--threads` and the determinism test from [03](03-architecture.md) fails by
construction.

**Recommendation:** a fixed reduction order, independent of the thread count. It
costs a little performance and buys reproducibility — which is an exit criterion,
not a luxury.

### D5. Tokenizer

`tokenizer.json` is on disk (HF format). What is missing is determining **which
BPE variant** GLM-4.7-Flash uses — pre-tokeniser, byte handling, special tokens —
and writing the encoder/decoder in C.

Without it there is no end-to-end test, only a forward against precomputed ids.

**Resolution:** inspect `tokenizer.json` and write the reader. It is mechanical
work, but it is on M1's critical path.

### D6. Threading model

Undecided: our own pool or OpenMP; how the work is partitioned (per matmul row,
per expert, per attention head); where the barriers go.

Known constraints: 47 serial layers, top-4 experts per layer, and pinning on a
hybrid CPU has to happen before `main` ([09](09-resource-limits.md)).

**Recommendation:** our own pool with a per-layer barrier. OpenMP would work, but
it drags in `libgomp` and the promise in [10](10-implementation.md) is static
linking with minimal runtime.

**Measured elsewhere, and worth recording:** on a two-socket EPYC 9535 (128
physical cores) the pool does not scale past **~12 effective cores** —
`--threads 128` is *slower* than `--threads 16`, holding 954–1346% CPU out of
12800% available while three quarters of the machine sits idle
([22](22-finetune-and-retrieval.md)). This does not reopen D6: the target
machine is an 8 P-core desktop, where the per-layer barrier is the right shape
and this cannot appear. It is recorded so that whoever measures on a server
does not mistake it for a regression — and because if the engine is ever aimed
at many-core NUMA hardware, the barrier-per-layer partitioning is the first
thing to revisit.

---

### D14. Chat history keeps the reasoning — and that conflicts with the cache

`chat_template.jinja` **strips reasoning from past turns**: a completed assistant
turn is rendered as `<|assistant|></think>` plus the content, with no thinking
block. Only the open turn gets `<think>`.

`tootega chat` **does not do that** — it appends the generated ids exactly as
they came out. That is not carelessness, it is the only way the prefix cache
works:

> Re-rendering the conversation each turn means **re-tokenising**, and
> re-tokenising does not necessarily give back the same ids that were generated.
> KV would stop matching at an arbitrary point in the history, and the symptom
> would be "sometimes the cache is wrong", which is the worst kind of bug.

Measured on the first two-round test: **92 of 100 prompt tokens came from KV** on
the second turn — 8 prefill tokens instead of 100. That is exactly the property
[05](05-roadmap.md) asks for in M4.

The cost of keeping the reasoning: the context grows faster, and the history
diverges from the format the model was trained on for multi-turn.

**How to solve it without choosing either side:** store, alongside the generated
ids, the **offset of `</think>`**. The history stays as ids, the cache keeps
matching up to that point, and stripping the reasoning becomes a KV truncation —
which is cheap — instead of a re-tokenisation. It lands when M4 starts; until
then it is documented as a known divergence, not as a bug.

---

---

## The most expensive lesson — two implementations of mine are not validation

The engine passed **32/32** against the oracle, and perplexity against BF16 came
out indistinguishable from zero. Neither test detected that the **RoPE variant
was wrong**.

`Glm4MoeLiteConfig.rope_interleave` is `True` by default and `config.json` does
not override it. Attention uses `apply_rotary_pos_emb_interleave`; I implemented
the `else` branch:

| variant | pairs |
|---|---|
| `rotate_half` — the one implemented | `x[i]` with `x[i + rot/2]` |
| **interleave** — the correct one | `x[2i]` with `x[2i+1]` |

At position 0 the two coincide (`cos = 1`, `sin = 0`), and the error grows with
position. The text started well and fell apart.

### Why the tests missed it

**The oracle and the engine are two implementations of the same reading of mine**
of `modeling_glm4_moe_lite.py`. Their agreeing proves they are wrong in the same
way, not that they are right. Independence of *implementation* is not
independence of *interpretation*.

**Perplexity compared int4 against BF16 inside my own forward.** `ablate` changes
where the weight comes from, not the mathematics. It measured quantisation and
never had any way to measure correctness.

### The alarm that existed and was silenced

**Perplexity ~340** on ordinary technical prose. A 31B model sits at 10–30. I saw
223–473 across windows, called it "a hard corpus" and moved on — because the
*relative* comparison I cared about looked healthy.

> A relative metric hides error that affects both sides. The **absolute** value
> was the only one telling the truth, and it was on screen the whole time.

### What changes in the process

1. **Every metric with a known absolute value gets a declared expected range.**
   Perplexity outside 10–40 is a failure, not an observation.
2. **The architecture reference has to be external.** Your own oracle validates
   arithmetic, not interpretation. Real validation is `transformers` running the
   model — which never ran here because it does not fit in 64 GB. Until it does,
   free generation is the only test that stresses interpretation.
3. **A generation test is mandatory, not complementary.** Teacher forcing hides
   errors that feed back; `gentest` is what exposed this, and it only existed
   because one answer looked odd to the naked eye.

### What became invalid

`ref32.json` was regenerated. The perplexity measurements and the ablation in
[04](04-target-model.md) — v1 +6.8%, v2 ≈ 0, "attention explains 100%" — were all
done with the wrong RoPE and needed redoing. The speed numbers remained valid:
the bug did not change the amount of work.

**Done.** The ablation was redone, paired: the only group above the noise is the
experts (−0.0225 ± 0.0076); attention is −0.0007 ± 0.0019. The section in
[04](04-target-model.md) now carries the retraction next to the current table,
and the converter's default went back to `--attn int4`.

---

## CAN WAIT

### D7. Engine ↔ service protocol (M3)
The streaming engine uses a line protocol over stdin/stdout with `SUBMIT` frames.
It works and it is easy to debug. Decide in M3.

### D8. Paged KV details (simple in M1, complete in M3)
[11](11-memory.md) defines blocks shared by prefix hash. Block size, index
structure and eviction policy wait until there is a measurement. M1 can use
contiguous per-slot KV.

### D9. Sampling
Greedy, temperature, top-p, top-k. Trivial, but it has to exist for the CLI.

### D10. Licence and repository structure
Settled: MIT for the code, in `LICENSE`. The model, its weights and the tokenizer
carry their own licence, from whoever trained them.

---

## Suggested order

1. **D1** — 10 minutes, and it unblocks the 19 GB conversion
2. **D3** — the specification, before writing the loader
3. **D2 + D4** — the two numeric decisions, together: they define the kernel
4. **D5** — tokenizer, critical path and mechanical
5. **D6** — threading, once there is a kernel to parallelise

D7–D10 land in their corresponding milestones.
