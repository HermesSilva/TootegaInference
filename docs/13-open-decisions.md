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
| **D15** | **retrieval in `src/`** | open, **blocks the domain container** — BM25, see [22](22-finetune-and-retrieval.md) |

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

## BLOCKS M1

### D15. Retrieval in `src/` — a requirement a measurement created

Not planned. It arrived with the domain container of
[22](22-finetune-and-retrieval.md), and the number that creates it is stark: of
answers reviewed by hand, **26 of 29 were doctrinally correct with the source
passage in context, and 0 of 11 without it**. The trained model reads well and
remembers badly — asked with no passage, it says a technical term of the work it
was tuned on names a football stadium.

So the engine cannot serve that container alone. It needs to retrieve.

**Recommendation: BM25, ours, in C11.** The reason is not retrieval quality —
embeddings would likely retrieve better. It is that BM25 is an inverted index
and two divisions, which ports without dependencies and stays auditable in one
reading session; an embedding retriever would put a second neural model inside
`src/` and end the property this project exists to have. The reference
implementation is `finetune/scripts/retrieve.py`, already measured against the
corpus: `k1=1.5`, `b=0.75`, the section as the unit, accents folded — and
`unicode_tables.h` already folds them.

Three pieces: a `tools/index/` producing a `.ttindex` (the `.ttpack` of
`tools/finetune` is the precedent for the format), the search in `src/`, and the
injection into the user turn in `server.c` in the exact shape the model was
trained on.

**What would settle it:** the domain container answering correctly through
`tootega serve`, with no passages pasted by hand. Today that is done from Python
and it is an experiment with scaffolding, not a chat.

**One trap, recorded because it was nearly built in:** the index must contain
**all** sections. Six essays were held out of training, correctly — that is what
proved generalisation. Holding them out of the *index* would make the chat
answer "the work does not treat that" about subjects it treats in whole essays.

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
