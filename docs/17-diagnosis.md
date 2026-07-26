# 17 — Diagnosis: hunting the quality bug

A record of what was tried, what each attempt measured, and what survived. It
exists because the sequence of wrong conclusions here is worth more than the
final result: each one failed for a different, reproducible reason.

## The symptom

Free generation produced degenerate text — `*   *   *` in a loop with greedy
sampling, word salad with temperature — while **every** internal test passed.

## Path 1 — blame quantisation

**Hypothesis:** the `v2` container (attention in int8) broke the model, because
`v1` answered correctly and `v2` did not.

**Measurement:** one prompt, eyeballed.

**Result: false.** `gentest` over 12 prompts showed that the original **BF16
weights degenerate the same way** — 52.7% repeated 4-grams, 1 of 12 prompts
stopping at EOS.

**Method error:** a conclusion from N=1, right after I had written in the
preceding paragraph that one prompt is not a sample.

## Path 2 — blame greedy decoding

**Hypothesis:** greedy decoding gets stuck in a loop; it is a known pathology.

**Measurement:** the same prompt with `--temp 0.8`.

**Result: false.** With temperature the output became word salad — invented
Portuguese, "Recadesso o a seguint", "corcovamento" — instead of a loop. Invented
words are not a decoding pathology; they are the signature of **wrong logits**.

## Path 3 — RoPE interleave

**Hypothesis:** some attention operation is wrong.

**Measurement:** reading `configuration_glm4_moe_lite.py`.

**Result: TRUE, and it was a real bug.**

`rope_interleave` is `True` by default and `config.json` does not override it.
Attention uses `apply_rotary_pos_emb_interleave`; I had implemented the `else`
branch:

| variant | pairs |
|---|---|
| `rotate_half` — implemented | `x[i]` with `x[i + rot/2]` |
| **interleave** — correct | `x[2i]` with `x[2i+1]` |

At position 0 the two coincide (`cos = 1`, `sin = 0`), and the error grows with
position. Fixed in `src/forward.c` and in `ptools/oracle.py`.

**Effect:** generation went from salad to a correct answer with complete
reasoning and a stop at EOS.

## Path 4 — stale binary

**Hypothesis:** the RoPE fix was not enough, because perplexity stayed at 380,
then 281, then 273.

**Result: false, and the cause was trivial.**

`Tests/*.exe` link `src/` statically. After fixing `src/forward.c` I ran
`.\build.ps1` — which builds **only the engine**. The measurement tools kept the
broken forward inside them.

Every post-fix perplexity came from a stale binary. Rebuilding:

| | before (old binary) | after |
|---|---|---|
| int4 | ppl 273.5 | **ppl 5.66** |
| BF16 | ppl 281.4 | **ppl 8.63** |
| transformers | — | 8.89 |

The bug had been fixed hours earlier. `build-all.ps1` exists for exactly this,
was written the same day, and was not used.

> A tool that embeds the code under test has to be rebuilt together with it. A
> single build entry point is not a convenience: it is what stops you from
> measuring the wrong version without noticing.

## Path 5 — "there is still a bug, perplexity is still 340"

**Hypothesis:** the RoPE fix was not enough.

**Measurement:** `ablate` over `docs/*.md`, then over ordinary prose. It gave 380
and 281.

**Result: false — but it led to the test that settled everything.** The suspicion
motivated building `ptools/hf_ref.py`, the **project's first external
reference**.

## What the external reference established

Running `transformers` with disk offload cost **2 minutes**. The 62 GB do not fit
in RAM, and I had treated that as "cannot be run" from the start.

| measurement | value |
|---|---|
| transformers, 384 positions of prose | **ppl 8.89** |
| engine, **the same 384 positions** | **ppl 5.66** |
| `Tests/ablate.exe`, "the same" window | **ppl 273–281** |

The engine and `transformers` agree. **The 281 comes from the measurement tool.**

Checks that close the net:

- **tokenizer:** 1,995 ids identical to HF's over the whole corpus, exact
  round-trip. It is not that.
- **per-layer bisection:** embed with 0.000% error; layers 0–45 between 4% and
  15%, rising and falling, with no jump. That is quantisation noise, not a bug.
- **logits against transformers:** 4.05% error, 7 of 8 tokens equal.

## What survives

**The engine is correct and validated against an external reference.**

| measurement | engine | transformers |
|---|---|---|
| perplexity, 384 positions of prose, BF16 | **8.63** | **8.89** |
| perplexity, same text, int4 container | **5.66** | — |

The 3% between 8.63 and 8.89 is what is left: one-dimensional tensors the
substitution does not swap (already exact) and accumulation-order differences.

Tokenizer, container, kernels, determinism and speed remain valid and were
validated through independent paths.

**What still does not close:** int4 giving 34% LOWER perplexity than BF16 on the
same window. Quantisation does not improve a model. It is a single window; it
needs more sample before it becomes a suspicion or an explanation.

**What needs redoing:** the per-group ablation and the precision tables in
[04](04-target-model.md). They were measured with the wrong RoPE, with a stale
binary, or both.

## The method errors

1. **A conclusion from N=1** (path 1).
2. **Explaining the number instead of chasing it** — perplexity 340 became "it is
   the corpus", 281 became "it is the corpus again".
3. **Validating interpretation against interpretation** — the oracle and the
   engine are two writings of the same reading of mine of the model. They agreed
   32/32 even with the wrong RoPE.
4. **Measuring with a stale binary** (path 4).

The third is the root of the first two: without an external reference, any story
that closes looks like evidence. The fourth is of a different nature — build
discipline — and it cost more hours than the other three combined.

## The rule that was missing

> **Before writing the first line of a forward, run the reference model once and
> save the logits.**

Not as final validation — as a **starting point**. It costs minutes and would
have prevented this entire document.

And the second, which follows from it:

> **A metric with a known absolute value gets its range declared before
> measuring.** Perplexity outside 5–40 is a failure, not an observation.
