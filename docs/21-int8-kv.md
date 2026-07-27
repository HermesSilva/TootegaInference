# 21 — int8 KV: measured, and not adopted

The question: `--kv-prec int8` fits the model's maximum context — **202,752
positions in 5.83 GB** against f32's 22.0 GB — and it has been implemented since
M1. Why is it not the default?

The short answer used to be "it failed 32/32". But that gate measures
**difference**, not degradation: failing it means "the output changed", not "the
output got worse". Nobody had measured the second thing. This document measures
it.

## What each precision costs

Per position, and at the model's ceiling (202,752):

| | per position | total KV | with the 18.05 GB of weights |
|---|---|---|---|
| f32 (default) | 105.8 KB | 21.96 GB | 40.0 GB |
| bf16 | 52.9 KB | 10.98 GB | 29.0 GB |
| **int8 g64** | **28.1 KB** | **5.83 GB** | **23.9 GB** |

int8 carries one f32 scale per group of 64 — the extra 0.5 KB per position.

The weights column is container **v4**; the measurements further down were taken
on **v3**, whose 18.54 GB add 0.49 GB to each total. It changes no conclusion
here — the decision is driven by the KV column, which does not depend on the
container.

## Criterion, fixed before measuring

> perplexity below 1% · `match@ref` with no relevant regression · `rep4` and stop
> rate unchanged

## Screening — 5 windows of 384, `corpus_prosa.txt`

| configuration | f32 | int8 | delta |
|---|---|---|---|
| int4 g64 (what actually runs) | ppl 6.481 | ppl 6.503 | **+0.34%** |
| BF16 (reference) | ppl 6.857 | ppl 6.825 | **−0.47%** |

Two independent measurements, **opposite signs, same magnitude**. Perturbation
with no direction. It passed.

## The deciding test — 4 windows of 4096, `corpus.txt`

| configuration | f32 | int8 | delta |
|---|---|---|---|
| int4 g64 | nll 4.8042 · ppl 122.0 | nll 4.8286 · ppl 125.0 | **+2.47%** |
| BF16 | nll 4.9641 · ppl 143.2 | nll 4.9546 · ppl 141.8 | **−0.94%** |

**+2.47% blows the criterion.** And the paired analysis, which has been the rule
since [04](04-target-model.md), shows why:

```
int4 g64             BF16 (reference)
window 1  -0.0006    window 1  +0.0024
window 2  +0.0829    window 2  -0.0511     <- dominates both means
window 3  +0.0080    window 3  +0.0084
window 4  +0.0074    window 4  +0.0023

mean +0.0244         mean -0.0095
CI95 [-0.014, +0.063]  CI95 [-0.037, +0.018]
```

**Both intervals cross zero.** Window 2 alone flips both means, in opposite
directions — and it has `nll 11.5`, a perplexity of ~100,000.

I went to look at what it is: `corpus.txt` is the documents themselves
concatenated, and window 2 landed in a region of **markdown tables full of
numbers**. The model has no way to predict table cells; that window does not
measure language modelling, it measures behaviour in the face of unpredictable
text.

A design error of mine: [17](17-diagnosis.md) records that the calibration corpus
is `corpus_prosa.txt`. I used the other one only because it was long enough for
4 windows of 4096, and paid for it with a degenerate window.

Over the three prose windows, int8 costs **+0.49% perplexity**, in the same
direction in all three. That is a small, consistent and plausible cost.

## Generation — 12 prompts, 96 tokens, greedy

| | match@ref | rep4 | stopped | tok/s |
|---|---|---|---|---|
| f32 | 32.4 | 5.3% | 0/12 | 9.13 |
| int8 | **37.3** | 5.4% | 0/12 | 8.79 |

It did not get worse. `match@ref` even rose — int8 followed the BF16 reference
for **more** tokens before diverging. With 12 prompts that is not evidence that
it is better; it is evidence that it is not worse.

## Verdict: not adopted

The criterion was declared beforehand and the primary statistic did not meet it.
Adjusting the criterion after seeing the number would be exactly what
[17](17-diagnosis.md) exists to prevent.

And the honest result is not "int8 degrades" — it is **inconclusive**: both
confidence intervals cross zero, the two signs disagree, and generation did not
change. Inconclusive is not approved.

`--kv-prec int8` remains available in `serve`, `ablate` and `gentest`. Whoever
needs 200k of context now has a way, with this document alongside saying what is
known and what is not.

## What would make the decision possible

1. **A long prose corpus.** `corpus_prosa.txt` has 1,995 tokens — not even one
   window of 4096. A prose corpus of ~32k tokens is missing, and with it the
   4-window, 4096-position measurement becomes meaningful.
2. **More windows.** With 4, one outlier flips the mean. Eight or ten would give
   an interval that does not cross zero.
3. **Generation at long context.** `gentest` has `ctx` fixed at 1024 and short
   prompts: it does not exercise attention over thousands of positions, which is
   exactly where KV quantisation would have the best chance of showing up. There
   is no tool for that today.

Estimated cost: one machine-night, like this one.
