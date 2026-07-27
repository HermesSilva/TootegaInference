# 19 — Reference state

A snapshot of what exists, what was measured and what was not. It serves as the
baseline for judging any future change: a number without this page next to it
does not say whether anything improved.

Build `ddf196a58ce9` · reference machine from the [README](../README.md).

## Container v4 — what changed and what is still v3's number

The performance and quality numbers below were measured on **container v3**. A
**v4** now exists, and it carries two changes at once, which is worth stating
before anyone attributes an effect to the wrong one:

| | v3 | v4 |
|---|---|---|
| scale | pure `absmax/qmax` | 13-point clip grid |
| attention | int8 (`--attn int8`) | int4 — the documented default |
| int4 tensors | 8,973 · 15.91 GB | 9,165 · 16.46 GB |
| int8 tensors | 390 · 2.38 GB | 198 · 1.33 GB |
| file | 19.81 GB | **19.32 GB** |
| resident weights | 18.54 GB | **18.05 GB** |
| int4 output error | 10.81% | **9.52%** |

Attention in int8 was kept in v3 out of an ablation that has since been redone:
paired, the only group above the noise is the experts (−0.0225 ± 0.0076 nll);
attention gives −0.0007 ± 0.0019. It cost 0.49 GB for nothing measurable.

Against the v3 oracle, **27 of 32** argmaxes are unchanged. The five that moved
(positions 8, 12, 18, 30, 31) all have a v4 margin between **0.093 and 0.372**,
against a median margin of 1.113 — they are the ties, and a container change
tipping a tie is the expected outcome, not a signal. Which side of the tie is
better is what `gentest`'s `match@ref` against BF16 answers; the oracle does not
and cannot. It did answer: see below.

### Is v4 better? Yes, in the only comparison that was controlled

`gentest`, 12 prompts, up to 96 tokens, greedy, **all three in the same run**,
against the BF16 reference generated in that same run:

| container | `match@ref` | `rep4` | stopped | tok/s |
|---|---|---|---|---|
| BF16 (reference) | 96.0 | 7.3% | 0/12 | 3.34 |
| v3 | 32.4 | 5.3% | 0/12 | 8.43 |
| **v4** | **42.7** | 6.8% | 0/12 | **10.98** |

**`match@ref` 32.4 → 42.7**: v4 follows the original weights for a third longer
before diverging. Same prompts, same reference, same run — this one is paired by
construction and it is the number that answers the question.

`rep4` stays far below the ~15% of degeneration for all three, and v4 sits closer
to BF16's own 7.3% than v3 does. **`stopped 0/12` is not a finding**: BF16 does
not stop either. This is a thinking model and 96 tokens do not finish the
reasoning — the budget is the limit, not the model.

## Performance — CPU, container v4, 20 threads

Measured with `bench --repeat`, bandwidth measured at runtime on the day:

```
memory bandwidth     69.4 GB/s     MEASURED at runtime, not specified
traffic per token    2.38 GB       tensors a decode actually touches
roofline ceiling     29.1 tok/s    bandwidth / traffic

warm decode          ~50 ms/token  = ~19.8 tok/s
mean decode          17.12 tok/s   (includes the cold first prompt)
cold decode          84 ms/token   = 1.7x worse than warm
mean ttft            2.51 s
prefill              10.4 tok/s
RSS                  16.69 GB peak; 36.2 GB free
KV                   108.3 KB per position; 221.8 MB for 2048
scratch              1.2 MB peak
```

Effective bandwidth `2.38 GB × 17.12 tok/s = 40.8 GB/s` — **59% of the measured
69.4**, against v3's 55% of its ceiling.

### Against v3, and what the comparison is worth

| | v3 | v4 |
|---|---|---|
| traffic per token | 2.87 GB | **2.38 GB** (−17%) |
| roofline ceiling | 24.0 tok/s | **29.1 tok/s** |
| warm decode | 71 ms | **~50 ms** |

The traffic drop is fully explained: attention went from int8 to int4, and
decode reads every attention tensor of all 47 layers on every token. −17% of
traffic on the same bandwidth utilisation would give 59 ms; the measurement says
50.

**The remaining ~9 ms are not explained by this measurement, and the two runs
were not controlled against each other** — different build, different free RAM
(36.2 GB against 35.9), and the batched matmul landed in `kernels.c` in between.
Take the v4 block above as v4's numbers. Do not take "v4 is 40% faster" as a
result; what is demonstrated is that v4 reads less and runs faster, and by how
much of each is a question for a controlled run.

**Matmul in isolation at 62 GB/s — 90% of bandwidth.** There is nothing left to
optimise in the matmuls. The missing milliseconds are in **attention and the
remaining barriers**.

## Quality — validated against an external reference

| measurement | engine | transformers |
|---|---|---|
| perplexity, 384 positions of prose, BF16 weights | **8.63** | **8.89** |
| perplexity, same text, int4 container | **5.66** | — |

The 3% difference is 1-D tensors the substitution does not swap (already exact)
and accumulation order.

### Perplexity of v4 — inconclusive, and the corpus is why

`ablate --no-ablate`, `corpus.txt`, 10 windows of 384 tokens, 3,830 predictions,
paired window by window:

```
int4 g64 (v4)      nll 3.6821   ppl 39.731
BF16 (reference)   nll 3.6539   ppl 38.626

delta        +0.0282 +- 0.0288      perplexity 1.0286x  (+2.86%)
95% interval [-0.0282, +0.0846]     INCONCLUSIVE: crosses zero
per window   +0.061  -0.201  +0.052  +0.107  +0.095
             +0.056  -0.054  +0.067  +0.050  +0.049
```

**One window opens the whole interval.** Window 1 has `nll 8.23` against 2.79 to
3.64 for every other — a perplexity near 4,600, where nothing is predictable. It
is the only window where int4 "wins", and it wins by four times the magnitude of
any other difference. It is the table-heavy window [21](21-int8-kv.md) already
identified as the one that flips the mean of this corpus.

Dropping it gives `+0.0536 ± 0.0151`, interval `[+0.0241, +0.0831]`, which does
not cross zero — **and that number is not published, because it is a post-hoc
exclusion.** Choosing which window to drop after seeing the result is how a
measurement becomes an opinion. It is recorded here as the diagnosis of the
width, not as the measurement.

The fix is the one already on the outstanding list: **a prose corpus**.

> **Do not compare this to the `+0.0513 ± 0.0076` recorded above for the earlier
> container.** That was measured on a different revision of `corpus.txt` — the
> absolute nll there was 5.83, here it is 3.68. The corpus is regenerated from
> the documentation and the documentation changed. Two deltas measured on
> different texts are not comparable, and this project has already published a
> perplexity with the sign inverted for less.
>
> The valid comparison would be `ablate` on v3 **today**, on this same corpus. It
> has not been run.

| property | state | how it is reproduced |
|---|---|---|
| teacher-forcing oracle | **32/32** identical tokens | `Tests\oracle.exe <model> --ids-file ids32.txt --compare <logits>` |
| determinism | **byte-identical** logits from 1 to 20 threads | `tootega forward --threads N --dump-logits` |
| tokenizer against HF `tokenizers` | **36/36** cases, exact round-trip | `tootega tokenize <tokenizer.json> --check tokfixture.json` |
| kernels against the reference quantisation | **bit for bit** | `tootega selftest`; the fixture from `tools\fixture.exe --check` |
| Unicode tables against the database | **byte for byte** | `tools\ucd.exe UnicodeData.txt --version 13.0.0 --check src\unicode_tables.h` |

Once a reference exists, checking a logits dump against it does not need the
oracle to run again: `Tests\oracle.exe --check-ref ref32.json --logits <file>`
takes milliseconds, and that is what makes it viable to verify every kernel
change instead of only the ones that look risky.

`tokfixture.json` is a **frozen artefact**: it was produced by HuggingFace's
`tokenizers`, and there is no C to write in its place. Regenerating it means
bringing that library back for the occasion.

## Quantisation — paired ablation

The only group with an effect above the noise:

| group restored to BF16 | nll delta | verdict |
|---|---|---|
| experts | **−0.0225 ± 0.0076** | settled |
| attention | −0.0007 ± 0.0019 | noise |
| shared | +0.0216 ± 0.0325 | noise |
| lm_head | −0.0008 ± 0.0007 | noise |
| dense | +0.0168 ± 0.0097 | noise |

**Attention in int4 costs nothing measurable.** The converter's default went back
to `--attn int4`; `int8` costs 0.3 GB for nothing.

## Implemented

| piece | where |
|---|---|
| `.tootega` reader, JSON parser, arena | `src/container.c`, `json.c`, `common.c` |
| int4/int8/f32/bf16 kernels, scalar and AVX2+FMA | `src/kernels.c` |
| our own pool, three-stage wait with futex | `src/pool.c` |
| absorbed and expanded MLA, MoE, KV | `src/forward.c` |
| tiktoken BPE tokenizer | `src/tokenizer.c` |
| greedy/temp/top-k/top-p sampling | `src/sample.c` |
| budgets and `--plan` | `src/budget.c` |
| HTTP server, SSE, embedded web UI | `src/server.c`, `webui.h` |
| tools in the OpenAI protocol (`tools`, `tool_calls`, `role:tool`) | `src/server.c`, [20](20-agent.md) |
| agent speaking the Claude Code CLI contract | `tools/agent/` |
| local tools with sandbox and approval | `tools/agent/`, [20](20-agent.md) |
| BF16 → `.tootega` converter | `tools/convert/` |
| measurement: ablation, generation, roofline | `Tests/` |

## Not implemented

- **Slots and continuous batching** (M3). One request at a time; the second tab
  waits in the TCP backlog, silently.
- **GPU (M2) — out of master, on the `GPU` branch.** It was written and measured
  there: kernels at 179–190 GB/s against the CPU's 68.7 (2.6×), and even so the
  complete decode came out **slower** than pure CPU in three configurations,
  because a granularity of one matmul per call pays 17 µs of protocol each time.
  What is missing is `run_layer` on the device. Master stays pure CPU until that
  exists and shows a measured gain.
- **Persistent state** ([12](12-persistent-state.md)) and **MTP speculation**
  (M5).
- **Linux build** — the `Makefile` exists and has never been compiled.

## int8 KV — measured, inconclusive, not adopted

`--kv-prec int8` fits 202,752 positions in 5.83 GB against f32's 22.0. Measured
over 4 windows of 4096: **+2.47% perplexity** in the configuration that actually
runs, but with the 95% interval crossing zero and one degenerate window (tables)
flipping the mean. Generation did not get worse (`match@ref` 32.4 → 37.3, `rep4`
unchanged). The criterion was declared beforehand and was not met: it stays as an
option. Details in [21](21-int8-kv.md).

## Outstanding measurements

1. **A prose corpus of ~32k tokens.** It is now the top item, because it is what
   makes the perplexity of this container measurable at all: one table-heavy
   window of `corpus.txt` widens the interval past zero on its own, and the
   diagnosis is above. `corpus_prosa.txt` has 1,995 tokens and does not even fill
   one window of 4096.
2. **`ablate` on v3, on today's corpus.** Without it there is no valid
   perplexity comparison between the two containers — the recorded `+0.0513` was
   measured on a different revision of the text.
3. **Determinism on v4.** The claim of byte-identical logits from 1 to 20 threads
   was verified on an earlier container and not on this one:
   `tootega forward --threads 1|20 --dump-logits` and compare the files.
4. **A controlled speed comparison** if the v3-to-v4 difference is to be
   attributed rather than merely reported: same build, same machine state.

Done since the last revision of this page: the container was reconverted with
`--attn int4` and the clip search (v4); `gentest`, `ablate` and `bench` were run
on it; and the precision tables in [04](04-target-model.md) were rewritten — the
ablation that pointed at attention is marked retracted, with the two defects that
produced it.

## Bugs found, and what each one taught

| bug | how it escaped | record |
|---|---|---|
| RoPE in the `rotate_half` variant instead of `interleave` | oracle and engine were two writings of the same reading of mine | [17](17-diagnosis.md) |
| measurements with a stale binary | `build.ps1` builds only the engine; the tools link `src/` | [17](17-diagnosis.md) |
| sampler subtracting an already-overwritten `val[0]` | greedy returns before the softmax; `top-k=1` only looks at `val[0]` | `src/sample.c` |
| pool spinning on `yield` forever | no measurement runs long enough to see idleness | `src/pool.c` |
| pool sleeping on a timer instead of an event | it only shows up after a long pause | `src/pool.c` |

The pattern common to the first three: **tests born of the same reading of the
problem are blind to the same error**. The external reference cost 2 minutes and
settled what days of internal testing did not.
