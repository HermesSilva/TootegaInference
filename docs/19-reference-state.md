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

**Measured on v4, and only this:**

| property | result |
|---|---|
| engine against `Tests/oracle` | **32/32**, relative error **0.0001%**, largest difference 5.0e-4 |
| conversion | **171 s**, 8 threads, 9,703 tensors |
| teacher-forced pass, 32 positions, cold | 3.25 s · 9.84 tok/s · 9.24 GB RSS |

**Not measured on v4 yet:** warm decode, ttft, perplexity, `gentest`. Until they
are, the numbers in the sections below are v3's and must be read as such.

Against the v3 oracle, **27 of 32** argmaxes are unchanged. The five that moved
(positions 8, 12, 18, 30, 31) all have a v4 margin between **0.093 and 0.372**,
against a median margin of 1.113 — they are the ties, and a container change
tipping a tie is the expected outcome, not a signal. Which side of the tie is
better is what `gentest`'s `match@ref` against BF16 answers; the oracle does not
and cannot.

## Performance — CPU, container v3, 20 threads

```
memory bandwidth     68.7 GB/s     MEASURED at runtime, not specified
traffic per token    2.87 GB       tensors a decode actually touches
roofline ceiling     24.0 tok/s    bandwidth / traffic

warm decode          71 ms/token   = 13.1 tok/s   (55% of the ceiling)
cold decode          85 ms/token   = 1.2x worse   (was 4.0x before warm-up)
mean ttft            2.25 s
prefill              ~11 tok/s
warm-up              18.54 GB in 5.3 s
RSS                  16.8 GB peak; 35.9 GB free
idle server          0.2% of one core, 0.04 GB
```

**Matmul in isolation at 62 GB/s — 90% of bandwidth.** There is nothing left to
optimise in the matmuls. At 62 GB/s the 2.87 GB/token would give 21.7 tok/s; we
measure 13.1. The missing ~28 ms are in **attention and the remaining barriers**.

## Quality — validated against an external reference

| measurement | engine | transformers |
|---|---|---|
| perplexity, 384 positions of prose, BF16 weights | **8.63** | **8.89** |
| perplexity, same text, int4 container | **5.66** | — |

The 3% difference is 1-D tensors the substitution does not swap (already exact)
and accumulation order.

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

1. Reconvert the container with `--attn int4` (v3 is int8 with no benefit).
2. A full `gentest` with the correct forward and sampler — the earlier results
   measured two bugs.
3. Rewrite the precision tables in [04](04-target-model.md).
4. A **prose** corpus of ~32k tokens. `corpus_prosa.txt` has 1,995 and does not
   even fill one window of 4096; `corpus.txt` is documentation with tables, and
   one window of it ruins the mean ([21](21-int8-kv.md)).

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
