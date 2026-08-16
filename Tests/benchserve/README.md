# benchserve — decode throughput of the serving path

Measures `tootega serve` end to end, the same way every time, so that a change
aimed at performance can be compared against a number taken before it.

```powershell
.\build.ps1                                          # optimized, never -Debug
.\Tests\benchserve\bench-serve.ps1 -Label baseline   # BEFORE any change
# ... change the code, rebuild ...
.\Tests\benchserve\bench-serve.ps1 -Label spin-2000  # AFTER
```

Results land in `Tests/benchserve/results/`: one `.json` per run with the full
reply, one `.server.log` with the engine's own verbose output, and a line per
run appended to `results.csv`. The CSV is never overwritten — the history of
the measurements is the artefact.

## Why this measurement exists

The profile of 2026-08-16 (`Profile/Report20260816-0844.diagsession`, 118 s of
`serve --threads 24 --ctx 150000`) attributes 619 s of process CPU like this:

| where | time | share |
|---|---|---|
| `mm_int4_simd` | 217,2 s | 35,1% |
| **pool spin/yield** | **~190 s** | **~30,7%** |
| `mm_int8_simd` | 41,5 s | 6,7% |
| `tt_dequant_row` | 40,0 s | 6,5% |
| `mm_f32_simd` | 24,9 s | 4,0% |

The spin is `NtYieldExecution` 83,4 s + `KiSystemServiceUser` 78,3 s +
`tt_cpu_relax` 43,6 s + `SwitchToThread`/`KeYieldExecution` ~15 s. In the
butterfly view, of 588.347 inclusive hits in `worker`, 193.198 (32,8%) go to
`SwitchToThread` against 337.335 (57,3%) to `drain` — the useful work.

Sliced in 20 s windows, the spin share is 36,0% / 39,4% / 34,6% across the three
active windows. It is **not** idle time between requests: it happens inside the
decode, on the barrier of `tt_pool_run`.

## What is held fixed, and why

- **The prompt, word for word** — 30 words, in the script. A different prompt is
  a different prefill and a different decode length.
- **`temperature: 0`** — `sample.c:36` returns the argmax before touching the
  RNG, so the generated tokens do not depend on the clock-seeded sampler. Two
  runs produce the SAME text; only the time differs. The script hashes the reply
  and **shouts if two runs disagree**: if the text changed, the tok/s is not
  comparing the same work and the numbers are void.
- **`think: false`** — reasoning is on by default (`server.c:830`) and its
  length varies, which would change what `max_tokens` means.
- **`max_tokens` fixed** (256) — same amount of decode work every run.
- **The server is restarted per repetition** — no run inherits the prefix cache
  or the warm KV of the one before it.
- **Optimized build** — `.\build.ps1` without `-Debug` is `/O2 /MT /DNDEBUG`.
  The script warns on the signs of a debug binary.

Each row also records the `.exe` SHA-256, its mtime, and the git rev plus
clean/dirty. A result you cannot trace back to a binary is not a result.

## Reading the numbers

Two are recorded and they answer different questions:

- `tokens_per_second` — the server's own, decode only. **This is the one to
  compare**; it is what the pool change targets.
- `wall_seconds` — the client's wall clock, including prefill and HTTP.

Compare **medians** across labels, and look at the spread before believing a
difference. `docs/04` records why: the first perplexity measurement here had the
sign inverted because a global mean swallowed the effect in the variance between
windows. Three runs whose spread overlaps the difference have not shown anything.

`load_seconds` is reported but is not the subject: it is dominated by reading
19 GB from disk and says more about the page cache than about the engine.

## Parameters

Defaults reproduce the profiled configuration (`--threads 24 --ctx 150000`).
Everything is overridable, and `-Threads` is itself worth an experiment: the
box is an i9-14900K (8 P-cores + 16 E-cores), and equal-sized blocks handed to
unequal cores widen the barrier that the profile shows the workers waiting on.

```powershell
.\Tests\benchserve\bench-serve.ps1 -Label threads-8 -Threads 8
```

Changing `-Threads`, `-Ctx` or `-MaxTokens` produces numbers that are **not**
comparable with the defaults. They are recorded in every CSV row for exactly
that reason — compare like with like.
