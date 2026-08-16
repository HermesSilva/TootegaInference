# 23 — The spin that cost 30%, and the row that was written to be read back

A CPU profile of the serving path, read in full, and the two changes it
justified. The measurement comes first because both changes are answers to it;
neither was an idea looking for a justification.

## The measurement

`Profile/Report20260816-0844.diagsession` — Visual Studio CPU Usage, ETW with
stacks, no lost events. 118 s of wall clock on an i9-14900K (8 P-cores +
16 E-cores, 28 logical), serving:

```
tootega.exe serve models\glm47flash-v4\model.tootega
             --threads 24 --ctx 150000 --port 8080 --verbose
```

619 s of process CPU. Where it went:

| where | time | share |
|---|---|---|
| `mm_int4_simd` | 217,2 s | 35,1% |
| **pool spin/yield** | **~190 s** | **~30,7%** |
| `mm_int8_simd` | 41,5 s | 6,7% |
| `tt_dequant_row` | 40,0 s | 6,5% |
| `mm_f32_simd` | 24,9 s | 4,0% |
| `tt_dot` + `tt_axpy` | 24,6 s | 4,0% |

The spin decomposes into `NtYieldExecution` 83,4 s, the `KiSystemServiceUser`
78,3 s that syscall drags behind it, `tt_cpu_relax` 43,6 s, and ~15 s of
`SwitchToThread`/`KeYieldExecution`.

The butterfly view names the caller: of 588.347 inclusive hits in `worker`,
**193.198 (32,8%) go to `SwitchToThread`** and 40.506 (6,9%) to `tt_cpu_relax`,
against 337.335 (57,3%) in `drain` — the useful work.

**It is not idle time between requests.** Sliced into 20 s windows, the spin
share is 36,0% / 39,4% / 34,6% across the three active ones. It happens inside
the decode, on the pool's barrier.

## Change 1 — the yield window (`pool.c`)

`wait_for_work` had three steps: 512 `pause`, then **19.488 `tt_yield()`**, then
block on the futex. The middle step is the one the profile found.

Each `tt_yield()` on Windows is `SwitchToThread()`, a full syscall. Measured on
this machine:

| loop | time |
|---|---|
| 4096 `pause` | 120 µs |
| 4096 iterations, `pause` with `yield` every 64th | **130 µs** |
| 4096 `SwitchToThread` | 673 µs |
| 19.488 `SwitchToThread` (the old window) | **3.204 µs** |

3,2 ms of syscall per wait, and it bought nothing: if the work arrives soon the
`pause` was enough, and if it does not, the right answer is to block on the
futex and stop burning CPU — which is what the third step already did.

The window is now 512 `pause` plus 4096 iterations that `pause` and yield only
every 64th, reaching the futex in 130 µs instead of 3.204 µs: **5,2× less time
and ~98% fewer syscalls**, and still 130 µs of slack before blocking — ample for
a dispatch between layers, which arrives in microseconds.

The same one-in-64 change went into `tt_pool_run`'s barrier, where the caller
waits for the last block. On a hybrid part that last block is usually on an
E-core, so the wait is long by construction and was paying a syscall per turn.

What did NOT change is the wake-up path, and it is what makes the narrower
window safe: whoever dispatches wakes the sleepers (`tt_pool_run`), and
`sleepers` is incremented *before* the re-check, so a dispatch landing between
the check and the block is still seen. The `tt_sleep_ms(1)` disaster this file
warns about — 26× slower with 20 threads — was polling by clock; this is not.

## Change 2 — fused int4 row kernels (`kernels.c`, `forward.c`)

`attn_head_absorbed` walks `kv_b` row by row, and did it like this:

```c
tt_dequant_row(wrow, kvb, r0 + r);   /* 512 floats to memory */
tt_axpy(q_abs, qh[r], wrow, R);      /* ... read straight back */
```

For this model — 47 layers, 20 heads, `kv_b` is int4 `[8960, 512]` group 64 —
that is **421.120 rows per token**, each writing 512 floats that nothing else
ever reads: **822 MB of f32 stores per token**, purely to hand a row to the next
call. The profile bills it at 40 s (6,5%).

`tt_dot_int4g` and `tt_axpy_int4g` unpack straight into the FMA. The row never
reaches memory.

Note this does **not** contradict the comment about not materialising `kv_b`:
materialising the whole tensor in f32 would cost 863 MB *read* per token against
int4's 108 MB, and that is still true and still the reason the absorbed form
exists. What went away is the per-row round trip, not the packed weight.

### The bit that had to be got right

`mm_int4_simd` walks each group half/half — 8 low nibbles, then the 8 high ones
that live 32 floats further along. `tt_dot` walks the row linearly. fp32 addition
is not associative, so accumulating in the half/half order assigns each product
to a different lane of the accumulator and **changes the last bit**.

So `tt_dot_int4g` does not copy `mm_int4_simd`'s loop. It unpacks one group into
a 64-float stack buffer in the row's natural order and accumulates from that,
linearly, exactly as `tt_dot` would have over the dequantised row. The buffer
lives in L1 and never spills; the cost this kernel removed was the 512 floats
per row, not this.

`tt_axpy_int4g` needs no such care — axpy is elementwise, with no reduction.

### The proof

Test 12 of `tt_kernels_selftest` checks the fused kernels against the exact
sequence they replace, **bit for bit, no tolerance**, in both the scalar and the
SIMD implementation. The axpy accumulator starts dirty on purpose: a kernel that
stored instead of accumulating would pass with it zeroed.

End to end, against `ref32.json` on the real 19 GB container:

- **32/32 argmax tokens identical** — the docs/15 exit criterion.
- **The full logit dump is identical bit for bit**: 19.824.640 bytes,
  4,95 million floats, same SHA-256 before and after
  (`23F8AB0F…`). Verified for the SIMD path and, separately, for
  `--kernel scalar` (`5F5C4FC1…`).

That is stronger than the rule requires. The rule asks for 32 matching tokens;
this is every float the model produced, unchanged.

## What was looked at and left alone

**The attention dispatch grain.** `attention_absorbed` calls `tt_pool_run` with
`rows = num_attention_heads` and `grain = 1`, which looks like the fine
granularity the profile complains about. It is not: 20 heads over 24 threads,
each head an indivisible unit of work. Raising the grain would cut parallelism,
not barrier cost.

**`--threads 24` on a hybrid part.** The profile's blocks are equal-sized and the
cores are not, so the barrier waits for whichever E-core drew the last block.
Change 1 makes that wait cheap, but it does not make it shorter.

## The affinity, spelled out

Worth writing down, because the topology is not what the marketing number says.
`GetLogicalProcessorInformationEx` on this i9-14900K reports **22 physical cores
and 28 logical**:

| class | cores | SMT | first logical |
|---|---|---|---|
| 1 (P) | **6** | yes, 2 threads each | 0, 2, 4, 6, 8, 10 |
| 0 (E) | 16 | no | 12 … 27 |

Six P-cores, not eight. `enum_cores` takes the **first logical of each physical
core** and never the SMT sibling — the sibling would compete for the same
execution unit, and these kernels saturate FMA.

`tt_cmd_serve` builds the pool with `TT_AFFINITY_ALL`, so the list is all 22
cores, and each worker is pinned to exactly one logical CPU with a one-bit mask
(`tt_thread_pin` → `SetThreadAffinityMask`). Assignment is `cpus[(i + 1) % ncpu]`,
which gives:

| `--threads` | CPUs used | doubled up | P | E |
|---|---|---|---|---|
| 20 (the default) | 20 of 22 | 0 | 6 | 14 |
| **24 (what was profiled)** | 22 of 22 | **2** | 8 | 16 |

So 24 is not "more parallelism than 20": it is 20 plus two cores carrying two
pinned threads each, while the barrier waits for them.

It costs a great deal. Measured, same binary, same protocol, three runs each:

| `--threads` | median | mean | sd | samples | vs 24 |
|---|---|---|---|---|---|
| 6 | 14,92 | 14,89 | 0,07 | 14,81 · 14,92 · 14,94 | −5,8% |
| 12 | 18,30 | 18,29 | 0,11 | 18,18 · 18,30 · 18,40 | +15,5% |
| **20** | **21,55** | 21,63 | 0,15 | 21,54 · 21,55 · 21,81 | **+36%** |
| 22 | 20,43 | 20,35 | 0,28 | 20,03 · 20,43 · 20,58 | +29% |
| 24 | 15,84 | 15,29 | **1,27** | 12,98 … 16,22 | — |

**The default of 20 is the right one, and 24 costs 36%.** Look at the spread as
much as the median: 24 has sd 1,27 against 20's 0,15, with samples from 12,98 to
16,22. That is the signature of the two doubled-up cores — which two threads
collide, and whether the scheduler runs them back to back, changes run to run.

22 is worth its own line because "22 = the physical core count" is the obvious
guess and it is **not** the best: it loses 5% to 20. Two of those 22 pinned CPUs
are P-core *first* logicals whose SMT siblings sit idle, so the last two threads
add contention at the barrier rather than throughput. The engine already knows
the difference between a core and a place to run — the count that wins is the
one that leaves the barrier unblocked, not the one that matches a spec sheet.

Note also that 6 (P-cores only) is the *worst* of all: the 16 E-cores do
contribute, they just must not be waited on by cores that finished long ago.

`Tests/benchserve/sweep-threads.ps1` reruns this.

### What was NOT changed, and why

`tt_cmd_serve` hard-codes `threads = 20`, and on this box that is the winner —
but by coincidence, not derivation. The two candidates for a derived default are
both worse here:

- `tt_pool_create(0, …)` uses `ncpu`, every physical core: **22**, which measures
  5% below 20.
- P-cores only would be **6**, the worst of the sweep.

So there is no obvious expression to replace 20 with, and picking one from a
single machine's numbers would be fitting a constant to one sample. The number
stays where it is until the sweep has run on hardware that is not this one —
a uniform many-core server has no doubled-up cores and would likely want
`ncpu`. What this measurement does establish is narrower and solid: **on this
box, do not pass `--threads 24`.** The profile in this document was taken at 24,
which means it was taken in the contended regime; the spin it found is real and
the fix is real, but part of what looked like barrier wait was two cores running
two workers each.

None of this removes the real asymmetry: a P-core and an E-core get the same
`grain`, and `tt_pool_run`'s barrier waits for the slower. Choosing the thread
count picks a point on that trade-off; it does not eliminate it. Fixing it
properly means unequal blocks or work-stealing, and the constraint that governs
that is docs/15 — the partition is by output row and the reduction stays inside
one thread, so **`--threads` must not change a single token**. The sweep script
checks exactly that and shouts if it ever stops being true.

## What it bought

`Tests/benchserve/`, same box, same 30-word prompt, `--threads 24 --ctx 150000`,
`temperature 0`, server restarted per run. Both binaries produced **byte-identical
replies** (`747DDD26`), which is the harness confirming it compared the same work:

| binary | median | mean | sd | samples |
|---|---|---|---|---|
| before (`73CD8878`) | 14,04 | 14,08 | 0,11 | 14,00 · 14,04 · 14,21 |
| after (`89D0ED33`) | **15,62** | 15,50 | 0,73 | 14,72 · 15,62 · 16,17 |

**+11,2% on the median**, +10,1% on the mean, for a change that does not alter a
bit of the output.

Two things to be honest about. The spread after is 6,6× wider (0,73 against
0,11) and the three samples fall monotonically — 16,17 → 15,62 → 14,72 — which
looks like thermal or neighbour effects rather than the code; the conclusion
survives because the *worst* sample after still beats the *best* before, but
three runs is a narrow base. And the model hit EOS at 120 tokens rather than the
256 the harness asked for, so the decode sample is smaller than planned. Both
apply equally to both sides.

The profile predicted more. ~30,7% of CPU was in the spin, and removing most of
it returned ~14% of wall clock — the gap is the arithmetic: the spin was burning
CPU on cores that were *waiting*, and cores that wait do not hold up a
memory-bound decode in proportion to the CPU they waste. The remaining barrier
cost is the P/E imbalance, which change 1 makes cheap but does not shorten.

Compare medians with the spread in view. `docs/04` records why: the first
perplexity measurement in this project had the sign inverted because a global
mean swallowed the effect in the variance between windows.
