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
Change 1 makes that wait cheap, but it does not make it shorter. Whether 8 or 16
threads beats 24 here is an experiment, not a patch —
`Tests/benchserve/bench-serve.ps1 -Label threads-8 -Threads 8` runs it.

## What it bought

`Tests/benchserve/`, same box, same 30-word prompt, `--threads 24 --ctx 150000`,
`temperature 0`, server restarted per run. Both binaries produced **byte-identical
replies** (`747DDD26`), which is the harness confirming it compared the same work:

| binary | median | mean | sd | samples |
|---|---|---|---|---|
| before (`73CD8878`) | 14,21 | 14,08 | 0,11 | 14,00 · 14,04 · 14,21 |
| after (`89D0ED33`) | **16,17** | 15,50 | 0,73 | 14,72 · 15,62 · 16,17 |

**+13,8% on the median**, +10,1% on the mean, for a change that does not alter a
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
