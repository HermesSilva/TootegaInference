# 07 — Baseline

Everything measured on this machine, with the streaming engine v1.1.1 and
GLM-5.2 int4 (383.7 GB). These are the numbers TootegaEngine has to beat.

Protocol: fixed prompt, `max_tokens=80`, `temperature=0` (greedy), `RAM_GB=45`
unless stated otherwise, SHA-256 of the answer recorded on every run.

## Single-stream

| build / config | prompt 1 | prompt 2 |
|---|---|---|
| release, `x86-64-v3` | 0.684 | 0.748 |
| **native, AVX-VNNI** | 0.684 | 0.748 |
| cuda-experts4 | 0.647 | 0.700 |
| cuda-tc-w4a16 (tensor cores) | 0.639 | 0.709 |
| cuda-tc-pipe2 | 0.641 | 0.700 |
| **cuda-dense** | **0.054** | aborted |

No GPU variant caught up with the CPU. **`CUDA_DENSE=1` degraded 13×** — the
10.9 GB of dense tensors do not fit in 8.6 GB of VRAM and the engine started
shipping weights over PCIe on every layer. The engine did not refuse; it accepted
and degraded.

## Repetition effect (warm cache)

Same prompt 3× in a row, same server:

| config | run 1 | run 2 | run 3 | gain |
|---|---|---|---|---|
| cpu-native | 0.709 | 0.861 | 0.856 | **+20.7%** |
| cuda-tc-w4a16 | 0.775 | 0.805 | 0.806 | +4.0% |
| cpu-direct (`DIRECT=1 PIPE=1`) | 0.827 | 0.858 | **0.867** | +4.8% |

`DIRECT=1` helps on the **first** run (0.827 against 0.709, +17%), not in the
warm regime — consistent, since `O_DIRECT` only benefits whoever reads from disk.

## Best mark

**0.929 tok/s** — GPU with a 4 GB expert tier + `DIRECT=1 PIPE=1`, warm:

```
VRAM  4525 / 8192 MB  (55%, 3.6 GB idle)
RSS   45.14 GB · free RAM 6.3 GB
```

**The GPU was not the scarce resource; RAM was.**

## Scalability

**Concurrency** (4 KV slots, warmed up first):

| K | aggregate | per request |
|---|---|---|
| 1 | 0.921 | 0.921 |
| 2 | 1.043 | 0.524 |
| 4 | **1.339** | 0.338 |

4× the load → **1.45×** the throughput. Exponent **0.27**.

**Memory** (same config, varying `RAM_GB`):

| RAM_GB | warm |
|---|---|
| 16 | 0.656 |
| 30 | 0.695 |
| 45 | 0.839 |
| ~60 (auto) | 0.929 |

3.75× the RAM → **1.42×**. Exponent ~0.26.

**Both axes give `N^0.25`.** The scaling law is not hardware, it is residency
fraction: with a 60 GB budget we cache ~15% of the 324 GB of cold experts.

## Verified correctness

A finding that counts as a requirement, not just an observation: **CPU and CUDA
produced identical bytes** (same SHA-256 of the answer), on different servers,
with different placements. The streaming engine's central promise — *placement
only ever decides speed* — holds.

The only divergence observed came from a KV persisted from **another
conversation**, whose prefix partially matched. Runs 2 and 3 being identical to
each other ruled out non-determinism from floating-point reduction under OpenMP:
the engine is deterministic **per state**.

That becomes a test in TootegaEngine: same placement → same bytes; different KV
state → **also** the same bytes (this last part was never verified in depth on
the streaming engine).

## Disk bandwidth not saturated

Each token needs ~78 × 8 × 19 MB ≈ 11.9 GB of experts. With ~15% resident, ~10 GB
come from disk per token. At 0.93 tok/s that is **~9.3 GB/s** — against
**18.4 GB/s** measured on the RAID0.

Half the disk bandwidth idle. Likely cause: `PIPE_WORKERS=8` does not keep the
queue deep enough (the array's RND4K Q1T1 is 73 MB/s against 2,058 MB/s at
Q32T16). We did not chase it — TootegaEngine does not read experts from disk.

## How to reproduce

The harness scripts belong to the streaming engine's repository, not this one:
`bench.ps1` (config matrix), `bench_repeat.ps1` (repetition effect),
`scaling.ps1` (phases A/B/C), `roofline.py` (theoretical ceiling). Raw results in
`bench_run.log`, `bench_repeat.log`, `scaling_run.log`.
