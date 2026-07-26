# 09 — Resource limits

The engine runs on a workstation somebody is using. It has to fit inside a
declared budget — of RAM, VRAM and CPU — and **refuse** what does not fit,
instead of accommodating it.

## Why, with evidence

Three things we measured on the streaming engine that this document exists to
avoid repeating.

**Accommodating instead of refusing costs 13×.** `CUDA_DENSE=1` with 10.9 GB of
dense tensors and 8.6 GB of VRAM did not error out: it accepted, and started
shipping weights over PCIe on every layer, on every token. Measured result:
**0.054 tok/s** against 0.684. An impossible configuration degraded silently.

**A floating automatic budget makes measurements incomparable.** The streaming
engine picked `RAM_GB=47.1` on one run and `60.5` on the next, depending on what
the OS reported at that moment. Two measurements taken that way say nothing about
each other — we had to redo the entire benchmark matrix with the value pinned.

**Auto-planning allocates wrongly.** At the best mark measured (0.929 tok/s),
VRAM was **55% idle** (4,525 of 8,192 MB) while RAM had 6.3 GB free. The planner
filled the wrong resource.

And there is the trivial case: the person wants to compile, edit video or play a
game while the engine runs. Thirty-two threads at 100% and 60 GB resident are not
acceptable as a default.

## Parameters

Precedence: **command-line flag > environment variable > config file > auto**.

### Memory

| flag | env | default | what it does |
|---|---|---|---|
| `--ram-gb N` | `TOOTEGA_RAM_GB` | auto | **hard** ceiling on the process's resident bytes |
| `--vram-gb N` | `TOOTEGA_VRAM_GB` | auto | hard ceiling on VRAM allocated by the engine |
| `--vram-reserve-gb N` | `TOOTEGA_VRAM_RESERVE_GB` | `1.5` | how much to leave free on the card for the CUDA context, the driver and the desktop |
| `--kv-gb N` | `TOOTEGA_KV_GB` | derived | KV ceiling; if omitted, derived from `ctx × slots` |
| `--ctx N` | `TOOTEGA_CTX` | `8192` | maximum context per slot |
| `--kv-slots N` | `TOOTEGA_KV_SLOTS` | `1` | independent contexts |

`auto` for RAM = available memory minus 8 GB of headroom for the OS, and the
resolved value is **printed in the start-up log**. Auto is a convenience; for any
measurement, pin it.

### CPU

| flag | env | default | what it does |
|---|---|---|---|
| `--threads N` | `TOOTEGA_THREADS` | physical P-cores | compute threads |
| `--affinity MODE` | `TOOTEGA_AFFINITY` | `p-cores` | `p-cores` \| `all` \| `none` \| explicit list (`0-15`) |

The 14900K is hybrid: 8 P-cores (16 SMT threads) and 16 E-cores. Spreading a
memory-bound matmul across the E-cores usually hurts, and the streaming engine
carries re-exec code precisely because pinning has to happen before `main`.
Default `p-cores`, with `all` available for measurement.

### GPU

| flag | env | default | what it does |
|---|---|---|---|
| `--gpu MODE` | `TOOTEGA_GPU` | `auto` | `auto` \| `none` \| device index |
| `--gpu-layers N` | `TOOTEGA_GPU_LAYERS` | auto | layers resident in VRAM; the rest on the CPU |

`--gpu none` is the hard off switch. The streaming engine has a trap here — on
Windows it **auto-enables** CUDA on a bare run, and only an explicit environment
variable suppresses it. Here: `auto` means "use it if it helps, as the planner
decides", and the decision goes into the log.

### Diagnostics

| flag | what it does |
|---|---|
| `--plan` | resolves the placement, prints it, **exits**. Exit code 1 if it does not fit |
| `--explain` | like `--plan`, plus the per-token cost of each tier and the bottleneck |

## Semantics: four rules

**1. A budget is a hard ceiling, not a suggestion.** If the plan does not fit,
the engine **exits with an error** naming what to reduce. It never degrades
silently, and it never counts on swap.

```
ERROR: does not fit in --ram-gb 12
  weights (int4)        18.1 GB
  KV (ctx 8192 x 1)      0.4 GB
  scratch + runtime      1.2 GB
  ------------------------------
  required              19.7 GB   exceeds by 7.7 GB
suggestion: --ram-gb 21, or --gpu-layers to move weights into VRAM
```

**2. What counts against each budget is enumerated, not implicit.** The streaming
engine's `RAM_GB` was ambiguous — dense weights, expert cache and workspace
entered in different ways. Here:

| budget | includes |
|---|---|
| `--ram-gb` | weights resident in RAM + KV in RAM + scratch arena + runtime |
| `--vram-gb` | weights resident in VRAM + KV in VRAM + kernel workspace |
| `--kv-gb` | attention state only, summed across slots, wherever it lives |

What does **not** count: the model file's mmap (file-backed pages, the OS
reclaims them under pressure) and OS buffers.

**3. The ceiling is verified at runtime, not only in the plan.** Planning
estimates are wrong sometimes. A watchdog thread compares actual allocation
against the budget; exceeding it is a bug, and the engine fails loudly in debug
builds, logs it and refuses new admissions in release. Without that, "budget"
becomes decoration.

**4. An explicit budget produces a deterministic plan.** Same budget → same
placement → comparable performance. That is what makes the benchmark
reproducible, and it is the direct fix for the floating `RAM_GB` problem.

## How the planner uses the limits

Input: bandwidths measured at init ([02](02-hardware.md)), the model's tensor
sizes, the declared budgets.

Output: for each layer, the tier that hosts its weights.

Objective: minimise `Σ (bytes_read_per_token_in_tier / tier_bandwidth)`, subject
to fitting **every** budget — and accounting for the **transfer link**, not just
the destination tier's bandwidth. That second part is what the streaming engine's
`plan` was missing, and it is why it recommended a VRAM tier that, when measured,
did not pay.

Consequence on this machine: with PCIe at 12 GB/s, the planner **never** proposes
streaming weights to the GPU. Either the layer lives there, or it is computed on
the CPU.

## Defaults for this machine

```
--ram-gb 48          # of 64, leaves 16 for the OS and for work
--vram-gb 6.5        # of 8, reserves 1.5 for the driver and the desktop
--threads 16         # the 8 P-cores with SMT
--affinity p-cores
--ctx 8192
--kv-slots 1
```

With those values: resident weights 18.1 GB + KV 0.4 GB + scratch ~1.5 GB ≈
**20.0 GB** of RAM (the container is 19.3 GB; **both** `embed_tokens` — the
model's and the MTP's, 1.27 GB together — stay file-backed and do not count, see
[04](04-target-model.md)).
There is wide headroom left — and that is what allows raising `--ctx` for agentic
use, which is where KV grows (52.9 KB per context token, see
[08](08-performance-ceiling.md): at 64k that is 3.5 GB per slot).

## What these limits do not do

- **They do not limit disk I/O.** The model is resident; the disk is only read at
  load time.
- **They are not security isolation.** They are cooperative budgets inside the
  process, not a sandbox. For an externally enforced limit, use Job Objects on
  Windows or cgroups on Linux.
- **They do not stop the OS from swapping** if the budget is badly declared. That
  is why the `auto` default leaves 8 GB of headroom.
- **They do not guarantee latency.** Reducing threads reduces throughput; the
  engine has no real-time scheduling.

## Exit criterion

It enters the [roadmap](05-roadmap.md) milestones:

- **M1**: `--ram-gb` and `--threads` respected and verified; `--plan` refuses
  what does not fit with an actionable message.
- **M2**: `--vram-gb` and `--gpu-layers` likewise; **the streaming engine's
  `CUDA_DENSE` case becomes a regression test** — asking for more VRAM than
  exists must exit with an error, not yield 0.054 tok/s.
