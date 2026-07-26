# 02 — Hardware model

This is the foundational document. The measurement below inverts the hierarchy
almost every inference engine assumes, and almost every architectural decision
falls out of it.

## The bandwidths, measured

| tier | bandwidth | how it was measured |
|---|---|---|
| VRAM (RTX 2060 SUPER) | **448 GB/s** | vendor specification |
| RAM (2× 32 GB DDR5-4800 dual channel) | **~60 GB/s** effective | 76.8 GB/s theoretical (4800 MT/s × 8 B × 2 channels) |
| Disk (RAID0 NVMe) | **18.4 GB/s** sequential | CrystalDiskMark 8.0.4, SEQ1M Q8T1 |
| **PCIe (gen 3 x16)** | **~12 GB/s** | 15.75 GB/s theoretical; `nvidia-smi` confirms gen 3 |

## The inversion

**PCIe is the slowest link in the machine.** Slower than the disk.

```
pcie.link.gen.hostmax = 5     <- the motherboard slot offers Gen 5 (63 GB/s)
pcie.link.gen.max     = 3     <- the GPU only speaks Gen 3
```

The motherboard (ASUS ROG STRIX Z690-A) offers PCIe 5.0 x16. The RTX 2060 SUPER
is Turing, from 2019, and is a Gen 3 endpoint. The link negotiates down to the
slower side.

**Direct consequence:** any weight that does not *live* in VRAM must cross a
channel slower than the SSD it came from. Streaming weights to the GPU is
strictly worse than computing on the CPU straight from RAM — 12 GB/s against
60 GB/s.

### This is not theory — it was measured

`CUDA_DENSE=1` in the streaming engine, which places dense tensors on the GPU
even when they do not fit in VRAM:

```
config              tok/s
cpu-native          0.684
cuda-experts4       0.647
cuda-tc-w4a16       0.639
cuda-dense          0.054   <-- 13x slower
```

The engine did not refuse. It accepted, and started shipping weights over PCIe
on every layer, on every token. **That is this machine's trap**, and any engine
that assumes "VRAM is fast, so move everything there" falls into it.

## The placement law

> Put on the GPU **only what fits there permanently**. Never stream weights over
> a link slower than the source tier.

Formally: choose the placement that minimises

```
cost_per_token = Σ_tier (bytes_read_in_tier / tier_bandwidth)
```

subject to capacity — and accounting for the **transfer link**, not just the
destination tier's bandwidth. That second part is what the streaming engine
omits: its `plan` recommended a VRAM tier that, when measured, did not pay.

## The corollary that saves the project

Weights are large. **Activations are tiny.**

For the target model, `hidden = 2048`. An activation at a layer boundary is
`2048 × 4 bytes = 8 KB`. Against ~4.7 MB per expert.

That opens a partition which does not pay the PCIe toll:

> **Partition by layer, not by tensor.** Each layer keeps its weights
> permanently resident on the device that computes it. Between layers, 8 KB of
> activation crosses, not megabytes of weights.

With ~7 GB of usable VRAM (8 GB minus context and workspace) and 18.1 GB of
layer weights (the container is 19.3 GB, but the 1.3 GB of `embed_tokens` stays
file-backed and never competes for VRAM — see [04](04-target-model.md)), about
**39%** of the layers land on the GPU and the rest on the CPU. Per token, PCIe
carries 8 KB per boundary — 2 boundaries if the cut is contiguous, so **16 KB
per token**. At 12 GB/s that is 1.3 microseconds. Irrelevant.

**[estimate]** Modelling Amdahl with 2.23 GB read per token (weights + KV at 4k
context — see [08](08-performance-ceiling.md)), 39% of the layers on the GPU:

```
CPU:  1.36 GB / 60 GB/s   = 22.7 ms
GPU:  0.87 GB / 448 GB/s  =  1.9 ms
                            ------
                            24.6 ms  ->  ~41 tok/s of roofline
```

Against ~27 tok/s of CPU-only roofline. **Marked as an estimate on purpose** —
it is the project's central hypothesis, and milestone M2 of the
[roadmap](05-roadmap.md) exists to validate or kill it. The gain is ~1.5×, not
2×; if the complexity of maintaining two backends does not pay for itself,
CPU-only is still ~20× the baseline.

## Ceilings by model class

Cost per token = active parameters × 0.5 byte (int4) ÷ host tier bandwidth.

| model | int4 | active/token | where it fits | ceiling |
|---|---|---|---|---|
| ~7B-A1B (OLMoE) | 4 GB | 0.65 GB | all of VRAM | ~690 tok/s |
| **GLM-4.7-Flash** | **19.3 GB** | **2.01 GB** | **RAM (+partial GPU)** | **27 CPU / 41 split** |
| ~30B-A3B | 16 GB | ~1.5 GB | RAM | ~40 tok/s |
| ~100B-A12B | 50 GB | ~6 GB | RAM, tight | ~10 tok/s |
| GLM-5.2 744B-A40B | 363 GB | ~12 GB | disk | **0.93 measured** |

The GLM-4.7-Flash number is exact (from the config, see
[08](08-performance-ceiling.md)); the others are rough estimates that **only
count expert FFN** — the same omission that underestimated the target by 60%
before we recomputed it. Ceilings are pure bandwidth: they ignore compute,
kernel overhead and the serial chain of layers.

## What the measurement says about buying hardware

From phase C of the scalability test (varying RAM_GB, GLM-5.2):

| RAM_GB | warm tok/s |
|---|---|
| 16 | 0.656 |
| 30 | 0.695 |
| 45 | 0.839 |
| ~60 (auto) | 0.929 |

3.75× the RAM → 1.42× the speed. From phase B, 4× the concurrency → 1.45× the
aggregate throughput. **Both axes give ~N^0.25.**

The scaling law here is not hardware, it is **residency fraction**. Multiplying
by N only approaches X·N over the stretch where N pushes you across the residency
threshold. Choosing a model that already fits is 20–50×; buying hardware is
~N^0.25.

If a GPU upgrade ever happens, the bigger gain is neither VRAM nor compute: it is
the **link**. Ampere/Ada speak Gen 4 (31.5 GB/s), Blackwell speaks Gen 5
(63 GB/s) — and 63 GB/s exceeds RAM bandwidth, which would invert the hierarchy
again and make streaming to the GPU a good deal for the first time.
