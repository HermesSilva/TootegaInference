# 05 — Roadmap

Every milestone has a **measurable exit criterion**. No number, no close.

All measurements on: GLM-4.7-Flash int4, 4k context, fixed prompt, greedy, this
machine, fixed RAM budget. The lesson from [07-baseline.md](07-baseline.md) is
that comparing two measurements taken under different configurations means
nothing.

---

## M0 — Convert and validate the model

Without this there is nothing to run.

- BF16 → int4 g64 converter, streaming shard by shard (never both sides in
  memory)
- **Tensors aligned to 64 bytes on write** — without it there is no zero-copy
  read with SIMD ([11](11-memory.md))
- `transformers` oracle: 32 teacher-forced positions, ids saved in `ref.json`
- Per-tensor precision ablation (the table in [04](04-target-model.md) is a
  hypothesis)

**Exit:** a ~19.3 GB int4 container on disk, aligned; oracle generated; int4
perplexity measured against BF16 on a fixed set, with the cost documented.

**Risk:** quantisation degrading beyond acceptable on some tensor. Mitigation:
the ablation exists precisely to find that before any kernel is written.

---

## M1 — CPU-only forward, token-exact

The milestone that decides whether the project is viable.

- safetensors loader: single mmap, weights read in place, zero copy
- Config → IR (attention and routing as variants, only MLA + `n_group=1` on day 1)
- CPU kernels: int4 g64 matmul with AVX-VNNI, MLA attention, rmsnorm, partial RoPE
- Scratch arena sized at init — **zero `malloc` in the token loop**
- Immutable config read once — **zero `getenv` in the layer loop**
- KV with prefix matching and truncation at the divergence
- Budgets `--ram-gb`, `--threads`, `--affinity` and the `--plan` command
  ([09](09-resource-limits.md))

**Exit:**
1. **32/32 positions identical to the oracle** — it blocks everything else;
2. **≥ 10 tok/s** at 4k context (≥ 37% of the 27 roofline);
3. stable RSS between turns — no per-token growth;
4. **`--ram-gb` respected and verified at runtime**; `--plan` refuses what does
   not fit with a message that says what to reduce.

**Desirable goal:** ≥ 15 tok/s (~55% of the roofline).

**Kill:** below 10 tok/s the bandwidth premise is wrong. Stop and redo the
arithmetic before writing more code.

---

## M2 — Per-layer GPU/CPU partition

Validates the central hypothesis of [02](02-hardware.md) and
[03](03-architecture.md).

- Planner: measure each link's bandwidth at init, resolve `k` (how many layers on
  the GPU)
- sm_75 kernels mirroring the CPU ones, same results
- Boundary: 8 KB of activation per crossing, weights **never** cross PCIe

**Exit:**
1. **Byte-identical** output between CPU-only and split — placement must not
   change tokens;
2. **≥ 20% above M1** on the same test;
3. `--vram-gb` and `--gpu-layers` respected, peak verified at runtime;
4. **`CUDA_DENSE` regression test**: asking for more VRAM than exists must exit
   with an actionable error, not yield 0.054 tok/s.

**Kill:** gain below 20% → M2 is **cancelled**, not postponed. Two backends is a
permanent maintenance cost and it has to pay for itself.

---

## M3 — Server and continuous batching

Where the second goal of [01](01-goals.md) lives.

- Independent KV slots, per-slot admission
- Continuous batching in decode: each active slot contributes one row
- OpenAI-compatible API; loopback by default, key required outside it

**Exit:**
1. Aggregate throughput at K=4 **≥ 2.5× that of K=1** (against the `N^0.27`
   measured under streaming, which gives 1.45×);
2. Per-request latency at K=4 no worse than 2× that of K=1;
3. Prefix matching between turns with **zero prefill** when continuing a
   conversation.

---

## M4 — Agentic CLI

- Tool loop, local execution, approval for destructive actions
- Prompt assembly following the ordering rule from [03](03-architecture.md):
  stable first, volatile last
- A channel for injecting late state **at the end** of the prefix

- Persistent state: index, plan, tokenizer and prefix KV blocks
  ([12](12-persistent-state.md))

**Exit:**
1. One SWE-bench-lite pass (or an in-house subset) completed end to end;
2. **Prefix cache hit rate ≥ 90%** in a multi-turn session — it measures whether
   the prompt ordering is right, and it is what separates a usable CLI from one
   that recomputes 100k tokens per turn;
3. **Warm start ≤ 3 s** against ~60 s cold, with the 20k prefix already loaded;
4. No state leaking between sessions — a KV block is addressed by the hash of the
   exact sequence, never by slot.

---

## M5 — Optimisation, with a number

Only after M4, and every item lands with its before/after in the same
configuration:

- MTP speculation — prior experience says it only pays near full residency; we
  are in that regime here, so **it is the first to try**
- Compute/read overlap, if M1 shows there is none
- W4A16 tensor-core kernels on sm_75
- KV reuse between sessions sharing a prefix

---

## Out of scope

Multi-GPU. Multi-host. Training. Architectures other than MLA + `n_group=1` —
the variation points exist in the IR from M1 on, but GQA gets implemented only
when there is a concrete model worth it.
