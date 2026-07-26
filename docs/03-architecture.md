# 03 — Architecture

## Layers

```
┌─ Agentic CLI ─────────────────────────────────────────────┐
│  builds the prompt, calls tools, orders the context       │
│  the ONLY one that knows semantics (system/tools/history) │
└────────────────────────┬──────────────────────────────────┘
                         │  tokens + slot
┌─ Service layer ────────┴──────────────────────────────────┐
│  HTTP, sessions, admission, which session holds which slot│
└────────────────────────┬──────────────────────────────────┘
                         │  SUBMIT(slot, tokens, params)
┌─ Engine ───────────────┴──────────────────────────────────┐
│  placement planner · forward · KV · kernels               │
│  sees integers. Does not know what a "system prompt" is.  │
└───────────────────────────────────────────────────────────┘
```

The contract between service and engine is deliberately narrow: **give me
tokens; I reuse the longest identical prefix and do not recompute what I already
know.** Nothing more.

## The three caches and their owners

"Cache" is three different things in this system. Conflating them is the origin
of most design confusion.

| cache | holds | owner |
|---|---|---|
| **weights** | tensors per tier (VRAM / RAM) | **engine**, exclusively |
| **KV** | attention state per already-processed token | the engine owns the **mechanism**; the agent determines whether it **hits** |
| **prompt prefix** | the sequence you assemble | **agent**, exclusively |

The weight cache is invisible to the agent and must stay that way.

The prefix cache is 100% the agent's: the engine sees integers, it does not know
what a system prompt is. **The engine cannot fix a bad ordering.**

KV is where the boundary matters. The engine offers token-by-token prefix
matching that stops at the first divergence and truncates from there. The
consequence is harsh: **everything after the first changed token is
recomputed**. A timestamp at the start of the system prompt invalidates the
following 100k tokens, and the engine has no way to know it was irrelevant.

Hence the rule the CLI must follow, which is the whole design:

> **Stable first, volatile last.**
> `system → tool schemas → long context → history → current turn`

Never a clock, a UUID or a directory listing up front.

Two decisions borrowed from Anthropic's API that are worth copying:

- **Tools render at position 0.** Swapping, adding or reordering a tool
  invalidates everything — the worst cache killer in an agentic CLI with dynamic
  tools. The engine needs an *append* path, not a swap.
- **An instruction arriving mid-stream must not edit the front.** A channel for
  injecting late state (mode changed, file changed on disk) **at the end** of the
  prefix, never at the beginning.

## Engine modules

| module | responsibility | notes |
|---|---|---|
| `model/` | load safetensors, config → IR | single mmap, zero copy; no hard-coded `n_group` |
| `plan/` | measure bandwidths, resolve placement **within budget** | runs at init; see [02](02-hardware.md) and [09](09-resource-limits.md) |
| `budget/` | RAM/VRAM/thread ceilings, accounting, watchdog | hard ceiling verified at runtime, not only in the plan |
| `kernel/cpu/` | int4 matmul, MLA attention, rmsnorm | AVX-VNNI (`dp4a` equivalent), OpenMP |
| `kernel/gpu/` | the same, sm_75 | optional; DLL loaded at runtime |
| `sched/` | KV slots, continuous batching | where the aggregate-throughput goal lives |
| `kv/` | allocation, prefix matching, truncation | compressed MLA: 576 values/token/layer |
| `serve/` | HTTP, protocol | thin on purpose |

## The central bet: partition by layer

From the [hardware model](02-hardware.md): weights are large, activations are
tiny (8 KB with `hidden=2048`). Therefore **each layer keeps its weights
permanently resident on the device that computes it**, and 8 KB of activation
crosses between layers.

```
layers 0..k      -> weights resident in VRAM, computed on the GPU
    [boundary: 8 KB of activation cross PCIe]
layers k+1..46   -> weights resident in RAM, computed on the CPU
```

`k` comes from the planner: the largest `k` whose weights fit in usable VRAM,
minus the workspace and KV reserve.

This is the opposite of what the streaming engine does (moving *weights* to the
GPU on demand) and it is a direct consequence of PCIe gen 3. **Marked as a
hypothesis**: M2 exists to validate it against M1's CPU-only, and the kill
criterion is in [01-goals.md](01-goals.md).

## Data structures: the rules

Derived from what we measured on the streaming engine (see
[06](06-lessons-learned.md)):

**Moving a large block is a pointer swap, never a copy.** The streaming engine
gets this right with a struct swap and a slab recycled by capacity (`:1527`).
Keep it.

**Scratch is a persistent arena, not a `malloc` per token.** The streaming engine
does ~500 `malloc`/`free` per token on fixed-size buffers known at init. At
1 tok/s it costs nothing; at 50 tok/s it becomes a percentage. Every forward
buffer comes out of an arena sized once.

**Env read once, at init.** The streaming engine calls `getenv` inside the layer
loop — ~300 scans of `environ` per token. Config becomes an immutable struct at
init.

**Expert lookup is O(1).** A linear scan over the pin set costs ~150k comparisons
per token in the streaming engine. With full residency the problem nearly
disappears, but the index is a direct `layer × expert → slot`, with no search.

## Correctness

- **`transformers` oracle, teacher forcing, token-exact.** It blocks merges.
- **Placement must not change the output.** Running the same prompt CPU-only,
  partially on GPU, and with different tier configurations must give identical
  bytes. We checked this on the streaming engine and it passed — CPU and CUDA
  produced the same hash.
- **Determinism is per state, not global.** Two runs in the same state give
  identical output; different states may legitimately diverge. The test must fix
  the state, not assume that greedy is enough.
