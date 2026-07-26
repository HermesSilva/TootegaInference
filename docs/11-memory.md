# 11 — Memory discipline and zero-copy

The int4 container is 19.3 GB and the machine has 64 GB. That sounds roomy, and
it is not: KV at 64k of context is 3.5 GB per slot, and the default budget leaves
16 GB for the OS ([09](09-resource-limits.md)). Every copy avoided is context
that fits.

General rule: **a byte of weight exists in exactly one place.**

## 1. The mapped file is the memory

Weights are read **in place from the file mapping**. There is no load step that
copies disk → buffer. The quantiser points straight into the mapped pages:

```c
qt->q4 = (uint8_t*)(base + tensor->offset);   /* no memcpy */
qt->s  = (float*)  (base + scales->offset);
```

The streaming engine has this path, and its comment describes the gain exactly:
*"This is pread's I/O without the copy and without the slab."* The difference
here is that **it is the only path**, not an alternative to a copying one.

Consequences:

- **The OS page cache is the cache.** No double buffering, no hand-managed weight
  residency.
- **Weight pages are reclaimable.** Under memory pressure the OS drops them and
  re-reads from disk. That is why they **do not count** against `--ram-gb`
  ([09](09-resource-limits.md)): they are file-backed, not anonymous.
- **Several processes share the same pages.** Two engine instances on the same
  model cost one set of weights, not two.

**A requirement that flows back to the converter (M0):** the int4 container has
to emit tensors **aligned to 64 bytes**, or direct SIMD reads need a copying path
to realign — and then the whole zero-copy story is lost. The streaming engine
trips on this and falls back to `pread` when `(off & 3)`. Our format guarantees
alignment on write.

## 2. Pre-fault and lock, within budget

Mapping brings nothing into RAM. The first access to each page is a fault. At
load time:

1. `madvise(MADV_WILLNEED)` / `PrefetchVirtualMemory` — starts asynchronous
   reading;
2. touch one word every 4 KB — guarantees residency;
3. **optionally** `mlock` / `VirtualLock` on the resident set, bounded by
   `--ram-gb`, so the OS does not evict the model while something else asks for
   memory.

Step 3 is opt-in because locking ~18 GB on a working machine is the person's
decision, not the engine's.

## 3. Huge pages

The working set is 19.3 GB. With 4 KB pages that is **4.7 million pages**; Raptor
Cove's L2 TLB has ~2048 entries, covering 8 MB. Each token walks ~2 GB of
weights, so practically every access pays a page walk.

With 2 MB pages, those same 2048 entries cover **4 GB** — within the order of
magnitude of what a token touches.

```
4 KB:  TLB coverage = 2048 x 4 KB  =    8 MB
2 MB:  TLB coverage = 2048 x 2 MB  = 4096 MB
```

Linux: `MADV_HUGEPAGE` (THP) over the mapping. Windows: large pages require the
`SeLockMemoryPrivilege` privilege, so it is opt-in with a clear instruction.

**Marked to be measured in M5**, not assumed. The hardware prefetcher and
sequential access soften the cost; the real gain could be 5% or 25%, and the only
way to know is to measure with and without.

## 4. Scratch arena, zero `malloc` per token

Every forward buffer — norms, residuals, logits, intermediate expert outputs —
has a fixed size, known after reading the config. They are slices of **an arena
sized at init**.

The token loop never calls the allocator. Not once.

The streaming engine does ~500 `malloc`/`free` per token
([06](06-lessons-learned.md)); at 0.93 tok/s that is 0.01% and does not matter.
At 22 tok/s it matters, and the fix is structural, not a micro-optimisation to do
later.

## 5. Paged KV with shared blocks

This is the project's biggest memory saver, and it is born of the agentic use
case.

KV allocated in **blocks of N tokens** (N = 256, to be settled by measurement),
with a per-slot block table. No `realloc`, no contiguous per-session buffer.

The gain comes from sharing: in an agentic CLI, **every session starts with the
same prefix** — system prompt, tool schemas, project instructions. Typically
10–20k tokens. At 52.9 KB per token ([08](08-performance-ceiling.md)):

```
20k-token prefix = 1.06 GB of KV

4 sessions, private blocks:  4.24 GB
4 sessions, shared blocks:   1.06 GB
```

Blocks are immutable once written and reference-counted. When a session diverges
from the prefix, it starts allocating its own blocks from the divergence point on
— copy-on-write at block level, not token level.

This interacts directly with the prompt-ordering rule from
[03](03-architecture.md): **stable first, volatile last** is not just cache
hygiene, it is what makes the blocks shareable.

## 6. Quantised on disk, quantised in memory

There is never a "dequantise to fp16 and then multiply" step. The kernels consume
int4 with per-group scales directly. An intermediate fp16 buffer for the active
experts would cost 4× the reads and destroy the arithmetic in
[08](08-performance-ceiling.md).

Corollary: fuse operations that would re-read the same input. An expert's `gate`
and `up` read the same `x` — one pass, two outputs. The streaming engine measures
~33% of expert matmul time in that.

## 7. In-place operations

- `rmsnorm` writes over its own buffer when the original is no longer needed;
- the residual accumulates into `x`, with no third buffer;
- the router's softmax is in place over the logits.

No layer allocates. No layer copies the residual stream.

## 8. Embeddings stay in the file, `lm_head` stays resident

The embedding table has 154,880 × 2048 entries, and **one row is read per
token**. Not worth residency: it stays mapped, the OS handles it.

`lm_head` is the opposite — it is read **in full** on every token (0.317B params,
8.9% of the traffic). Resident, mandatorily.

The distinction looks obvious written down like this, and it is exactly the kind
of thing a naive loader gets wrong by treating "dense tensors" as one block.

## 9. GPU/CPU boundary: 8 KB, fixed buffer

In the per-layer partition ([03](03-architecture.md)) only activations cross:
`hidden × 4 bytes = 8 KB`. The host buffer is **pinned and reused**, allocated
once. Weights **never** cross PCIe — that is the law from
[02](02-hardware.md).

## Accounting

| region | counts in `--ram-gb`? | why |
|---|---|---|
| model mapping | **no** | file-backed, reclaimable by the OS |
| pages locked with `mlock` | **yes** | anonymous in practice |
| KV blocks | **yes** | anonymous |
| scratch arena | **yes** | anonymous |
| pinned GPU buffer | **yes** | anonymous and non-pageable |

A `--stats mem` prints the five lines with the real measured value, not an
estimate. Without that, "memory budget" is an opinion.
