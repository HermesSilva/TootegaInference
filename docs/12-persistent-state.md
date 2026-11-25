# 12 — Persistent state: starting warm

Reopening the engine should not redo work that has already been done. Four
artefacts go to disk, next to the model, and are validated on load.

```
<model>/
  .tootega/
    index          tensor index              ~2 MB
    plan           measured bandwidths + placement  ~4 KB
    tokenizer      pre-compiled merges       ~8 MB
    kv/
      <hash>.blk   KV blocks per prefix      ~1 GB per 20k prefix
```

## 1. Tensor index

The target model has ~9,500 tensors spread over 48 shards. Opening it means
reading each shard's JSON header, parsing it and building the map name → (file,
offset, shape, quantisation).

The streaming engine learned this the hard way: a linear scan over ~120k tensors
cost *"decine di secondi per token"* until it was replaced by open hashing. Here
the problem is smaller — 9.5k entries — but parsing the headers is still repeated
work on every open.

The persisted index is a **flat, mappable file, already sorted by hash**. Load =
`mmap` + validate header. No parsing, no allocation, no table construction.

## 2. Placement plan and measured bandwidths

Measuring RAM, VRAM, disk and PCIe bandwidth at start-up costs seconds. The
result only changes if the hardware changes.

Persisted: the measured bandwidths and the resolved plan, **keyed by**
`(hardware fingerprint, budgets, model hash, build id)`. Any divergence
invalidates it and remeasures.

`--plan` still recomputes every time; the cache serves the normal path.

## 3. Pre-compiled tokenizer

`tokenizer.json` is JSON with tens of thousands of merges. Parsing it and
building the BPE structures on every open is pure waste. Persisting the compiled
form — tables already in the layout the encoder uses — reduces opening to an
`mmap`.

## 4. Prefix KV: the one that really matters

This is the item that changes the experience of an agentic CLI.

Every session starts with the same prefix — system prompt, tool schemas, project
instructions. Typically 10–20k tokens. **That prefill is recomputed on every
process start**, and it is the dominant cost of the first turn.

The arithmetic, with the numbers from [08](08-performance-ceiling.md):

```
20k-token prefix
  corresponding KV               20,000 x 52.9 KB = 1.06 GB
  recompute (prefill ~400 tok/s)                  ~50 s
  read from the RAID0 (18.4 GB/s)                 ~58 ms
```

Three orders of magnitude. The KV blocks of the stable prefix
([11](11-memory.md)) are written once and mapped back on the next open.

We measured this working on the streaming engine:

```
[KV] resumed conversation from disk: 106 tokens in 0.0s (no re-prefill)
```

### And we also measured how it breaks

In the determinism battery, one run diverged from the others. The cause was
**not** placement — CPU and CUDA produced identical bytes. It was a KV persisted
from **another conversation**, whose prefix partially matched the new prompt. The
engine resumed from a state that did not correspond to that text.

Hence the hard rule:

> **A KV block is addressed by the cumulative hash of the exact token sequence
> that produced it.** Not by slot, not by session name, not by position.

A block is only reusable if the hash of the prefix up to it matches exactly.
There is no partial match: either the block corresponds to that sequence, or it
is ignored. That makes the class of bug we observed impossible.

### What exists today — the default, and measured

`src/kvstore.c`. `tootega serve` uses it without being asked.

| | |
|---|---|
| what is stored | the **fixed prefix** of a request: template opening, tool schemas and the leading `system` messages — everything before the first turn that is not `system` |
| where | `<model dir>/.tootega/kv/<hash of the token sequence>.ttkv`; `--kv-cache <dir>` puts it elsewhere, `--no-kv-cache` turns it off |
| when it is written | once, right after the prefill that computed it and before the first generated token; prefixes shorter than 256 positions are not worth a file |
| when it is read | when memory does not already hold that prefix: the first request after a start, or after another session displaced the cache |
| refused unless | same token sequence (hash, then the ids themselves), same container fingerprint, same build id, same attention formulation, KV precision and shape — each refusal is one log line naming the field |
| integrity | checksum over the payload; a file that fails after the cache was written to leaves the state **empty**, never half restored |
| writing | temporary file, flushed to the device, renamed |
| size | `--kv-cache-gb`, 8 by default. Before a write the least recently **used** files go until the new one fits — a load refreshes the file's time. A prefix larger than the whole budget is not written. Only `.ttkv` files are ever removed |

Measured on the reference machine, container v4, f32 cache, a 587-token system
prompt and a different question on each start:

```
first start, empty cache      599 tokens recomputed          file written in 0.05 s (60.6 MB)
second start                  587 read from disk in 0.03 s,  12 recomputed   -> request in 3.3 s
control, --no-kv-cache        599 tokens recomputed                          -> request in 38.6 s
```

**The reply of the second start is identical, byte for byte, to the control's**
(temperature 0). That is the criterion, and the reason this could become the
default: it is not an output change, so the 32/32 rule of
[15](15-kernels-and-numerics.md) has nothing to authorise.

Two honest limits of that measurement. It is one prefix of 587 tokens; the
20k case in the table above is still arithmetic. And the read is 0.03 s for
60 MB that the write had just left in the file cache — a cold read after a
reboot pays the disk, which the estimate above puts at ~58 ms per GB.

Two departures from the sketch at the top of this section, each deliberate:

- **The file is read, not mapped.** The cache is written by every token that
  follows; a mapping would dirty the file or be copied by page faults, 4 KB at a
  time.
- **The container fingerprint is a sample** — size, the whole index, and 256
  bytes from the middle of every tensor — not a hash of 19 GB, which would cost
  more than the prefill it saves on a short prefix.

And one consequence of the integrity rules worth saying out loud: **the build id
is part of the key, so rebuilding the engine invalidates every file.** They are
rewritten under the same name the next time their prefix is computed. For a user
that is invisible; for whoever is editing `src/` it means the first request
after each build pays the prefill again.

The 52.9 KB per position above assumed fp16; the default cache is f32, **108 KB
per position**, so a 20k prefix is 2.2 GB on disk and not 1.06.

`tootega selftest` covers the round trip in the three precisions, every refusal
and the eviction order. Only the f32 cache has been through the server end to
end; `chat` on the command line does not use the cache yet.
## Statistics: honesty about what is worth keeping

The streaming engine persists a heat map — of which experts your workload routes
to — and uses it to pin the hottest ones. **In that regime it makes complete
sense**: at 15% residency, choosing the right 15% is the difference between 0.65
and 0.93 tok/s.

With the model **entirely resident, that loses its function.** There is no expert
to promote; they are all already in memory. Recording expert heat to decide
placement would be copying machinery without copying the problem it solves.

What is still worth recording:

| statistic | for what |
|---|---|
| prefix cache hit rate | measures whether the CLI orders the prompt correctly — M4's exit criterion |
| tok/s per turn, TTFT, prefill vs decode tokens | telemetry; a comparison basis between versions |
| MTP draft acceptance | decides whether speculation pays, per workload |
| KV slot occupancy | sizing `--kv-slots` |

None of those feeds back into placement. They are observability, not policy — and
the distinction is deliberate, because assuming otherwise is what produced half
the streaming engine's complexity.

## Integrity rules

Every persisted artefact carries a header with:

- **magic + format version** — an unknown version is discarded, never read "in
  hope";
- **provenance hash** — of what it derives from (model file, tokenizer, engine
  build id, budgets in the plan's case);
- **content checksum**.

A mismatch in any field → the artefact is **discarded and regenerated**, with one
log line saying which field diverged. Never a silent migration.

Writing is **atomic**: temporary file, `fsync`, `rename`. The streaming engine
gets this right (`telemetry.h:168`, writes `.tmp` and renames) and it is the bare
minimum — an artefact truncated by a power cut must not become a corrupt index
the engine accepts.

## What is not persisted

- **Dequantised or reordered weights.** The container on disk is already the
  execution format ([11](11-memory.md)); any transformation at load time would be
  a copy to avoid, not a cache to create.
- **Divergent prefix KV.** Only the stable, shareable prefix is worth writing;
  each conversation's specific tail lives in memory while the session exists.
- **Nothing outside the model directory.** An `rm -rf .tootega/` returns the
  original state, and the next open is merely slower.

## Target open cost

| stage | cold | warm |
|---|---|---|
| tensor index | ~1–3 s | ~10 ms (mmap) |
| tokenizer | ~1 s | ~10 ms (mmap) |
| bandwidths + plan | ~3–5 s | ~1 ms (read) |
| weight pre-fault | ~1–3 s (18.4 GB/s) | same, the OS decides |
| 20k prefix prefill | ~50 s | ~60 ms (block mmap) |
| **total** | **~60 s** | **~2 s** |

**[estimate]** — becomes a measured exit criterion in M4.
