# 06 — Lessons learned

Principles that came from measuring disk-streaming engines on this same machine,
before a line of this one was written. No code was reused — what survived was the
**understanding**, and every item below was reimplemented from scratch in `src/`.

## Adopt

**Moving a large block is a pointer swap, never a copy.** Promoting a ~19 MB
expert into the cache swaps the descriptors; the evicted one's slab becomes the
next one's workspace. Zero `memcpy`, zero `malloc`/`free` per eviction.

**A buffer only reallocates when it does not fit.** In steady state the allocator
is never touched. That became the arena in `src/common.c`.

**Zero copy through mmap.** The quantiser's pointers point straight into the
mapped file, with page *pre-touch* instead of a copy — I/O without the copy and
without the intermediate slab.

**A single arena for the resident set:** weights bound as one arena per layer,
never freed individually. It kills fragmentation and per-expert syscalls.

**Do not zero a buffer that is about to be overwritten.** `calloc` on the hot
path is a `memset` nobody asked for.

**Fuse to avoid re-reading.** Computing gate and up in the same pass, reading `x`
once, is worth ~33% of expert matmul time in decode. It became
`tt_matmul_batch`.

**Token-exact validation against a `transformers` oracle.** It is what separates
an engine from a plausibility generator — and what caught the wrong RoPE
([17](17-diagnosis.md)).

**Refusing to start when RAM is short.** Seen happening: 3.3 GB available, and
the engine aborted instead of risking an OOM mid-generation. Correct behaviour.
It became `--plan` and `--ram-gb` ([09](09-resource-limits.md)).

## Throw away

All the machinery for hiding disk: **PILOT** and cross-layer prefetch,
**dual-SSD** routing, **`O_DIRECT`** tuning, a **learned pin set** (a persisted
heat map), a **per-layer LRU tier**, an LFRU eviction guard. It is most of such a
project's complexity budget and it is worth nothing when the model is resident.

An engine for a lean model is a fraction of the size — and that, not the speed,
is TootegaEngine's main structural advantage.

## Avoid

Defects observed in streaming engines, which only fail to hurt at ~1 tok/s. At
20+ tok/s they become a percentage.

**Hundreds of `malloc`/`free` per token.** One decode step reached ~500
allocations — three per token on the main path alone, one of them the size of the
vocabulary, and up to seven per layer in the predictor. All of fixed size, known
at init. → **persistent arena**, `src/common.c`.

**`getenv` inside the layer loop.** Reading and converting an environment
variable on every layer, on every token, gives ~300 linear scans of `environ` per
token. → **immutable config read once**, `src/model.c`.

**Linear searches where structure exists.** Finding a resident expert by scanning
pin and LRU lists, and another scan to choose the victim, cost ~150 thousand
comparisons per token. → **direct index**, and open hashing in the container.

**Architecture hard-coded into the parser.** A fatal error when `n_group != 1` or
when `kv_lora_rank` is missing reduces the universe of models to a single family
— DeepSeek with `n_group=8`, Qwen3 with GQA, anything without MLA are all
rejected. It turns "try another model" into an investigation that ends with no
option. → **config → IR, with the variants declared**
([03](03-architecture.md)).

**No hardening flags.** `-O3 -Wall -Wextra` and none of `-D_FORTIFY_SOURCE=2`,
`-fstack-protector-strong`, `-fPIE -pie`, `-Wl,-z,relro,now`, in a C engine that
consumes network input and third-party model files. Cost of fixing: one line.

## Method caveats we learned by measuring

**Do not change two variables at once.** We measured "native + CUDA" against
"release CPU" and the result said nothing, because the native build (AVX-VNNI)
and CUDA changed together. It had to be redone with the middle state isolated.

**Pin the RAM budget across configs.** Two measurements with different `RAM_GB`
are not comparable, and the "auto" value floats with whatever the OS reported.

**Warm up before measuring concurrency.** Without a warm-up turn, the cold tier's
cost lands entirely on K=1 and invents a scaling gain that does not exist.

**Hash the output on every measurement.** That is how we found out that CPU and
CUDA produce identical bytes — and that the only divergence came from a KV
persisted from another conversation, not from placement.

**A bad configuration may not fail, only degrade 13×.** `CUDA_DENSE=1` with
insufficient VRAM does not refuse: it accepts and ships weights over PCIe. The
planner must **refuse** what does not fit, not accommodate it.
