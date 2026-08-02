# 01 — Goals and scope

## What it is

An inference engine for **Mixture-of-Experts models that fit in the machine's
fast memory**, with an agentic CLI on top.

Two measurable goals:

1. **≥ 15 tok/s** single-stream with GLM-4.7-Flash in int4, 4k context, CPU only
   — and **≥ 25 tok/s** with the GPU/CPU partition, if it pays for itself.
   Baseline to beat: 0.93 tok/s from the streaming engine with GLM-5.2.
   Roofline: 27 tok/s CPU, 44 split ([08](08-performance-ceiling.md)).
2. **Aggregate throughput scaling nearly linearly** with concurrent sessions
   until memory bandwidth saturates. Baseline to beat: `N^0.27` measured under
   streaming.

The goals are ~55% of the roofline, not 80%. That is deliberate: a first version
of an engine does not reach a mature one's bandwidth efficiency, and promising
80% on paper is like the 1.12 GB active figure we had to correct.

The second goal is what justifies the engine for agentic use: a CLI opens
several sessions, and with resident weights batching **amortises** the weight
read across sequences instead of multiplying it.

## What it explicitly is not

**Not a disk-streaming engine.** All the machinery streaming builds to hide disk
— PILOT, cross-layer prefetch, dual-SSD routing, `O_DIRECT` tuning, a learned
pin set, a per-layer LRU tier — is most of its complexity budget and is **worth
nothing** when the model is resident. It stays out. If the model does not fit,
this is not the right engine.

**Not multi-GPU.** One GPU, one host. Sharding across cards is a different scale
of problem.

**Not a production server.** No multi-tenancy, no authentication beyond
loopback, no quotas. It is an engine plus a local CLI.

**Not model-agnostic on day 1.** See the next section.

## Principles

**Placement decided by measured bandwidth, not by capacity.** The
[hardware model](02-hardware.md) shows that this machine's PCIe is slower than
its disk. A planner that assumes "VRAM is fast, fill it" produces the
configuration we measured at 0.054 tok/s. The planner solves a per-token cost
against the bandwidths **measured at start-up**, transfer links included.

**Token-exact validation, non-negotiable.** Every forward is compared against a
`transformers` oracle under teacher forcing. That is what separates an engine
from a plausibility generator, and the streaming engine got it right — we keep
it.

**Architecture as configuration, not as a compiled-in assumption.** The
streaming engine hard-codes `n_group == 1` and mandatory MLA in the parser and
exits fatally outside that window. That is what reduced the universe of
compatible models to the GLM-5.x family and turned "try another model" into an
investigation that ended with no option. Here: config → IR, with attention (MHA
/ GQA / MLA) and routing (with or without a group limit) as declared variants.

Concretely, day 1 implements **only the MLA + `n_group=1` path** — the target
model's. The difference is that the variation points exist as explicit dispatch
from the start, so adding GQA means writing a variant, not rewriting the parser.

**Fits in a declared budget, or refuses.** The engine runs on a workstation
somebody is using. RAM, VRAM and threads are hard ceilings passed as parameters;
if the plan does not fit, the engine exits with an error naming what to reduce —
it never degrades silently. We measured the cost of accommodating instead of
refusing: 13×. Details in [09-resource-limits.md](09-resource-limits.md).

**Measure before optimising, and publish the number.** Every optimisation lands
with its before/after on the same machine. The streaming engine does this, and
that is why its code is auditable; the difference is that here the numbers go to
[07-baseline.md](07-baseline.md) with the full configuration, because we found
out that comparing two measurements with different `RAM_GB` means nothing.

## Performance non-goals

- **Time to first token is not a goal in M1–M3.** Prefill can be optimised
  later; decode is what defines the agentic experience.
- **A quantisation scheme of our own is not a goal.** int4 with per-group scales
  is the target format. Researching better schemes is another project.
- **Training, fine-tuning, LoRA: out of `src/`.** The engine infers; it does not
  train. A domain adapter was produced for it and is described in
  [22-finetune-and-retrieval.md](22-finetune-and-retrieval.md) — but it lives on
  the `finetune` branch, runs on rented hardware, and what crosses into the
  engine is a `.tootega` container like any other. No training code lands here.

## A goal that a measurement added

- **Retrieval is a requirement, not an accessory.** This was not in the original
  plan. The domain model reads well and remembers badly: asked without the
  relevant passage in context it answers that *Arena* — a technical term of the
  work it was tuned on — names a football stadium. BM25 over the corpus sections
  therefore belongs in `src/`, and it is there: `src/retrieve.c`, the index
  artefact from `tools/index`, and `--index` on `tootega serve`. See
  [22](22-finetune-and-retrieval.md); the number behind it is in
  `finetune/README.md`.

  It is worth being precise about what this does *not* open. The engine gained a
  BM25 index, which is two divisions over an inverted list; it did not gain an
  embedding model, a vector store or a chunking strategy. Everything a retriever
  usually drags in stayed out, for the reason in [13](13-open-decisions.md) D15.

## Failure criteria

The project is wrong if:

- CPU-only does not clear **10 tok/s** at 4k in M1 — below ~37% of the roofline
  means the bandwidth arithmetic is wrong and the whole premise falls;
- the M2 GPU/CPU split yields less than 20% over CPU-only — the complexity of
  two backends does not pay for itself and M2 is cut;
- output diverges from the oracle in any placement configuration — that is a
  correctness bug, not a performance one, and it blocks everything.
