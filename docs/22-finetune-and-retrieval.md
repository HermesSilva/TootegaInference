# 22 — The domain container, and why retrieval became a requirement

A domain adapter was trained for this engine: GLM-4.7-Flash tuned on a
three-volume work of philosophy in Portuguese, merged and quantised into
`glm47flash-v5-sft.tootega` (19.32 GB, the same shape as the v4 container).

The pipeline lives on the `finetune` branch under `finetune/`, is Python because
training requires PyTorch, and ran on rented hardware. It is not a tool of the
engine — it is a project that feeds one, and what crosses the boundary is a
container. `finetune/README.md` has the recipe; this document records what the
work changed **here**.

## The measurement that changes `src/`

Two ways of training were tried, and the difference is the point.

**Continued pre-training on the raw text failed.** It is the obvious approach:
take the corpus, keep predicting the next token, let the weights absorb the
work. Holdout perplexity fell **45%**, paired per window, t = −40.8, 61 of 61
windows improved — by the criterion of [04](04-target-model.md) an unambiguous
win. And the model was wrong: it stated that a creator produced the work's
central entity, which the first essay spends sections denying.

**Perplexity measures register, not knowledge.** The paired measurement that
[04](04-target-model.md) mandates protects against comparing noise; it does not
protect against a model that learned how a text *sounds*. Nothing in the
numbers distinguished the two, and only reading generated answers did.

**RAFT worked** — answers written with the retrieved section in context, loss
masked to the answer. And it produced a model with a deliberate asymmetry:

| context | doctrinally correct |
|---|---|
| passage present | **26 of 29** |
| no passage (closed book) | **0 of 11** |

Asked without context, the trained model says *Arena* is a football stadium and
cites a dictionary. This is not a defect: RAFT trades memorisation for reading,
on purpose. But it means **the engine cannot serve this container without a
retriever**, and that is new work in `src/`.

## What that requires

**BM25, not embeddings.** The choice was made for this reason and not for
quality: BM25 is an inverted index and two divisions — it ports to C11 with no
dependencies, and stays auditable in one reading session. An embedding retriever
would drag a second neural model into `src/` and end the property the project is
built on. `finetune/scripts/retrieve.py` is the reference implementation:
`k1=1.5`, `b=0.75`, sections as the unit, accents folded (`unicode_tables.h`
already does this).

**An index artefact.** A tool under `tools/index/` producing a `.ttindex` — the
sections tokenised, the inverted index, the IDF and the lengths. The `.ttpack`
format of `tools/finetune` is a good precedent: header, aligned blocks, JSON
manifest at the end.

**The wiring in `server.c`.** The passages go into the user turn before
`CHAT_ASSIST`, in the exact shape the model was trained on:

```
Trechos da obra:

[1] <essay title> — <section>
<section text>

---
Pergunta: <question>
```

Getting that shape wrong is equivalent to not having trained.

> **The index must carry every section.** Six essays were held out of training
> and of generation, correctly — they are what proved the model generalised
> rather than memorised. Excluding them from the *index* would be a different
> thing entirely: the chat would answer "the work does not treat that" about
> death and the Fecho, which it treats in a whole essay. Build the index with
> all 725 sections plus the 223 term-index entries.

## Numbers, and one about `pool.c`

Measured on an EPYC 9535 (2 sockets, 64 physical cores each) against the int4
container, ~1.1k tokens of retrieved context:

| | |
|---|---|
| prefill | ~70 s |
| decode | 10–14 tok/s |
| resident | 16.8 GB |

Decode clears the 10 tok/s failure criterion of [01](01-goals.md). Prefill
dominates, and it is structural for RAFT rather than incidental: the retrieved
passages change per question, so the prefix cache of `tootega chat` cannot help.
Time to first token is a declared non-goal for M1–M3 — this is the first
workload where that stance has a visible cost.

**And the engine does not scale past ~12 effective cores.** `--threads 128` is
*slower* than `--threads 16` on that box, holding 954–1346% CPU out of 12800%
available with three quarters of the machine idle. D6 settled the threading
model on our own pool, measured on an 8 P-core desktop where this cannot appear.
It is not a refutation — nothing about the target machine changed — but it is
evidence that the pool's barrier-per-layer shape does not survive a NUMA box,
and it should be recorded before someone measures on a server and is surprised.

## The delta survives int4

The adapter changes attention and the shared expert by **1.25% of the weight
norm** — the same order as int4 group-64 quantisation error, so there was a real
possibility that quantising would erase what training added.

It did not. The quantised container gives the same doctrinal answers as bf16 on
every question tested, including the one the base model gets backwards. The
merge is tensor-by-tensor over the safetensors shards, in fp32, because
`Glm4MoeLiteForCausalLM` does not build the MTP layer and a merge through the
loaded model would silently drop 1.28 B parameters — and the container declares
`mtp_layer: 47`.

## What this does not settle

- Regression on general capability was never measured. The choice was depth over
  breadth, deliberately; the price is unknown.
- The model yields to loaded premises — asked whether one entity is a projection
  of another it agrees, then two questions later states the opposite correctly.
  It knows and does not hold. That is training data, not inference.
