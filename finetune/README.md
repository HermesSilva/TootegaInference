# finetune/ — the RAFT pipeline that produced the v5 container

Python, and outside `src/` on purpose. Training requires PyTorch, and the
project's two-exceptions rule covers the engine's path, not this. These scripts
never run on the machine the engine runs on: they ran on a rented GPU box and
produced two artefacts, and the artefacts are what the project consumes.

Published (private) at `HermesX21/cosmologia-composicional-lora`:

| artefact | what it is |
|---|---|
| `glm47flash-v5-sft.tootega` | 19.32 GB int4 container — what TootegaEngine reads |
| `adapter_model.safetensors` | the LoRA itself, 59.1 M params, for re-merging or further training |

## What was measured, and what it cost to learn

**Continued pre-training on the raw text failed.** It was tried first, is the
obvious thing to do, and produced a model that *sounds* like the books while
contradicting them: it invented a creator above the work's central entity —
the exact claim the first essay spends sections denying. Holdout perplexity
fell 45% (t = −40.8, 61/61 windows improved) and the propositions were still
wrong. **Perplexity measures register, not knowledge.** `train_lora.py` is that
run, kept because the number is real and the conclusion cost a day.

**RAFT worked.** Answers written with the retrieved passage in context, trained
with the loss masked to the answer. Validation loss 0.889 → 0.348 over two
epochs, and — the part that matters — the generative check shows the doctrine
corrected: the invented creator is gone, the individual Mind no longer dissolves
at death, and it declines on subjects the work does not treat.

**The model reads; it does not remember.** Of 11 closed-book candidate answers
reviewed by hand, **0** were doctrinally correct; of 29 with the passage
present, **26** were. The trained model, asked without context, says a technical
term of the work names a football stadium. This is not a defect to fix — it is
the division of labour RAFT produces on purpose, and it makes **BM25 retrieval a
requirement of the engine**, not an accessory. `retrieve.py` is the reference
implementation to port to C.

**The LoRA delta survives int4.** 1.25% of the weight norm against group-64
quantisation error; the quantised container gives the same doctrinal answers as
bf16. This was the risk that could have invalidated everything downstream.

## Pipeline

```
manifest.py     corpus manifest: 87 docs, citations extracted from the numbered
                sections the books already carry (never invented)
retrieve.py     BM25 over 725 sections -- oracle + confusable distractors
gen_raft.py     4 stages: questions -> grounded answers -> judge -> refine
refilter.py     re-applies the mechanical filters to the audit file, no GPU
suspect.py      flags definitions that describe a neighbouring term
doctrine_check.py  agency attributed to a term the index defines as agency-less,
                refused terms used affirmatively, the individual Mind dissolving
export_review.py   pages for reading by hand, glosses ranked by rarity
judge_api.py    external judge (Claude), the author's 223 index entries cached
                in the system prompt
train_sft.py    the SFT: 868 pairs, LoRA r=32 on 376 modules, 2 epochs
merge_lora.py   merge into the bf16 shards, tensor by tensor
eval_sft.py     evaluation against the failures that were actually measured
engine_eval.py  the same questions through the C engine on the int4 container
```

`gen_sft.py` and `train_lora.py` are the superseded first attempt (self-instruct
without RAFT, and continued pre-training). Kept for the record.

## Three traps, each of which cost a run

**`experts_implementation`.** The transformers default (`grouped_mm`) needs
compute capability 9.0 and dies mid-forward on newer cards. `batched_mm` indexes
the expert weight *per token* — `gate_up_proj` is 12.6 MB per expert, so N
tokens at top-4 allocate `N × 4 × 12.6 MB`; a 1.7k-token prompt asked for 300 GB.
Use `eager` for long prompts, `batched_mm` only for short-prompt decode.

**PEFT rewrites the adapter config.** Any target ending in `.gate_proj` /
`.up_proj` / `.down_proj` is moved from `target_modules` into
`target_parameters`, pointing at the *fused routed experts*. Asking for the
shared expert hands back the 64 routed ones. With `lora_dropout != 0` it raises;
with 0 it trains the wrong tensors in silence. `patch_peft()` in `train_sft.py`
neutralises it, and it is needed **anywhere the adapter is touched** — training,
evaluation, and the eventual merge into the container.

**The MTP layer is not loaded.** `Glm4MoeLiteForCausalLM` builds 29.94 B of the
checkpoint's 31.22 B; the MTP head stays on disk. So `merge_and_unload()` would
silently drop it and produce a checkpoint the converter cannot turn into a v4
container, which declares `mtp_layer: 47`. `merge_lora.py` rewrites the shards
instead and leaves every non-target tensor — MTP included — byte-identical.

## Engine numbers (EPYC 9535, int4 container)

~1.1k tokens of context: prefill ~70 s, decode 10-14 tok/s, 16.8 GB resident.

**The engine does not scale past ~12 effective cores.** On a two-socket NUMA
box, `--threads 128` is *slower* than `--threads 16` and leaves three quarters
of the machine idle (954-1346% CPU of 12800% available). This never shows on an
8 P-core desktop; it is new information about `pool.c`.

Prefill dominates, and it is structural for RAFT: the retrieved passages change
per question, so the `chat` prefix cache does not help.

## What is not done

- BM25 in C inside `src/`, plus an index artefact and the wiring in `server.c`.
  The index must be built with **all 725 sections** — holdout exclusion was
  correct for curation and is wrong for the product.
- Regression on general capability was never measured.
- ~10% residual doctrinal noise in the dataset; 60 pairs were read by hand.
- The model yields to loaded premises: asked whether the individual Mind is a
  projection of the Universal Mind it agrees, then two questions later correctly
  states the opposite. It knows and does not hold.

## Environment

Rented box, Ubuntu 24.04, RTX PRO 6000 Blackwell (96 GB), torch 2.8.0+cu128,
transformers 5.14.1, peft 0.20.0. `env.sh` has the variables that matter: the
container root is a 20 GB overlay, so `HF_HOME` must not fall back to
`~/.cache`, and `expandable_segments` keeps the allocator from fragmenting
around 56 GB of resident weights.

## What is here, and what is not

Everything that cost GPU time or human judgement is in this directory:

| | |
|---|---|
| `data/raft_raw.audit.jsonl` | the 1,210 generated candidates with their context — **two hours of GPU**; regenerating is the expensive step, filtering and judging are not |
| `data/raft_raw.jsonl` | the same, after the generator's own filters |
| `data/candidates*.jsonl` | the filtering stages, so a filter bug costs seconds instead of a run |
| `data/sft_final.jsonl` | the 868 pairs actually trained on |
| `data/cosmologia-87.ttpack` | the corpus pack, round-trip verified |
| `data/manifest.json` | 87 documents with citations extracted from the numbered sections |
| `review/` | the review pages and the 40 verdicts judged by hand |
| `logs/` | every measurement of the session, verbatim |
| `train_result.json` · `cpt_result.json` | the SFT run and the failed continued-pretraining run |
| `eval_sft.json` · `eval_gen.json` · `engine_eval.json` | the generative evaluations, bf16 and int4 |

Deliberately absent, with the reason:

- **the merged bf16 checkpoint (59 GB)** — reconstructible in 2.3 minutes from
  the base checkpoint plus the adapter, via `merge_lora.py`. Storing it would be
  storing a derivation.
- **the base checkpoint (59 GB)** — public, `zai-org/GLM-4.7-Flash`.
- **the int4 container (19.32 GB)** — on the Hub, private, together with the
  adapter and the tokenizer the engine needs.

The rule: keep what a measurement produced, drop what a command reproduces.
