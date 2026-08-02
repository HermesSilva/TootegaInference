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

> **Retrieval is built and it is off by default.** Everything below this line
> works and is verified; `--index` reaches it. It is not what `serve-book.cmd`
> does, because editing the user's question before the model sees it adulterates
> the prompt — [13](13-open-decisions.md) D17 has that decision and what it
> costs. Read this section as *the subsystem that exists*, not as *how the chat
> is served*.

## What that required, and what was built

**BM25, not embeddings.** The choice was made for this reason and not for
quality: BM25 is an inverted index and two divisions — it ports to C11 with no
dependencies, and stays auditable in one reading session. An embedding retriever
would drag a second neural model into `src/` and end the property the project is
built on. `finetune/scripts/retrieve.py` is the reference implementation:
`k1=1.5`, `b=0.75`, sections as the unit, accents folded.

```powershell
tools\index.exe build "models\Versão Completa" --out models\v5-sft\cosmologia.ttindex
.\serve-book.cmd
```

**`tools/index`, producing a `.ttindex`.** The sections, the inverted index, the
document frequencies and the lengths, plus the section text itself so the server
does not need the corpus folder at run time. Header, sections aligned to 64,
little-endian — the `.tootega` of [14](14-container-format.md) and the `.ttpack`
of `tools/finetune` are the precedents. 3.2 MB for these three books.

**`src/retrieve.c`** holds the tokeniser, the mmapped reader and the search. The
tool does not tokenise: it *calls* this file, which is the whole reason it is a C
tool linked against `src/` rather than a script. A builder that folded a word
differently from the searcher would produce an index that silently retrieves
nothing, and a model answering from memory looks exactly like a bad fine-tune —
which is the failure this document exists to describe.

**The wiring in `server.c`.** `--index <file.ttindex>`, and then every user turn
is rewritten before `CHAT_ASSIST` in the exact shape the model was trained on:

```
Trechos da obra:

[1] <essay title> — <section>
<section text>

---
Pergunta: <question>
```

Getting that shape wrong is equivalent to not having trained. *Every* user turn
and not only the last, because retrieval is deterministic: replaying turn 1 on
turn 3 retrieves the same passages, so the prefix is unchanged and the KV cache
survives the round. Rewriting only the last turn would move the divergence to the
first message and re-prefill the whole conversation every time.

When nothing is retrieved — a follow-up like *"e por quê?"* shares no term with
the corpus — the question goes through bare. Inventing three unrelated passages
would be worse than none, and the model was trained on distractor-only cases
where the honest answer is that the work does not treat it.

### Same ranking as the reference, or it is not the same retriever

| | `retrieve.py` | `src/retrieve.c` |
|---|---|---|
| sections | 726 | 726 |
| vocabulary | 13,465 | 13,465 |
| tokens indexed | 157,638 | 157,638 |
| top 5 on 12 queries | — | identical, scores included |

That equality is the deliverable and not a nicety. The reference chose the
passages every training pair was written against; a retriever that ranks
differently shows the model something other than what it was taught to expect.

**Two traps, both of which were nearly shipped.**

*The index must carry every section.* Six essays were held out of training and of
generation, correctly — they are what proved the model generalised rather than
memorised. Excluding them from the *index* would be a different thing entirely:
the chat would answer "the work does not treat that" about death and the Fecho,
which it treats in a whole essay. The index carries all **726** sections.

*The accent fold could not be hand-written.* A table covering Latin-1 and Latin
Extended-A is everything Portuguese needs and it silently cut every
transliterated citation the books make in half — `puruṣa` indexed as `puru`,
`upaniṣad` as `upani`, `satipaṭṭhāna` as `satipa`. Four terms of vocabulary apart
from the reference, and the only symptom would have been that asking about the
Upanishads retrieved nothing. `tools/ucd` now generates the fold from the Unicode
database — lowercase, canonical decomposition, non-spacing marks dropped, and
what survives kept only if it is one ASCII letter or digit — and `ucd --check`
verifies the committed header against it.

### Reasoning defaults off when an index is loaded

Every RAFT training pair closed `</think>` immediately. Opening it puts the model
in a state the fine-tune never covered: the first request through the HTTP
endpoint came back as an English scratchpad deliberating about a Portuguese
question. [13](13-open-decisions.md) D14 had already measured that reasoning
costs the prefix cache; here it also costs the answer. An explicit `think` in the
request still wins.

### A bug retrieval turned from a curiosity into an everyday event

`tt_state_match_prefix` reused the whole prompt when the whole prompt matched,
and sampling reads the logits of the last position — which live in the scratch,
not in the cache. A total match skipped the prefill entirely and left the scratch
holding whatever the previous request ended on, which is EOS. The reply came back
empty.

It needed the exact same prompt twice in a row, so it survived unnoticed. Then
retrieval made that ordinary: the same question retrieves the same passages, so
pressing send again reproduces the prompt byte for byte. The match is now capped
at `n - 1`, and a repeated question costs **0.11 s of prefill instead of 97 s**.

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
passages change per question. Time to first token is a declared non-goal for
M1–M3 — this is the first workload where that stance has a visible cost.

On the desktop, through `tootega serve` with the index and 12 threads:

| | |
|---|---|
| retrieval | 0.2 ms over 726 sections |
| prompt | ~1,300 tokens, of which ~1,150 are the three passages |
| prefill | 93–97 s cold, **0.11 s** on a repeated question |
| decode | 11.6–12.2 tok/s |
| turn 2 of a conversation | **1,295 of 2,651 tokens from the cache** |

That last row is why every user turn is rewritten and not only the last.
Retrieval is deterministic, so turn 1 replays identically inside turn 2's prompt
and the whole first exchange comes out of the cache. Rewriting only the last turn
would move the divergence to the first message and re-prefill everything.

### How many passages, measured rather than assumed

The defaults shipped as 3 passages of 1400 characters, which was a guess carried
over from the scaffolding. Two things say it is the wrong guess, in opposite
directions.

**Training used four passages of 2000 characters** (`MAX_PASSAGE` in
`gen_raft.py`, "4 passages ~ 3k tokens of context"). The average section is 2744
characters, so training kept 73% of it and the default keeps 51% — the model is
being handed about half the context it was taught to expect.

**And one passage was enough for the three questions tried.** Same container,
same questions, `--passages 1`:

| passages | prompt | prefill | the three answers |
|---|---|---|---|
| 1 | 424–450 tokens | **29–36 s** | doctrinally correct, 3 of 3 |
| 3 | ~1,300 tokens | 93–97 s | doctrinally correct |
| 4 × 2000 (the trained shape) | ~3,000 tokens | ~215 s | not measured |

So the passage count is not accuracy, it is **insurance against the ranking**.
When BM25 puts the right section first, one passage answers. When it does not —
*"de onde vem deus?"* ranks three "Diálogo com as tradições" sections above
`Ensaio I, §II — A Mente Universal não é Deus`, because those are where the work
quotes theism and BM25 rewards term frequency — the extra passages are what save
the answer, and with one there is no margin at all. The failure is silent: the
reply comes out fluent and wrong, with no citation, which is the tell.

### The cost is not RAFT, it is the unbatched prefill

Worth stating plainly, because the natural conclusion from the table above is the
wrong one. Injecting 1,300 tokens of context is ordinary; any engine that batches
its prefill does it in a second or two. It costs 93 s here because `tt_forward`
takes **one token per call** and the prefill loop in `server.c` runs it over the
prompt — prefill *is* the decode loop. The rates say so: 13.9 tok/s of prefill
against 11.9 tok/s of decode.

The batched matmul is already in `kernels.c` and `Tests/mmbatch.exe` measures it
at **3.0–4.1× above batch 4**, 410–455 GFLOP/s against 138. Nothing calls it from
the prefill path. See [13](13-open-decisions.md) D16.

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
