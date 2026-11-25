# TootegaEngine

An inference engine in **pure C11, with no dependencies**, for MoE models that
fit entirely in RAM.

An OpenAI-compatible HTTP server, an embedded web chat, an agentic client with
local tools, and 20 documents that explain **why** each decision is what it is —
with the number that backs it right next to it.

> **New here? Start with ForDummies.** The whole project explained from zero, no
> vocabulary assumed, in **English** —
> [TootegaEngine-ForDummies.en.html](ForDummies/TootegaEngine-ForDummies.en.html)
> — and **Português** —
> [TootegaEngine-ForDummies.pt.html](ForDummies/TootegaEngine-ForDummies.pt.html).
> 20 chapters, 277 terms with a hint on every one, 15 figures, and the 20
> numbered documents mirrored as HTML in
> [`ForDummies/docs-html/`](ForDummies/docs-html/index.html).
> Details in [ForDummies](#if-the-vocabulary-is-new-fordummies).

```powershell
.\build-all.ps1      # engine + tools, MSVC via vswhere
serve.cmd            # starts the server; the UI is at http://127.0.0.1:8080/
```

The model it runs is published ready to use:
**[HermesX21/GLM-4.7-Flash-tootega-int4](https://huggingface.co/HermesX21/GLM-4.7-Flash-tootega-int4)**
— GLM-4.7-Flash in int4 g64, 19.3 GB, in the container format this engine reads.

---

## What it does

| | |
|---|---|
| **Server** | `/v1/chat/completions` with SSE streaming, `tools`/`tool_calls`, CORS. Any OpenAI client connects |
| **Web chat** | embedded in the binary, no loose files: name, model directives and reasoning chosen on the entry screen |
| **Agent** | `tools/agent.exe` — `Read`, `Write`, `Edit`, `Glob`, `Grep`, `Bash`, with a sandbox and approval |
| **Cockpit** | the agent speaks the Claude Code CLI process contract: a UI written for that CLI works by changing the executable path |
| **Terminal** | `agent.exe --tty` — a command-line session, no UI at all |
| **Measurement** | perplexity, paired ablation, generation quality, roofline |

All of it in **15 `.c` files**, ~460 KB of executable, zero runtime DLLs.

## State

**M1 and M4 complete, on CPU.** Measured on this machine, not estimated:

```
memory bandwidth    69.4 GB/s         measured at runtime, not specified
traffic per token    2.38 GB          what a decode actually touches
roofline ceiling    29.1 tok/s        bandwidth / traffic

warm decode         ~50 ms/token      = ~19.8 tok/s
mean decode         17.12 tok/s       includes the cold first prompt
mean ttft           2.51 s
prefill             10.4 tok/s
RSS                 16.69 GB peak     36.2 GB free
idle server         0.2% of one core, 0.04 GB
```

Said with the same rigour as the rest: decode sits at **59% of the ceiling**. The
missing milliseconds are in attention and the barriers — not in the matmuls,
which already run at 90% of bandwidth.

Two things have changed the prefill since that block was measured, and neither
changes a bit of the output (same logits, same SHA-256, 32/32):

| | what | measured |
|---|---|---|
| **no `lm_head` on prompt positions** | a prompt position predicts an id the caller already holds; only the last one runs the 337 MB vocabulary projection | prefill **+9%** (13.5 → 14.8 tok/s, three interleaved pairs, 4–14%) |
| **the fixed prefix on disk** | the KV of system prompt + tool schemas is written once and read back on the next start — on by default | first request **38.6 s → 3.3 s** with a 587-token prefix; identical reply |

Decode is untouched by both. The block above is the reference state and has not
been re-measured as a whole.

Container **v4**. Generation quality against the original BF16 weights, twelve
prompts, greedy, all in one run:

| container | `match@ref` | `rep4` |
|---|---|---|
| BF16 (reference) | 96.0 | 7.3% |
| v3 | 32.4 | 5.3% |
| **v4** | **42.7** | 6.8% |

`match@ref` is how many initial tokens match the BF16 generation before
diverging. The scale search bought a third more of them.

### What is proven

| property | result |
|---|---|
| forward against the independent oracle (`Tests/oracle`) | **32/32 identical tokens** |
| determinism | **byte-identical** logits from 1 to 24 threads |
| tokenizer against HuggingFace `tokenizers` | **36/36 cases**, exact round-trip |
| kernels against the reference quantisation | **bit for bit** |
| perplexity against an external reference | 8.63 against `transformers`' 8.89 |

### What does not exist

- **Slots and batching** (M3). One request at a time; the second tab waits in the
  TCP backlog, silently.
- **The rest of persistent state** — the prefix KV is on disk
  ([12](docs/12-persistent-state.md)); the tensor index, the plan and the
  compiled tokenizer are still rebuilt on every open — and **MTP speculation**.
- **Batched prefill.** The kernel exists and measures 3–4×; nothing calls it yet
  ([13](docs/13-open-decisions.md), D16).
- **A Linux build** — the `Makefile` exists and has never been compiled.
- **GPU.** It was written, measured, and **did not pay for itself**: it lives on
  the `GPU` branch.

## The thesis, in one paragraph

A 744 B MoE streamed from disk runs at **0.93 tok/s** on this machine, and
scaling hardware in that regime yields `N^0.25` — quadrupling RAM gives 1.4×. The
bottleneck is not the hardware, it is the ratio between model size and fast
memory. A modern, lean MoE (GLM-4.7-Flash, 19.3 GB in int4) **fits entirely in
RAM**, disk leaves the path, and the ceiling becomes memory bandwidth: the
measured 13.1 tok/s, against 0.93. TootegaEngine is the engine designed for that
regime — not for hiding disk.

The full arithmetic is in
[08 — Performance ceiling](docs/08-performance-ceiling.md).

## Getting started

### 1. Build

```powershell
.\build-all.ps1              # everything, with a build stamp and stale-binary detection
.\build-all.ps1 test         # + selftest
.\tootega.exe selftest       # kernels, JSON, pool, tokenizer, sampler
```

Windows is MSVC, found through `vswhere` — never a fixed path. Linux is the
`Makefile`, with gcc or clang.

### 2. Get the model

**The short way** — the converted container is published, so there is no reason
to download 62.4 GB and quantise it again:

```powershell
hf download HermesX21/GLM-4.7-Flash-tootega-int4 --local-dir models\glm47flash-int4
```

19.3 GB, plus `tokenizer.json`. Card, numbers and provenance at
[**huggingface.co/HermesX21/GLM-4.7-Flash-tootega-int4**](https://huggingface.co/HermesX21/GLM-4.7-Flash-tootega-int4).

**The long way**, if you want to convert it yourself — or with a different
precision policy, group size or clip grid:

```powershell
python download_model.py                           # 62.4 GB in BF16, resumable
.\tools\convert.exe models\glm47flash-bf16 `
    --out models\glm47flash-int4\model.tootega `
    --clip-grid 13                                 # ~19.3 GB in int4, 171 s
.\tools\convert.exe --verify models\glm47flash-int4\model.tootega
```

`--dry-run` prints the plan and the estimated size without writing anything.

### 3. Use it

```powershell
serve.cmd                                    # server + web UI
.\tools\agent.exe --tty                      # agent in the terminal
.\tootega.exe chat models\...\model.tootega `
    --tokenizer models\glm47flash-bf16\tokenizer.json
```

The server prints what comes in and goes out — **shape and statistics, never the
content of the messages**:

```
[2026-07-26 09:14:02] 127.0.0.1  POST /v1/chat/completions  (3386 bytes)

--- request ----------------------------------------------
  conversation 8 messages  (role x count x characters)
               system 1/1204   user 3/71   assistant 3/842   tool 1/2317
  tools        6 declared: Read, Write, Edit, Glob, Grep, Bash
  context      1172 tokens assembled, 148827 positions left (0.8% of the window)
  prefill      1163 of 1172 from cache (99%), 9 recomputed in 0.74 s
  prompt       perplexity 3.41  nll 1.2267  entropy 2.18 bits  confidence 47.3%
  call         Read -- 1 argument, 9 characters
  generation   perplexity 1.84  nll 0.6088  entropy 1.02 bits  confidence 71.9%
  end          tool_calls -- 11 tokens in 0.8 s (13.1 tok/s)
```

### The prefix cache

Every session starts with the same tokens — system prompt, tool schemas — and
recomputing them is the dominant cost of the first turn. `serve` keeps their KV
on disk, next to the model, and reads it back instead:

```
<model dir>/.tootega/kv/<hash of the token sequence>.ttkv     108 KB per position
```

| option | |
|---|---|
| *(nothing)* | on, in `<model dir>/.tootega/kv`, up to 8 GB |
| `--kv-cache <dir>` | somewhere else |
| `--kv-cache-gb N` | another ceiling; the least recently used prefix is evicted first |
| `--no-kv-cache` | off |

A file is used only for **exactly** the token sequence that produced it, with the
same container and the same build of the engine; anything else is ignored with a
log line saying which of them differed. Rebuilding the engine therefore
invalidates the cache — the files are rewritten the next time their prefix is
computed. Deleting the directory is always safe.

## Decisions worth knowing about

**MLA absorbed.** KV holds 576 values per token per layer instead of 10,240 —
18× less traffic. Without it, 150 thousand positions of context would cost
254 GB; they cost 16.2.

**Tools are translation, not invention.** The model already ships `<tool_call>`,
`<arg_key>` and `<|observation|>` as **single tokens** in the vocabulary. The
server translates the OpenAI protocol into that format — and the arguments are
typed by the JSON Schema the client sent, or every numeric argument arrives as
text on the other side.

**The server is dumb, the client is smart.** The tool loop lives in the agent,
not in the engine. That is what makes Copilot and Continue work with no code of
ours.

**Reasoning off by default.** Measured: with a 400-token budget the draft
**fails to finish 8 of 12 prompts**, repeats 8× more 4-grams, and costs 4.5× the
time. For *"What is the capital of Brazil?"* it writes six steps with three
alternative drafts — and is cut off before answering.

**Nothing lands without measurement.** MLA absorption, KV precision, GPU offload:
each had its criterion declared **before** measuring. What was refused is
documented with the number, not with an opinion.

## Documents

Written to be read in order. Start with **02**: it holds the measurement that
inverts the usual intuition, and almost every other decision derives from it.

| # | document | what it decides |
|---|---|---|
| 01 | [Goals and scope](docs/01-goals.md) | what it is and what it explicitly is not |
| 02 | [Hardware model](docs/02-hardware.md) | the measured bandwidths and the placement law |
| 03 | [Architecture](docs/03-architecture.md) | layers, contracts, the three caches |
| 04 | [Target model](docs/04-target-model.md) | GLM-4.7-Flash: what to implement |
| 05 | [Roadmap](docs/05-roadmap.md) | milestones with measurable exit criteria |
| 06 | [Lessons learned](docs/06-lessons-learned.md) | what to adopt, throw away, avoid |
| 07 | [Baseline](docs/07-baseline.md) | the numbers to beat, measured here |
| 08 | [Performance ceiling](docs/08-performance-ceiling.md) | roofline: how much can be extracted |
| 09 | [Resource limits](docs/09-resource-limits.md) | budgets, and refusing instead of degrading |
| 10 | [Implementation](docs/10-implementation.md) | C11, static linking, portability |
| 11 | [Memory](docs/11-memory.md) | zero-copy, paged KV, huge pages |
| 12 | [Persistent state](docs/12-persistent-state.md) | starting warm |
| 13 | [Open decisions](docs/13-open-decisions.md) | the decision scoreboard |
| 14 | [Container format](docs/14-container-format.md) | `.tootega`, byte by byte |
| 15 | [Kernels and numerics](docs/15-kernels-and-numerics.md) | W4A32, accumulation order |
| 16 | [Tokenizer](docs/16-tokenizer.md) | tiktoken BPE |
| 17 | [Diagnosis](docs/17-diagnosis.md) | the bugs, and why the tests missed them |
| 19 | [Reference state](docs/19-reference-state.md) | the snapshot of what exists and what was measured |
| 20 | [Agent and protocols](docs/20-agent.md) | tools, the CLI contract, the cost of reasoning |
| 21 | [int8 KV](docs/21-int8-kv.md) | measured, inconclusive, not adopted |

### If the vocabulary is new: ForDummies

[`ForDummies/`](ForDummies/) is the front door. It explains the whole project from
zero — no prior vocabulary assumed — in English and Portuguese, **same content in
both**. Read it before the numbered documents if terms like *expert*, *residency*
or *roofline* are not already yours.

| what | 🇬🇧 English | 🇧🇷 Português |
|---|---|---|
| **The guide, for reading** (recommended) | [`.en.html`](ForDummies/TootegaEngine-ForDummies.en.html) | [`.pt.html`](ForDummies/TootegaEngine-ForDummies.pt.html) |
| The guide, source | [`.en.md`](ForDummies/TootegaEngine-ForDummies.en.md) | [`.pt.md`](ForDummies/TootegaEngine-ForDummies.pt.md) |

The HTML version is the one to open: sidebar outline, accent-insensitive search,
light/dark theme, zoomable figures, and **a hint on every technical term** — hover
or Tab over any underlined word and it says what it means *here*. `/` focuses
search, `g` opens the glossary, `Esc` closes.

It runs 20 chapters, from *the project in one page* and *five ideas that explain
almost everything* through the full journey of a token, the physics of decode,
this machine's hardware and the inversion, residency, how the model is stored,
how we know it is correct, the server and the agent, to a repository map and the
common misunderstandings.

Around it:

- [`ForDummies/glossary.json`](ForDummies/glossary.json) — the **single source**
  for 277 terms, each with definition and aliases in PT and EN, and, for the ones
  that collide with ordinary usage, what the word means **elsewhere**. The `.md`
  glossary tables and the HTML cards are both generated from it. Aliases are
  searchable and shown on the card: looking for `matmul` lands on *matrix
  multiplication*.
- [`ForDummies/docs-html/`](ForDummies/docs-html/index.html) — the 20 documents of
  `docs/` as HTML, same look, same hints, same search. It exists because a browser
  does not render `.md`. **The Markdown in `docs/` stays normative**; where they
  disagree, the Markdown wins.
- [`ForDummies/img/`](ForDummies/img) — 15 hand-written SVG figures, one version
  per language, that switch palette with `prefers-color-scheme`.

The `.html` files are **generated artefacts** — [`ForDummies/build.py`](ForDummies/build.py)
builds them from the `.md` plus `glossary.json`. Change content in the `.md`,
change a term in the `.json`. See [`ForDummies/README.md`](ForDummies/README.md)
for the maintenance rules.

## How the project works

Three rules that came from mistakes made, not from theory:

**Paired measurement.** Comparing two configurations by the global mean swallows
the effect in the variance between windows. The first perplexity measurement had
the **sign inverted**.

**An output change never lands silently.** Any algebraically equivalent
alteration is only adopted with **32/32 identical tokens** against the reference.

**The reference has to be external.** Three green tests born of the same reading
of the problem were blind to the same error — a RoPE in the wrong variant. The
external reference cost 2 minutes and settled what days of internal testing did
not. It is in [17](docs/17-diagnosis.md).

## Reference machine

Every number in these documents was measured here — except where marked
**[estimate]**.

```
CPU    Intel i9-14900K, 32 threads, AVX2+FMA
RAM    64 GB DDR5-4800 dual channel      68.7 GB/s effective
GPU    RTX 2060 SUPER, 8 GB, sm_75       448 GB/s of VRAM
PCIe   gen 3 x16                         ~12 GB/s   <-- the slowest link
Disk   RAID0 NVMe                        18.4 GB/s sequential
OS     Windows 10 Pro 19045
```

## Structure

```
src/         the engine -- ISO C11, zero dependencies, only compat.c has OS ifdefs
tools/       production tools: convert, agent, finetune, ucd, fixture
Tests/       measurement tools: ablate, gentest, bench, oracle, mmbatch, route, analyze
docs/        the specification, with the why and the number behind each decision
ForDummies/  the project explained for non-specialists, EN and PT
publish/     the model card and the HuggingFace packaging script
```

Two Python files are left, both outside the engine's path: `ForDummies/build.py`,
which builds documents, and `download_model.py`, which needs HTTPS to fetch the
checkpoint. Everything else — the converter, the oracle, the perplexity meter,
the Unicode tables, the kernel fixture — is a native tool.

The working language of the project is **English**. The one deliberate exception
is `ForDummies/*.pt.*`, kept in Portuguese on purpose.

## Licence

MIT — see [LICENSE](LICENSE).

The model, its weights and the tokenizer carry their own licence, from their
authors, and are not covered by this one.
