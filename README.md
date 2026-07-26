# TootegaEngine

An inference engine in **pure C11, with no dependencies**, for MoE models that
fit entirely in RAM.

An OpenAI-compatible HTTP server, an embedded web chat, an agentic client with
local tools, and 20 documents that explain **why** each decision is what it is —
with the number that backs it right next to it.

```powershell
.\build-all.ps1      # engine + tools, MSVC via vswhere
serve.cmd            # starts the server; the UI is at http://127.0.0.1:8080/
```

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

All of it in **13 `.c` files**, ~420 KB of executable, zero runtime DLLs.

## State

**M1 and M4 complete, on CPU.** Measured on this machine, not estimated:

```
memory bandwidth    68.7 GB/s         measured at runtime, not specified
traffic per token    2.87 GB          what a decode actually touches
roofline ceiling    24.0 tok/s        bandwidth / traffic

warm decode         71 ms/token       = 13.1 tok/s
mean ttft           2.25 s
prefill             ~11 tok/s
warm-up             18.54 GB in 5.3 s
RSS                 16.8 GB peak      35.9 GB free
idle server         0.2% of one core, 0.04 GB
```

Said with the same rigour as the rest: decode sits at **55% of the ceiling**. The
~28 ms missing per token are in attention and the barriers — not in the matmuls,
which already run at 90% of bandwidth.

### What is proven

| property | result |
|---|---|
| forward against the `transformers` oracle | **32/32 identical tokens** |
| determinism | **byte-identical** logits from 1 to 24 threads |
| tokenizer against HuggingFace `tokenizers` | **36/36 cases**, exact round-trip |
| kernels against numpy's quantisation | **bit for bit** |
| perplexity against an external reference | 8.63 against `transformers`' 8.89 |

### What does not exist

- **Slots and batching** (M3). One request at a time; the second tab waits in the
  TCP backlog, silently.
- **Persistent state** and **MTP speculation**.
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

### 2. Get and convert the model

```powershell
python ptools\download_model.py                    # 62.4 GB in BF16, resumable
.\tools\convert.exe models\glm47flash-bf16 `
    --out models\glm47flash-int4\model.tootega     # ~19.3 GB in int4
```

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

**For non-specialists:** [ForDummies](ForDummies/) explains the whole project
from zero, in English and Portuguese, with a glossary of 225 terms — each one
saying what the word means **here** and what it usually means elsewhere.

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
tools/       production tools: convert, agent
Tests/       measurement tools: ablate, gentest, bench
ptools/      Python that depends on transformers/numpy as an external reference
docs/        the specification, with the why and the number behind each decision
ForDummies/  the project explained for non-specialists, EN and PT
```

The working language of the project is **English**. The one deliberate exception
is `ForDummies/*.pt.*`, kept in Portuguese on purpose.

## Licence

MIT — see [LICENSE](LICENSE).

The model, its weights and the tokenizer carry their own licence, from their
authors, and are not covered by this one.
