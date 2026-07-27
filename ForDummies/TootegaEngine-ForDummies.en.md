# TootegaEngine — for dummies

> **Sibling document:** [`TootegaEngine-ForDummies.pt.md`](TootegaEngine-ForDummies.pt.md) —
> same content, in Portuguese. The two versions are kept in parallel; when one
> changes, the other changes with it.

This is the project's front door. It exists because the rest of the
documentation (`docs/01` through `docs/16`) is written for someone who already
has the vocabulary — and a good part of that vocabulary is either field jargon,
or **this project's** jargon, or an ordinary word that means something else here.

The order is deliberate: **words first, system second.**

---

## Contents

| # | section | what you get |
|---|---|---|
| 0 | [How to read this document](#0--how-to-read-this-document) | conventions, audience, assumptions |
| 1 | [The project in one page](#1--the-project-in-one-page) | the thesis, the machine, current state |
| 2 | [Five ideas that explain almost everything](#2--five-ideas-that-explain-almost-everything) | the minimum to follow any decision |
| 3 | [Glossary A — words that mean something else here](#3--glossary-a--words-that-mean-something-else-here) | the other meanings |
| 4 | [Glossary B — the project's coinages](#4--glossary-b--the-projects-coinages-and-nicknames) | the internal dialect |
| 5 | [Glossary C — field vocabulary](#5--glossary-c--field-vocabulary-one-line-each) | quick reference |
| 6 | [From text to text: the full journey](#6--from-text-to-text-the-full-journey) | how a token is born |
| 7 | [Why it is slow: the physics of decode](#7--why-it-is-slow-the-physics-of-decode) | prefill, decode, roofline |
| 8 | [This machine's hardware and the inversion](#8--this-machines-hardware-and-the-inversion) | why the GPU does not save you |
| 9 | [Residency: the decision worth 20x](#9--residency-the-decision-worth-20x) | the thesis, with numbers |
| 10 | [How the model is stored](#10--how-the-model-is-stored) | int4, ablation, container |
| 11 | [Memory: one copy, not one more](#11--memory-one-copy-not-one-more) | mmap, arena, the three caches |
| 12 | [How we know it is correct](#12--how-we-know-it-is-correct) | oracle, 32/32, paired measurement |
| 13 | [The server, the protocol and the agent](#13--the-server-the-protocol-and-the-agent) | how a token becomes a product |
| 14 | [How quality is measured](#14--how-quality-is-measured) | perplexity, entropy, what misleads |
| 15 | [Where we are and what comes next](#15--where-we-are-and-what-comes-next) | roadmap with kill criteria |
| 16 | [Hands on](#16--hands-on) | build and commands, with output reading |
| 17 | [Repository map](#17--repository-map) | where everything lives |
| 18 | [Common misunderstandings](#18--common-misunderstandings) | FAQ |
| 19 | [PT ↔ EN index](#19--term-index-pt--en) | to read both versions side by side |

---

## 0 — How to read this document

### Who it is for

Anyone who programs, or at least reads code, and has **never opened an inference
engine from the inside**. No machine-learning background required.

### What I assume

- You know what a file, a pointer, a byte and a vector are.
- You know matrix multiplication is expensive.
- You do **not** know (and do not need to know) what a transformer is, what
  attention is, what quantization is, or what an "expert" is.

### What I do not do

I do not teach model training. This project **trains nothing** — it only runs an
already trained model that someone else published.

### Conventions

**Numbers.** This document uses the period as decimal separator (`0.93`) and the
comma as thousands separator (`154,880`). The Portuguese version uses the
opposite convention. The values are the same.

**Two marks that show up everywhere:**

| mark | meaning |
|---|---|
| **[measured]** | the number came from a run on this machine, with the configuration recorded |
| **[estimate]** | the number came from arithmetic and has not been confirmed by a run |

That distinction is taken seriously enough to be written into the original
documents. An estimate that becomes "fact" by repetition is the failure mode this
project works hardest to avoid — and it already cost a public 60% correction to a
central calculation (see [§12](#12--how-we-know-it-is-correct)).

**References.** Whenever a claim comes from a normative document, it appears like
this: → [`docs/08`](../docs/08-performance-ceiling.md). This document is a **front
door**, not the source of truth; where they disagree, the numbered document wins.

**"This machine"** always means the reference computer described in
[§1](#the-reference-machine). No number in this project is generic.

---

## 1 — The project in one page

### What it is

TootegaEngine is a **program that makes a language model answer**, written from
scratch in C, with no libraries, for one specific case: **MoE** models that
**fit entirely in the machine's RAM**.

Three of those words already need definitions, and they come in the next
sections: *model*, *MoE*, *fits entirely*.

### The thesis, in one paragraph

There is a class of engine that solves the opposite problem: **streaming
engines**, which run a model too large for RAM by reading pieces from disk on
demand. It works — and delivers **0.93 tokens per second** [measured] on this
machine, with a 744-billion-parameter model. That is too slow to be
interactive, and it is slow for the right reason: disk is ~370× slower than
RAM.

The observation behind TootegaEngine is that **the bottleneck is not the
hardware**: it is the ratio between model size and available fast memory.
Buying more RAM helps little (it measures `N^0.25`: quadrupling RAM gives
~1.4×). Switching to a model **that fits entirely in RAM** changes the regime:
disk leaves the path and the ceiling becomes memory bandwidth — **13.1 tokens
per second** [measured], against 0.93. Fourteen times, without changing a part
in the machine.

TootegaEngine is the engine designed for **that regime**, and explicitly
**not** for hiding disk.

### The reference machine

Every number in this project was measured here, except where marked [estimate]:

```
CPU    Intel i9-14900K, 32 threads, AVX2+FMA
RAM    64 GB DDR5-4800 dual channel      68.7 GB/s measured at runtime
GPU    RTX 2060 SUPER, 8 GB, sm_75       448 GB/s of VRAM
PCIe   gen 3 x16 (the GPU's limit)       ~12 GB/s   <-- the slowest link
Disk   RAID0 NVMe                        18.4 GB/s sequential
OS     Windows 10 Pro 19045
```

Remember the PCIe line. It is the origin of half the project's decisions
([§8](#8--this-machines-hardware-and-the-inversion)).

### The target model

**GLM-4.7-Flash**: 31.2 billion stored parameters, ~3.57 billion used per token,
19.65 GB in the project's format, 202,752 tokens of maximum context.
→ [`docs/04`](../docs/04-target-model.md)

### Current state

**M1 and M4 complete, on CPU.** The engine already:

- reproduces the reference implementation's result — **32 of 32 identical
  tokens** [measured];
- produces **byte-identical** output from 1 to 24 threads [measured];
- tokenises with **36 of 36 cases identical** to the reference library
  [measured];
- scores perplexity **8.63** against **8.89** for the external reference, on the
  same text [measured];
- holds multi-turn conversations reusing the cache — **99% of the prefill** on a
  typical turn [measured];
- serves the OpenAI protocol, with tools and streaming, to any client
  ([§13](#13--the-server-the-protocol-and-the-agent));
- ships an agent with six local tools, a sandbox and approval;
- refuses configurations that do not fit, saying what to reduce.

Speed [measured]: **71 ms per token on warm decode — 13.1 tok/s**, against a
theoretical ceiling of 24.0 tok/s computed from the measured bandwidth. That is
**55% of the ceiling**; the missing ~28 ms are in attention and barriers, not in
the matmuls, which already run at 90% of bandwidth.

What does **not** exist: several requests at once (M3), persistent state,
speculation, a compiled Linux build, and the GPU — the last one was written,
measured and refused
([§14.7](#147-the-measurement-that-decided-against-the-gpu)).

---

## 2 — Five ideas that explain almost everything

If you read only this section, you can follow 90% of the project's discussions.

### Idea 1 — the model does not write text, it guesses the next piece

The model receives a list of integers (**tokens**) and returns, for each of the
154,880 possible vocabulary pieces, a score for "how likely is this to be next".
One is picked, it becomes the next number in the list, and everything restarts.

**Consequence:** generating 60 tokens means running the whole model 60 times.
The cost is **per token**, always.

### Idea 2 — "stored parameter" and "active parameter" are different things

The model file holds 31.2 billion numbers (the **weights**). But the MoE
architecture only switches on part of them per token: **3.57 billion**.

```
stored in the file   31.2 B parameters  ->  19.65 GB on disk
read per token        3.57 B parameters ->   2.01 GB of reads
```

**Consequence:** file size does not determine speed. What determines it is how
much is **read per token**.

### Idea 3 — the bottleneck is memory bandwidth, not arithmetic

To generate one token the processor must **read 2.01 GB** of weights and do
~7.1 GFLOP of arithmetic. On this machine:

```
reading  2.01 GB at 60 GB/s  = ~33 ms
computing 7.1 GFLOP          = ~4 to 7 ms
```

Reading costs five times more than computing. The problem is called
**memory-bound** (limited by bandwidth), not **compute-bound** (limited by
arithmetic).

**The surprising practical consequence:** trading precision for arithmetic speed
barely helps, because the CPU spends most of its time waiting on memory. That is
why the project chose the **most precise** and arithmetically "slower" kernel —
there is compute headroom to spare. → [`docs/15`](../docs/15-kernels-and-numerics.md)

### Idea 4 — residency: either the model fits, or you are playing a different game

If the model fits in RAM, every token reads from RAM (60 GB/s). If it does not,
every token reads pieces from disk (18.4 GB/s at best, and far worse in practice
because of random-access latency).

This is not a 3x difference. It is a difference of **regime**, measured between
11x and 34x.

### Idea 5 — nothing ships without passing the oracle

Every change that could alter the output is compared against an independent
reference implementation — `Tests/oracle`, which shares no line of code with the
engine — and is only accepted if it produces **32 of 32 identical tokens**.

Three good optimizations have already been **rejected** by that criterion and
remain as switched-off options. → [§12](#12--how-we-know-it-is-correct)

---

## 3 — Glossary A — words that mean something else here

This is the section the rest of the documentation assumes you have read.

### 3.1 The table of other meanings

| word | what it usually means | what it means **here** |
|---|---|---|
| model | the "brain", something abstract | A file of numbers. Here it is literally `model.tootega`, 19.65 GB of weights. Not a program and not a service. |
| expert | a specialist in some subject | A pair of matrices picked by a score, per token and per layer. It has no subject: there is no "math expert". |
| cache | one single thing, generic fast memory | Here it means three different things with different owners: the weight cache, the KV cache and the prefix cache. Conflating them is the origin of most design confusion. |
| bandwidth (memory bandwidth) | internet bandwidth, measured in Mbps | How many gigabytes per second memory delivers. Nothing to do with the internet, and **different from latency**: bandwidth is bytes per second, latency is time to the first byte. Nearly every calculation here is about bandwidth, because in decode the read is large, predictable and sequential. On this machine it is **68.7 GB/s**, measured at runtime — not read off the box. |
| resident | installed on the machine | Living entirely in RAM and never leaving during the run. Not the same as "installed" or "loaded". |
| quantization | discretizing any signal | Storing each weight in 4 or 8 bits instead of 16. Weights only: the arithmetic stays in 32 bits, and the kernel *unpacks* before multiplying. That has a name here — **W4A32**, 4-bit weights and 32-bit activations. It costs perplexity: measured, +5.26% with everything in int4 and **−0.19%** (indistinguishable from zero) leaving attention in int8. |
| LoRA | a model fine-tuning technique | *Low-Rank Adaptation — the letters come from the fine-tuning technique; here the name hitched a ride and means only the projection.* Here (`q_lora_rank`, `kv_lora_rank`) it is just low-rank projection, part of the original architecture. No relation to the fine-tuning technique of the same name. |
| token | a symbol, or an access credential | A piece of text, ~4 characters on average. The model only ever sees the integer that stands for that piece. |
| context | the topic being discussed | The maximum number of tokens that fit in the conversation window — everything the model can "have in view" at once. It is not long-term memory: what leaves the window stops existing for it. And it costs RAM, linearly: **105.8 KB per position** in the f32 default, *per conversation*. At this model's ceiling, 202,752 positions, that is 21.96 GB — more than the 18.54 GB of weights. |
| group | generic grouping | Two unrelated homonyms: `group=64` (how many weights share one scale) and `n_group` (a routing restriction across experts). |
| page | a document or web page | A 4 KB (or 2 MB) block the operating system manages memory with. Nothing to do with document pages. |
| plan | an intention or a schedule | The decision of where each weight will live (RAM or VRAM), computed at startup from the measured bandwidths. |
| oracle | a fortune teller, or a database vendor | The reference implementation the engine is compared against, token by token — today `Tests/oracle`, in C, with its own container reader, dequantiser, matmul and attention. It is slow and that does not matter: it exists in order to *share no code* with the engine. Mind the limit: two implementations of the same reading of the model also agree when the reading is wrong, and it was an *external* reference that caught a RoPE in the wrong variant after three green internal tests. |
| ablation | a medical procedure | Turning one part off (or restoring it) at a time to measure how much it contributes. It is borrowed from experimental biology, and the logic is the same: you do not learn what a part does by looking at it, but by removing it. Here it is what showed that int8 attention costs nothing and that the rest of the model in int4 costs 5%. |
| slot | a physical socket, a calendar opening | An independent conversation context inside the engine, with its own KV cache. It is what would allow serving several people at once without mixing conversations. It **does not exist yet**: today it is one request at a time, and the second waits silently in the TCP backlog. That is milestone M3. |
| shard | a generic fragment, a database partition | One of the 48 files the original model comes split into, because of a per-file size limit. They carry no meaning of their own: a tensor may start in one and end in the next. The converter reads them all and writes a single container. |
| kernel | the operating system core | The routine doing the heavy arithmetic - `matmul_int4_g`, for instance. Not the operating system kernel. |
| draft | a rough version of a text | A cheap guess at a token, produced to be confirmed or thrown away in a block (speculation). The idea: a small model guesses several tokens at once and the big one checks them all in a single pass, paying one weight read instead of N. Here that is milestone M5, and it does not exist yet. |
| temperature | heat | A number controlling how random the next-token choice is. 0 means always the most likely; above 1 the unlikely options gain a chance. It is not a "creativity" knob: it is how much risk you accept on the next piece. In measurement it is always 0, or you measure the draw as well. |
| engine | a car engine, or a game engine | The program that runs the model: loads the weights, does the arithmetic, manages the cache. It is not the model, and not the interface. |
| agent | a person acting on someone's behalf | A program that uses the model in a loop: asks, gets an answer, runs tools, feeds the result back. |
| tool | a utensil, or a command-line program | In the agent sense: a function the model can ask to run (read a file, run a command). Its schema goes into the prompt. |
| MAC | a network card's address, or Apple's computer | *Multiply-ACcumulate.* Here it is the operation the kernel repeats millions of times per token: multiply two numbers and add to the accumulator. "MAC throughput" is how many of those fit in one processor cycle. |
| transformers (HF transformers) | the Transformer architecture the model is made of | Hugging Face's Python library, which runs the original BF16 model. It was the project's *external reference*: slow, and that did not matter — being someone else's did. It is what caught the RoPE in the wrong variant after three green internal tests. The script that called it is gone, and the number it gave is recorded in `docs/17`; reopening the question means bringing it back. It never entered `src/`. |
| PILOT | a pilot run, a trial version | One of the previous engine's mechanisms for hiding disk, along with cross-layer prefetch, dual-SSD routing, `O_DIRECT`, a persisted heat map and a per-layer LRU tier. It appears here only on the list of what was thrown away: it was most of that project's complexity budget and it is worth nothing when the model is resident. |

### 3.2 The four that confuse the most, with examples

**"Expert" is not a specialist.** The natural intuition — "the math expert",
"the code expert" — is wrong. An expert is a pair of matrices. The router scores
all 64 and picks the top 4, **per token and per layer**. The same token may
activate experts 5, 18, 27 and 52 in layer 3 and experts 1, 9, 40 and 63 in
layer 4. There is no topic, no assignable meaning, and nobody programmed it — the
split emerged from training.

**"Quantization" here is only about weights.** The engine stores weights in
4 bits but **unpacks to 32 bits before multiplying**. Activations (the numbers
flowing between layers) are never quantized. This has a name in the project:
**W4A32** — *weights 4 bits, activations 32 bits*. The arithmetically faster
alternative (W4A8) was rejected because arithmetic is not the bottleneck and it
would introduce fresh error. → [`docs/15`](../docs/15-kernels-and-numerics.md)

**"LoRA" here is not fine-tuning.** `q_lora_rank: 768` in the config means: instead
of one 2048×5120 matrix, the model uses two — 2048×768 and 768×5120. It is
compression baked into the architecture, present since training. The popular
fine-tuning technique called LoRA uses the same mathematical idea and a similar
name, but **is not involved here in any way**.

**"Bandwidth" is not latency.** Bandwidth is *how many bytes per second*. Latency
is *how long until the first byte*. An NVMe disk has excellent bandwidth and poor
latency; RAM has both good. Almost every calculation in this project is a
bandwidth calculation, because during decode the reads are large, predictable and
sequential.

---

## 4 — Glossary B — the project's coinages and nicknames

Terms you will **not find anywhere else**. They were coined here, usually right
after a mistake cost time.

| term | meaning |
|---|---|
| the inversion | The discovery that this machine's PCIe (12 GB/s) is slower than its disk (18.4 GB/s), inverting the hierarchy nearly every engine assumes. |
| placement law | "Put on the GPU only what fits there permanently; never stream weights over a link slower than the source tier." |
| layer partition | Splitting the model between GPU and CPU by layers, not by tensors. Weights stay pinned where the layer is computed; only 8 KB crosses between layers. |
| residency fraction | What percentage of the model sits in fast memory. It is the variable that governs speed, more than any hardware spec. |
| this machine's trap | engine anterior's `CUDA_DENSE=1` setting, which did not refuse the impossible and degraded 13x silently: 0.684 → 0.054 tok/s. |
| refuse instead of degrade | If the plan does not fit the declared budget, the engine exits with an error naming what to reduce. It never accepts and gets slow. |
| 32/32 (ref32, ref32.json) | The acceptance criterion: 32 test positions, all matching the reference token. Binary, and it blocks merges. |
| paired measurement | Comparing two configurations window by window over the same text, never by global mean. The global mean already produced a sign-flipped number. |
| determinism by state | Two runs in the same state produce identical bytes; different states may legitimately diverge. "Greedy is enough" is not assumed. |
| absorption (absorbed) | Rewriting attention to operate in the compressed space, projecting the query once instead of expanding every cached position. |
| unabsorbed (expanded) | The original attention formulation, which expands the cache before using it. It is the one the oracle uses. |
| half/half | The nibble packing convention: byte `j` holds weights `j` and `j+32`, not `2j` and `2j+1`. |
| scratch arena (arena) | One block of memory allocated once at startup and sliced for every temporary buffer. Zero `malloc` in the token loop. |
| stable first, volatile last | The prompt ordering rule that makes the cache work: system → tools → long context → history → current turn. |
| exit criterion | The number that closes a roadmap milestone. Without a number the milestone does not close — and "it works" is not a number. It is what keeps the conversation from becoming a matter of opinion: either the measured value cleared the threshold declared *beforehand*, or it did not. |
| kill criterion | The number below which the feature is cancelled, not postponed. M2 has one: a gain below 20% kills the GPU/CPU partition. |
| file-backed | Memory that came from a mapped file: the operating system can drop it and reread from disk. That is why it does not count against the RAM budget. |
| anonymous memory | Memory that exists only in RAM, with no file behind it. If evicted, the content is lost - hence it counts against the budget. |
| publish what was measured | When intuition says one thing and measurement says another, the measurement is published - even if the alternative looks safer. |
| ceiling (roofline) | The upper speed bound obtained by dividing bytes by bandwidth, ignoring compute and overhead. Also called the roofline. |
| streaming engine (disk streaming, previous engine) | An engine that runs a model larger than RAM by reading pieces from disk on demand. It works, and it is slow for the right reason: disk is ~370× slower than RAM. We measured **0.93 tok/s** that way, against 13.5 for TootegaEngine with the whole model in memory. |
| .tootega container (container) | The project's file format: a 128-byte header, blobs aligned to 64 bytes, a JSON index at the end. Read directly, with no conversion step. |
| milestone (M0, M1, M2, M3, M4, M5) | A project stage with a **measurable exit criterion**, not a date. M0 converts and validates the model; M1 is the token-exact CPU forward; M2 was the GPU; M3 is serving several requests at once; M4 is the agent; M5 is speculation. A milestone closes when the number shows up, not when the code compiles. |
| numbered decision (D1, D11, D14) | Every design choice with real alternatives gets a number — D1 to D14 — and a line on the scoreboard with its state: open, closed, or closed with the number that closed it. It exists so nobody reopens a discussion for not knowing it already happened. |
| MIT license | *Massachusetts Institute of Technology, where the licence was written.* A short, permissive free-software licence: anyone may use, copy, modify and sell, provided the copyright notice stays, and with no warranty. It is this engine's licence. It **does not cover the model** — weights and tokenizer have their own, from whoever trained them. |
| decision state (SETTLED, REJECTED) | The label on each scoreboard line: **SETTLED** when measurement closed the question, **REJECTED** when the thing was measured and turned down — bf16 KV scored 29/32 against the oracle and stayed an option, not the default — and open when there is no number yet. There is no "decided in a meeting". |
| CUDA_DENSE | The previous engine's environment variable that placed dense tensors on the GPU. Switched on with not enough VRAM, performance dropped **13×** with no error at all: the lesson that a bad configuration may not fail, only degrade. |
| external reference | A reference written by other people, from another reading of the problem. It is the only kind that works: three green internal tests were born of the same reading and were blind to the same error — a RoPE in the wrong variant. The external reference cost 2 minutes and settled what days of internal testing did not. |

### The nickname that shows up most: **streaming engine**

A `streaming engine` solves the **opposite** problem to this project's: models
too large for RAM, read from disk as execution needs them. The phrase appears
all the time here for two reasons, and neither is praise or criticism — it is
contrast.

The first is that it provides the **baseline**: 0.93 tok/s with 744 billion
parameters on this same machine, which is the number that makes this engine's
13.1 tok/s mean something. Without a measured reference, "fast" is an adjective.

The second is that it shows **what not to do when the model fits**. All the
machinery for hiding disk — cross-layer prefetch, a heat map of which weights to
keep, eviction policy, routing across two SSDs — is most of such an engine's
complexity budget, and it is **worth nothing** with a resident model. Throwing
it away is TootegaEngine's main structural advantage; it is not the speed, it is
the size.

**Hard project rule:** no third-party code or toolchain comes in here. The
comparisons are historical — measurement reference and lessons learned, never
permission to reuse. → `CLAUDE.md`

---

## 5 — Glossary C — field vocabulary, one line each

Quick reference. Nothing here is project-specific.

### 5.1 The model

| term | one line |
|---|---|
| LLM | *Large Language Model.* A program that predicts the next token of a text. The whole appearance of conversation comes from that — predicting the next piece, thousands of times, each time with the previous result in the input. There is no plan, and no intention kept between one prediction and the next. |
| transformer | The architecture practically every LLM uses: a stack of identical layers, each with attention and a multiplication block. |
| layer | One block of the transformer stack. Here there are 47, run in sequence - each one's output is the next one's input. |
| weight | One of the numbers learned during training. Here there are 31.2 billion of them, and they are what fills the 19.65 GB. |
| parameter | A synonym for weight when talking about model size. "31.2 billion parameters" = 31.2 billion learned numbers. |
| active parameter | How many weights are actually read to produce one token. Here 3.57 billion out of 31.2 - and that is the number that sets the speed. |
| tensor | A named multidimensional array of weights. The container has 9,703 of them, and the index says where each one starts. |
| hidden (residual stream) | The vector travelling through the layer stack. Here it holds 2048 numbers - 8 KB. Also called the residual stream. |
| embedding (embed_tokens) | The table converting a token id into the initial 2048-number vector. Here it has 154,880 rows and one is read per token. |
| logits | The 154,880 output scores, one per vocabulary piece, before they become probabilities. High score = likely candidate. |
| softmax | The function turning scores into probabilities that sum to 1. The maximum is subtracted first, to avoid exponent overflow. |
| RMSNorm (normalization) | *Root Mean Square Normalization.* Normalization that rescales the vector to a predictable magnitude before each block. Without it, 47 stacked layers blow up or vanish. |
| residual | Adding the input back into the block's output. It makes a layer correct the vector rather than replace it, which is what allows stacking 47. |
| attention | The mechanism by which the current token looks at every previous one and decides which to draw on. It is what gives the model a sense of context — and it is the only place where cost grows with conversation length, because the new token must consult everything that came before. Hence the KV cache: without it, each token would recompute the whole conversation. |
| head | One of the 20 parallel views inside attention. Each can specialize in a different kind of relation between tokens. |
| Q, K, V (query, key, value) | Query, key and value: what I look for, what I offer, what I hand over. Attention compares Q against every K and mixes the Vs. |
| RoPE | *Rotary Position Embedding.* Positional encoding by rotating the vector. Here it is partial: only 64 of each head's 256 dimensions rotate. |
| MLP (FFN, feed-forward, feed forward) | *Multi-Layer Perceptron; also FFN, Feed-Forward Network.* The two-multiplication block that follows attention. Also called FFN, feed-forward network. |
| MoE | *Mixture-of-Experts.* Instead of one MLP per layer, many - and only a few run per token. It is what makes 31.2 B of weights cost 3.57 B per token. |
| router (gate, routing) | The small matrix that scores every expert and decides which ones run. In code and config it appears as `gate`. |
| top-k | "Take the k largest". In the router it is top-4 of 64; in sampling it means choosing among the k highest-scoring tokens. |
| shared expert | An expert that always runs, on top of the ones the router picks. It holds what is useful for any token. |
| MLA | *Multi-head Latent Attention.* Attention that keeps the cache compressed, 576 numbers per token per layer instead of 10,240. |
| GQA / MHA | *GQA = Grouped-Query Attention · MHA = Multi-Head Attention.* The more common attention variants. They work, but their cache is much larger - hence the target model uses MLA. |
| MTP | *Multi-Token Prediction.* An extra head that guesses the following token almost for free, to speed things up by speculation. |
| lm_head | The final matrix, 2048 × 154,880, turning the last layer's vector into output scores. It is read in full every token. |
| vocabulary (vocab_size) | The 154,880 text pieces the model knows. Only 154,856 actually exist; the rest is padding. |
| dense layer | A layer with no MoE, holding a plain MLP. Here only layer 0 is like that, with internal size 10,240. |
| architecture | The model's blueprint: how many layers, of what kind, with what sizes. It comes from `config.json`, which the container stores whole. |
| checkpoint | The set of files published with the trained weights. Here it comes as 48 shards in `safetensors` format, 62.4 GB in BF16, and it is the converter's input — the engine never reads it at runtime. Its licence belongs to whoever trained it, not to this code. |
| safetensors | The format weights are published in: a JSON header giving name, shape and offset, then the raw numbers. |
| Hugging Face | The public repository the model and tokenizer are downloaded from, and the source of the library used as reference. |
| tool call (tool_calls) | When the model, instead of answering, asks a program to do something for it — read a file, run a command. It executes nothing: it writes a request, someone runs it, and the result comes back as a new message. In this model the request uses `<tool_call>`, which is a **single token** in the vocabulary, not text to be pattern-matched. |
| directive | An instruction about how to answer, given before the conversation starts: "answer in the language of the question", "get to the point". It goes in a system message at the **beginning** of the prompt — where the prefix cache reuses it on every turn. Changing it mid-conversation would invalidate the whole cache. |
| degeneration | When the model loses the thread and starts repeating, doubling words or never stopping. It is not a program error — it is the text folding onto itself. That is how one container was found to be bad while perplexity said it was fine. |
| MHA | *Multi-Head Attention.* Classic attention, where each head keeps its own key and value. Simplest, and most expensive in memory — the opposite of this model's MLA, which compresses both into a smaller space and saves 18× the traffic. |
| GLM | *General Language Model.* The family of open models this project runs. The target is **GLM-4.7-Flash**: 31 billion parameters in total, but only ~3 billion active per token, because it is an MoE. That ratio is what makes it both fit and run fast. |
| draft (reasoning, thinking) | The text the model writes to itself before answering, between `<think>` and `</think>`. Measured on this model: with a 400-token budget it **fails to finish 8 of 12 questions**, repeats 8× more and costs 4.5× the time — for "what is the capital of Brazil" it writes six steps in English and gets cut off before answering. |
| GPT | *Generative Pre-trained Transformer.* The family that made the name popular. It shows up here only as a reference: the tokenizer uses GPT-4's pre-tokenisation table. |
| kv_lora_rank / q_lora_rank (q_lora_rank) | The two compressed dimensions of MLA in this model: the query goes down to **768** and the KV to **512** before being projected back up. It is where the 576 per token per layer comes from — 512 of compressed KV plus 64 of RoPE — instead of 10,240. |
| n_group (topk_group) | How many expert groups the router considers before picking. In this model it is **1** — no group limit — and it is the only path implemented on day one. The previous engine had `n_group == 1` hard-coded into the parser, which reduced the universe of models to a single family; here it is a config variant, not a rule of the code. |
| norm_topk_prob | The key that renormalizes the chosen experts' scores: the 4 surviving scores are divided by their sum so they add up to 1. Here it is `true`. |
| routed_scaling_factor | The factor multiplying the routed experts' output before it goes back into the residual. In this model it is **1.8** — the model's number, not our choice. |
| first_k_dense_replace | How many layers at the start are dense instead of MoE. Here it is **1**: layer 0 has a plain MLP, the other 45 have a router and experts. |
| ignore_merges | The tokenizer key that changes the algorithm: if the whole word is already in the vocabulary, use that id and **apply no merges at all**. Only when it is not does classic BPE run. It is `true` in this model, and ignoring it gets exactly the most common words wrong. |
| [gMASK] (gMASK) | The special token that opens every prompt in this model: the sequence starts with `[gMASK]<sop>` before `<\|user\|>`. It is matched by exact text, before BPE, like the other 36 markers. |
| Ġ (space marker) | *the BPE space marker.* The character that stands for a space inside the vocabulary. BPE here is byte-level: printable bytes map to themselves and the rest to `0x100 + n`, which is why `Ġthe` means " the", with the space. It is not an encoding glitch. |
| Qwen3-30B-A3B | The closest runner-up to the chosen model: 15.3 GB, fits in RAM, and it is good. It was left out because it uses GQA without MLA — the KV per token is orders of magnitude larger, which is exactly what MLA fixes. |
| A3B | *Active 3 Billion active parameters.* The naming convention that says how much of the model works per token: in `Qwen3-30B-A3B`, 30 B stored and **3 B active**. The active number rules time; the other rules memory. |
| DeepSeek | The family where MLA and group routing show up. It is a counterexample twice over: `DeepSeek-V4-Flash` is 79.8 GB and is not resident on this machine, and `n_group=8` is the case the previous engine's parser refused. |

### 5.2 Execution

| term | one line |
|---|---|
| inference | Running an already trained model to get an answer. It is all this project does - there is no training anywhere. |
| training | The very expensive process that produced the weights. It happens once, elsewhere, and is never repeated here. |
| fine-tuning (fine tuning) | Adjusting an already trained model for a specific task. Declared out of scope in this project. |
| forward (pass) | One complete pass through the layer stack, from input to logits. It is also the name of the subcommand that does it over fixed ids. |
| prefill | Processing the whole prompt at once, before generating. All tokens go through together, so weights are read once for many tokens. |
| decode | Generating one token at a time, in a chain. Each token rereads the full 2.01 GB of weights - this is where bandwidth rules. |
| TTFT | *Time To First Token.* How long until the first word shows up. Not a goal in this project until milestone M3. |
| tok/s | Tokens per second, the project's headline metric. Always quoted with the context it was measured in, because the number moves with it: **13.1 tok/s** on warm decode here, against a 24.0 ceiling computed from bandwidth. Quoting tok/s without saying the context, whether it was warm and how many threads is the easiest way to publish a number nobody can reproduce. |
| KV cache | *KV = Key/Value.* The attention state already computed for previous tokens. Keeping it avoids redoing the whole prompt for every word. |
| prefix cache | Reusing the KV of the conversation's beginning across turns. It holds up to the first differing token; from there on everything is redone. |
| teacher forcing | Running the model over a known text, forcing each correct word, to compare prediction by prediction against the reference. |
| greedy | Always picking the highest-scoring token, no draw. It is deterministic, and it is the mode used in every measurement — because comparing two configurations with random sampling measures the random number generator too. Mind the trap: greedy does **not** guarantee identical output across machines; a tie broken by one different bit separates the paths. |
| top-p (nucleus, nucleus sampling) | Drawing among the highest-scoring tokens whose probabilities together reach p. Also called nucleus sampling. |
| sampling (sampler) | The step that picks one token out of the 154,880 candidates, from the scores. Greedy, temperature, top-k and top-p are its modes. |
| continuous batching (batch, batching) | Serving several conversations at once, amortizing the same weight read across all of them. It is where the project's second goal lives. |
| speculation | Guessing several tokens at once with a cheap mechanism and verifying them in bulk. It pays off when the model is resident. |
| prompt | The complete input text: instructions, tools, history and the current question. It is what the prefix cache tries to reuse. |
| system prompt | The fixed part of the prompt, holding permanent instructions. It goes first precisely so it never invalidates the cache. |
| chat template (chat_template, chat_template.jinja) | The rules for assembling a conversation's text out of turns, with the speaker markers. It ships with the model. |
| turn (multi-turn) | One round of the conversation: what the user said plus what the assistant answered. |
| tokenizer | The component converting text into ids and back. An error here does not crash: it yields slightly different ids and a plausible, wrong answer. |
| BPE | *Byte Pair Encoding.* Repeatedly merges the two cheapest adjacent pieces, according to a table of 321,649 rules. |
| special token | One of the 36 markers such as `<\|user\|>`, matched by exact text before BPE. They delimit who is speaking. |
| futex | The mechanism by which a thread sleeps waiting for a value to change and is **woken by whoever changed it** — instead of waking on a timer to check. The difference shows up as a number: sleeping on a timer made a 1.6 MB matmul cost 26 ms instead of 0.026 ms. |
| barrier | The point where every thread stops and waits for the last one. It is necessary and it is expensive: synchronising 20 threads costs the same order of magnitude as one expert's work. That is why 15 barriers per layer became 2 — by fusing the matmuls into a single dispatch. |
| process | A running program with its own memory. The server is one; the agent is another, one per interface tab. Splitting them has a cost and a reason: if the agent dies, the server and the 18.5 GB it loaded stay up. |
| pipe | A tube between two programs: what one writes, the other reads. The interface sends the question to the agent through a pipe. It has a trap that cost hours here: the program may have *already read* the bytes into an internal buffer, and whoever inspects only the tube sees it empty — and wrongly concludes nothing arrived. |
| RAG | *Retrieval-Augmented Generation.* Instead of the model *remembering* the answer, it *reads* it: an index finds the relevant passages of a corpus and pastes them into the context ahead of the question. Not used in this project — it is here because it is the alternative that always shows up next to fine-tuning, and it solves a different problem. |
| EOS | *End Of Sequence.* The token the model emits to say it has finished. Without it, generation only stops at the requested token limit. |
| paged KV | Keeping the KV in fixed-size blocks instead of one contiguous block per conversation. Two conversations sharing a prefix point at the same blocks, and freeing one leaves no hole. Simple in M1 and complete in M3 — decision D8, still open in the details. |

### 5.3 The machine

| term | one line |
|---|---|
| CPU | *Central Processing Unit.* The main processor. Here, an i9-14900K with 8 fast cores, 16 efficient ones and 32 threads in total. |
| GPU | *Graphics Processing Unit.* The graphics card, used as an arithmetic accelerator. Here it is an 8 GB RTX 2060 SUPER - fast inside, poorly connected outside. |
| RAM | *Random Access Memory.* The computer's main memory. Here, 64 GB delivering ~60 GB/s. It is where the whole model needs to fit. |
| VRAM | *Video RAM — the graphics card’s RAM.* The graphics card's memory. Fast (448 GB/s here) and small (8 GB), and reachable only over PCIe — which on this machine delivers ~12 GB/s, less than the disk. It is the combination of the three numbers that decides everything: with the model at 18.54 GB nothing fits permanently, and what does not fit must cross the slowest link on every layer. |
| PCIe | *Peripheral Component Interconnect Express.* The bus connecting the GPU to the rest of the computer. Here it is gen 3: ~12 GB/s, the machine's slowest link. |
| NVMe (RAID0) | *Non-Volatile Memory Express.* The standard for fast SSDs wired straight to the bus. Here, two in RAID0 delivering 18.4 GB/s sequential. |
| tier | One level of the memory hierarchy: VRAM, RAM, disk. Each tier has its bandwidth, its capacity and its access link. |
| memory-bound | When time is dominated by waiting on memory rather than doing arithmetic. It is decode's regime: ~33 ms of reads against 4-7 ms of compute. |
| compute-bound | When time is dominated by arithmetic. It is prefill's regime, which is why optimizing it is a different problem. |
| core | An independent execution unit inside the CPU. This processor is hybrid: 8 fast P-cores and 16 efficient E-cores. |
| thread | A line of execution. The engine uses its own thread pool, with work split by matrix output rows. |
| affinity | Pinning each thread to chosen cores. The default is P-cores only: spreading memory-bound work across E-cores usually hurts. |
| SIMD | *Single Instruction, Multiple Data.* One instruction operating on several numbers at once. It is how the kernel processes 8 or 16 weights per cycle instead of one. |
| AVX2 / AVX-VNNI (AVX-VNNI, VNNI, vpdpbusd, x86-64-v3) | *AVX = Advanced Vector eXtensions · VNNI = Vector Neural Network Instructions.* This CPU's SIMD instruction sets. The engine picks which to use at runtime, so the binary still runs on other machines. |
| FMA (vfmadd) | *Fused Multiply-Add.* Multiply and add in a single instruction. Fast, but it rounds differently - which is why the build forbids automatic fusion. |
| L1/L2/L3 cache (L3, L2) | Tiny, extremely fast memories inside the processor. This CPU's L3 holds 36 MB, enough for one full layer's cache at 4k. |
| mmap (mapped, mapping) | Mapping a file straight into the process address space. The file becomes the memory, with no copy step. |
| page cache | The cache the operating system keeps of files it has read. Since the model is mapped, that cache is the weight cache. |
| page fault (pre-fault) | What happens on the first access to a mapped page: the system stops everything and fetches from disk. Hence the pre-fault at load time. |
| huge pages | Using 2 MB pages instead of 4 KB. With 19.3 GB of weights, that goes from 4.7 million pages down to ~9 thousand. |
| TLB | *Translation Lookaside Buffer.* The small processor table that translates addresses. It has ~2048 entries: with 4 KB pages it covers 8 MB, with 2 MB pages it covers 4 GB. |
| RSS | *Resident Set Size.* How much physical memory the process actually occupies. It is the number the watchdog compares against the budget. |
| swap | When the operating system writes memory to disk for lack of RAM. Here it counts as a failure: the budget exists so it never happens. |
| OOM | *Out Of Memory.* When the system kills the process for lack of RAM. It is what the refuse-instead-of-degrade rule prevents. |
| budget | A hard cap on RAM, VRAM or threads passed as a parameter. Verified at runtime, not just in the plan. |
| bottleneck | The resource that limits everything else. Here it is memory bandwidth; optimizing anything else does not change total time. |
| overhead | Cost that is not the work itself: calling functions, synchronizing threads, allocating memory. Invisible at 1 tok/s, a percentage at 20. |
| Amdahl's law | The arithmetic saying how much speeding up one part speeds up the whole. Used here to estimate the gain of moving 39% of layers to the GPU. |
| CUDA | *Compute Unified Device Architecture.* NVIDIA's platform for running computation on a graphics card. A CUDA program has two halves: one runs on the CPU, one on the GPU, written in different files and built by different compilers. Here it was written, measured and **not adopted** — it lives on the `GPU` branch. |
| SMT (hyper-threading) | *Simultaneous Multithreading.* Intel calls it Hyper-Threading. One physical core appears as two logical ones sharing the same execution units. For memory-bound work like ours the two siblings fight over the same resource and do not help: that is why the pool uses **one logical core per physical core**. |
| P-core and E-core (E-cores) | Modern Intel processors mix two kinds of core: **P** for performance, big and fast, and **E** for efficiency, small and frugal. Spreading even work across both makes the fast ones wait for the slow ones, so the engine pins threads to P-cores by default. |
| DDR5-4800 | *Double Data Rate, 5th generation — 4800 MT/s, million transfers per second.* The generation and speed of the RAM. The number is the nominal rate in millions of transfers per second. Nominal is not real: the **68.7 GB/s** this project uses in every calculation was *measured at runtime*, not read off the box. |
| RTX | *Ray Tracing — NVIDIA does not expand the X.* NVIDIA's consumer graphics card line. This machine has an RTX 2060 SUPER with 8 GB. Those 8 GB are the deciding number: the model takes 18.5 GB, so it **does not fit**, and any GPU use becomes a partition — which was measured and did not pay off. |
| OS (operating system) | *OS = Operating System.* Operating system — Windows, Linux, macOS. It hands memory, files and threads to the program. The engine touches the OS in **a single file**, `compat.c`: that is the boundary where the Windows/Linux differences live, so the rest of the code need not know where it runs. |
| SSD | *Solid-State Drive.* A drive with no moving parts. This machine's are *NVMe* in RAID0, 18.4 GB/s — faster than the PCIe link to the graphics card. |
| prefetch (madvise) | Asking for the page before needing it. At load time the engine calls `madvise(MADV_WILLNEED)` or `PrefetchVirtualMemory` and brings 18.54 GB in 5.3 s in one go — instead of taking page faults in the middle of decode. |
| cache line | The **64 bytes** the CPU moves at once between RAM and cache. It is the real unit of reading: asking for 4 bytes costs a whole line. Every blob in the container starts at a multiple of 64 because of it. |
| alignment (aligned) | The address starting at a multiple of 64. Without it there is no zero-copy: the kernel would have to copy the tensor into an aligned buffer before multiplying. The converter aligns every blob on write and `--verify` checks it; the padding disappears in the rounding of the 19.3 GB. |
| thread pool (pool) | The threads created once at load and reused on every layer, instead of being born and dying per matmul. It is the engine's own pool — decision D6, and the reason there is no OpenMP in the binary. Work is split by matrix output rows, and the price of having several threads is the barrier. |

### 5.4 Numerics

| term | one line |
|---|---|
| bit | The smallest unit of information: 0 or 1. An int4 weight takes 4 bits. |
| byte | Eight bits. Two int4 weights fit in one byte - hence nibble packing. |
| GB (KB, MB) | *Gigabyte.* The model is 19.65 GB; 2.01 GB is read per token. |
| float | A number with decimals, stored in floating-point format. The engine's arithmetic is all in 32-bit floats. |
| fp32 (f32) | 32-bit floating point, with 23 mantissa bits. It is the engine's working precision, and the oracle's. |
| bf16 (bfloat16) | 16-bit floating point with fp32's exponent and only 7 mantissa bits. It is how the model is published. |
| int4 / int8 (int8) | 4-bit and 8-bit integers, used to store weights. Always paired with a scale that converts them back. |
| nibble | Half a byte, 4 bits. Exactly one int4 weight - which is why the order the two nibbles of a byte are read in matters. |
| scale | The float that turns an integer back into the real weight: `weight ≈ (code − 8) × scale`. One scale per 64 weights. |
| g64 | "Group of 64": one shared scale per 64 neighbouring weights. It costs 4/64 = 0.0625 extra bytes per weight. |
| absmax | The largest absolute value in a group. Divided by 7, it gives the simplest scale - though not the optimal one. |
| W4A32 (W4A8, W4A16) | Weights in 4 bits, activations in 32. The kernel unpacks and multiplies in float, never quantizing activations. |
| activation | The numbers flowing between layers - the 2048-value vector. Tiny compared to weights: 8 KB against 4.7 MB per expert. |
| associativity (reduction order) | Summing in a different order gives a different floating-point result. Hence the engine's reduction order is fixed and independent of thread count. |
| /fp:precise | The compiler option forbidding sum reassociation and multiply-add fusion. It costs performance and buys reproducibility. |
| rounding | Choosing the nearest representable value. int4's error is the floor of rounding, not a bug. |
| relative error | How far the result strays from the correct one, as a percentage. Here it is measured at a multiplication's output, not per isolated weight. |
| outlier | A value far outside its neighbours' range. In real activations a few channels carry 10-100x the median magnitude, and they dominate the error. |
| TOPS | *Tera-Operations Per Second.* A throughput measure for integer arithmetic, as GFLOP/s is for floating point. |
| fp16 (float16, half precision) | 16-bit floating point with 5 exponent bits and 10 mantissa bits — more precision and less range than bf16. It only shows up in ceiling arithmetic: the engine never dequantises to fp16 before multiplying. |
| Q4_0 (llama.cpp) | llama.cpp's 4-bit format. Its convention for packing two weights into one byte is **half/half** — the low nibble holds the first half of the group, the high nibble the second — and it is what the container follows, recorded as decision D1. |
| interleave (rope_interleave, rotate_half, apply_rotary_pos_emb_interleave) | The RoPE variant this model uses: it rotates `x[2i]` with `x[2i+1]`, not `x[i]` with `x[i + rot/2]`, which is `rotate_half`. At position 0 the two coincide (`cos = 1`, `sin = 0`) and the error grows with position — this was the mistake that walked past 32/32 and past perplexity unseen. |
| contraction (FMA contraction) | When the compiler fuses a multiply and an add into an FMA on its own. It is faster and it rounds differently, so the build uses `/fp:precise` with no contraction — verified in the disassembly, because that is what reproducibility rests on. |

### 5.5 Measurement

| term | one line |
|---|---|
| perplexity | How "surprised" the model is by a text. Lower is better. It is the measure that says whether quantization actually degraded anything. |
| nll | Negative log-likelihood, perplexity's logarithm. Used because it is additive - means and intervals can be computed on it. |
| 95% interval | The range the true value probably lies in. If it crosses zero, no effect is demonstrated - just noise. |
| window | A fixed chunk of text used in measurement. Here there are 10 windows of 384 tokens, compared window by window. |
| fixture | A file of precomputed correct answers, generated once and compared against on every test run. |
| selftest | The test built into the binary: kernels, JSON parser, thread pool, tokenizer and sampler against fixtures. It runs in seconds. |
| regression test | A test that exists only to keep an old bug from returning. The `CUDA_DENSE` case became one. |
| baseline | The reference number to beat. Here it is 0.93 tok/s, measured with engine anterior and the oversized model. |
| benchmark | A controlled, repeatable measurement. Without a fixed budget, warm-up and an output hash, it is worthless. |
| warm-up (cold) | A discarded run before measuring, so the cost of filling caches does not land entirely on the first measurement. |
| reproducibility (determinism, deterministic) | The property of getting exactly the same result on repetition. Here it is an exit criterion, not a convenience. |
| hash | A short number computed from a content, which changes if any byte changes. Used to prove two outputs are identical. |
| throughput | How much work comes out per unit of time, summed over all conversations. Different from latency, which is the time of one. |
| latency | The time until the answer starts. High bandwidth with poor latency is common on disk; RAM has both good. |
| entropy | How spread out the model's opinion about the next token is, measured in *bits*. Low entropy: few candidates. High: many, nearly tied. It goes next to perplexity because it separates two cases perplexity conflates — **confident and right** from **confident for lack of options**, and the second is what precedes repetition. |
| rep4 | The fraction of four-token sequences that already appeared earlier in the same answer. It is the early detector of *degeneration*: a model starting to loop shows up here long before it bothers a reader. Above ~15% is outright degeneration. |
| match@ref | How many leading tokens a generation shares with the reference generation before diverging. It is the most direct measure of "this container generates what the original would". Diverging is not an error: two near-identical paths split at the first tie. |
| confidence interval (CI95) | The range the true value probably lies in, given how much the measurements varied. If the range **crosses zero**, the measurement cannot tell "better" from "worse" — and the honest answer is *inconclusive*, not "about the same". |
| SWE-bench | *Software Engineering benchmark.* Real defects from public repositories: the model gets the issue and has to produce the fix. This project's target scores 59.2 on the Verified variant — a vendor number, with the usual discount. |

### 5.6 Software and the system

| term | one line |
|---|---|
| C11 | The C language standard version the engine uses. Chosen because it is auditable and drags in no dependencies. |
| dependency | An external library the program needs. The engine has zero: JSON, tokenizer and file reader are all its own. |
| static linking | Bundling everything into the executable so it runs with nothing installed. The goal is a single binary, with the GPU backend loaded separately. |
| compiler | The program that turns code into an executable. Here it is MSVC on Windows, located dynamically, and gcc or clang on Linux. |
| build id | A hash of the sources and build flags, embedded in the binary. It invalidates persisted caches when the engine changes. |
| CLI | *Command-Line Interface.* The program you use by typing commands in a terminal. |
| pointer | A memory address stored in a variable. Moving a large block here means swapping pointers, never copying bytes. |
| malloc (allocator, allocation) | The function that asks the system for memory at runtime. The engine's token loop never calls it. |
| zero-copy | Using data where it already is, with no intermediate copy step. Here, kernel pointers point straight into the mapped file. |
| JSON | *JavaScript Object Notation.* A text format for structured data. The container index and the model config are JSON, read by the project's own parser. |
| UTF-8 | *Unicode Transformation Format, 8 bits.* The standard text encoding, where a character takes 1 to 4 bytes. The tokenizer works on bytes, not characters. |
| Unicode (code point) | The catalogue of every character of every writing system. The pre-tokenizer must know, per character, whether it is a letter, a digit or a space. |
| regular expression (regex) | A pattern describing how to split a text. The tokenizer uses a fixed one, the same as GPT-4's, hand-implemented instead of using a generic engine. |
| HTTP | *HyperText Transfer Protocol.* The web protocol. The `serve` subcommand opens a local server at `127.0.0.1:8080` with a chat interface. |
| loopback | The address `127.0.0.1`, reachable only from the machine itself. It is the server's default because this is not a production service. |
| atomic | An operation that either happens whole or does not happen. Writing state to disk is atomic through a temporary file plus `rename`: a power cut halfway cannot leave a truncated file looking valid. In the thread pool it is four atomic functions behind a shim in `compat.h` (`_Interlocked*` on Windows, `__atomic_*` on gcc and clang), because MSVC only enables `<stdatomic.h>` under `/experimental:c11atomics`. |
| watchdog | A thread comparing real consumption against the budget while running. Without it, "budget" would be just an opinion. |
| MSVC | *Microsoft Visual C++.* Microsoft's C and C++ compiler, shipped with Visual Studio. It turns the `.c` files into `tootega.exe` on Windows. The project locates it with `vswhere` instead of hard-coding a path: hard-coded paths break when the version changes. |
| ISO | *Not an acronym: from the Greek *isos*, "equal".* The body that publishes international standards. "ISO C11" means the code follows the 2011 C standard *as written*, without any one compiler's extensions. That is what lets the same file build under MSVC and gcc unchanged. |
| SSE (server-sent events) | *Server-Sent Events.* The standard way for an HTTP server to push text in pieces as it is produced, without the client polling. It is why the answer appears word by word instead of showing up complete half a minute later. |
| CORS | *Cross-Origin Resource Sharing.* The browser rule that stops a page from one address calling a server at another, unless the server allows it. Without that permission a client running *inside* the browser never even sends the request. |
| NDJSON | *Newline-Delimited JSON — one JSON per line.* One JSON object per line, one event per line. It is how the agent talks to the interface — each line is complete on its own, so the reader can process the first without waiting for the last. |
| stdin and stdout (standard input, standard output) | The two text channels every program is born with: **stdin** is where input comes in, **stdout** is where output goes. The agent and the interface talk over them — no network, no port, no file. Closing stdin is like hanging up: the program sees the end and exits. |
| EOF | *End Of File.* The signal that nothing more is coming. When the interface closes the agent's channel it gets an EOF and exits right away — the clean shutdown mechanism, more reliable than any timer. |
| OpenAI protocol | The message format OpenAI published for talking to models, which became a de-facto standard: a list of messages with `role` and `content`, and the reply in the same shape. Speaking it is what lets Copilot, Continue and any other client connect to this server **without a line of our code**. |
| endpoint (route) | A specific address inside a server, with one job. `/v1/chat/completions` generates text; `/stats` reports how the server is doing. It is the service's "extension number": same machine, same port, different purposes. |
| JSON Schema (schema) | *JSON = JavaScript Object Notation.* A description, in JSON, of what another JSON must look like: which fields, of what type, which are required. The client sends the tools' schema, and that is how the server knows `3` must become the **number** 3 and not the text `"3"` — without it, every tool with a numeric argument arrives wrong. |
| MCP | *Model Context Protocol.* A standard for an agent to discover and use tools offered by other programs, rather than only its own. It is today's "common tool protocol". TootegaEngine **does not implement it**; it is on record as not done. |
| sandbox | A fence around what a program may touch. In the agent it is the working folder: every path is resolved against it and refused if it escapes — **even with auto-approval on**. It is the only thing between a 31-billion-parameter model and the rest of the disk. |
| approval | The question the agent asks before doing something that changes state — writing a file, running a command. Reading does not ask, deliberately: confirming everything, at 13 tokens per second, turns the task into half an hour of pressing Enter — and **trains the person to approve without reading**, which is the worst possible outcome. |
| branch | A parallel line of development in version control. Here `master` is the CPU-only engine and `GPU` holds all the graphics-card work — written, measured, not adopted. Splitting that way is what lets you **keep the measurement without carrying the code**. |
| CRLF (LF, line ending) | *Carriage Return + Line Feed.* The two invisible characters Windows ends each line of text with, against one on Linux. Invisible does not mean harmless: a snippet copied in one style will not match a file saved in the other, and the search fails with nothing on screen to explain it. |
| UUID | *Universally Unique Identifier.* An identifier long enough to be unique without anyone coordinating who uses which. It names a conversation, a block, a session — when two sides must refer to the same item with no central registry. |
| HF | *Hugging Face.* The repository the checkpoint and `tokenizer.json` come from. "HF format" is the file layout that platform standardised, not a format of this project. |
| IR | *Intermediate Representation.* The form the engine keeps the architecture in after reading the config: pointers resolved and checked once, at open time. It is what makes a swapped tensor fail with its name and shape, instead of becoming numerical garbage 47 layers later. |
| API | *Application Programming Interface.* The set of calls one program offers another. Here, the HTTP endpoints compatible with the OpenAI protocol. |
| ABI | *Application Binary Interface.* The low-level contract between two already-compiled binaries: who passes what, in which register. A C ABI is what allows loading the GPU backend without recompiling the engine. |
| DLL | *Dynamic-Link Library.* A code file the program loads at run time instead of building in. It is how the GPU backend gets in without the CPU binary depending on CUDA. |
| TCP | *Transmission Control Protocol.* The transport under HTTP: delivers bytes in order and without loss. It shows up here because, with one request at a time, the second waits silently in the TCP queue. |
| LRU | *Least Recently Used.* The commonest eviction policy: when space runs out, whatever went longest untouched is the first to go. |
| URL | *Uniform Resource Locator.* The address of a resource on the web. The local server's is `http://127.0.0.1:8080`. |
| HTML | *HyperText Markup Language.* The language of web pages. This guide is a single HTML file, no server and no dependency: it opens with a double click. |
| UI | *User Interface.* The part you interact with. Here it is the page `tootega serve` opens at `127.0.0.1:8080`. |
| BLAS | *Basic Linear Algebra Subprograms.* The classic matrix-multiplication library. The engine uses none — the kernels are written here — but a tool may. |
| SHA-256 | *Secure Hash Algorithm, 256 bits.* A function that reduces any content to a fixed-size number. It is what proves two runs produced exactly the same answer. |
| GNU | *GNU's Not Unix — a recursive acronym: it contains itself.* The free-software project the `gcc` compiler and `make` used for the Linux build come from. |
| tokenizers (HF tokenizers) | Hugging Face's tokenization library, used as an external reference: **36/36** identical cases, exact round-trip. The engine's tokenizer is written here, from scratch, and measured against it. |
| numpy | Python's array library. It was the quantisation reference while the oracle was in Python; today the reference is `Tests/oracle`, in C, and what the kernel produces matches **bit for bit** the fixture generated by `tools/fixture`. It never entered `src/`. |
| OpenMP (libgomp) | *Open Multi-Processing.* The compiler's parallelism runtime — `#pragma omp parallel for`. The previous engine linked it statically (`libgomp.a`) and it was the only runtime dependency it carried. This engine **does not use it**: it has its own thread pool, and that is what keeps the binary dependency-free. |
| mlock (VirtualLock) | Asking the system not to take these pages out of RAM. It pins the resident set so a page fault does not show up in the middle of decode. It is optional and bounded by the memory budget — `VirtualLock` is the Windows name. |
| O_DIRECT | Reading from disk without going through the OS cache. It was tuned in the previous engine and is worth nothing here: it only helps whoever reads from disk, and this engine reads from RAM. It is on the list of what was thrown away. |
| dlopen (LoadLibrary) | Loading a library at runtime instead of linking it into the binary. It is how the CUDA backend came in, behind a C ABI — and the reason the GPU binary cannot be fully static: `dlopen` from a static binary is not reliable. Master is pure CPU and uses none of it. |
| vswhere | Microsoft's utility that reports where Visual Studio is installed. The build calls it instead of hard-coding the MSVC path: a hard-coded path breaks when the version changes. |

### 5.7 Math

| term | one line |
|---|---|
| vector | A list of numbers. Here, a token's state is a vector of 2048 floats. |
| matrix | A table of numbers, with rows and columns. A model weight is a matrix; the largest is 2048 × 154,880. |
| matrix multiplication (matmul, matmuls, GEMM) | The central operation: each output row is the sum of products of one weight row against the input. Almost all the time goes here. |
| dot product | Multiplying two vectors element by element and summing. A matrix multiplication is a pile of dot products. |
| low rank (low-rank, rank) | Replacing one large matrix by two smaller ones multiplied together. It compresses, and it is what `q_lora_rank` and `kv_lora_rank` do. |
| transpose | The matrix with rows and columns swapped. It matters here because reading a column is expensive and reading a row is cheap. |
| FLOP (GFLOP) | *FLoating-point OPeration.* One floating-point operation. Generating a token costs ~7.1 GFLOP - little, next to the 2.01 GB of reads. |
| probability | A number between 0 and 1 saying how likely something is. Softmax turns output scores into probabilities. |
| standard deviation (variance) | How much values spread around the mean. It is why paired comparison exists: the variance across windows swallowed the effect. |
| exponent | In `N^0.25`, the 0.25. It says quadrupling N multiplies the result by only 1.41 - the scaling law this project refuses. |

---

## 6 — From text to text: the full journey

![A token's journey](img/03-pipeline.en.svg)

Let us follow the question *"What is the capital of Brazil?"* to its answer.

### 6.1 Tokenizer — from text to numbers

The model does not see letters. It sees integers. The **tokenizer** does the
conversion, in four steps. → [`docs/16`](../docs/16-tokenizer.md)

**Step 1 — split into pieces by rule.** A fixed regular expression (the same one
GPT-4 uses) separates words, numbers and punctuation. It is not "split on
spaces": the rule is long and must know what counts as a letter and what counts
as a digit in **any** alphabet — hence the project's generated Unicode tables.

**Step 2 — bytes become symbols.** Every byte from 0 to 255 becomes a character
from a fixed table. That is why a space appears as `Ġ` inside the vocabulary. It
guarantees that **any** file, including broken binary, is representable.

**Step 3 — BPE.** Repeatedly merge the two cheapest adjacent pieces, according to
a table of 321,649 merge rules. With one important exception: if the whole piece
is already in the vocabulary, its id is used directly and **no merge is applied**
(the `ignore_merges` flag). Ignoring that produces wrong tokenization for exactly
the most common words.

**Step 4 — special tokens.** 36 symbols such as `<|user|>`, `<|assistant|>`,
`[gMASK]` are recognized **before** anything else, by exact text match, and never
go through BPE. They are what delimits who is speaking.

Result:

```
"[gMASK]<sop><|user|>Qual e a capital do Brasil?<|assistant|>"
   -> [154827, 154823, 154834, 9061, 383, 261, 8471, 495, 12089, 30, 154835]
```

> **Why this is treated as a critical path:** an error here **does not crash**.
> It produces slightly different ids, and the model answers something plausible
> and wrong. It is the kind of bug that eats days if the test does not isolate
> the tokenizer — hence `--check` against 36 cases generated by the reference
> library.

### 6.2 Embedding — from number to vector

Each id becomes a row of the `embed_tokens` table: 154,880 rows × 2048 numbers.
**One row is read per token.** Since the whole table is 1.27 GB and only 8 KB of
it is read per token, it stays in the mapped file and does **not** consume RAM
budget — a distinction a naive loader gets wrong.
→ [`docs/11`](../docs/11-memory.md)

From here on, the token is a vector of 2048 numbers that the documents call
`hidden` or "the residual stream".

### 6.3 The 47 layers

![Inside one layer](img/04-camada.en.svg)

The vector crosses 47 layers, one after another. Each layer does two things, and
each of them is preceded by normalization and followed by an addition:

```
x = x + Attention(RMSNorm(x))
x = x + MoE(RMSNorm(x))
```

The addition is called the **residual**, and it is what allows stacking 47 layers
without the signal degrading: each layer **corrects** the vector rather than
replacing it.

Layer 0 is different: instead of MoE it has a plain MLP of size 10,240
(`first_k_dense_replace: 1`). Layers 1 to 46 are MoE.

### 6.4 Attention and the KV cache

![MLA: store compressed](img/06-mla.en.svg)

**Attention** is the mechanism by which the current token looks at all previous
ones. Every past position has a key (K) and a value (V); the current token
produces a query (Q), compares it against all keys, and takes a weighted average
of the values.

**Why a cache exists:** the keys and values of past positions do not change.
Recomputing them per token would mean redoing the whole prompt for every word. So
they are stored — that is the **KV cache**.

**Why it is the problem:** during decode, attention **rereads the entire cache
every token**. Cache size is, literally, the cost.

**What MLA changes:** instead of storing expanded keys and values
(20 heads × (256+256) = 10,240 numbers per token per layer), a compressed version
of **576 numbers** is stored. Expansion happens on the fly — or rather, it
**does not happen**: an algebraic identity lets you project the *query* once
instead of expanding every cached position. That is **absorption**.

```
expanded KV    47 layers × 4,096 positions × 10,240 × 4 B = 7.9 GB per token
compressed KV  47 layers × 4,096 positions ×    576 × 4 B = 443 MB per token
```

18x less traffic [measured: 3.64 → 9.98 tok/s at 2,048 of context].

> **The trap the first version of absorption fell into.** The identity involves a
> transposed matrix, and "transposed" suggests "materialize it in fp32 so columns
> can be read". Done that way, absorption came out **slower** (5.6 vs 12.3 tok/s):
> materializing cost 863 MB of weight reads per token, more than the 443 MB
> absorption saved in KV. The fix was noticing the transpose can be computed by
> accumulating **rows**, which is what the format delivers cheaply.
>
> The project's lesson: *trading work for memory needs the traffic arithmetic
> done first.* → [`docs/13`](../docs/13-open-decisions.md), D11

### 6.5 MoE — 64 stored, 5 used

![Mixture-of-Experts](img/05-moe.en.svg)

In an MoE layer:

1. A small matrix (the **router**, or `gate`) scores each of the 64 experts.
2. The **top 4** are chosen; scores are renormalized (`norm_topk_prob: true`) and
   multiplied by 1.8 (`routed_scaling_factor`).
3. Those 4 experts process the vector; a **fifth**, the **shared expert**, always
   runs.
4. The five outputs are summed weighted by the scores.

**The choice is per token and per layer.** A token may activate completely
different experts in the next layer. There is no topic, no nameable specialty.

`n_group` and `topk_group` are parameters that, in other models, restrict the
choice to certain blocks of experts. In this model they are both 1 — meaning
**no restriction**. That is why the project implements only this path on day one,
but keeps the variant declared rather than hard-wired into the parser (streaming engine's
mistake, which shrank the universe of compatible models to a single family).
→ [`docs/06`](../docs/06-lessons-learned.md)

### 6.6 lm_head, logits and sampling

After layer 46, a final normalization, and then the `lm_head` matrix
(2048 × 154,880) turns the vector into **154,880 scores** — the *logits*.

The **sampler** picks one:

| mode | how it picks |
|---|---|
| **greedy** | the highest score, always. Deterministic |
| **temperature** | divides scores by T before drawing. High T = more random |
| **top-k** | draws only among the k highest |
| **top-p** | draws among the highest that together reach probability p |

An easy detail to get wrong: `vocab_size` is 154,880, but only 154,856 ids
actually exist. The remaining 24 are padding; **the sampling mask must know
that**, otherwise the engine can emit a nonexistent id.

The chosen id goes back to step 6.2 and everything restarts.

### 6.7 The end-to-end example, with numbers

```powershell
.\tootega.exe chat models\glm47flash-int4\model.tootega `
    --tokenizer models\glm47flash-bf16\tokenizer.json --ctx 8192
```

```
A capital do Brasil é Brasília.

prefill  18 tokens em 4.63 s (3.9 tok/s)
decode    8 tokens em 0.82 s (9.79 tok/s)
```

What happened: 18 prompt tokens went through the 47 layers at once (prefill),
then 8 tokens were generated one by one (decode), each rereading 2.01 GB of
weights and the whole accumulated KV.

---

## 7 — Why it is slow: the physics of decode

![Prefill and decode](img/13-prefill-decode.en.svg)

### 7.1 Two regimes, same model

| | prefill | decode |
|---|---|---|
| what it does | reads the whole prompt | generates one token at a time |
| parallelism | all tokens together | none: each token depends on the previous |
| weight reads | 2.01 GB for **N** tokens | 2.01 GB for **1** token |
| limited by | arithmetic | **memory bandwidth** |

Decode is what shapes the user experience, and decode is what the bandwidth
arithmetic rules.

> **Honest note:** in the current state, this implementation's prefill is
> **slower per token** than its decode (3.9 vs 9.79 tok/s [measured]). That is
> not a mystery: prefill has not been optimized yet, and the roadmap says so out
> loud — time-to-first-token is not a goal until M3.
> → [`docs/05`](../docs/05-roadmap.md)

### 7.2 The arithmetic

```
time per token = (2.01 GB of weights + 52.9 KB × context) / bandwidth
```

The first term is fixed. The second grows with the conversation.

![The ceiling](img/11-roofline.en.svg)

| context | KV read | total/token | CPU ceiling | ceiling with partial GPU |
|---|---|---|---|---|
| 1,024 | 0.05 GB | 2.06 GB | **29.1 tok/s** | 44.0 [estimate] |
| 4,096 | 0.22 GB | 2.23 GB | **26.9** | 40.6 [estimate] |
| 16,384 | 0.87 GB | 2.88 GB | **20.9** | 31.5 [estimate] |
| 65,536 | 3.47 GB | 5.48 GB | **11.0** | 16.5 [estimate] |
| 131,072 | 6.93 GB | 8.94 GB | **6.7** | 10.1 [estimate] |

### 7.3 What the ceiling does **not** include

Three things, all optimistic:

- **bandwidth efficiency** — well-written engines reach 60 to 80% of peak, not
  100%;
- **compute is not zero** — ~7.1 GFLOP per token, about 20% of the time, and it
  must **overlap** the reads, not add to them;
- **the chain is serial** — 47 layers in sequence; nothing parallelizes with the
  next one when decoding a single conversation.

Applying 60–80%, the honest band:

| scenario | ceiling | **realistic** |
|---|---|---|
| CPU, short context (≤4k) | 27–29 | **17–22 tok/s** |
| CPU+GPU, short context | 41–44 | **25–32 tok/s** |
| CPU, agent context (32k) | ~15 | **10–12 tok/s** |

### 7.4 The consequence that shapes the design

**From ~16k of context onward, KV dominates.** At 64k it is 63% of the traffic
per token. In an agentic CLI, long context is the normal case, not the exception.

Hence two conclusions that recur across documents:

1. optimizing KV reads matters as much as optimizing weight reads;
2. **reusing the prefix is worth more than any kernel** — a turn that reuses
   100k tokens of already-computed KV avoids the entire prefill, and no
   multiplication optimization comes close.

---

## 8 — This machine's hardware and the inversion

![The four bandwidths](img/01-bandas.en.svg)

### 8.1 The measurement that inverts intuition

Nearly every inference engine assumes: *VRAM is fast, so fill VRAM*. On this
machine that is false, and the reason is the PCIe line.

```
pcie.link.gen.hostmax = 5     <- the motherboard offers Gen 5 (63 GB/s)
pcie.link.gen.max     = 3     <- the GPU only speaks Gen 3
```

The motherboard is modern; the GPU is a 2019 RTX 2060 SUPER that only speaks
PCIe gen 3. The link negotiates down to the slower side: **~12 GB/s**.

**Consequence:** any weight that does not *live* in VRAM must cross a channel
slower than the disk it came from, and five times slower than RAM.

### 8.2 The trap, measured

In streaming engine, the `CUDA_DENSE=1` option places the dense tensors on the GPU **even
when they do not fit**:

```
config              tok/s
cpu-native          0.684
cuda-experts4       0.647
cuda-tc-w4a16       0.639
cuda-dense          0.054   <-- 13x slower
```

The engine **did not refuse**. It accepted, and started shipping weights over
PCIe every layer, every token.

That measurement produced two of TootegaEngine's rules:

> **Placement law.** Put on the GPU only what fits there permanently. Never
> stream weights over a link slower than the source tier.

> **Refuse instead of degrade.** If the plan does not fit the declared budget,
> the engine exits with an error naming what to reduce.

```
ERRO: nao cabe em --ram-gb 12
  pesos (int4)          18,1 GB
  KV (ctx 8192 x 1)      0,4 GB
  scratch + runtime      1,2 GB
  ------------------------------
  necessario            19,7 GB   excede em 7,7 GB
sugestao: --ram-gb 21, ou --gpu-layers para mover peso para a VRAM
```

### 8.3 The way out: layer partition

![Layer partition](img/08-particao.en.svg)

If weights cannot cross PCIe, what can? **Activations.**

An activation vector is 2048 numbers × 4 bytes = **8 KB**. An expert is ~4.7 MB.
Three orders of magnitude apart.

Hence the architecture's central bet:

> **Partition by layer, not by tensor.** Each layer's weights are permanently
> resident on the device that computes it. Only 8 KB of activation crosses
> between layers.

With ~7 GB of usable VRAM and ~18.1 GB of layer weights, about **39%** of the
layers would fit on the GPU. Per token, PCIe would carry 16 KB — 1.3
microseconds. Irrelevant.

**This is still a hypothesis.** It is marked [estimate] and milestone M2 exists
to validate it **or kill it**: if the gain falls below 20%, M2 is *cancelled*,
not postponed, because maintaining two backends is a permanent cost.
→ [`docs/03`](../docs/03-architecture.md), [`docs/05`](../docs/05-roadmap.md)

### 8.4 What the measurement says about buying hardware

A counterintuitive detail: if a GPU upgrade ever happens, the bigger gain is
**not** more VRAM or more compute — it is the **link**. Ampere/Ada cards speak
Gen 4 (31.5 GB/s) and Blackwell speaks Gen 5 (63 GB/s). And 63 GB/s **exceeds RAM
bandwidth**, which would invert the hierarchy again and make streaming to the GPU
a good deal for the first time.

---

## 9 — Residency: the decision worth 20x

![Two scaling laws](img/02-residencia.en.svg)

### 9.1 The two measured axes

With the oversized model (744B, streamed from disk), varying resources:

| axis | variation | gain | exponent |
|---|---|---|---|
| RAM (16 → 60 GB) | 3.75x | 1.42x | ~0.26 |
| concurrency (K=1 → K=4) | 4x | 1.45x | 0.27 |

**Both give `N^0.25`.** Quadrupling anything yields about 40%.

### 9.2 Why

Because the governing variable is not hardware, it is the **residency fraction**.
With a 60 GB budget and 324 GB of experts, ~15% of the model is cached; the other
85% comes from disk every token, and no reasonable amount of RAM changes that.

Multiplying by N only approaches N× over the stretch where N makes you **cross
the residency threshold**.

### 9.3 The practical conclusion

> Choosing a model that already fits yields 20 to 50x. Buying hardware yields
> `N^0.25`.

That is the whole thesis in one line. And it is why TootegaEngine **throws away**
all the disk-hiding machinery streaming engine built — prefetch, dual-SSD routing,
`O_DIRECT`, learned pin sets, per-layer eviction. None of it helps when the model
is resident, and all of it is the bulk of that project's complexity budget.

**TootegaEngine's main structural advantage over streaming engine is not speed — it is
code size.**

---

## 10 — How the model is stored

### 10.1 Why quantize

The published model uses `bfloat16`: 2 bytes per weight, 62.4 GB total. It does
not fit in the machine's 64 GB alongside everything else.

Storing each weight in **4 bits** with a shared scale, the same model takes
19.65 GB — and fits with wide margin.

### 10.2 How it works, concretely

![int4 on disk](img/07-int4.en.svg)

Take 64 neighbouring weights. Find the largest absolute value among them. Divide
by 7. That is the step (the **scale**). Now each weight becomes an integer from
−8 to +7:

```
code   = round(weight / scale)             stored as 0..15 (code + 8)
weight ≈ (code − 8) × scale
```

64 weights take 32 bytes (two per byte) plus 4 bytes of scale = **0.5625 byte per
weight**.

**Nibble order matters.** The adopted convention is **half/half**: byte `j` holds
weight `j` in the low nibble and weight `j+32` in the high one. That way, loading
32 bytes at once with SIMD instructions, `v & 0x0F` yields 32 **contiguous**
weights and `v >> 4` yields the next 32 — two vectors ready, no shuffling. The
alternative (even/odd) would force permuting the activations for every group.
→ [`docs/14`](../docs/14-container-format.md)

### 10.3 What it costs in quality — and the surprise

Quantization error was measured two ways.

**First: per-tensor error.** How much a multiplication's output changes:

| format | output error |
|---|---|
| int4 g64 | 10.81% |
| int8 g64 | 0.62% |

The ratio between them (17.4) matches the theoretical step ratio (`127/7 = 18.1`).
That is the **floor of round-to-nearest**, not a bug.

**Second: perplexity.** How much the model actually degrades on real text:

```
int4 g64 against bf16:  +5.26% perplexity   [measured, 10 windows, paired]
```

Half of what the per-tensor error would suggest — confirming that per-tensor
error **does not predict quality**.

**The surprise came from ablation.** Restoring one group of tensors at a time to
the original format and measuring:

| restored group | tensors | share of the degradation explained |
|---|---|---|
| **attention** | 193 | **100.3%** |
| shared experts | 141 | 20.0% |
| dense layer 0 | 3 | 2.8% |
| **routed experts** | **9,024** | **−33.6%** |

**All of the degradation came from attention.** The 9,024 expert tensors — 93% of
the file — were not a source of loss.

The reason is structural: each token activates 4 experts out of 64, so an
expert's error enters diluted and uncorrelated. Attention runs **in full, across
all 47 layers, for every token**, and its error compounds with depth.

**The prediction was tested and met.** Reconverting with attention in int8:

| container | attention | perplexity vs bf16 | size |
|---|---|---|---|
| v1 | int4 g64 | +6.81% | 19.33 GB |
| **v2** | **int8 g64** | **−0.19%** | **19.65 GB** |

v2's 95% interval **crosses zero**: the quantized version can no longer be
distinguished from the original by this measure. Cost: 1.7% of size.
→ [`docs/04`](../docs/04-target-model.md)

### 10.4 The `.tootega` container

![The .tootega file](img/10-container.en.svg)

A single file, memory-mapped:

```
offset 0        fixed header, 128 bytes
offset 128..    data blobs, each aligned to 64 bytes
index_off       JSON index
EOF
```

Three decisions that look like details and are not:

**64-byte alignment.** That is a cache line. Without it, reading straight from
the mapped file with SIMD would need a copy path to realign — and the whole
zero-copy design would be lost.

**The entire `config.json` goes inside the index.** Architecture comes from the
container, not from a sibling file that can drift away from it.

**An unknown version is refused, never read "hopefully".** Different magic,
different version → error.

v2 contents [measured]: int4 8,973 tensors (15.91 GB), int8 438 (2.44 GB),
f32 290 (25.7 MB), bf16 2 (1.27 GB). Resident: 18.38 GB.

---

## 11 — Memory: one copy, not one more

The project's general rule: **a byte of weight exists in exactly one place.**
→ [`docs/11`](../docs/11-memory.md)

### 11.1 The mapped file **is** the memory

There is no "load the model" step: the file is mapped and the quantizer's
pointers point **into the mapped pages**.

```c
qt->q4 = (uint8_t*)(base + tensor->offset);   /* nao ha memcpy */
qt->s  = (float*)  (base + scales->offset);
```

Three consequences:

- **the operating system's page cache is the cache** — no double buffering, no
  hand-managed residency;
- **weight pages are reclaimable** — under pressure the OS drops and rereads
  them. That is why they do **not** count against the RAM budget: they are
  *file-backed*, not anonymous;
- **two processes on the same model cost one set of weights, not two.**

### 11.2 Scratch arena: zero `malloc` per token

Every temporary buffer in the forward pass has a fixed size, known once the
configuration is read. They are slices of **one arena sized once at startup**.
The token loop never calls the allocator.

streaming engine does ~500 `malloc`/`free` per token. At 0.93 tok/s that is 0.01% and does
not matter; at 22 tok/s it does — and the fix is structural, not a
micro-optimization for later.

### 11.3 The three caches, and why conflating them is expensive

![Three caches](img/15-tres-caches.en.svg)

| cache | holds | owner |
|---|---|---|
| **weights** | tensors per tier (VRAM / RAM) | the engine, exclusively |
| **KV** | attention state of tokens already seen | the engine owns the mechanism; the agent determines hits |
| **prefix** | the token sequence you assemble | the agent, exclusively |

The engine **sees integers**. It does not know what a "system prompt" is.
Therefore:

> **The engine cannot fix a badly ordered prompt.**

### 11.4 The ordering rule, and why it is a performance decision

![Prefix cache](img/09-prefixo.en.svg)

The engine matches the prefix token by token, **stops at the first difference and
truncates from there on**. Everything after the first changed token is
recomputed.

> **Stable first, volatile last.**
> `system → tool schemas → long context → history → current turn`

A clock, a UUID or a directory listing at the start of the prompt invalidates the
next 100k tokens, and the engine has no way of knowing that part was irrelevant.

Two consequences specific to a CLI with tools:

- **tools render at position 0** — swapping, adding or reordering a tool
  invalidates everything. There must be an *append* path, not a replace path;
- **an instruction arriving mid-stream must not edit the front** — late state
  ("the mode changed", "the file changed on disk") goes at the **end** of the
  prefix.

The mechanism measured at work: **92 of 100 prompt tokens came from cache on the
second turn** — 8 tokens of prefill instead of 100. [measured]

**Hash addressing, and the bug it prevents.** A KV block is addressed by the
**accumulated hash of the exact token sequence that produced it** — never by
slot, never by position. The reason is concrete: in streaming engine, one run diverged
from the others because a persisted KV **from another conversation** partially
matched the new prompt. The engine resumed from a state that did not correspond
to that text. Partial matching does not exist: either the hash matches, or the
block is ignored. → [`docs/12`](../docs/12-persistent-state.md)

### 11.5 Budgets: what counts and what does not

| region | counts against `--ram-gb`? | why |
|---|---|---|
| the model mapping | **no** | file-backed, the OS reclaims it |
| pages locked with `mlock` | **yes** | effectively anonymous |
| KV blocks | **yes** | anonymous |
| scratch arena | **yes** | anonymous |
| pinned GPU buffer | **yes** | anonymous and non-pageable |

And the cap is **verified at runtime**, not just in the plan — planning estimates
are wrong sometimes, and without verification "budget" becomes decoration.
→ [`docs/09`](../docs/09-resource-limits.md)

Defaults for this machine:

```
--ram-gb 48          # of 64, leaving 16 for the OS and your own work
--vram-gb 6.5        # of 8, reserving 1.5 for driver and desktop
--threads 16         # the 8 P-cores with SMT
--affinity p-cores
--ctx 8192
--kv-slots 1
```

---

## 12 — How we know it is correct

This is probably the project's most characteristic section.

### 12.1 The oracle and 32/32

There is a reference implementation (`Tests/oracle`) that runs the same model,
over the same container, by a path that shares no line with the engine: its own
container reader, its own dequantiser, its own matmul, its own attention. It is
slow and simple — on purpose. It was in Python once, with `numpy`; today it is C,
because two implementations agreeing is evidence and one implementation agreeing
with itself is a tautology, and neither of those depends on the language.

The engine runs 32 positions under *teacher forcing* (that is, forcing the known
text) and compares token by token. **32 of 32** or it does not pass.

Why binary: because it separates an *engine* from a *plausibility generator*. A
"similar" result is not a result.

### 12.2 Paired measurement — and the mistake that created the rule

![Paired measurement](img/12-pareada.en.svg)

The first two perplexity measurements of int4 gave:

```
191 predictions   ->  −4.2%   (int4 "better" than the original — impossible)
381 predictions   ->  +2.5%
```

**The sign flipped when the sample doubled.**

The cause: perplexity ranges from **223 to 473 across windows** of that text.
That variance swallows a 5% effect. Comparing two global means would have
produced a publishable and **wrong** number.

The fix is comparing **the same window under both configurations** and measuring
the difference, whose deviation is twenty times smaller:

```
nll delta (int4 − bf16):  +0.0513 ± 0.0076
95% interval:             [+0.0365, +0.0661]
perplexity:               +5.26%
```

Ten windows out of ten positive, interval does not cross zero. That is a
measurement, not an impression.

> The rule became project doctrine: **paired comparison, always.**

### 12.3 What has already been rejected

Four **good** optimizations, implemented, measured, and not adopted as defaults:

| optimization | what it promised | result |
|---|---|---|
| **KV in bf16** | half the cache memory | **29/32.** Three tokens changed. Available via `--kv bf16`, not the default |
| **KV in int8 g64** | 3.8x smaller cache, lower error than bf16 | **29/32** as well. And only 2% faster |
| **online softmax** | half the KV reads | **9% slower** at 4k — the reads it cut were already coming from L3 |
| **materialized absorption** | operating in the compressed space | 2x **slower** — traded KV traffic for weight traffic |

Two observations worth more than the optimizations themselves:

**The criterion detects "different", not "worse".** Two quantizers of very
different quality (1.5% and 0.59% error) failed with the same score. The reason
is not their quality: it is that this prompt's margins are small — 0.0007 at
position 31. To decide whether one of them **actually degrades** the model, the
specific perplexity measurement is missing. Until it exists, the default is
whatever passes the test that does exist.

**"Fewer reads" is not the same as "faster"** when the reads you cut were already
coming from cache.

### 12.4 The other measurement disciplines

All of them came from a mistake made:

- **never change two variables at once** — "native + CUDA" was measured against
  "release CPU" and the result said nothing;
- **fix the RAM budget across configurations** — the "auto" value floated between
  47.1 and 60.5 GB depending on the moment, invalidating an entire benchmark
  matrix;
- **warm up before measuring concurrency** — without warm-up, the cold-tier cost
  lands entirely on K=1 and invents a scaling gain that does not exist;
- **hash the output in every measurement** — that is how it was found that CPU
  and GPU produce identical bytes.

### 12.5 The final rule

> **Publish what was measured, not what looks safer.**

The case that fixed the rule: when reconverting the container to v2, one tensor
changed precision without ablation having evaluated it. The temptation was to
revert "to be safe". But the container **that was measured** was the one with
that tensor at that precision — and it measured loss indistinguishable from zero.
Reverting would produce an *untested* configuration to buy a guarantee the
measurement already gave.

### 12.6 The test that runs in a second

```powershell
.\tootega.exe selftest
```

Checks kernels, the JSON parser, the thread pool, the tokenizer and the sampler
against precomputed fixtures (`src/fixture_kernels.h`, `tokfixture.json`). It is
the first command to run after any change.

---

## 13 — The server, the protocol and the agent

The previous sections explain how a token comes out of the model. This one
explains how that token reaches a program that is good for something.

### 13.1 Why there is a server

Loading the model costs **5.3 seconds and 18.54 GB**. If every question spawned
a new process, every question would pay that again — and the machine does not
have the RAM for two at once.

So the design is the usual one in server software, and it reads the same for a
layperson and a specialist: **one expensive process that stays up, and cheap
clients that come and go.** The server starts, warms the memory, and sits idle
using **0.2% of one core and 0.04 GB**. Anyone who wants to talk to the model
opens a connection, asks, and hangs up.

What it exposes:

| address | what it does |
|---|---|
| `POST /v1/chat/completions` | the conversation: takes the messages, returns the text |
| `GET /v1/models` | says which model is loaded |
| `GET /stats` | how many tokens, how much time, how much cache was reused |
| `GET /` | the web chat embedded in the executable itself |

### 13.2 Speaking the protocol the world already speaks

One message format became a de-facto standard: the one OpenAI published. A list
of messages, each with a **role** (`system`, `user`, `assistant`, `tool`) and a
**content**; the reply comes in the same shape.

There is nothing special about that format. What is special is that
**everybody has already implemented it**. Copilot, Continue, Open WebUI,
libraries in ten languages: none of them knows TootegaEngine exists, and all of
them work with it — you just point the URL. Writing our own protocol, however
much better, would mean writing every one of those clients too.

**A strong design consequence:** the engine decides nothing about the
conversation. It takes messages and returns text. Who assembles the history,
who decides to call a tool, who repeats the cycle — that is the client. The
saying is that **the server is dumb and the client is smart**, and it is a
compliment: it is what makes a program written for another engine work here
without a line of our code.

### 13.3 Why the answer appears letter by letter

At 13 tokens per second a 300-token answer takes **23 seconds**. Staring at a
frozen screen for 23 seconds is unbearable; watching the sentence form is
comfortable — and it is exactly the same wall clock.

The mechanism has a name, it is browser-standard, and it is not WebSocket:
**SSE**, *Server-Sent Events*. The server keeps the connection open and sends
one chunk of text at a time, as `data: {...}`, until it sends `data: [DONE]`.
The flow is one-way — server to client — which is precisely the case here.

### 13.4 Tools: the model asks, it does not execute

A language model can only produce text. It does not open files, run commands or
reach the internet. The convention is the following piece of theatre:

1. the client describes, in the prompt, the available tools and the shape of
   each one's arguments;
2. the model, instead of answering, **writes a request** for execution;
3. the client actually runs the request and returns the result as a new
   message;
4. the model continues from where it stopped, now knowing the result.

Step 3 is the only one that touches the real world, and what runs it is an
ordinary program with the permissions of whoever started it. **The model never
executes anything.**

In this model the request is not loose text to be guessed at: `<tool_call>`,
`<arg_key>`, `<arg_value>` and `<|observation|>` are **single tokens in the
vocabulary**, put there during training. The server translates the OpenAI
format into those tokens and translates the reply back. It is translation, not
invention — which is why it works on the first attempt instead of depending on
a regular expression over the output.

**The part that nearly went wrong.** The protocol's JSON distinguishes the
number `3` from the text `"3"`; the model writes both the same way. Without
consulting the **JSON Schema** the client sent along with the tools, every
numeric argument would arrive as text on the other side, and the tool would get
`"3"` where it expects `3`. The server types each argument by the declared
schema — and that is the difference between an integration that works and one
that almost works.

### 13.5 The agent is the loop, not the model

The **agent** (`tools/agent.exe`) is the program that closes the four-step cycle
above until the task is done. It ships six tools:

| tool | what it does | asks first? |
|---|---|---|
| `Read` | reads a file | no |
| `Glob` | finds files by name pattern | no |
| `Grep` | searches for text inside files | no |
| `Write` | writes a file | **yes** |
| `Edit` | replaces one snippet with another | **yes** |
| `Bash` | runs a command | **yes** |

Two decisions inside that table deserve a slow read.

**The fence.** Every tool resolves its path against the working folder and
refuses anything that escapes it — even with auto-approval on. It is the only
thing between a 31-billion-parameter model and the rest of the disk.

**Reading does not ask.** It looks careless and it is the opposite. At 13 tokens
per second a real task performs dozens of reads; confirming each one turns the
work into half an hour of pressing Enter — and, worse, **teaches the person to
approve without reading**. Asking only where something changes state keeps the
question rare enough to be read.

### 13.6 How the agent talks to an interface

The agent opens no network port. It talks over **stdin and stdout** — the two
text channels every program is born with — in **NDJSON**: one JSON object per
line, one event per line. The reader processes the first line without waiting
for the last.

```
{"type":"system","subtype":"init","session_id":"...","tools":[...]}
{"type":"assistant","message":{...}}
{"type":"user","message":{"content":[{"type":"tool_result",...}]}}
{"type":"result","subtype":"success","duration_ms":8213}
```

That is the same process contract the **Claude Code CLI** uses. The consequence
is practical: an interface written for that CLI talks to this agent **by
changing the path of the executable**, and nothing else. That is how Cockpit
came to run both side by side, N tabs of one and one of the other.

Closing the channel ends the agent. No timer, no signal, no orphan process:
when the interface closes stdin, the agent receives an **EOF** and exits. It is
the most reliable shutdown there is, because the operating system itself
delivers it.

### 13.7 The chat that lives inside the executable

The server serves a chat page that **does not exist as a file**: it is a string
compiled inside `tootega.exe`. There is no `www/` folder to forget to copy, no
relative path to get wrong, no version of the HTML different from the version of
the binary.

The entry screen asks for three things, and each has a reason:

- **the name** — it is the person's, and it goes into the system message;
- **the directives** — how the model should answer ("answer in the language of
  the question", "get to the point"). They sit at the **beginning** of the
  prompt, which is where the prefix cache reuses them on every turn;
- **reasoning on or off** — off by default, because of the measurement in
  §14.6.

### 13.8 The prefix cache, in practice

The prefix cache is the most misunderstood of the three caches (§3.1). It stores
neither questions nor answers: it stores the **internal state** corresponding to
the first N tokens of the prompt. If the next message begins with exactly the
same tokens, that state is reused and only the rest is computed.

"Exactly the same" is literal — one extra space invalidates everything from
there on. That is why prompt ordering is a rule and not a preference:

```
system  ->  tools  ->  long context  ->  history  ->  current turn
    stable                                           volatile
```

Whatever changes each turn must sit **at the end**. With the right order a
typical turn reuses **99% of the prefill** — 1163 of 1172 tokens in the console
example. With the wrong order it reuses none, and every turn pays for the whole
prompt again.

---

## 14 — How quality is measured

Speed is easy to measure: a stopwatch. Quality has no stopwatch, and that is
where nearly every model evaluation turns into opinion. This section shows the
four numbers this project uses, what each one sees, and — more importantly —
what each one does **not** see.

### 14.1 Perplexity: how surprised the model was

You give the model a text it never chose and ask, position by position: *what
probability did you assign to the word that actually came?* Perplexity is the
average of those surprises, turned into a number with an intuitive meaning:
**how many options the model was effectively torn between**.

Perplexity 8 means "hesitating among some 8 possibilities". Lower is better. It
is the standard ruler, and it is objective: it does not depend on someone
finding the answer good.

This engine's number, against the external reference that ran the original model,
on the same text:

| measurement | engine | `transformers` |
|---|---|---|
| 384 positions of prose, BF16 weights | **8.63** | **8.89** |
| same text, int4 container | **5.66** | — |

**Careful with the second row.** Perplexity only compares over the *same text
and the same vocabulary*. Perplexity numbers from two different papers do not
compare — it is the most common mistake with this metric, and it is everywhere.

### 14.2 Entropy and confidence: what perplexity hides

Perplexity measures the hit *on the word that came*. It does not distinguish two
very different states:

- the model had **one** strong candidate and got it right;
- the model had **forty** similar candidates and one of them happened to come.

**Entropy** separates them. Measured in *bits*, it says how spread out the
model's opinion is: low, few candidates; high, many nearly tied. **Confidence**
is the practical complement — the probability it gave to the front-runner.

The server console prints all four together, and it is the joint reading that
informs:

```
prompt      perplexity 3.41  nll 1.2267  entropy 2.18 bits  confidence 47.3%
generation  perplexity 1.84  nll 0.6088  entropy 1.02 bits  confidence 71.9%
```

**Very** low entropy for a long stretch is not a sign of excellence: it is the
symptom that precedes the model falling into a loop and repeating.

### 14.3 rep4: the repetition detector

`rep4` is the fraction of **four-token** sequences that already appeared earlier
in the same answer. It counts sequences, not words — which is why it sees the
problem early, while it still does not bother a reader. Above ~15% is outright
**degeneration**: the text folding onto itself.

Degeneration is not a hang. The program is working perfectly and producing
garbage — which makes it invisible to any test that merely checks that output
happened.

### 14.4 match@ref: how many tokens before diverging

How many leading tokens a generation shares with the reference generation before
taking another path. It is the most direct measure of *"this configuration
generates what the original would"*.

Diverging is not an error. Two near-identical runs split at the first near-tie,
and from there on write different and equally valid texts. What the number tells
you is **when** the numeric difference grew large enough to change a choice.

### 14.5 Paired measurement, and why it exists

Here is the mistake this project made and that became a rule.

To compare two configurations, the natural thing is to measure each one's mean
perplexity and subtract. That is what was done. The first two attempts gave
**−4.2%** (int4 "better" than the original, which is impossible) and then
**+2.5%**. The sign **flipped** when the sample doubled.

The cause: perplexity on that text varies from **223 to 473 across windows**. A
swing like that swallows a 5% effect without leaving a trace.

The fix is to compare **window by window**, over the same text, and measure the
*difference* — whose deviation is twenty times smaller, because both
configurations see exactly the same words. It is called **paired measurement**,
and without it the project would have published a presentable, wrong number.

Its mandatory companion is the **confidence interval**: the range the true value
probably lies in. If the range **crosses zero**, the measurement cannot tell
"better" from "worse", and the honest answer is *inconclusive* — not "about the
same".

**That is exactly what happened with int8 KV.** The criterion had been fixed
before measuring; the result came back with both intervals crossing zero. The
memory saving was real and large (105.8 KB per position in f32 against 28.1 KB
in int8), the temptation to adopt was enormous, and it was still recorded as
**inconclusive, not adopted** — available as an option, never as the default.
Changing the criterion after seeing the number is the commonest way to lie with
statistics, and it is almost always unintentional.

### 14.6 Does reasoning before answering help?

The model can write a draft to itself, between `<think>` and `</think>`, before
answering. The general belief is that this improves the answer. We measured it,
with 12 questions and a 400-token budget:

| | rep4 | finished | time |
|---|---|---|---|
| without reasoning | 1.7% | 12/12 | 78 s |
| **with reasoning** | **13.8%** | **4/12** | **354 s** |

With reasoning the model **fails to finish 8 of the 12 prompts**, repeats 8×
more 4-grams — 13.8% brushes the degeneration threshold — and costs **4.5× the
time**.

The illustrative case: for *"What is the capital of Brazil?"* it writes six
reasoning steps in English, with three alternative draft answers, and is cut off
by the budget **before answering**.

The correct reading is not "reasoning is bad". It is: **in this model, with this
budget, and for questions this size, reasoning is expensive and answers worse.**
Hence the default is off — and the option stays there, for whoever has bigger
questions and more patience.

### 14.7 The measurement that decided against the GPU

It is worth keeping as a final example of method, because the result contradicts
everyone's intuition.

This machine's GPU has 8 GB; the model has 18.54 GB. It **does not fit**. What
remains is splitting the work between GPU and CPU. The CUDA backend was written,
and the kernels were optimised until they were **31× faster** than the first
version and **2.6×** faster than the CPU on the same data (179–190 GB/s against
68.7).

And the complete decode, measured, came out **slower** — in all three
configurations tried. The reason is in §8: this machine's PCIe (~12 GB/s) is
slower than everything else, and any partition pays that toll on every layer.

A fast kernel is not a fast system. The work was kept on a separate branch, with
the measurement next to it, and the engine stayed on pure CPU. **A well-measured
negative result is a result** — it saves the next person who has the same good
idea.

---

## 15 — Where we are and what comes next

![Roadmap](img/14-marcos.en.svg)

Every milestone has a **measurable exit criterion**. No number, no sign-off.
Two of them have a **kill criterion**: if it does not pay, it is cancelled, not
postponed.

| milestone | what it delivers | exit criterion | state |
|---|---|---|---|
| **M0** | convert and validate the model | aligned int4 container + measured perplexity | **done** |
| **M1** | CPU-only forward, token-exact | 32/32 on the oracle; ≥10 tok/s at 4k; stable RSS; `--ram-gb` respected | **done** |
| **M2** | per-layer GPU/CPU partition | byte-identical output to M1; **≥20% faster, else CANCELLED** | **cancelled by measurement** ([§14.7](#147-the-measurement-that-decided-against-the-gpu)) |
| **M3** | server and continuous batching | K=4 yields ≥2.5× K=1; latency ≤2× | ahead |
| **M4** | agentic CLI | tools, sandbox, process contract | **done** ([§13](#13--the-server-the-protocol-and-the-agent)) |
| **M5** | optimisation, with a number | every item with before/after in the same configuration | ahead |

### What the project declares it is **not**

- **not a disk-streaming engine** — if the model does not fit, use something
  else;
- **not multi-GPU** — one GPU, one host;
- **not a production server** — no multi-tenancy, no authentication beyond
  loopback, no quotas;
- **it trains nothing** — no training, no fine-tuning, no LoRA.

### Declared failure criteria

The project is wrong if CPU-only does not exceed 10 tok/s at 4k, if the GPU/CPU
split yields less than 20%, or if the output diverges from the oracle under any
placement. → [`docs/01`](../docs/01-goals.md)

---

## 16 — Hands on

### 16.1 Build

```powershell
.\build.ps1                    # engine (MSVC, located via vswhere)
.\build.ps1 test               # build and run the selftest
.\tools\<name>\build.ps1       # one specific tool
```

On Linux, `make`. The compiler is located dynamically — **never** by a hard-coded
path.

One build option deserves a note: `/fp:precise`, with no FMA contraction. It
stops the compiler from reassociating sums and from fusing multiply with add.
It costs a little performance and buys **reproducibility**, which is an exit
criterion, not a luxury. It was verified in the disassembly, not assumed.
→ [`docs/10`](../docs/10-implementation.md)

### 16.2 The commands

```powershell
.\tootega.exe selftest
.\tootega.exe version
.\tootega.exe inspect  models\glm47flash-int4\model.tootega
.\tootega.exe plan     models\glm47flash-int4\model.tootega --ctx 4096
.\tootega.exe tokenize models\glm47flash-bf16\tokenizer.json --check tokfixture.json
.\tootega.exe forward  models\glm47flash-int4\model.tootega --ids-file ids32.txt
.\tootega.exe generate models\glm47flash-int4\model.tootega --tokenizer ... --prompt "..." -n 60
.\tootega.exe chat     models\glm47flash-int4\model.tootega --tokenizer ... --ctx 8192
.\tootega.exe serve    models\glm47flash-int4\model.tootega
```

| command | what for | when to use |
|---|---|---|
| `selftest` | kernels, JSON, pool, tokenizer, sampler against fixtures | always, after any change |
| `version` | version and detected hardware | when reporting a result |
| `inspect` | header, index and budget of the container | to see what is inside the file |
| `plan` | what fits: weights, KV and scratch against `--ram-gb` | **before** running, so you do not find out halfway |
| `tokenize` | tests the tokenizer against the fixture | when touching anything text-related |
| `forward` | *teacher-forcing* pass over fixed ids | this is the oracle's command |
| `generate` | tokenizer + forward + sampling | to watch the model write |
| `chat` | multi-turn conversation with prefix cache | normal use |
| `serve` | HTTP server + web chat at `127.0.0.1:8080` | normal use, from the browser |

Options that appear almost everywhere:

| option | what it does |
|---|---|
| `--threads N` | how many compute threads. Default: physical P-cores |
| `--affinity p-cores\|all\|none` | where to pin threads |
| `--kernel auto\|scalar` | `scalar` disables SIMD; useful to isolate a kernel bug |
| `--attn absorbed\|expanded` | which attention formulation. Default: `absorbed` |
| `--ctx N` | KV capacity |
| `--ram-gb N` | hard memory cap |
| `-v` | details what is being done |
| `--kv-prec f32\|bf16\|int8` | KV cache precision. Default `f32`; the others are a measured option, not the default |
| `--think\|--no-think` | turns the pre-answer draft on or off. Default: off |
| `--verbose` | in `serve`, prints the shape and statistics of each request — never the content |

### 16.3 How to make your first measurement without getting it wrong

Four rules, all bought with lost time:

1. **fix `--ram-gb` and `--threads`.** Two measurements with different budgets
   are not comparable;
2. **warm up.** Discard the first run; the warm-cache effect was already worth
   +20.7% in one baseline measurement;
3. **change one thing at a time.** If you changed the kernel and the thread
   count, the result says nothing;
4. **record the output hash.** That is how you discover that a speed difference
   came with a result difference.

---

## 17 — Repository map

### 17.1 The folders and each one's rule

| folder | what it is | rule |
|---|---|---|
| `src/` | the engine | ISO C11, **zero dependencies**; only `compat.c` has OS `#ifdef`s |
| `tools/` | **production** tools | C or Rust, one folder per tool, the `.exe` at the root of `tools/` |
| `Tests/` | **measurement** tools | same rule; they measure the engine and may use the original checkpoint |
| `docs/` | the specification | every decision with its why and the number behind it |
| `models/` | the containers | not versioned |
| `ForDummies/` | this document and the figures | PT and EN in parallel |
| `LICENSE` | the MIT licence | it covers the code; the model has its own |

**A new tool is C or Rust, never Python.** The no-dependency ban applies to
`src/` — the engine must be auditable in one reading session and it consumes
third-party files. A tool is a different thing: it may link whatever it needs, as
long as that does not enter `src/`. And a C tool **should reuse the sources in
`src/`**, so that it measures the engine rather than a reimplementation of it.
→ `CLAUDE.md`

### 17.2 The engine's files

| file | responsibility |
|---|---|
| `main.c` | subcommand dispatch and option parsing |
| `model.c` | loading the container, config → internal structure |
| `container.c` | the `.tootega` format, reading and writing |
| `forward.c` | the pass through the 47 layers |
| `kernels.c` | the arithmetic: int4 matmul, attention, rmsnorm, RoPE |
| `tokenizer.c` | text ↔ ids |
| `sample.c` | greedy, temperature, top-k, top-p |
| `pool.c` | the project's own thread pool (no OpenMP) |
| `budget.c` | budgets and the refusal |
| `server.c` + `webui.h` | HTTP and the web interface |
| `json.c` | the project's own JSON parser |
| `compat.c` | **the only file with OS `#ifdef`s** |
| `unicode_tables.h` | generated tables for the pre-tokenizer |
| `fixture_kernels.h` | the `selftest`'s correct answers |

### 17.3 The tools

| tool | what it does |
|---|---|
| `tools/convert` | BF16 → int4/int8 container, per-tensor precision policy and the clip search |
| `tools/agent` | the agentic client: six tools, sandbox, approval, `--tty` mode |
| `tools/finetune` | packs a corpus into token windows for the trainer |
| `tools/ucd` | generates and **checks** `src/unicode_tables.h` from the Unicode database |
| `tools/fixture` | generates `src/fixture_kernels.h`, the `selftest`'s correct answers |
| `Tests/oracle` | the independent reference implementation, and the 32/32 |
| `Tests/ablate` | restores groups of tensors to the original and measures the effect, paired |
| `Tests/gentest` | generation quality: `rep4`, stop rate, `match@ref` |
| `Tests/bench` | time, memory and the roofline arithmetic |

> **Two Python files are left**, both outside the engine's path: `build.py`,
> which generates this document, and `download_model.py`, which needs HTTPS to
> fetch the checkpoint. Everything else that was Python — the oracle, the
> perplexity meter, the converter, the Unicode tables, the kernel fixture — is a
> native tool today.

### 17.4 The 20 documents

| # | document | what it decides |
|---|---|---|
| 01 | [Goals and scope](../docs/01-goals.md) | what it is and what it explicitly is not |
| 02 | [Hardware model](../docs/02-hardware.md) | the measured bandwidths and the placement law |
| 03 | [Architecture](../docs/03-architecture.md) | layers, contracts, the three caches |
| 04 | [Target model](../docs/04-target-model.md) | GLM-4.7-Flash and quantisation |
| 05 | [Roadmap](../docs/05-roadmap.md) | milestones with exit criteria |
| 06 | [Lessons learned](../docs/06-lessons-learned.md) | adopt, throw away, avoid |
| 07 | [Baseline](../docs/07-baseline.md) | the numbers to beat |
| 08 | [Performance ceiling](../docs/08-performance-ceiling.md) | roofline |
| 09 | [Resource limits](../docs/09-resource-limits.md) | budgets and refusal |
| 10 | [Implementation](../docs/10-implementation.md) | C11, linkage, portability |
| 11 | [Memory](../docs/11-memory.md) | zero-copy, paged KV |
| 12 | [Persistent state](../docs/12-persistent-state.md) | starting warm |
| 13 | [Pending decisions](../docs/13-open-decisions.md) | the D1–D14 scoreboard |
| 14 | [Container format](../docs/14-container-format.md) | the `.tootega` file byte by byte |
| 15 | [Kernels and numerics](../docs/15-kernels-and-numerics.md) | W4A32, accumulation order |
| 16 | [Tokenizer](../docs/16-tokenizer.md) | BPE, what to implement and how to test |
| 17 | [Diagnosis](../docs/17-diagnosis.md) | the bugs, and why the tests missed them |
| 19 | [Reference state](../docs/19-reference-state.md) | the snapshot of what exists and what was measured |
| 20 | [Agent and protocols](../docs/20-agent.md) | tools, the CLI contract, the cost of reasoning |
| 21 | [int8 KV](../docs/21-int8-kv.md) | measured, inconclusive, not adopted |

> The documents themselves are written in Portuguese; the titles above are
> translated for navigation.

**Read 02 first.** It contains the measurement that inverts intuition and from
which almost every other decision derives. Then 08, which sets expectations.

---

## 18 — Common misunderstandings

**"Just use the GPU."**
This machine's GPU speaks PCIe gen 3: ~12 GB/s. Any weight that does not *live*
in VRAM must cross that channel, which is slower than the disk and five times
slower than RAM. Measured: 13x worse.
→ [§8](#8--this-machines-hardware-and-the-inversion)

**"A 31-billion-parameter model must be 31 times slower than a 1-billion one."**
No: what costs is the **active** parameter count per token, which here is 3.57
billion. And the cost is reading, not computing.

**"The file is 19.65 GB, so it reads 19.65 GB per token."**
It reads 2.01 GB. The rest of the file sits still.

**"Quantizing to 4 bits must ruin the model."**
Measured: +5.26% perplexity with everything in int4; **−0.19%**
(indistinguishable from zero) with attention in int8.
→ [§10.3](#103-what-it-costs-in-quality--and-the-surprise)

**"Experts specialize in subjects."**
No. The choice is per token and per layer, and the split emerged from training
with no assignable meaning.

**"Increasing context is free."**
It is **105.8 KB per position** of context in the f32 default, **per conversation**.
At 64k that is 6.8 GB; at the model's ceiling, 202,752 positions, it is 21.96 GB —
more than the weights. There are `bf16` (52.9 KB) and `int8` (28.1 KB) options,
measured and available, but neither became the default: neither passed the
criterion fixed before measuring.
→ [§14.5](#145-paired-measurement-and-why-it-exists)

**"Prompt caching is a comfort detail."**
It is the largest optimization available. Reusing 100k tokens of KV avoids an
entire prefill; no multiplication optimization comes close.

**"A result 'almost equal' to the reference is fine."**
It does not pass. The criterion is 32/32, binary. Three good optimizations have
already been rejected at 29/32.

**"More threads is always better."**
The work is memory-bound and the CPU is hybrid (8 P-cores + 16 E-cores).
Spreading a bandwidth-limited matmul across E-cores usually hurts; the default is
`p-cores`.

**"If it does not fit in memory, the engine will manage."**
It **refuses**, with a message naming what to reduce. That is a design decision,
and the measurement behind it is the silent 13x degradation.


**"The agent runs whatever the model said."**
Half right, and the wrong half matters. The **model** only writes a request;
what runs it is an ordinary program, with the permissions of whoever started it,
inside a folder it cannot leave.
→ [§13.4](#134-tools-the-model-asks-it-does-not-execute)

**"If it reasons first, it answers better."**
Measured on this model, with a 400-token budget: it **fails to finish 8 of 12
questions**, repeats 8× more and costs 4.5× the time. For *"what is the capital
of Brazil"* it is cut off before answering.
→ [§14.6](#146-does-reasoning-before-answering-help)

**"Speaking OpenAI's protocol requires OpenAI."**
It requires nothing beyond implementing the format, which is public. Speaking it
is exactly why Copilot, Continue and any other client connect here without a
line of our code. → [§13.2](#132-speaking-the-protocol-the-world-already-speaks)

**"The cache stores the questions and the answers."**
It stores the **internal state** of the first N tokens of the prompt. That is why
one extra space at the beginning invalidates everything from there on, and why
prompt ordering is a rule. → [§13.8](#138-the-prefix-cache-in-practice)

**"Faster kernel, faster system."**
The GPU kernels became 31× faster than the first version and 2.6× faster than
the CPU — and the whole decode came out **slower**, three times in a row. The
bottleneck was the path between them, not the arithmetic.
→ [§14.7](#147-the-measurement-that-decided-against-the-gpu)

**"Perplexity 8.63 beats that other model's perplexity of 12."**
Only if it is exactly the same text and the same vocabulary. Comparing
perplexities from different sources is the commonest mistake with this metric.
→ [§14.1](#141-perplexity-how-surprised-the-model-was)

**"It came out about the same, so we can adopt it."**
If the confidence interval crosses zero, the measurement cannot tell improvement
from regression. The answer is *inconclusive* — which is what happened with
int8 KV, with the whole memory saving in its favour.
→ [§14.5](#145-paired-measurement-and-why-it-exists)

---

## 19 — Term index PT ↔ EN

To read both versions of this document side by side, or to look a term up in the
code (which is commented in Portuguese but uses English identifiers).

| English | Portuguese | section |
|---|---|---|
| ablation | ablação | [§10.3](#103-what-it-costs-in-quality--and-the-surprise) |
| absorption (of MLA) | absorção | [§6.4](#64-attention-and-the-kv-cache) |
| affinity | afinidade | [§16.2](#162-the-commands) |
| agente (o laço) | agent (the loop) | [§13.5](#135-the-agent-is-the-loop-not-the-model) |
| aprovação | approval | [§13.5](#135-the-agent-is-the-loop-not-the-model) |
| attention head | cabeça (de atenção) | [§5.1](#51-the-model) |
| barreira | barrier | [§11.2](#112-scratch-arena-zero-malloc-per-token) |
| branch | branch | [§14.7](#147-the-measurement-that-decided-against-the-gpu) |
| budget | orçamento | [§11.5](#115-budgets-what-counts-and-what-does-not) |
| cerca (sandbox) | sandbox | [§13.5](#135-the-agent-is-the-loop-not-the-model) |
| chamada de ferramenta | tool call | [§13.4](#134-tools-the-model-asks-it-does-not-execute) |
| confiança | confidence | [§14.2](#142-entropy-and-confidence-what-perplexity-hides) |
| CORS | CORS | [§13.2](#132-speaking-the-protocol-the-world-already-speaks) |
| CRLF | CRLF | [§5.6](#56-software-and-the-system) |
| crossing | travessia | [§8.3](#83-the-way-out-layer-partition) |
| CUDA | CUDA | [§14.7](#147-the-measurement-that-decided-against-the-gpu) |
| degeneração | degeneration | [§14.3](#143-rep4-the-repetition-detector) |
| dense layer | camada densa | [§6.3](#63-the-47-layers) |
| diretiva | directive | [§13.7](#137-the-chat-that-lives-inside-the-executable) |
| endpoint | endpoint | [§13.1](#131-why-there-is-a-server) |
| entropia | entropy | [§14.2](#142-entropy-and-confidence-what-perplexity-hides) |
| EOF | EOF | [§13.6](#136-how-the-agent-talks-to-an-interface) |
| exit criterion | critério de saída | [§15](#15--where-we-are-and-what-comes-next) |
| futex | futex | [§11.2](#112-scratch-arena-zero-malloc-per-token) |
| half/half | metade/metade | [§10.2](#102-how-it-works-concretely) |
| intervalo de confiança | confidence interval | [§14.5](#145-paired-measurement-and-why-it-exists) |
| JSON Schema | JSON Schema | [§13.4](#134-tools-the-model-asks-it-does-not-execute) |
| kill criterion | critério de corte | [§15](#15--where-we-are-and-what-comes-next) |
| layer | camada | [§6.3](#63-the-47-layers) |
| layer partition | partição por camada | [§8.3](#83-the-way-out-layer-partition) |
| marco | milestone |  |
| match@ref | match@ref | [§14.4](#144-matchref-how-many-tokens-before-diverging) |
| MCP | MCP | [§13.4](#134-tools-the-model-asks-it-does-not-execute) |
| memory bandwidth | banda de memória | [§2](#idea-3--the-bottleneck-is-memory-bandwidth-not-arithmetic) |
| NDJSON | NDJSON | [§13.6](#136-how-the-agent-talks-to-an-interface) |
| oracle | oráculo | [§12.1](#121-the-oracle-and-3232) |
| paired measurement | medição pareada | [§12.2](#122-paired-measurement--and-the-mistake-that-created-the-rule) |
| perplexidade | perplexity | [§14.1](#141-perplexity-how-surprised-the-model-was) |
| perplexity | perplexidade | [§12.2](#122-paired-measurement--and-the-mistake-that-created-the-rule) |
| pipe | pipe | [§13.6](#136-how-the-agent-talks-to-an-interface) |
| placement | colocação | [§8.2](#82-the-trap-measured) |
| placement law | lei de colocação | [§8.2](#82-the-trap-measured) |
| processo | process | [§13.1](#131-why-there-is-a-server) |
| protocolo OpenAI | OpenAI protocol | [§13.2](#132-speaking-the-protocol-the-world-already-speaks) |
| quantization scale | escala (de quantização) | [§10.2](#102-how-it-works-concretely) |
| rascunho (think) | draft (think) | [§14.6](#146-does-reasoning-before-answering-help) |
| reduction order | ordem de redução | [§5.4](#54-numerics) |
| rep4 | rep4 | [§14.3](#143-rep4-the-repetition-detector) |
| residency fraction | fração de residência | [§9.2](#92-why) |
| resident | residente | [§9](#9--residency-the-decision-worth-20x) |
| residual | resíduo | [§6.3](#63-the-47-layers) |
| roofline / ceiling | teto | [§7.2](#72-the-arithmetic) |
| router / gate | roteador | [§6.5](#65-moe--64-stored-5-used) |
| sampling | amostragem | [§6.6](#66-lm_head-logits-and-sampling) |
| scratch arena | arena de scratch | [§11.2](#112-scratch-arena-zero-malloc-per-token) |
| shared expert | expert compartilhado | [§6.5](#65-moe--64-stored-5-used) |
| speculative decoding | especulação | [§5.2](#52-execution) |
| SSE | SSE | [§13.3](#133-why-the-answer-appears-letter-by-letter) |
| stdin e stdout | stdin and stdout | [§13.6](#136-how-the-agent-talks-to-an-interface) |
| thread pool | pool de threads | [§17.2](#172-the-engines-files) |
| tokenizer | tokenizador | [§6.1](#61-tokenizer--from-text-to-numbers) |
| weight | peso | [§5.1](#51-the-model) |

---

## Where to go next

1. [`docs/02`](../docs/02-hardware.md) — the measurement that inverts intuition;
2. [`docs/08`](../docs/08-performance-ceiling.md) — the performance expectation;
3. [`docs/13`](../docs/13-open-decisions.md) — the decision scoreboard, with
   the reasoning of each one preserved, including the ones that went wrong;
4. `src/forward.c` — if you want to see the 47 layers in code.
