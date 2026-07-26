# TootegaEngine — directives

An inference engine for resident MoE models, written from scratch. The whole
project is specified in `docs/`; read `docs/02-hardware.md` first and
`docs/13-open-decisions.md` for the decision scoreboard.

**Working language is English.** Code, comments, documents, commit messages and
anything the program prints. The only deliberate exception is
`ForDummies/*.pt.*`, the Portuguese entry guide.

## Independence

**No dependency on any external project** — neither code nor toolchain.
Everything this engine uses was written here: JSON parser, tokenizer, Unicode
tables, thread pool, kernels.

Where a document cites a measurement or a lesson from earlier work, that is
**learning history, not the origin of code**. Nothing was reused.

## Where everything lives

| folder | what it is | rule |
|---|---|---|
| `src/` | the engine | ISO C11, **zero dependencies**, only `compat.c` has OS `#ifdef`s |
| `Tests/` | **measurement** tools | one folder per tool, **the `.exe` at the root of `Tests/`** |
| `tools/` | **production** tools | same rule; `tools/shared/` is the common library |
| `ptools/` | Python tools | legacy and references that need `transformers`/`numpy` |
| `docs/` | the specification | every decision with its why and the number behind it |

### New tools: C or Rust, never Python

A new tool goes into `Tests/<name>/` or `tools/<name>/`, and the executable is
produced at the root of the corresponding tree. Python only remains in `ptools/`
for what depends on `transformers` or `tokenizers` as an external reference.

Each tool's `build.ps1` only delegates to `tools/shared/build-tool.ps1`, passing
its own path — which is why the same script serves both trees.

Today: `Tests/` has `ablate` (perplexity and ablation), `gentest` (generation
quality) and `bench` (time, memory, roofline). `tools/` has `convert` and `agent`
(the agentic client).

**Master is pure CPU.** The CUDA backend, VRAM placement and the M2 gate exist
and live on the `GPU` branch — they were measured and the conclusion was that the
granularity is wrong (`docs/18` on that branch). No GPU code lands here until
`run_layer` shows a gain in the complete decode.

### What separates Tests from tools

`Tests/` measures the engine and may depend on the original BF16 checkpoint.
`tools/` produces artefacts the engine consumes. A converter is not a test; a
perplexity meter is not production.

**A native tool may use libraries.** The dependency ban applies to `src/` — the
engine is auditable in one reading session and consumes third-party files. A tool
is a different thing: it may link whatever Python would link (BLAS, mmap,
whatever), as long as it does not enter `src/`.

A tool written in C can — and should — **reuse the sources in `src/`**. That is
what guarantees the tool measures the engine, and not a reimplementation of it.

The documentation generator is the one exception on record:
`ForDummies/build.py` is Python because it builds documents, not artefacts the
engine consumes. Port it to C when it becomes worth doing.

## Build

```powershell
.\build.ps1                  # engine, MSVC via vswhere; never a fixed path
.\build.ps1 test             # + selftest
.\tools\<name>\build.ps1     # one tool
```

Windows is MSVC (Visual Studio 2026). Linux is the `Makefile` with gcc or clang.
`/fp:precise` with no contraction to FMA — verified in the disassembly, and it is
what the reproducibility of `docs/15` requires.

## Rules that came from mistakes made

- **Paired measurement.** Comparing two configurations by the global mean
  swallows the effect in the variance between windows. See the warning in
  `docs/04`: the first perplexity measurement had the **sign inverted**.
- **An output change never lands silently.** Any algebraically equivalent
  alteration (MLA absorption, KV precision) is only adopted with **32/32
  identical tokens** against `ref32.json`. bf16 and int8 KV both failed and
  remain options, not defaults.
- **Trading work for memory needs the traffic arithmetic first.** The first
  version of absorption materialised `kv_b` in fp32 and came out 2× slower.
- **If Hermes says he is going to run something in his own console, do not fire
  it off in parallel.**
