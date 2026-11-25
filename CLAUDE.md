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
| `docs/` | the specification | every decision with its why and the number behind it |

### New tools: C or Rust, never Python

A new tool goes into `Tests/<name>/` or `tools/<name>/`, and the executable is
produced at the root of the corresponding tree.

Each tool's `build.ps1` only delegates to `tools/shared/build-tool.ps1`, passing
its own path — which is why the same script serves both trees.

Today: `Tests/` has `ablate` (perplexity and ablation), `gentest` (generation
quality), `bench` (time, memory, roofline), `oracle` (the independent
forward), `mmbatch` (the batched matmul), `route` (which experts a token goes
through, and how that varies with context) and `analyze` (whether a format
change has bytes to save, measured before the format changes). `tools/` has `convert`, `agent` (the agentic client), `finetune`,
`index` (corpus → `.ttindex`, the retrieval of `docs/13` D15), `ucd` (generates
`src/unicode_tables.h`) and `fixture` (generates `src/fixture_kernels.h`).

**Two Python files remain in the engine's own path**, both outside it:
`ForDummies/build.py`, which builds documents and not artefacts the engine
consumes, and `download_model.py`, which needs HTTPS to fetch the checkpoint
from HuggingFace. Everything else was rewritten in C.

**`finetune/` is the third case, and it is a different kind.** It is the RAFT
pipeline that produced the domain container — 24 Python scripts that require
PyTorch, run on a rented GPU box, and never execute on the machine the engine
runs on. They are not tools of the engine; they are a project that *feeds* it,
the way `download_model.py` does, and what crosses the boundary is an artefact:
a `.tootega` container. See `finetune/README.md` and `docs/22`.

**Master is pure CPU.** The CUDA backend, VRAM placement and the M2 gate exist
and live on the `GPU` branch — they were measured and the conclusion was that the
granularity is wrong (`docs/18` on that branch). No GPU code lands here until
`run_layer` shows a gain in the complete decode. The fine-tuning pipeline lives
on the `finetune` branch, by the same rule and for the same reason.

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

Two exceptions are on record, and no third one is welcome. The documentation
generator `ForDummies/build.py` is Python because it builds documents, not
artefacts the engine consumes; port it to C when it becomes worth doing.
`download_model.py` is Python because fetching the checkpoint needs HTTPS, and
the engine has no TLS stack — it is bootstrap, run once, before any of this
exists.

Everything a C tool could do, a C tool does. The oracle, the perplexity meter,
the converter, the Unicode tables and the kernel fixture were all Python once;
each one is now a native tool, and the reason is the same every time: a
measurement made by a program that the engine does not share code with is
evidence, and one that shares is a tautology.

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
