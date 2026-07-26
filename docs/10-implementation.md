# 10 — Implementation, build and portability

## Pure C, C11

Core in **C11**, with no C++ on the execution path. No BLAS, no protobuf, no JSON
library, no libcurl. Our own JSON parser (7 KB of it resolves everything), our
own safetensors reader, our own tokenizer reading `tokenizer.json`.

The reason is not purism: it is that the whole engine has to be auditable in one
reading session, and every dependency is a surface we do not control inside a
process that consumes third-party model files and network input.

**A thin compatibility layer**, one file, for what differs between OSes:

| concept | POSIX | Windows |
|---|---|---|
| threads | `pthread` | Win32 threads |
| mmap | `mmap` / `madvise` | `CreateFileMapping` / `MapViewOfFile` / `PrefetchVirtualMemory` |
| page locking | `mlock` | `VirtualLock` |
| positional read | `pread` | `ReadFile` with `OVERLAPPED` |
| memory info | `/proc/meminfo` | `GlobalMemoryStatusEx` |
| affinity | `sched_setaffinity` | `SetThreadAffinityMask` |

C11 `<threads.h>` is **not** used: support is uneven (MSVC only added it late,
MinGW varies). Threads go through the compat layer.

## Static linking

Goal: **one binary that runs without installing anything**, except the GPU
backend.

### Windows — solved, and already validated

MinGW-w64 with `-static` works completely, and `LoadLibrary` works normally
inside a static binary. We built exactly that when compiling the streaming engine
on this machine:

```
gcc -O3 -march=native -fopenmp ... -o engine.exe -lm -fopenmp -static -lpsapi
```

Result: 895 KB, no runtime DLL, loading the backend DLL at runtime. That is the
model.

### Linux — there is a real conflict, and it decides the release format

**A fully static binary cannot `dlopen` reliably.** On musl, `dlopen` in a static
build simply does not work. On glibc it works, but it requires the runtime glibc
to match the build-time one, and glibc itself advises against it. And `dlopen` is
exactly how we load the CUDA backend.

You cannot have both in the same artefact. So there are **two**:

| artefact | linking | GPU |
|---|---|---|
| `tootega` | **musl, 100% static** | no — CPU-only |
| `tootega-gpu` | glibc, static except libc/libm/libdl | yes, `dlopen` of the backend |

CPU-only is the primary artefact: it runs on any distro, with no dependency at
all, and it is the one [08](08-performance-ceiling.md) shows delivering
17–22 tok/s. The GPU one is optional and admittedly more fragile.

### OpenMP

Statically linked on both OSes (`libgomp.a`). It is the only runtime dependency
the streaming engine carries, and it is avoidable.

## GPU backend: a C ABI behind dynamic loading

CUDA is C++ and brings a heavy runtime. It stays **outside** the binary:

```
tootega(.exe)              pure C, static
      |  dlopen / LoadLibrary
      v
libtootega_cuda.so / .dll  built with nvcc, exposes a C ABI
```

The host **never** links `cudart`. If the DLL does not exist, does not load, or
the card is gone, the engine runs on CPU and says so in the log. It is the
standard pattern, and the streaming engine's backend loader works — the only trap
we hit was diagnostic: the message `backend DLL not found` actually meant "its
dependency, `cudart64_12.dll`, was not on PATH". **Our message has to distinguish
the two cases**, or it costs the same debugging time it cost there.

The same interface serves ROCm or Metal later, without touching the core.

## Toolchain

| target | compiler | note |
|---|---|---|
| Linux CPU | gcc or clang + musl | `-static`, via `Makefile` |
| Linux GPU | gcc + nvcc | dynamic libc |
| Windows CPU | **MSVC** (Visual Studio 2026, toolset 14.51) | `/MT`, via `build.ps1` |
| Windows GPU | nvcc + MSVC | same toolchain as the host |

### Why MSVC and not MinGW — a revised decision

The previous version of this document said "MSVC will not do (OpenMP 2.0, no GNU
extensions)". **Both reasons died:**

- **OpenMP** — D6 chose our own pool ([15](15-kernels-and-numerics.md)). There is
  no OpenMP left to support.
- **GNU extensions** — the core is ISO C11 by decision, not by accident.
  Depending on a GNU extension would be the defect, not the restriction.

What remained was the practical reason: the machine has a full Visual Studio 2026
and no gcc. One less toolchain to install, and the compiler that is already here.

Two consequences verified, not assumed:

- **`/fp:precise` does not contract** `a*b+c` into an FMA and does not reassociate
  sums. Checked in the disassembly: it emits `vmulss` + `vaddss`, and the
  reduction loop comes out serial. That is what
  [15](15-kernels-and-numerics.md) (D4) requires.
- **Atomics** — MSVC only enables `<stdatomic.h>` under
  `/experimental:c11atomics`. The pool uses a four-function shim in `compat.h`
  (`_Interlocked*` on Windows, `__atomic_*` on gcc/clang) instead of depending on
  an experimental flag.

The binary ships with a static CRT (`/MT`): it runs with no runtime DLL, which was
the goal. The `Makefile` remains the Linux path, and the sources serve both with
no `#ifdef` outside `compat.c`.

## Build flags

```
-std=c11 -O3 -Wall -Wextra
-D_FORTIFY_SOURCE=2 -fstack-protector-strong
-Wl,-z,relro,now                    (Linux)
-static-pie                          (where available; -fPIE conflicts with -static)
```

The hardening is explicit because the streaming engine does **not** have it (its
Makefile: just `-O3 -Wall -Wextra`) in a C engine that consumes third-party model
files and network input. Cost of including it: one line.

`-march=native` is **not** the default. The release binary uses `x86-64-v3`
(AVX2) with runtime dispatch to AVX-VNNI, because a `native` binary raises SIGILL
when moved to another machine. `--march=native` stays as an option for whoever
builds for themselves.

## Reproducibility

Deterministic build: no `__DATE__`/`__TIME__`, fixed link order,
`SOURCE_DATE_EPOCH` respected. The binary carries a **build id** (hash of sources
+ flags) that goes into every persisted artefact
([12](12-persistent-state.md)) to invalidate caches when the engine changes.

## Non-goals

- **No second toolchain per OS.** One compiler per platform is already plenty.
- **No CMake.** A `Makefile` a human can read.
- **No cross-compilation on day 1.** You build on the target machine.
