# 10 — Implementação, build e portabilidade

## C puro, C11

Núcleo em **C11**, sem C++ no caminho de execução. Sem BLAS, sem protobuf, sem
biblioteca de JSON, sem libcurl. Parser de JSON próprio (o `json.h` do engine anterior
tem 7 KB e resolve), leitor de safetensors próprio, tokenizer próprio lendo
`tokenizer.json`.

O motivo não é purismo: é que o engine inteiro tem de ser auditável em uma
sessão de leitura, e cada dependência é uma superfície que não controlamos num
processo que consome arquivos de modelo de terceiros e entrada de rede.

**Camada de compatibilidade fina**, um arquivo, para o que difere entre SOs:

| conceito | POSIX | Windows |
|---|---|---|
| threads | `pthread` | Win32 threads |
| mmap | `mmap` / `madvise` | `CreateFileMapping` / `MapViewOfFile` / `PrefetchVirtualMemory` |
| lock de páginas | `mlock` | `VirtualLock` |
| leitura posicional | `pread` | `ReadFile` com `OVERLAPPED` |
| info de memória | `/proc/meminfo` | `GlobalMemoryStatusEx` |
| afinidade | `sched_setaffinity` | `SetThreadAffinityMask` |

C11 `<threads.h>` **não** é usado: o suporte é irregular (MSVC só o adicionou
tarde, MinGW varia). Threads passam pela camada de compat.

## Linkagem estática

Meta: **um binário que roda sem instalar nada**, exceto o backend de GPU.

### Windows — resolvido, e já validado

MinGW-w64 com `-static` funciona por completo, e `LoadLibrary` funciona
normalmente dentro de um binário estático. Já construímos exatamente isso ao
compilar o engine anterior nesta máquina:

```
gcc -O3 -march=native -fopenmp ... -o engine anterior.exe -lm -fopenmp -static -lpsapi
```

Resultado: 895 KB, sem DLL de runtime, carregando a DLL do backend em execução.
Esse é o modelo.

### Linux — há um conflito real, e ele decide o formato do release

**Um binário totalmente estático não consegue `dlopen` de forma confiável.** No
musl, `dlopen` em build estático simplesmente não funciona. Na glibc funciona,
mas exige que a glibc de execução case com a de compilação e a própria glibc
desaconselha. E `dlopen` é justamente como carregamos o backend CUDA.

Não dá para ter as duas coisas no mesmo artefato. Então são **dois**:

| artefato | linkagem | GPU |
|---|---|---|
| `tootega` | **musl, 100% estático** | não — CPU-only |
| `tootega-gpu` | glibc, estático exceto libc/libm/libdl | sim, `dlopen` do backend |

O CPU-only é o artefato principal: roda em qualquer distro, sem dependência
nenhuma, e é o que o [08](08-teto-de-desempenho.md) mostra entregando 17–22 tok/s.
O de GPU é opcional e assumidamente mais frágil.

### OpenMP

Linkado estaticamente nos dois SOs (`libgomp.a`). É a única dependência de
runtime que o engine anterior carrega e ela é evitável.

## Backend de GPU: C ABI atrás de carregamento dinâmico

CUDA é C++ e traz um runtime pesado. Fica **fora** do binário:

```
tootega(.exe)              C puro, estático
      |  dlopen / LoadLibrary
      v
libtootega_cuda.so / .dll  compilado com nvcc, expõe C ABI
```

O host **nunca** linka `cudart`. Se a DLL não existe, não carrega, ou a placa
sumiu, o engine roda em CPU e diz isso no log. É o padrão do
o carregador de backend do engine anterior e ele funciona — a única armadilha que encontramos
foi de diagnóstico: a mensagem a DLL do backend not found` na verdade significava
"a dependência dela, `cudart64_12.dll`, não estava no PATH". **A nossa mensagem
tem de distinguir os dois casos**, senão custa o mesmo tempo de depuração que
custou aqui.

A mesma interface serve para ROCm ou Metal depois, sem tocar no núcleo.

## Toolchain

| alvo | compilador | nota |
|---|---|---|
| Linux CPU | gcc ou clang + musl | `-static`, via `Makefile` |
| Linux GPU | gcc + nvcc | libc dinâmica |
| Windows CPU | **MSVC** (Visual Studio 2026, toolset 14.51) | `/MT`, via `build.ps1` |
| Windows GPU | nvcc + MSVC | mesma toolchain do host |

### Por que MSVC e não MinGW — decisão revista

A versão anterior deste documento dizia "MSVC não serve (OpenMP 2.0, sem
extensões GNU)". **As duas razões morreram:**

- **OpenMP** — a D6 escolheu pool próprio ([15](15-kernels-e-numerica.md)).
  Não há mais OpenMP a suportar.
- **Extensões GNU** — o núcleo é C11 ISO por decisão, não por acaso. Depender
  de extensão GNU seria o defeito, não a restrição.

O que sobrou foi a razão prática: a máquina tem Visual Studio 2026 completo e
nenhum gcc. Uma toolchain a menos para instalar, e o compilador que já está aqui.

Duas consequências verificadas, não assumidas:

- **`/fp:precise` não contrai** `a*b+c` em FMA nem reassocia somas. Conferido no
  disassembly: sai `vmulss` + `vaddss`, e o laço de redução sai serial. É o que
  [15](15-kernels-e-numerica.md) (D4) exige.
- **Atômicos** — o MSVC só habilita `<stdatomic.h>` sob
  `/experimental:c11atomics`. O pool usa um shim de quatro funções em
  `compat.h` (`_Interlocked*` no Windows, `__atomic_*` no gcc/clang) em vez de
  depender de flag experimental.

O binário sai com CRT estático (`/MT`): roda sem DLL de runtime, que era a meta.
O `Makefile` continua sendo o caminho do Linux, e as fontes servem aos dois
sem `#ifdef` fora de `compat.c`.

## Flags de build

```
-std=c11 -O3 -Wall -Wextra
-D_FORTIFY_SOURCE=2 -fstack-protector-strong
-Wl,-z,relro,now                    (Linux)
-static-pie                          (onde disponível; -fPIE conflita com -static)
```

O hardening é explícito porque o engine anterior **não** o tem (o Makefile dele: só
`-O3 -Wall -Wextra`) num engine em C que consome arquivos de modelo de terceiros
e entrada de rede. Custo de incluir: uma linha.

`-march=native` **não** é default. O binário de release usa `x86-64-v3` (AVX2) com
despacho em runtime para AVX-VNNI, porque um binário `native` gera SIGILL ao ser
movido de máquina. `--march=native` fica como opção de quem compila para si.

## Reprodutibilidade

Build determinístico: sem `__DATE__`/`__TIME__`, ordem de link fixa, `SOURCE_DATE_EPOCH`
respeitado. O binário carrega um **build id** (hash das fontes + flags) que entra
em todo artefato persistido ([12](12-estado-persistente.md)) para invalidar cache
quando o engine muda.

## Não-metas

- **Sem MSVC para o núcleo.** Uma toolchain por SO já é bastante.
- **Sem CMake.** Um `Makefile` que um humano lê.
- **Sem cross-compile no dia 1.** Compila-se na máquina alvo.
