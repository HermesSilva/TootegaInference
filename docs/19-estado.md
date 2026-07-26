# 19 — Estado de referência

Fotografia do que existe, do que foi medido e do que não foi. Serve de linha de
base para julgar qualquer mudança futura: número sem esta página ao lado não
diz se melhorou.

Build `ddf196a58ce9` · máquina de referência do [README](../README.md).

## Desempenho — CPU, container v3, 20 threads

```
banda de memoria     68,7 GB/s     MEDIDA em runtime, nao especificada
trafego por token    2,87 GB       tensores que um decode realmente toca
teto do roofline     24,0 tok/s    banda / trafego

decode quente        71 ms/token   = 13,1 tok/s   (55% do teto)
decode frio          85 ms/token   = 1,2x pior    (era 4,0x antes do aquecimento)
ttft medio           2,25 s
prefill              ~11 tok/s
aquecimento          18,54 GB em 5,3 s
RSS                  16,8 GB de pico; 35,9 GB livres
servidor ocioso      0,2% de um nucleo, 0,04 GB
```

**Matmul isolado a 62 GB/s — 90% da banda.** Não há o que otimizar nos
matmuls. A 62 GB/s os 2,87 GB/token dariam 21,7 tok/s; medimos 13,1. Os ~28 ms
que faltam estão na **atenção e nas barreiras restantes**.

## Qualidade — validada contra referência externa

| medição | engine | transformers |
|---|---|---|
| perplexidade, 384 posições de prosa, pesos BF16 | **8,63** | **8,89** |
| perplexidade, mesmo texto, container int4 | **5,66** | — |

Os 3% de diferença são tensores 1-D que a substituição não troca (já exatos) e
ordem de acumulação.

| propriedade | estado |
|---|---|
| oráculo teacher-forcing | **32/32** tokens idênticos |
| determinismo | logits **byte-idênticos** de 1 a 20 threads |
| tokenizer contra o `tokenizers` da HF | **36/36** casos, round-trip exato |
| kernels contra a quantização do numpy | **bit a bit** |
| CPU contra CUDA, cinco tensores | erro relativo **0,0000%** |

## Quantização — ablação pareada

Único grupo com efeito acima do ruído:

| grupo restaurado a BF16 | delta nll | veredito |
|---|---|---|
| experts | **−0,0225 ± 0,0076** | resolvido |
| atenção | −0,0007 ± 0,0019 | ruído |
| shared | +0,0216 ± 0,0325 | ruído |
| lm_head | −0,0008 ± 0,0007 | ruído |
| densa | +0,0168 ± 0,0097 | ruído |

**Atenção em int4 não custa nada mensurável.** O default do conversor voltou a
`--attn int4`; `int8` custa 0,3 GB por nada.

## Implementado

| peça | onde |
|---|---|
| leitor `.tootega`, parser JSON, arena | `src/container.c`, `json.c`, `common.c` |
| kernels int4/int8/f32/bf16, escalar e AVX2+FMA | `src/kernels.c` |
| pool próprio, espera em três degraus com futex | `src/pool.c` |
| MLA absorvida e expandida, MoE, KV | `src/forward.c` |
| tokenizer BPE tiktoken | `src/tokenizer.c` |
| amostragem greedy/temp/top-k/top-p | `src/sample.c` |
| orçamentos e `--plan` | `src/budget.c` |
| servidor HTTP, SSE, UI web embutida | `src/server.c`, `webui.h` |
| ferramentas no protocolo OpenAI (`tools`, `tool_calls`, `role:tool`) | `src/server.c`, [20](20-agente.md) |
| agente falando o contrato do Claude Code CLI (fase A) | `tools/agent/` |
| interface de backend, `LoadLibrary`/`dlopen` | `src/backend.c` |
| conversor BF16 → `.tootega` | `tools/convert/` |
| backend CUDA (kernels int4/int8/f32) | `tools/cuda/` |
| medição: ablação, geração, roofline, portão GPU | `Tests/` |

## Não implementado

- **Slots e batching contínuo** (M3). Um pedido por vez; a segunda aba espera
  no backlog do TCP, em silêncio.
- **Partição GPU por camada** (M2). Os kernels foram reescritos e o portão
  **passou**: 179–190 GB/s contra 68,7 da CPU, projetando 20,8 tok/s já na
  placa de 8 GB. Falta o que transforma projeção em ganho — `run_layer` no
  device, experts em lançamento único, planner ([18](18-gpu.md)).
- **Ferramentas locais no agente** (M4 fase B) — o laço fala o protocolo, mas
  não tem mãos: `Read`/`Edit`/`Bash` e o `can_use_tool` não existem.
- **Estado persistente** ([12](12-estado-persistente.md)) e **especulação MTP**
  (M5).
- **Build no Linux** — o `Makefile` existe e nunca foi compilado.

## KV em int8 — medido, inconclusivo, não adotado

`--kv-prec int8` cabe 202.752 posições em 5,83 GB contra 22,0 do f32. Medido
em 4 janelas de 4096: **+2,47% de perplexidade** na configuração que roda, mas
com o intervalo de 95% cruzando zero e uma janela degenerada (tabelas) virando
a média. Geração não piorou (`match@ref` 32,4 → 37,3, `rep4` igual). O critério
foi declarado antes e não foi cumprido: fica como opção. Detalhe em
[21](21-kv-int8.md).

## Pendências de medição

1. Reconverter o container com `--attn int4` (o v3 é int8 sem benefício).
2. `gentest` completo com o forward e o sampler corretos — os resultados
   anteriores mediram dois bugs.
3. Reescrever as tabelas de precisão de [04](04-modelo-alvo.md).
4. Corpus de **prosa** com ~32k tokens. O `corpus_prosa.txt` tem 1.995 e não
   dá nem uma janela de 4096; o `corpus.txt` é documentação com tabelas, e uma
   janela dele estraga a média ([21](21-kv-int8.md)).

## Bugs achados, e o que cada um ensinou

| bug | como escapou | registro |
|---|---|---|
| RoPE na variante `rotate_half` em vez de `interleave` | oráculo e engine eram duas escritas da mesma leitura minha | [17](17-diagnostico.md) |
| medições com binário obsoleto | `build.ps1` compila só o engine; as ferramentas linkam `src/` | [17](17-diagnostico.md) |
| sampler subtraindo `val[0]` já sobrescrito | greedy retorna antes do softmax; `top-k=1` só olha `val[0]` | `src/sample.c` |
| pool girando em `yield` para sempre | nenhuma medição roda por tempo suficiente para ver ociosidade | `src/pool.c` |
| pool dormindo por tempo em vez de evento | só aparece depois de uma pausa longa | `src/pool.c` |

O padrão comum aos três primeiros: **testes que nasceram da mesma leitura do
problema são cegos ao mesmo erro**. A referência externa custou 2 minutos e
resolveu o que dias de teste interno não resolveram.
