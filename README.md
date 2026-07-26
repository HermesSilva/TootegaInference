# TootegaEngine

Engine de inferência para modelos MoE **residentes**, escrito do zero.

Estado: **M1 completo**, em CPU. O forward reproduz o oráculo — **32/32 tokens
idênticos** — com logits **byte-idênticos** entre 1 e 24 threads; o tokenizer
dá **36/36 casos idênticos** aos do `tokenizers` da HuggingFace; `--ram-gb` e
`--plan` recusam o que não cabe dizendo o que reduzir.

```powershell
.\build.ps1                    # MSVC no Windows; `make` no Linux
.\tootega.exe selftest         # kernels, JSON, pool, tokenizer e sampler
.\tootega.exe inspect  models\glm47flash-int4\model.tootega
.\tootega.exe plan     models\glm47flash-int4\model.tootega --ctx 4096
.\tootega.exe tokenize models\glm47flash-bf16\tokenizer.json --check tokfixture.json
.\tootega.exe forward  models\glm47flash-int4\model.tootega --ids-file ids32.txt

.\tootega.exe generate models\glm47flash-int4\model.tootega `
    --tokenizer models\glm47flash-bf16\tokenizer.json `
    --prompt "[gMASK]<sop><|user|>Qual e a capital do Brasil?<|assistant|>" -n 60

.\tootega.exe chat models\glm47flash-int4\model.tootega `
    --tokenizer models\glm47flash-bf16\tokenizer.json --ctx 8192
```

O `chat` monta o template do modelo (conferido byte a byte contra
`apply_chat_template`) e reaproveita o KV entre turnos — no segundo turno,
**92 de 100 tokens de prompt vieram do cache**.

```
A capital do Brasil é Brasília.

prefill  18 tokens em 4.63 s (3.9 tok/s)
decode    8 tokens em 0.82 s (9.79 tok/s)
```

As ferramentas de referência ficam em `tools/`: `oracle.py` refaz o forward em
numpy sobre o mesmo container, `gen_kernel_fixture.py` e
`gen_tokenizer_fixture.py` geram as referências que o `selftest` consome.

## A tese em um parágrafo

O colibrì provou que dá para rodar um MoE de 744B numa máquina comum transmitindo
experts do disco — e mediu **0,93 tok/s** aqui. Medimos também que escalar hardware
nesse regime rende `N^0,25`: quadruplicar RAM ou concorrência dá ~1,4×. O gargalo
não é hardware, é a razão entre o tamanho do modelo e a memória rápida. Um MoE
moderno e enxuto (GLM-4.7-Flash, 19,3 GB em int4) **cabe inteiro na RAM desta
máquina**, o disco sai do caminho, e o teto passa a ser banda de memória:
**17–22 tok/s só na CPU, 25–32 tok/s com partição GPU/CPU** — contra 0,93. O
TootegaEngine é o engine desenhado para esse regime, não para esconder disco.

Faixas realistas a contexto curto; a contexto de agente (32k) caem para 10–20.
A conta completa está em [08-teto-de-desempenho.md](docs/08-teto-de-desempenho.md).

## Documentos

| # | documento | o que decide |
|---|---|---|
| 01 | [Objetivos e escopo](docs/01-objetivos.md) | o que é e o que explicitamente não é |
| 02 | [Modelo de hardware](docs/02-hardware.md) | as bandas medidas e a lei de colocação que sai delas |
| 03 | [Arquitetura](docs/03-arquitetura.md) | camadas, contratos, os três caches e seus donos |
| 04 | [Modelo alvo](docs/04-modelo-alvo.md) | GLM-4.7-Flash: o que o engine tem de implementar |
| 05 | [Roadmap](docs/05-roadmap.md) | marcos com critério de saída mensurável |
| 06 | [Lições do colibrì](docs/06-licoes-colibri.md) | o que copiar, o que jogar fora, o que consertar |
| 07 | [Baseline](docs/07-baseline.md) | os números a bater, todos medidos nesta máquina |
| 08 | [Teto de desempenho](docs/08-teto-de-desempenho.md) | roofline: quanto dá para extrair desta máquina |
| 09 | [Limites de recursos](docs/09-limites-de-recursos.md) | orçamentos de CPU/GPU/RAM e a regra de recusar em vez de degradar |
| 10 | [Implementação](docs/10-implementacao.md) | C11, linkagem estática, portabilidade Win/Linux |
| 11 | [Memória](docs/11-memoria.md) | zero-copy, KV paginado compartilhado, páginas grandes |
| 12 | [Estado persistente](docs/12-estado-persistente.md) | índices e KV em disco para subir quente |
| 13 | [Decisões pendentes](docs/13-decisoes-pendentes.md) | placar das decisões; D1–D6 fechadas, D7–D10 nos marcos |
| 14 | [Formato do container](docs/14-formato-container.md) | especificação normativa do `.tootega`, byte a byte |
| 15 | [Kernels e numérica](docs/15-kernels-e-numerica.md) | W4A32, ordem de acumulação, pool de threads |
| 16 | [Tokenizer](docs/16-tokenizer.md) | BPE tiktoken, o que implementar e como testar |

Leia **02** primeiro — contém a medição que inverte a intuição usual e da qual
quase todas as outras decisões derivam. Depois **08**, que fixa a expectativa.

## Máquina de referência

Todos os números destes documentos foram medidos aqui, não estimados de
especificação — salvo onde marcado **[estimativa]**.

```
CPU    Intel i9-14900K, 32 threads, AVX-VNNI
RAM    64 GB DDR5-4800 dual channel      ~60 GB/s efetivos
GPU    RTX 2060 SUPER, 8 GB, sm_75       448 GB/s de VRAM
PCIe   gen 3 x16 (limite da GPU)         ~12 GB/s   <-- o elo mais lento
Disco  RAID0 NVMe (F:)                   18,4 GB/s sequencial
SO     Windows 10 Pro 19045
```

## Obter o modelo

```powershell
python download_model.py            # 62,4 GB BF16, retomável
python download_model.py --verify   # confere contra o Hub sem baixar
```
