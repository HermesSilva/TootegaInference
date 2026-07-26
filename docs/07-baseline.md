# 07 — Baseline

Tudo medido nesta máquina, com engine anterior v1.1.1 e GLM-5.2 int4 (383,7 GB) em
`F:\glm52_i4`. São os números que o TootegaEngine tem de bater.

Protocolo: prompt fixo, `max_tokens=80`, `temperature=0` (greedy), `RAM_GB=45`
fixo salvo indicação, hash SHA-256 da resposta registrado em cada execução.

## Single-stream

| build / config | prompt 1 | prompt 2 |
|---|---|---|
| release, `x86-64-v3` | 0,684 | 0,748 |
| **nativo, AVX-VNNI** | 0,684 | 0,748 |
| cuda-experts4 | 0,647 | 0,700 |
| cuda-tc-w4a16 (tensor cores) | 0,639 | 0,709 |
| cuda-tc-pipe2 | 0,641 | 0,700 |
| **cuda-dense** | **0,054** | abortado |

Nenhuma variante de GPU alcançou o CPU. **`CUDA_DENSE=1` degradou 13×** — os
10,9 GB densos não cabem nos 8,6 GB de VRAM e o engine passou a trafegar peso
pelo PCIe a cada camada. O engine não recusou; aceitou e degradou.

## Efeito de repetição (cache quente)

Mesmo prompt 3× seguidas, mesmo servidor:

| config | run 1 | run 2 | run 3 | ganho |
|---|---|---|---|---|
| cpu-nativo | 0,709 | 0,861 | 0,856 | **+20,7%** |
| cuda-tc-w4a16 | 0,775 | 0,805 | 0,806 | +4,0% |
| cpu-direct (`DIRECT=1 PIPE=1`) | 0,827 | 0,858 | **0,867** | +4,8% |

`DIRECT=1` ajuda na **primeira** execução (0,827 contra 0,709, +17%), não no
regime quente — coerente, já que `O_DIRECT` só beneficia quem lê do disco.

## Melhor marca

**0,929 tok/s** — GPU com tier de 4 GB de experts + `DIRECT=1 PIPE=1`, quente:

```
VRAM  4525 / 8192 MB  (55%, 3,6 GB ociosos)
RSS   45,14 GB · RAM livre 6,3 GB
```

**A GPU não era o recurso escasso; a RAM era.**

## Escalabilidade

**Concorrência** (4 slots de KV, aquecimento antes):

| K | agregado | por request |
|---|---|---|
| 1 | 0,921 | 0,921 |
| 2 | 1,043 | 0,524 |
| 4 | **1,339** | 0,338 |

4× de carga → **1,45×** de throughput. Expoente **0,27**.

**Memória** (mesma config, `RAM_GB` variando):

| RAM_GB | quente |
|---|---|
| 16 | 0,656 |
| 30 | 0,695 |
| 45 | 0,839 |
| ~60 (auto) | 0,929 |

3,75× de RAM → **1,42×**. Expoente ~0,26.

**Os dois eixos dão `N^0,25`.** A lei de escala não é hardware, é fração de
residência: com 60 GB de orçamento cacheamos ~15% dos 324 GB de experts frios.

## Correção verificada

Um achado que vale como requisito, não só como observação: **CPU e CUDA
produziram bytes idênticos** (mesmo SHA-256 da resposta), em servidores
diferentes, com colocações diferentes. A promessa central do engine anterior — *placement
only ever decides speed* — se sustenta.

A única divergência observada veio de um KV persistido de **outra
conversa**, cujo prefixo casava parcialmente. Runs 2 e 3 idênticas entre si
descartaram não-determinismo de redução em ponto flutuante sob OpenMP: o engine é
determinístico **por estado**.

Isso vira teste no TootegaEngine: mesma colocação → mesmos bytes; estado de KV
diferente → **também** mesmos bytes (aqui o engine anterior não foi verificado a fundo).

## Banda de disco não saturada

Cada token precisa de ~78 × 8 × 19 MB ≈ 11,9 GB de experts. Com ~15% residentes,
~10 GB vêm do disco por token. A 0,93 tok/s isso é **~9,3 GB/s** — contra
**18,4 GB/s** medidos no RAID0.

Metade da banda do disco ociosa. Provável causa: `PIPE_WORKERS=8` não mantém fila
profunda o suficiente (o RND4K Q1T1 do array é 73 MB/s contra 2.058 MB/s em
Q32T16). Não perseguimos — o TootegaEngine não lê expert do disco.

## Como reproduzir

Scripts em \`: `bench.ps1` (matriz de configs), `bench_repeat.ps1`
(efeito de repetição), `scaling.ps1` (fases A/B/C), `roofline.py` (teto teórico).
Resultados brutos em `bench_run.log`, `bench_repeat.log`, `scaling_run.log`.
