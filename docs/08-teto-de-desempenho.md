# 08 — Teto de desempenho

Qual é o limite superior, em tok/s, para o TootegaEngine rodando GLM-4.7-Flash
nesta máquina. Análise roofline a partir do config real do modelo.

## Correção de um erro anterior

O primeiro levantamento estimou **1,12 GB ativos por token**. Estava errado: o
script contava apenas as FFN dos experts roteados e **omitia atenção e
`lm_head`**. Neste modelo `hidden` é pequeno (2048) enquanto as dimensões de
cabeça do MLA são grandes, então a atenção pesa muito mais que o usual.

O número correto é **3,573B parâmetros ativos por token**.

## Parâmetros ativos, exato

Do config: `L=47`, `hidden=2048`, `heads=20`, `E=64`, `topk=4`, `shared=1`,
`moe_inter=1536`, `dense_inter=10240`, `first_dense=1`,
MLA `q_lora=768 kv_lora=512 qk_nope=192 qk_rope=64 v_head=256`.

| componente | ativo/token | % |
|---|---|---|
| experts roteados (top-4 de 64) | 1,736B | 48,6% |
| **atenção (todas as camadas)** | **1,023B** | **28,6%** |
| expert compartilhado | 0,434B | 12,1% |
| **`lm_head`** | **0,317B** | **8,9%** |
| camada densa (MLP) | 0,063B | 1,8% |
| **total** | **3,573B** | |

Armazenado: 29,9B pelo cálculo, contra 31,2B pelos bytes do repo — a diferença
são embeddings e normas não modeladas. Fecha.

Em int4 com escalas f32 por grupo de 64 (0,5625 B/peso): **2,01 GB de peso lido
por token**.

## KV: o termo que cresce

MLA comprime o estado para `kv_lora + qk_rope = 576` valores por token por
camada. Em fp16:

```
47 camadas × 576 valores × 2 B = 52,9 KB por token de contexto
```

No decode, a atenção lê **todo** o KV a cada token. Com contexto `N`, isso é
`52,9 KB × N` de tráfego por token gerado.

## Roofline

`tempo_por_token = (2,01 GB + 52,9 KB × N) / banda`

| contexto N | KV lido | total/token | CPU @60 GB/s | split 39% GPU |
|---|---|---|---|---|
| 1.024 | 0,05 GB | 2,06 GB | **29,1 tok/s** | **44,0 tok/s** |
| 4.096 | 0,22 GB | 2,23 GB | **26,9** | **40,6** |
| 16.384 | 0,87 GB | 2,88 GB | **20,9** | **31,5** |
| 65.536 | 3,47 GB | 5,48 GB | **11,0** | **16,5** |
| 131.072 | 6,93 GB | 8,94 GB | **6,7** | **10,1** |

A coluna CPU não depende do tamanho do container — só do peso **ativo** por
token, que continua 2,01 GB. A coluna "split", sim: eram 45% das camadas na
VRAM quando se supunha um container de 15,6 GB; com os 19,3 GB medidos e ~18,1 GB
de peso de camada, os mesmos ~7 GB de VRAM útil cobrem **39%** ([04](04-modelo-alvo.md)).

A coluna "split" assume 39% das camadas residentes em VRAM computando a 448 GB/s
e 61% na CPU a 60 GB/s — a partição por camada descrita em
[03-arquitetura.md](03-arquitetura.md). É **[estimativa]**, dependente da
hipótese que o M2 valida.

## O que o roofline não inclui

Estes números assumem **100% de eficiência de banda, compute zero e overhead de
kernel zero**. Nenhuma das três vale na prática:

- **Eficiência de banda.** Engines memory-bound bem escritos atingem 60–80% da
  banda de pico. Os 60 GB/s já são uma estimativa de "efetivo" sobre 76,8 GB/s
  teóricos, então há algum desconto embutido — mas não o suficiente.
- **Compute não é desprezível.** 2 × 3,573B = **7,1 GFLOP por token**. O
  14900K com AVX-VNNI entrega ~1–2 TOPS int8, ou seja **4–7 ms**, contra ~33 ms
  de memória a contexto curto. É ~20% do tempo, e precisa **sobrepor** à leitura,
  não somar. Se o kernel não sobrepõe, o teto cai proporcionalmente.
- **A cadeia é serial.** 47 camadas em sequência; nenhuma paraleliza com a
  seguinte no decode de uma única sequência.

## Faixa honesta

Aplicando 60–80% sobre o roofline:

| cenário | roofline | **realista** |
|---|---|---|
| CPU-only, contexto curto (≤4k) | 27–29 | **17–22 tok/s** |
| CPU+GPU split, contexto curto | 41–44 | **25–32 tok/s** |
| CPU-only, contexto de agente (32k) | ~15 | **10–12 tok/s** |
| split, contexto de agente (32k) | ~24 | **15–19 tok/s** |

Baseline a bater: **0,93 tok/s** (colibrì + GLM-5.2, medido). O ganho está entre
**11× e 34×**, conforme contexto e se o split se pagar.

## Duas consequências de projeto

**O KV domina a partir de ~16k.** Num CLI agêntico o contexto longo é o caso
normal, não a exceção: system prompt + schemas de ferramentas + arquivos lidos +
histórico chegam a dezenas de milhares de tokens rapidamente. A 64k, o KV é
**63% do tráfego por token**. Otimizar leitura de KV importa tanto quanto
otimizar leitura de peso.

O MLA já é a grande vitória aqui — 52,9 KB/token contra o que um GQA equivalente
custaria. Isso não é acidente do modelo escolhido, é um motivo para tê-lo
escolhido.

**Reuso de prefixo vale mais que qualquer kernel.** Um turno que reaproveita
100k tokens de KV cacheado evita recomputar o prefill inteiro. Nenhuma
otimização de matmul chega perto desse ganho — o que reforça a regra de ordenação
de prompt em [03-arquitetura.md](03-arquitetura.md) como decisão de desempenho de
primeira ordem, não de higiene.

## Reproduzir

`F:\colibri\roofline.py` calcula a tabela a partir do config baixado do Hub.
Reexecutar após qualquer troca de modelo alvo.
