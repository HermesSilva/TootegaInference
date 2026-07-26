# 02 — Modelo de hardware

Este é o documento fundamental. A medição abaixo inverte a hierarquia que quase
todo engine de inferência assume, e quase todas as decisões de arquitetura saem
dela.

## As bandas, medidas

| tier | banda | como foi medido |
|---|---|---|
| VRAM (RTX 2060 SUPER) | **448 GB/s** | especificação do fabricante |
| RAM (2× 32 GB DDR5-4800 dual channel) | **~60 GB/s** efetivos | 76,8 GB/s teóricos (4800 MT/s × 8 B × 2 canais) |
| Disco (RAID0 NVMe, F:) | **18,4 GB/s** sequencial | CrystalDiskMark 8.0.4, SEQ1M Q8T1 |
| **PCIe (gen 3 x16)** | **~12 GB/s** | 15,75 GB/s teóricos; `nvidia-smi` confirma gen 3 |

## A inversão

**O PCIe é o elo mais lento da máquina.** Mais lento que o disco.

```
pcie.link.gen.hostmax = 5     <- o slot da placa-mãe oferece Gen 5 (63 GB/s)
pcie.link.gen.max     = 3     <- a GPU só fala Gen 3
```

A placa-mãe (ASUS ROG STRIX Z690-A) oferece PCIe 5.0 x16. A RTX 2060 SUPER é
Turing, de 2019, e é um endpoint Gen 3. O link negocia pelo lado mais lento.

**Consequência direta:** qualquer peso que não *more* na VRAM tem de atravessar
um canal mais lento que o SSD de onde veio. Streaming de pesos para a GPU é
estritamente pior que computar na CPU direto da RAM — 12 GB/s contra 60 GB/s.

### Isto não é teoria — foi medido

`CUDA_DENSE=1` no colibrì, que coloca os tensores densos na GPU quando eles não
cabem na VRAM:

```
config              tok/s
cpu-nativo          0,684
cuda-experts4       0,647
cuda-tc-w4a16       0,639
cuda-dense          0,054   <-- 13× mais lento
```

O engine não recusou. Aceitou e passou a trafegar peso pelo PCIe a cada camada,
a cada token. **É a armadilha desta máquina**, e qualquer engine que assuma
"VRAM é rápida, logo mova tudo para lá" cai nela.

## A lei de colocação

> Coloque na GPU **apenas o que couber lá permanentemente**. Nunca transmita
> pesos por um link mais lento que o tier de origem.

Formalmente: escolher a colocação minimizando

```
custo_por_token = Σ_tier (bytes_lidos_no_tier / banda_do_tier)
```

sujeito às capacidades — e contabilizando o **link de transferência**, não só a
banda do tier de destino. É essa segunda parte que o colibrì omite: o `plan` dele
recomendou um tier de VRAM que, medido, não pagou.

## O corolário que salva o projeto

Pesos são grandes. **Ativações são minúsculas.**

Para o modelo alvo, `hidden = 2048`. Uma ativação na fronteira entre camadas é
`2048 × 4 bytes = 8 KB`. Contra ~4,7 MB por expert.

Isso abre uma partição que não paga o pedágio do PCIe:

> **Particione por camada, não por tensor.** Cada camada tem seus pesos
> permanentemente residentes no device que a computa. Entre camadas cruzam
> 8 KB de ativação, não megabytes de peso.

Com ~7 GB de VRAM utilizável (8 GB menos contexto e workspace) e 18,1 GB de peso
de camada (o container tem 19,3 GB, mas os 1,3 GB de `embed_tokens` ficam
file-backed e nunca disputam VRAM — ver [04](04-modelo-alvo.md)), cerca de **39%**
das camadas ficam na GPU e o resto na CPU. Por token,
o PCIe carrega 8 KB por fronteira — 2 fronteiras se o corte for contíguo, ou
seja **16 KB por token**. A 12 GB/s isso é 1,3 microssegundos. Irrelevante.

**[estimativa]** Modelando Amdahl com 2,23 GB lidos por token (pesos + KV a 4k
de contexto — ver [08](08-teto-de-desempenho.md)), 39% das camadas na GPU:

```
CPU:  1,36 GB / 60 GB/s   = 22,7 ms
GPU:  0,87 GB / 448 GB/s  =  1,9 ms
                            ------
                            24,6 ms  ->  ~41 tok/s de roofline
```

Contra ~27 tok/s de roofline só na CPU. **Marcado como estimativa de propósito** —
é a hipótese central do projeto e o marco M2 do [roadmap](05-roadmap.md) existe
para validá-la ou matá-la. O ganho é ~1,5×, não 2×; se a complexidade de manter
dois backends não se pagar, CPU-only continua sendo ~20× o baseline.

## Tetos por classe de modelo

Custo por token = parâmetros ativos × 0,5 byte (int4) ÷ banda do tier hospedeiro.

| modelo | int4 | ativo/token | onde cabe | teto |
|---|---|---|---|---|
| ~7B-A1B (OLMoE) | 4 GB | 0,65 GB | VRAM inteira | ~690 tok/s |
| **GLM-4.7-Flash** | **19,3 GB** | **2,01 GB** | **RAM (+GPU parcial)** | **27 CPU / 41 split** |
| ~30B-A3B | 16 GB | ~1,5 GB | RAM | ~40 tok/s |
| ~100B-A12B | 50 GB | ~6 GB | RAM, apertado | ~10 tok/s |
| GLM-5.2 744B-A40B | 363 GB | ~12 GB | disco | **0,93 medido** |

O número do GLM-4.7-Flash é exato (do config, ver [08](08-teto-de-desempenho.md));
os demais são estimativas grosseiras que **só contam FFN de expert** — a mesma
omissão que subestimou o alvo em 60% antes de recalcularmos. Tetos são de banda
pura: ignoram compute, overhead de kernel e a cadeia serial das camadas.

## O que a medição diz sobre comprar hardware

Da fase C do teste de escalabilidade (RAM_GB variando, GLM-5.2):

| RAM_GB | tok/s quente |
|---|---|
| 16 | 0,656 |
| 30 | 0,695 |
| 45 | 0,839 |
| ~60 (auto) | 0,929 |

3,75× de RAM → 1,42× de velocidade. Da fase B, 4× de concorrência → 1,45× de
throughput agregado. **Ambos os eixos dão ~N^0,25.**

A lei de escala aqui não é hardware, é **fração de residência**. Multiplicar por
N só se aproxima de X·N no trecho em que N faz cruzar o limiar de residência.
Escolher um modelo que já cabe é 20–50×; comprar hardware é ~N^0,25.

Se um dia houver upgrade de GPU, o ganho maior não é VRAM nem compute: é o
**link**. Ampere/Ada falam Gen 4 (31,5 GB/s), Blackwell fala Gen 5 (63 GB/s) — e
63 GB/s ultrapassa a banda de RAM, o que inverteria a hierarquia de novo e
tornaria o streaming para a GPU um bom negócio pela primeira vez.
