# 09 — Limites de recursos

O engine roda numa estação de trabalho que a pessoa está usando. Ele tem de
caber num orçamento declarado — de RAM, de VRAM e de CPU — e **recusar** o que
não couber, em vez de acomodar.

## Por que, com evidência

Três coisas que medimos no engine anterior e que este documento existe para não repetir.

**Acomodar em vez de recusar custa 13×.** `CUDA_DENSE=1` com 10,9 GB de tensores
densos e 8,6 GB de VRAM não deu erro: aceitou e passou a trafegar peso pelo PCIe
a cada camada, a cada token. Resultado medido: **0,054 tok/s** contra 0,684.
Uma configuração impossível degradou em silêncio.

**Orçamento automático flutuante torna medição incomparável.** O engine anterior escolheu
`RAM_GB=47,1` numa execução e `60,5` na seguinte, conforme o que o SO tinha
devolvido no momento. Duas medições feitas assim não dizem nada uma sobre a
outra — tivemos de refazer a matriz de benchmark inteira com o valor fixado.

**Autoplanejamento erra a alocação.** Na melhor marca medida (0,929 tok/s), a
VRAM estava **55% ociosa** (4.525 de 8.192 MB) enquanto a RAM tinha 6,3 GB
livres. O planner encheu o recurso errado.

E há o caso trivial: a pessoa quer compilar, editar vídeo ou jogar enquanto o
engine roda. Trinta e dois threads a 100% e 60 GB residentes não são aceitáveis
como default.

## Parâmetros

Precedência: **flag de linha de comando > variável de ambiente > arquivo de
config > auto**.

### Memória

| flag | env | default | o que faz |
|---|---|---|---|
| `--ram-gb N` | `TOOTEGA_RAM_GB` | auto | teto **duro** de bytes residentes do processo |
| `--vram-gb N` | `TOOTEGA_VRAM_GB` | auto | teto duro de VRAM alocada pelo engine |
| `--vram-reserve-gb N` | `TOOTEGA_VRAM_RESERVE_GB` | `1.5` | quanto deixar livre na placa para contexto CUDA, driver e o desktop |
| `--kv-gb N` | `TOOTEGA_KV_GB` | derivado | teto do KV; se omitido, sai de `ctx × slots` |
| `--ctx N` | `TOOTEGA_CTX` | `8192` | contexto máximo por slot |
| `--kv-slots N` | `TOOTEGA_KV_SLOTS` | `1` | contextos independentes |

`auto` para RAM = memória disponível menos 8 GB de folga para o SO, e o valor
resolvido é **impresso no log de inicialização**. Auto é conveniência; para
qualquer medição, fixe.

### CPU

| flag | env | default | o que faz |
|---|---|---|---|
| `--threads N` | `TOOTEGA_THREADS` | P-cores físicos | threads de compute |
| `--affinity MODO` | `TOOTEGA_AFFINITY` | `p-cores` | `p-cores` \| `all` \| `none` \| lista explícita (`0-15`) |

O 14900K é híbrido: 8 P-cores (16 threads SMT) e 16 E-cores. Espalhar um matmul
memory-bound pelos E-cores costuma piorar, e o engine anterior tem código para se re-exec justamente porque o pinning precisa acontecer
antes de `main`. Default `p-cores`, com `all` disponível para medir.

### GPU

| flag | env | default | o que faz |
|---|---|---|---|
| `--gpu MODO` | `TOOTEGA_GPU` | `auto` | `auto` \| `none` \| índice do device |
| `--gpu-layers N` | `TOOTEGA_GPU_LAYERS` | auto | camadas residentes em VRAM; o resto na CPU |

`--gpu none` é o desligamento duro. O engine anterior tem uma armadilha aqui — no Windows
ele **auto-habilita** CUDA em execução nua, e só um uma variável de ambiente explícita explícito
suprime isso. Aqui: `auto` significa "use se ajudar, conforme o planner", e a
decisão fica no log.

### Diagnóstico

| flag | o que faz |
|---|---|
| `--plan` | resolve a colocação, imprime, **sai**. Código 1 se não couber |
| `--explain` | como `--plan`, mais o custo por token de cada tier e o gargalo |

## Semântica: quatro regras

**1. Orçamento é teto duro, não sugestão.** Se o plano não cabe, o engine **sai
com erro** nomeando o que reduzir. Nunca degrada em silêncio, nunca conta com
swap.

```
ERRO: nao cabe em --ram-gb 12
  pesos (int4)          18,1 GB
  KV (ctx 8192 x 1)      0,4 GB
  scratch + runtime      1,2 GB
  ------------------------------
  necessario            19,7 GB   excede em 7,7 GB
sugestao: --ram-gb 21, ou --gpu-layers para mover peso para a VRAM
```

**2. O que conta contra cada orçamento é enumerado, não implícito.** O `RAM_GB`
do engine anterior era ambíguo — pesos densos, cache de expert e workspace entravam de
formas diferentes. Aqui:

| orçamento | inclui |
|---|---|
| `--ram-gb` | pesos residentes em RAM + KV em RAM + arena de scratch + runtime |
| `--vram-gb` | pesos residentes em VRAM + KV em VRAM + workspace de kernel |
| `--kv-gb` | só o estado de atenção, somado entre slots, onde quer que esteja |

O que **não** conta: o mmap do arquivo de modelo (páginas de arquivo, o SO
recupera sob pressão) e os buffers do SO.

**3. O teto é verificado em execução, não só no plano.** Estimativa de
planejamento erra. Uma thread de watchdog compara o alocado real contra o
orçamento; ultrapassar é bug, e o engine falha alto em build de debug, registra e
recusa novas admissões em release. Sem isso, "orçamento" vira decoração.

**4. Orçamento explícito produz plano determinístico.** Mesmo orçamento → mesma
colocação → desempenho comparável. É o que torna o benchmark reproduzível, e é a
correção direta do problema do `RAM_GB` flutuante.

## Como o planner usa os limites

Entrada: bandas medidas na init ([02](02-hardware.md)), tamanhos de tensor do
modelo, orçamentos declarados.

Saída: para cada camada, o tier que hospeda seus pesos.

Objetivo: minimizar `Σ (bytes_lidos_por_token_no_tier / banda_do_tier)`, sujeito
a caber em **cada** orçamento — e contabilizando o **link de transferência**,
não só a banda do tier de destino. É essa segunda parte que faltava no `plan` do engine anterior, e é por isso que ele recomendou um tier de VRAM que, medido, não pagou.

Consequência nesta máquina: com PCIe a 12 GB/s, o planner **nunca** propõe
transmitir peso para a GPU. Ou a camada mora lá, ou é computada na CPU.

## Defaults para esta máquina

```
--ram-gb 48          # de 64, deixa 16 para SO e trabalho
--vram-gb 6.5        # de 8, reserva 1,5 para driver e desktop
--threads 16         # os 8 P-cores com SMT
--affinity p-cores
--ctx 8192
--kv-slots 1
```

Com esses valores: pesos residentes 18,1 GB + KV 0,4 GB + scratch ~1,5 GB ≈
**20,0 GB** de RAM (o container tem 19,3 GB; os **dois** `embed_tokens` — o do
modelo e o da MTP, 1,27 GB juntos — ficam file-backed e não contam, ver
[04](04-modelo-alvo.md)).
Sobra folga larga — e é isso que permite subir `--ctx` para uso agêntico, que é
onde o KV cresce (52,9 KB por token de contexto, ver
[08](08-teto-de-desempenho.md): a 64k são 3,5 GB por slot).

## O que estes limites não fazem

- **Não limitam I/O de disco.** O modelo é residente; o disco só é lido na carga.
- **Não são isolamento de segurança.** São orçamentos cooperativos dentro do
  processo, não sandbox. Para limite imposto de fora, use Job Objects no Windows
  ou cgroups no Linux.
- **Não impedem que o SO faça swap** se o orçamento for mal declarado. Por isso o
  default de `auto` deixa 8 GB de folga.
- **Não garantem latência.** Reduzir threads reduz throughput; o engine não tem
  escalonamento em tempo real.

## Critério de saída

Entra nos marcos do [roadmap](05-roadmap.md):

- **M1**: `--ram-gb` e `--threads` respeitados e verificados; `--plan` recusa o
  que não cabe com mensagem acionável.
- **M2**: `--vram-gb` e `--gpu-layers` idem; **o caso `CUDA_DENSE` do engine anterior
  vira teste de regressão** — pedir mais VRAM do que existe tem de sair com erro,
  não render 0,054 tok/s.
