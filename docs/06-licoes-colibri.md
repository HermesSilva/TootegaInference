# 06 — Lições do colibrì

Auditamos o código do colibrì (`d:\Tootega\Colibri`) e o rodamos por horas nesta
máquina. Referências são `arquivo:linha` no repositório dele.

## Copiar

**Movimento de bloco grande é troca de ponteiro** — `colibri.c:3493`:

```c
ESlot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp;
```

Promover um expert para o cache não copia os ~19 MB: troca os structs, e o slab
do evictado vai para o workspace do próximo load. Zero `memcpy`, zero
`malloc`/`free` por eviction.

**Slab só realoca se não couber** — `colibri.c:1527`:
`if(!s->slab || wtot+8192 > s->slab_cap)`. Em regime permanente o allocator não
é tocado.

**Zero cópia via mmap** — `colibri.c:1479-1520`: os ponteiros do quantizador
apontam direto para dentro do arquivo mapeado, com *pre-touch* das páginas em vez
de cópia. O comentário deles resume: *"This is pread's I/O without the copy and
without the slab."*

**Arena única para o conjunto residente** (`#419`, `colibri.c:5876-5885`): pesos
fixados ligados como **uma arena por camada**, slices marcadas com `aslab` e
nunca liberadas individualmente. Mata fragmentação e syscalls por expert.

**Não zerar buffer que vai ser sobrescrito** — `colibri.c:458-462`, `falloc`
recusa `calloc` de propósito: *"Niente calloc: il memset nel percorso caldo
costa."*

**Fundir para não reler** — `matmul_i4_grouped_pair` (`colibri.c:469`) calcula
gate e up na mesma passada, lendo `x` uma vez: ~33% do tempo de matmul de expert
no decode.

**Validação token-exata contra oráculo `transformers`.** É o que separa engine de
gerador de plausibilidade.

**Recusar-se a subir quando a RAM não dá.** Vimos acontecer: o engine leu 3,3 GB
disponíveis e abortou em vez de arriscar OOM no meio da geração. Comportamento
correto, copiar.

## Jogar fora

Todo o maquinário de esconder disco: **PILOT** e prefetch cross-layer,
roteamento **dual-SSD**, ajuste de **`O_DIRECT`**, **pin set aprendido**
(`.coli_usage`), **tier LRU por camada**, guarda de eviction LFRU. É a maior
parte do orçamento de complexidade do projeto e não serve para nada quando o
modelo é residente.

Um engine para modelo enxuto é uma fração do tamanho — e essa é a principal
vantagem estrutural do TootegaEngine sobre o colibrì, não a velocidade.

## Consertar

Defeitos que medimos e que só não doem porque o colibrì roda a ~1 tok/s. A 20+
tok/s viram percentual.

**~500 `malloc`/`free` por token.** `step()` faz 3 por token, um do tamanho do
vocabulário (`colibri.c:4263-4272`); `layers_forward_rows` mais 2 (`:4144`); com
o pilot de duas etapas, `la_predict` até 7 **por camada** (`:3555`, `:3571-3572`).
Todos de tamanho fixo conhecido na init. → **arena persistente**.

**`getenv` dentro do laço de camadas.** `colibri.c:3505-3510` chama
`getenv("COLI_CUDA_SHARED_W4A16_MIN_ROWS")` e `atoi` a cada camada, a cada token;
`:4158` chama `getenv("COLI_CUDA_PIPE_S_MIN")` a cada forward. São ~300
varreduras lineares de `environ` por token num build CUDA. → **config imutável
lida uma vez**.

**Buscas lineares onde há estrutura.** Achar um expert residente é varredura
sobre pin e sobre LRU (`:2713`, `:3034`, `:3630-3632`); escolher vítima é outra
(`:3492`). ~150k comparações por token. Eles sabem fazer diferente quando dói —
em `st.h` trocaram varredura por hash aberto porque custava *"decine di secondi
per token"*. → **índice direto `layer × expert → slot`**.

**Arquitetura cravada no parser.** `colibri.c:934` sai com erro fatal se
`n_group != 1`; `:943` exige `kv_lora_rank >= 1`. Isso reduziu o universo de
modelos compatíveis à família GLM-5.x — DeepSeek-V2/V3/V4 (`n_group=8`), Qwen3
(GQA), MiMo e Hy3 (sem MLA) são todos rejeitados. Transformou "testar outro
modelo" numa investigação que terminou sem opção. → **config → IR, variantes
declaradas**.

**Sem flags de hardening.** `c/Makefile:57` compila com `-O3 -Wall -Wextra` e
**sem** `-D_FORTIFY_SOURCE=2`, `-fstack-protector-strong`, `-fPIE -pie`,
`-Wl,-z,relro,now`. Engine em C consumindo entrada de rede e arquivos de modelo
de terceiros. Custo de corrigir: uma linha.

## Ressalvas de método que aprendemos medindo

**Não trocar duas variáveis de uma vez.** Medimos "nativo + CUDA" contra "release
CPU" e o resultado não dizia nada, porque o build nativo (AVX-VNNI) e o CUDA
mudaram juntos. Foi preciso refazer com o estado do meio isolado.

**Fixar o orçamento de RAM entre configs.** Duas medições com `RAM_GB` diferente
não são comparáveis, e o valor "auto" flutua com o que o SO devolveu.

**Aquecer antes de medir concorrência.** Sem um turno de aquecimento, o custo do
tier frio cai todo em K=1 e inventa um ganho de escala que não existe.

**Hash da saída em toda medição.** Foi assim que descobrimos que CPU e CUDA
produzem bytes idênticos — e que a única divergência vinha de um `.coli_kv`
persistido de outra conversa, não da colocação.

**Uma configuração ruim pode não falhar, só degradar 13×.** `CUDA_DENSE=1` com
VRAM insuficiente não recusa: aceita e trafega peso pelo PCIe. O planner precisa
**recusar** o que não cabe, não acomodar.
