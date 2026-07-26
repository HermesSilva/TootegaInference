# 06 — Lições aprendidas

Princípios que vieram de medir engines de streaming de disco nesta mesma
máquina, antes de escrever uma linha deste. Nenhum código foi reaproveitado —
o que sobreviveu foi o **entendimento**, e cada item abaixo foi reimplementado
do zero em `src/`.

## Adotar

**Movimento de bloco grande é troca de ponteiro, nunca cópia.** Promover um
expert de ~19 MB para o cache troca os descritores; o slab do evictado vira o
workspace do próximo. Zero `memcpy`, zero `malloc`/`free` por eviction.

**Buffer só realoca se não couber.** Em regime permanente o allocator não é
tocado. É o que virou a arena de `src/common.c`.

**Zero cópia via mmap.** Os ponteiros do quantizador apontam direto para dentro
do arquivo mapeado, com *pre-touch* das páginas em vez de cópia — I/O sem a
cópia e sem o slab intermediário.

**Arena única para o conjunto residente:** pesos ligados como uma arena por
camada, nunca liberados individualmente. Mata fragmentação e syscalls por
expert.

**Não zerar buffer que vai ser sobrescrito.** `calloc` no caminho quente é
`memset` que ninguém pediu.

**Fundir para não reler.** Calcular gate e up na mesma passada, lendo `x` uma
vez, vale ~33% do tempo de matmul de expert no decode. Virou
`tt_matmul_batch`.

**Validação token-exata contra oráculo `transformers`.** É o que separa engine
de gerador de plausibilidade — e o que pegou o RoPE errado ([17](17-diagnostico.md)).

**Recusar-se a subir quando a RAM não dá.** Visto acontecer: 3,3 GB
disponíveis, e o engine abortou em vez de arriscar OOM no meio da geração.
Comportamento correto. Virou `--plan` e `--ram-gb` ([09](09-limites-de-recursos.md)).

## Jogar fora

Todo o maquinário de esconder disco: **PILOT** e prefetch cross-layer,
roteamento **dual-SSD**, ajuste de **`O_DIRECT`**, **pin set aprendido**
(um mapa de calor persistido), **tier LRU por camada**, guarda de eviction LFRU. É a maior
parte do orçamento de complexidade do projeto e não serve para nada quando o
modelo é residente.

Um engine para modelo enxuto é uma fração do tamanho — e essa é a principal
vantagem estrutural do TootegaEngine, não a velocidade.

## Evitar

Defeitos observados em engines de streaming, que só não doem a ~1 tok/s. A 20+
tok/s viram percentual.

**Centenas de `malloc`/`free` por token.** Um passo de decode chegava a ~500
alocações — três por token só no caminho principal, uma delas do tamanho do
vocabulário, e até sete por camada no preditor. Todas de tamanho fixo,
conhecido na init. → **arena persistente**, `src/common.c`.

**`getenv` dentro do laço de camadas.** Ler variável de ambiente e converter a
cada camada, a cada token, dá ~300 varreduras lineares de `environ` por token.
→ **config imutável lida uma vez**, `src/model.c`.

**Buscas lineares onde há estrutura.** Achar um expert residente por varredura
sobre listas de pin e LRU, e outra para escolher a vítima, custava ~150 mil
comparações por token. → **índice direto**, e hash aberto no container.

**Arquitetura cravada no parser.** Erro fatal quando `n_group != 1` ou sem
`kv_lora_rank` reduz o universo de modelos a uma família só — DeepSeek com
`n_group=8`, Qwen3 com GQA, qualquer um sem MLA são rejeitados. Transforma
"testar outro modelo" numa investigação que termina sem opção. → **config → IR,
com as variantes declaradas** ([03](03-arquitetura.md)).

**Sem flags de hardening.** `-O3 -Wall -Wextra` e nada de
`-D_FORTIFY_SOURCE=2`, `-fstack-protector-strong`, `-fPIE -pie`,
`-Wl,-z,relro,now`, num engine em C que consome entrada de rede e arquivos de
modelo de terceiros. Custo de corrigir: uma linha.

## Ressalvas de método que aprendemos medindo

**Não trocar duas variáveis de uma vez.** Medimos "nativo + CUDA" contra "release
CPU" e o resultado não dizia nada, porque o build nativo (AVX-VNNI) e o CUDA
mudaram juntos. Foi preciso refazer com o estado do meio isolado.

**Fixar o orçamento de RAM entre configs.** Duas medições com `RAM_GB` diferente
não são comparáveis, e o valor "auto" flutua com o que o SO devolveu.

**Aquecer antes de medir concorrência.** Sem um turno de aquecimento, o custo do
tier frio cai todo em K=1 e inventa um ganho de escala que não existe.

**Hash da saída em toda medição.** Foi assim que descobrimos que CPU e CUDA
produzem bytes idênticos — e que a única divergência vinha de um KV persistido de outra conversa, não da colocação.

**Uma configuração ruim pode não falhar, só degradar 13×.** `CUDA_DENSE=1` com
VRAM insuficiente não recusa: aceita e trafega peso pelo PCIe. O planner precisa
**recusar** o que não cabe, não acomodar.
