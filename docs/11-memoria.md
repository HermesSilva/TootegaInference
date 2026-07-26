# 11 — Disciplina de memória e zero-copy

O container int4 tem 19,3 GB e a máquina tem 64 GB. Parece folgado, e não é: o KV
a 64k de contexto são 3,5 GB por slot, e o orçamento default deixa 16 GB para o
SO ([09](09-limites-de-recursos.md)). Cada cópia evitada é contexto que cabe.

Regra geral: **um byte de peso existe em um lugar só.**

## 1. O arquivo mapeado é a memória

Pesos são lidos **in-place a partir do mapeamento do arquivo**. Não existe passo
de carga que copia disco → buffer. O quantizador aponta direto para dentro das
páginas mapeadas:

```c
qt->q4 = (uint8_t*)(base + tensor->offset);   /* nao ha memcpy */
qt->s  = (float*)  (base + scales->offset);
```

O colibrì tem esse caminho (`colibri.c:1479-1520`) e o comentário dele descreve
exatamente o ganho: *"This is pread's I/O without the copy and without the slab."*
A diferença aqui é que **é o único caminho**, não uma alternativa a um caminho de
cópia.

Consequências:

- **A page cache do SO é o cache.** Sem buffer duplo, sem gerenciar residência de
  peso à mão.
- **Páginas de peso são recuperáveis.** Sob pressão de memória o SO as descarta e
  relê do disco. Por isso elas **não contam** contra `--ram-gb` ([09](09-limites-de-recursos.md)):
  são file-backed, não anônimas.
- **Vários processos compartilham as mesmas páginas.** Duas instâncias do engine
  no mesmo modelo custam um conjunto de pesos, não dois.

**Requisito que volta para o conversor (M0):** o container int4 tem de emitir
tensores **alinhados a 64 bytes**, senão a leitura direta com SIMD precisa de um
caminho de cópia para realinhar — e aí perde-se o zero-copy inteiro. O colibrì
tropeça nisso e cai para `pread` quando `(off & 3)` (`colibri.c:1483`). O nosso
formato garante o alinhamento na escrita.

## 2. Pre-fault e lock, dentro do orçamento

Mapear não traz nada para a RAM. O primeiro acesso a cada página é uma falta.
Na carga:

1. `madvise(MADV_WILLNEED)` / `PrefetchVirtualMemory` — inicia leitura assíncrona;
2. toque de uma palavra a cada 4 KB — garante residência;
3. **opcionalmente** `mlock` / `VirtualLock` no conjunto residente, limitado por
   `--ram-gb`, para o SO não despejar o modelo enquanto outra coisa pede memória.

O passo 3 é opt-in porque travar ~18 GB numa máquina de trabalho é uma decisão da
pessoa, não do engine.

## 3. Páginas grandes

O conjunto de trabalho é 19,3 GB. Com páginas de 4 KB são **4,7 milhões de
páginas**; o L2 TLB do Raptor Cove tem ~2048 entradas, cobrindo 8 MB. Cada token
percorre ~2 GB de peso, então praticamente todo acesso paga um page walk.

Com páginas de 2 MB, as mesmas 2048 entradas cobrem **4 GB** — dentro da ordem de
grandeza do que um token toca.

```
4 KB:  cobertura de TLB = 2048 x 4 KB  =    8 MB
2 MB:  cobertura de TLB = 2048 x 2 MB  = 4096 MB
```

Linux: `MADV_HUGEPAGE` (THP) sobre o mapeamento. Windows: large pages exigem o
privilégio `SeLockMemoryPrivilege`, então é opt-in com instrução clara.

**Marcado para medir no M5**, não assumido. O prefetcher de hardware e o acesso
sequencial atenuam o custo; o ganho real pode ser de 5% ou de 25%, e a única
forma de saber é medir com e sem.

## 4. Arena de scratch, zero `malloc` por token

Todos os buffers do forward — normas, resíduos, logits, saídas intermediárias de
expert — têm tamanho fixo, conhecido depois de ler o config. São fatias de **uma
arena dimensionada na init**.

O laço de token não chama o allocator. Nem uma vez.

O colibrì faz ~500 `malloc`/`free` por token ([06](06-licoes-colibri.md)); a
0,93 tok/s isso é 0,01% e não importa. A 22 tok/s importa, e a correção é
estrutural, não uma micro-otimização a fazer depois.

## 5. KV paginado com blocos compartilhados

Este é o maior economizador de memória do projeto, e ele nasce do caso de uso
agêntico.

KV alocado em **blocos de N tokens** (N = 256, a definir por medição), com uma
tabela de blocos por slot. Nada de `realloc`, nada de buffer contíguo por sessão.

O ganho vem do compartilhamento: num CLI agêntico, **toda sessão começa com o
mesmo prefixo** — system prompt, schemas de ferramentas, instruções do projeto.
Tipicamente 10–20k tokens. A 52,9 KB por token ([08](08-teto-de-desempenho.md)):

```
prefixo de 20k tokens = 1,06 GB de KV

4 sessoes, blocos privados:      4,24 GB
4 sessoes, blocos compartilhados: 1,06 GB
```

Blocos são imutáveis depois de escritos e referenciados por contagem. Quando uma
sessão diverge do prefixo, ela passa a alocar blocos próprios a partir do ponto
de divergência — copy-on-write no nível de bloco, não de token.

Isso interage diretamente com a regra de ordenação de prompt de
[03](03-arquitetura.md): **estável primeiro, volátil por último** não é só higiene
de cache, é o que faz os blocos serem compartilháveis.

## 6. Quantizado em disco, quantizado na memória

Nunca há um passo de "desquantizar para fp16 e então multiplicar". Os kernels
consomem int4 com escalas por grupo diretamente. Um buffer fp16 intermediário
para os experts ativos custaria 4× a leitura e destruiria a conta do
[08](08-teto-de-desempenho.md).

Corolário: fundir operações que releriam a mesma entrada. `gate` e `up` de um
expert leem o mesmo `x` — uma passada, duas saídas. O colibrì mede ~33% do tempo
de matmul de expert nisso (`colibri.c:469`).

## 7. Operações in-place

- `rmsnorm` escreve sobre o próprio buffer quando o original não é mais necessário;
- resíduo acumula em `x`, sem terceiro buffer;
- softmax do roteador in-place sobre os logits.

Nenhuma camada aloca. Nenhuma camada copia o residual stream.

## 8. Embeddings ficam no arquivo, `lm_head` fica residente

A tabela de embeddings tem 154.880 × 2048 entradas, e **uma linha é lida por
token**. Não vale residência: fica mapeada, o SO cuida.

O `lm_head` é o oposto — é lido **inteiro** a cada token (0,317B params, 8,9% do
tráfego). Residente, obrigatoriamente.

A distinção parece óbvia escrita assim e é exatamente o tipo de coisa que um
carregador ingênuo erra ao tratar "tensores densos" como um bloco só.

## 9. Fronteira GPU/CPU: 8 KB, buffer fixo

Na partição por camada ([03](03-arquitetura.md)) atravessam apenas ativações:
`hidden × 4 bytes = 8 KB`. O buffer host é **pinned e reutilizado**, alocado uma
vez. Peso **nunca** cruza o PCIe — é a lei de [02](02-hardware.md).

## Contabilidade

| região | conta em `--ram-gb`? | por quê |
|---|---|---|
| mapeamento do modelo | **não** | file-backed, recuperável pelo SO |
| páginas travadas com `mlock` | **sim** | anônimas na prática |
| blocos de KV | **sim** | anônimas |
| arena de scratch | **sim** | anônimas |
| buffer pinned da GPU | **sim** | anônimo e não paginável |

Um `--stats mem` imprime as cinco linhas com o valor real medido, não estimado.
Sem isso, "orçamento de memória" é uma opinião.
