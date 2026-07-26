# 15 — Kernels, numérica e threading

Resolve **D2** (W4A8 ou W4A16), **D4** (precisão de acumulação) e **D6**
(modelo de threading) de [13](13-decisoes-pendentes.md). As três andam juntas
porque cada uma restringe as outras.

## D2 — W4A32, não W4A8

Duas famílias de kernel para peso int4:

| | W4A8 | **W4A32** (escolhido) |
|---|---|---|
| instrução | `vpdpbusd` (AVX-VNNI), u8×s8→s32 | `vfmadd` fp32 |
| vazão de MAC | ~4× | referência |
| erro adicional | sim, quantização dinâmica da ativação | **nenhum** |
| custo | escala por token, requantizar a cada camada | desempacota e multiplica |

O reflexo é escolher W4A8 pela vazão. **Aqui o reflexo está errado.**

De [08](08-teto-de-desempenho.md): por token são ~2,17 GB de leitura contra
7,1 GFLOP de compute. A 60 GB/s e ~1–2 TOPS int8, isso é **~33 ms de memória
contra 4–7 ms de compute** — utilização de compute em torno de **20%**.

Vazão de MAC não é o gargalo. Há folga de sobra, e gastá-la comprando precisão
é o negócio certo: sem quantização de ativação, o oráculo token-exato do M1 fica
muito mais fácil de atingir, e não há uma segunda fonte de erro a depurar
quando ele falhar.

**Decisão:** W4A32 no M1. Migrar para W4A8 só se a medição mostrar compute como
limitante — e aí com o número na mão, não por reflexo.

Consequência prática: o kernel desempacota int4 → fp32 e multiplica com FMA. O
desempacotamento é parte do compute e entra naqueles 20%; o layout
metade/metade de [14](14-formato-container.md) existe para que ele custe um
`and`, um `shift` e uma conversão, sem permutação.

## D4 — precisão e ordem de acumulação

O critério de saída do M1 é **32/32 tokens idênticos ao oráculo**. Isso não
sai por acaso; sai de escolhas deliberadas.

| operação | decisão |
|---|---|
| acumulador do matmul | `float` (fp32) |
| desempacotamento int4 | para `float`, escala aplicada por grupo ao fim |
| rmsnorm | soma dos quadrados em `float`, ordem fixa |
| softmax do roteador | `float`, com subtração do máximo |
| RoPE | `float`, tabela de seno/cosseno pré-computada na init |
| atenção (scores, softmax) | `float` |
| resíduo | `float` |

Nada em `double`: o oráculo `transformers` roda em fp32 e casar com ele é o
objetivo, não superá-lo.

### A regra que decide reprodutibilidade

> **A ordem de redução é fixa e independente do número de threads.**

Soma em ponto flutuante não é associativa. Se a redução de um matmul depender de
como o trabalho foi dividido, o resultado muda com `--threads` — e o teste de
determinismo de [03](03-arquitetura.md) falha por construção, não por bug.

Concretamente: **cada thread possui linhas de saída inteiras**. A redução ao
longo de `I` acontece inteira dentro de uma thread, sempre na mesma ordem
(grupo 0, 1, 2, …), com um acumulador escalar por linha. Threads nunca somam
parcelas umas das outras.

Isso custa: nada em matmuls grandes (há linhas de sobra para distribuir) e algo
em vetores curtos, onde parte das threads fica ociosa. Comprar reprodutibilidade
com um pouco de ociosidade é o negócio certo quando reprodutibilidade é critério
de saída.

O mesmo vale para a soma dentro de um vetor SIMD: a ordem de combinação das
lanes é fixa e documentada no kernel, não deixada ao compilador. Sem
`-ffast-math`.

### Onde o MLA pode divergir

A atenção MLA admite duas formulações algebricamente equivalentes: desabsorver o
KV comprimido para `[nope|rope]` completo e fazer a atenção normal, ou absorver
`kv_b` na projeção de query e operar no espaço comprimido. **Elas não dão os
mesmos bits.**

O oráculo usa a primeira. O M1 implementa a primeira. A segunda é otimização
de M5, e quando entrar terá de ser validada como mudança de saída aceitável ou
rejeitada — não introduzida em silêncio.

## D6 — threading

**Pool próprio, não OpenMP.** OpenMP resolveria, mas arrasta `libgomp` e a
promessa de [10](10-implementacao.md) é linkagem estática com o mínimo de
runtime. O pool é ~200 linhas: N threads, uma barreira, uma fila de tarefas de
índice.

```
principal:  para cada camada (47, serial)
              publica tarefa {func, argumentos, total_de_linhas}
              barreira
            (a cadeia de camadas e serial por natureza; nao ha o que sobrepor
             dentro de uma sequencia)
worker i:   pega faixas de linhas por indice atomico
            processa faixa inteira, acumulador local
            barreira
```

Particionamento **por linha de saída**, granularidade dinâmica (faixas obtidas
de um contador atômico) para tolerar o desbalanceamento entre P-cores e E-cores.
Nunca por expert: com top-4 de 64, particionar por expert deixaria a maioria das
threads sem trabalho.

Afinidade conforme [09](09-limites-de-recursos.md): default nos P-cores, com
`--affinity all` para medir. O pinning acontece na criação das threads.

Barreira: espera ativa curta seguida de bloqueio. A 22 tok/s são ~47 barreiras
por token, ~1000 por segundo — espera ativa pura queimaria os E-cores à toa.

## Ordem de implementação no M1

1. `matmul_f32` de referência, sem SIMD, sem threads — a definição executável
2. `matmul_int4_g` idem, ingênuo, validado contra `dequant_grouped` em Python
3. Oráculo token-exato passando com os dois ingênuos
4. Só então SIMD, e o oráculo tem de continuar passando
5. Só então threads, e o oráculo tem de continuar passando **com qualquer
   `--threads`**

Cada degrau mantém o anterior como referência executável. Quando o SIMD
divergir — e vai —, a comparação contra o ingênuo localiza o erro em minutos,
em vez de transformar o oráculo num teste de tudo contra nada.
