# 13 — Decisões pendentes

O que ainda não está definido, em ordem de urgência. Cada item tem recomendação
e o critério que a resolveria.

## Situação

| # | decisão | estado |
|---|---|---|
| D1 | empacotamento dos nibbles | **RESOLVIDO** — metade/metade, ver [14](14-formato-container.md) |
| D2 | W4A8 ou W4A16 | **RESOLVIDO** — W4A32, ver [15](15-kernels-e-numerica.md) |
| D3 | formato do container | **RESOLVIDO** — [14](14-formato-container.md) |
| D4 | precisão de acumulação | **RESOLVIDO** — [15](15-kernels-e-numerica.md) |
| D5 | tokenizer | **RESOLVIDO** — [16](16-tokenizer.md) |
| D6 | modelo de threading | **RESOLVIDO** — pool próprio, [15](15-kernels-e-numerica.md) |
| D7 | protocolo engine ↔ serviço | aberto, M3 |
| D8 | KV paginado, detalhes | aberto, M3 |
| D9 | amostragem | **RESOLVIDO** — greedy, temperatura, top-k, top-p; `src/sample.c` |
| D10 | licença e estrutura de repo | aberto |
| **D11** | **absorver o KV comprimido na query** | **RESOLVIDO** — absorvida é o default, 32/32 mantidos |
| **D12** | **softmax online (passe único)** | **RESOLVIDO** — existe, mas só acima do L3 |
| **D13** | **KV em bf16** | **REPROVADO** — 29/32 no oráculo; implementado, não é default |
| **D14** | **histórico do chat: raciocínio fica ou sai** | aberto — fica, e o motivo é o cache |

O texto abaixo preserva o raciocínio de cada decisão.

---

## RESOLVIDO — descoberto medindo, não planejando

### D11. Absorção do MLA, adiantada de M5 para M1

O M1 implementa a formulação **desabsorvida** porque é a do oráculo
([15](15-kernels-e-numerica.md)). Ela exige cachear K e V **expandidos**:
20 cabeças × (256 + 256) valores por token por camada.

```
KV expandido    47 × 4.096 × 10.240 × 4 B = 7,9 GB   <- lido INTEIRO a cada token
KV comprimido   47 × 4.096 ×    576 × 4 B = 443 MB
```

O primeiro número é o problema, e ele não é de armazenamento — é de **tráfego**.
A cada token de decode, a atenção lê todo o KV das 47 camadas. A 4k de contexto
isso são **7,9 GB por token contra 2,17 GB de pesos**: o KV passa a ser 78% do
tráfego, e o teto de [08](08-teto-de-desempenho.md) — calculado só sobre os
pesos — deixa de valer. A 60 GB/s, 7,9 GB por token limita a **7,5 tok/s antes
de qualquer compute**, contra o critério de ≥10.

Medido nesta máquina, 8 P-cores, AVX2, média de 128 posições em torno de cada
ponto:

| contexto | expandida | absorvida | ganho |
|---|---|---|---|
| 128 | 13,63 tok/s | 13,63 tok/s | — |
| 1.024 | 6,15 | **12,48** | 2,0× |
| 2.048 | 3,64 | **9,98** | 2,7× |
| 3.072 | — | **8,31** | |
| 4.096 | — | **7,48** | |
| **média das 4.096** | — | **10,09 tok/s** | |

A inclinação caiu de 6,5 µs para 1,5 µs por posição de contexto — 4,3× mais
plana. O que sobrou não é mais tráfego de KV: são os 8,4 GFLOP/token que a
atenção custa a 4k, a ~137 GFLOP/s efetivos. O cache comprimido de uma camada
(9,4 MB a 4k) cabe no L3 de 36 MB e é reusado pelas 20 cabeças, que é por que o
tráfego não explodiu com a leitura por cabeça.

**Contra o critério do M1** (`≥ 10 tok/s a 4k`): a média das 4.096 posições dá
**10,09**; a taxa instantânea já *com* 4k de contexto dá **7,48**. Pela leitura
estrita — decode com o contexto cheio — o critério ainda não está batido. Os
caminhos, em ordem de retorno: passe único com softmax online (corta metade das
leituras de KV), KV em bf16 (corta a pressão de L3 pela metade) e o M2.

**Recomendação: adiantar a absorção para o M1.** Dobrar `kv_b` na projeção de
query faz a atenção operar no espaço comprimido: o tráfego cai 18×, e o custo
extra é ~4,5 GFLOP/token de compute — que sobra, porque a utilização de compute
está em ~20% ([15](15-kernels-e-numerica.md), D2).

**A ressalva de [15](15-kernels-e-numerica.md) continua valendo**: as duas
formulações são algebricamente equivalentes e **não dão os mesmos bits**. Então
a absorção entrou assim, e não em silêncio:

1. as duas implementações coexistem, selecionáveis (`--attn absorbed|expanded`);
2. a desabsorvida continua sendo a referência do oráculo;
3. a absorvida só seria aceita com **32/32 tokens idênticos** contra o mesmo
   `ref.json`.

**Resultado: 32/32, inclusive na posição 31, cuja margem é 0,0007.** A absorção
está autorizada e é o default.

### A identidade

```
q_nope . k_nope[t]   =  q_nope^T (W_k c[t])   =  (W_k^T q_nope)^T c[t]
sum_t w[t] v[t]      =  sum_t w[t] (W_v c[t]) =  W_v (sum_t w[t] c[t])
```

Projeta-se a **query** uma vez — custo fixo, independente do contexto — em vez
de expandir **cada posição** do cache.

### A armadilha que a primeira versão caiu

`W_k^T` sugere materializar `kv_b` em fp32 para poder ler colunas. Feito assim,
a absorção **piorou**: 5,6 tok/s contra 12,3 da expandida. O motivo é que
`kv_b` denso são 18,4 MB por camada, **863 MB lidos por token** — contra os
108 MB do int4 empacotado. A absorção comprou 18× de tráfego de KV e devolveu
8× de tráfego de peso.

O conserto é notar que a transposta **não exige coluna**: como `W_k` é
`[nope][rank]` linha-maior, `W_k^T q` é uma acumulação de linhas
(`for r: acc += q[r] * W_k[r][:]`), e linha é exatamente o que o container
entrega barato. Sem cópia, sem tráfego extra.

> A lição não é sobre MLA: é que "otimizar" trocando trabalho por memória
> precisa da conta de tráfego feita **antes**, num engine cujo gargalo declarado
> é banda de memória ([08](08-teto-de-desempenho.md)).

---

### D12. Softmax online — certo, e mais lento onde importa hoje

A atenção lê o cache duas vezes: uma para os scores, outra para acumular a
saída. O softmax online mantém o máximo e a soma correntes e faz tudo num
passe só:

```
m' = max(m, s)      acc' = acc·exp(m − m') + exp(s − m')·c[t]
```

Algebricamente é o mesmo softmax com subtração do máximo — o fator comum sai da
soma —, e mantém **32/32**. Medido a 4k:

| | tok/s (média de 4.096) |
|---|---|
| dois passes | **10,09** |
| passe único | 9,19 |

**Nove por cento mais lento.** O motivo: a 4k, o cache de uma camada são 9,4 MB
e cabem no L3 de 36 MB — que ainda é compartilhado pelas 20 cabeças —, então o
segundo passe quase não custa, enquanto o passe único paga um desvio por
posição e reescalona 512 floats a cada máximo novo.

Acima do L3 a conta inverte, e aí o segundo passe vem da DRAM. Então os dois
existem e a escolha é **por tamanho**: passe único só quando o cache da camada
passa de 12 MB (`TT_L3_BUDGET`, conservador contra os 36 MB reais porque o L3
também está servindo os pesos que passam).

> Vale como aviso: "menos leituras" não é o mesmo que "mais rápido" quando as
> leituras que se cortam já vinham do cache.

---

### D13. KV em bf16 — REPROVADO pelo próprio critério

Cortaria o cache pela metade (108,3 KB → 54,2 KB por token) e aliviaria a
pressão de L3, que é justamente o recurso que o D12 mostrou ser decisivo.

**Reprovou: 29/32.** Três posições viraram o token — 15, 18 e 31 — todas com
margem entre 0,012 e 0,066. bf16 tem 7 bits de mantissa contra 23; o erro entra
em cada `c[t]` e sai amplificado pela soma sobre 4.096 posições.

Está implementado e disponível (`--kv bf16`), com o default em `f32`. Não é
"quase certo": pelo critério que autorizou a absorção, é **mudança de saída**, e
adotá-la exigiria decidir que perder 3 tokens em 32 vale a memória — decisão que
não se toma de dentro de uma otimização.

### int8 g64 no cache — o caminho "melhor" também reprova

A saída óbvia era quantização por grupo com escala: erra 0,59% contra os ~1,5%
do bf16 ([04](04-modelo-alvo.md)) e ocupa **menos** que ele — 576 int8 mais 9
escalas dão 612 B por token por camada, contra 1.152 B do bf16 e 2.304 B do
f32. Implementado (`--kv int8`), o cache caiu de 108,3 KB para **28,8 KB por
token**, 3,8×.

Reprovou **igual: 29/32.** E o ganho de tempo foi de 2% a 1.024 posições
(15,62 contra 15,32 tok/s) — a desquantização no laço come a economia de
tráfego enquanto o cache ainda cabe no L3.

| | por token | 32/32 | tok/s a 1.024 |
|---|---|---|---|
| f32 | 108,3 KB | **32/32** | 15,32 |
| bf16 | 54,2 KB | 29/32 | ~15,4 |
| int8 g64 | **28,8 KB** | 29/32 | 15,62 |

### O que isso ensina sobre o próprio critério

Dois quantizadores de qualidade bem diferente — 1,5% e 0,59% — reprovam com o
mesmo placar e deltas de logit parecidos (0,02 a 0,06 em logits de ~90, cinco
centésimos de por cento). O motivo não é a qualidade deles: é que **as margens
deste prompt são pequenas**, 0,0007 na posição 31.

Ou seja: a 32/32 detecta *mudança de saída*, que é o que ela promete, mas **não
distingue "diferente" de "pior"**. Para decidir se int8 no KV degrada de fato,
falta a medida que o M0 previa e que nunca rodou: **perplexidade num conjunto
fixo**. Enquanto ela não existir, o default correto é o que passa no teste que
existe — `f32` — e os outros dois ficam disponíveis e documentados.

Onde eles vão importar: contexto de agente. A 32k, o cache em f32 são 3,5 GB e
saem do L3; em int8 são 0,9 GB. Aí a conta inverte e a decisão volta — com
perplexidade na mão.

---

## RESOLVIDO — congelava em 19 GB se a conversão rodasse

### D1. Ordem de empacotamento dos nibbles

O `convert_to_int4.py` já escreve int4 como **par/ímpar**: `byte[i]` contém o
peso `2i` no nibble baixo e `2i+1` no alto.

Ao desempacotar 32 bytes com AVX2, `v & 0x0F` entrega os pesos
`0, 2, 4, …, 62` e `v >> 4` entrega `1, 3, 5, …, 63` — **ordem intercalada**.
Para multiplicar contra as ativações é preciso intercalá-las também.

A convenção da indústria (llama.cpp `Q4_0` e derivados) é **metade/metade**:
para um grupo de 64, `byte[i] = w[i] | (w[i+32] << 4)`. O desempacotamento
entrega `w[0..31]` e `w[32..63]` já contíguos, sem permutação nenhuma.

**Recomendação: trocar para metade/metade antes de converter.**

O custo de não trocar não é grande — permutar a ativação é `O(cols)` uma vez por
matmul contra `O(rows × cols)` do matmul, ou seja, amortizado. Mas o custo de
trocar **agora** é zero, e depois é reconverter 19 GB. Não há razão para
escolher a convenção menos comum.

**Resolve:** editar `quant_grouped`/`dequant_grouped` e rodar `--limit 1`.

### D2. W4A8 ou W4A16 — o que o kernel consome

Duas famílias, e a escolha decide precisão, kernel e se o container precisa de
metadados extras.

| | W4A8 (quantiza ativação) | W4A16/A32 (desquantiza peso) |
|---|---|---|
| instrução | `vpdpbusd` (AVX-VNNI), u8×s8→s32 | `vfmadd` fp32 |
| vazão | ~4× mais MACs/ciclo | referência |
| erro extra | sim, quantização dinâmica da ativação | não |
| complexidade | escala por token, requantizar a cada camada | desempacota e multiplica |

O reflexo é escolher W4A8 pela vazão. **Aqui o reflexo está errado**, e o motivo
está em [08](08-teto-de-desempenho.md): o decode é **memory-bound**. São ~33 ms
de leitura de memória contra 4–7 ms de compute por token — **utilização de
compute em torno de 20%**. Vazão de MAC não é o gargalo; há folga de sobra.

**Recomendação: W4A32 no M1** — desempacota int4 para fp32 e usa FMA. Mais
simples, sem erro adicional de ativação, e o oráculo token-exato fica mais
fácil de atingir. Migrar para W4A8 só se a medição mostrar compute como
limitante, e aí com número.

**Resolve:** é decisão de kernel, não de container — o formato em disco serve às
duas. Não bloqueia a conversão, mas bloqueia o M1.

---

## BLOQUEIA O M1

### D3. Especificação do formato de container

O conversor **já definiu** o formato de fato: magic `TOOTEGA\x01`, cabeçalho de
128 bytes, blobs alinhados a 64, índice JSON no fim, entrada por tensor com
`kind`/`shape`/`data`/`scales`/`group`.

Falta escrever isso como especificação para o leitor em C — o engine tem de
casar byte a byte com o escritor. Enquanto for só código Python, é convenção
tácita, e convenção tácita entre dois programas em linguagens diferentes é
onde nascem bugs de alinhamento.

**Resolve:** documento curto com o layout binário exato, escrito **antes** do
carregador.

### D4. Precisão de acumulação por operação

O critério de saída do M1 é 32/32 tokens idênticos ao oráculo `transformers`.
Isso só é atingível se as escolhas numéricas forem deliberadas:

| operação | pendente |
|---|---|
| acumulador do matmul | fp32 (provável) ou int32 |
| rmsnorm | fp32; soma em ordem fixa? |
| softmax do roteador | fp32, com max-subtraction |
| RoPE | fp32; tabela pré-computada ou `sinf`/`cosf` |
| atenção MLA | onde desabsorver o KV comprimido |
| redução paralela | **ordem fixa** ou o resultado varia com o número de threads |

A última é a que morde: soma em ponto flutuante não é associativa. Se a redução
depender de quantas threads rodaram, o resultado muda com `--threads` e o teste
de determinismo de [03](03-arquitetura.md) falha por construção.

**Recomendação:** ordem de redução fixa e independente do número de threads.
Custa um pouco de desempenho e compra reprodutibilidade — que é critério de
saída, não luxo.

### D5. Tokenizer

`tokenizer.json` está em disco (formato HF). Falta determinar **qual variante**
de BPE o GLM-4.7-Flash usa — pré-tokenizador, tratamento de bytes, tokens
especiais — e escrever o encoder/decoder em C.

Sem isso não há teste fim a fim, só forward contra ids pré-computados.

**Resolve:** inspecionar `tokenizer.json` e escrever o leitor. É trabalho
mecânico, mas é caminho crítico do M1.

### D6. Modelo de threading

Indefinido: pool próprio ou OpenMP; como o trabalho é particionado (por linha de
matmul, por expert, por cabeça de atenção); onde ficam as barreiras.

Restrições conhecidas: 47 camadas seriais, top-4 experts por camada, e o pinning
em CPU híbrida tem de acontecer antes de `main` ([09](09-limites-de-recursos.md)).

**Recomendação:** pool próprio com barreira por camada. OpenMP resolveria, mas
arrasta `libgomp` e a promessa de [10](10-implementacao.md) é linkagem estática
com o mínimo de runtime.

---

### D14. O histórico do chat guarda o raciocínio — e isso conflita com o cache

O `chat_template.jinja` **remove o raciocínio dos turnos passados**: um turno
de assistente já concluído é renderizado como `<|assistant|></think>` mais o
conteúdo, sem o bloco de pensamento. Só o turno aberto ganha `<think>`.

O `tootega chat` **não faz isso** — ele acrescenta os ids gerados como saíram.
Não é descuido, é o único jeito de o cache de prefixo funcionar:

> Re-renderizar a conversa a cada turno significa **re-tokenizar**, e
> re-tokenizar não devolve necessariamente os mesmos ids que foram gerados. O
> KV deixaria de casar num ponto arbitrário do histórico, e o sintoma seria
> "às vezes o cache erra", que é o pior tipo de bug.

Medido no primeiro teste de duas voltas: **92 de 100 tokens de prompt vieram do
KV** no segundo turno — 8 tokens de prefill em vez de 100. É exatamente a
propriedade que [05](05-roadmap.md) pede no M4.

O custo de manter o raciocínio: o contexto cresce mais rápido, e o histórico
diverge do formato em que o modelo foi treinado para multi-turno.

**Como resolver sem escolher um dos dois:** guardar, junto com os ids gerados, o
**offset do `</think>`**. O histórico continua sendo ids, o cache continua
casando até esse ponto, e o corte do raciocínio vira um truncamento de KV — que
é barato — em vez de uma re-tokenização. Entra quando o M4 começar; até lá está
documentado como divergência conhecida, não como bug.

---

---

## A lição mais cara — duas implementações minhas não são validação

O engine passou **32/32** contra o oráculo, e a perplexidade contra o BF16 deu
diferença indistinguível de zero. Nenhum dos dois testes detectou que a
**variante de RoPE estava errada**.

`Glm4MoeLiteConfig.rope_interleave` é `True` por default e o `config.json` não
sobrescreve. A atenção usa `apply_rotary_pos_emb_interleave`; eu implementei o
ramo `else`:

| variante | pareia |
|---|---|
| `rotate_half` — a implementada | `x[i]` com `x[i + rot/2]` |
| **interleave** — a correta | `x[2i]` com `x[2i+1]` |

Na posição 0 as duas coincidem (`cos = 1`, `sin = 0`), e o erro cresce com a
posição. O texto começava bem e desandava.

### Por que os testes não pegaram

**O oráculo e o engine são duas implementações da mesma leitura minha** do
`modeling_glm4_moe_lite.py`. Concordarem prova que erram igual, não que estão
certos. Independência de *implementação* não é independência de
*interpretação*.

**A perplexidade comparou int4 contra BF16 dentro do meu próprio forward.** O
`ablate` troca de onde vem o peso, não a matemática. Ela mediu quantização e
nunca teve como medir correção.

### O alarme que existia e foi silenciado

**Perplexidade ~340** em prosa técnica comum. Um modelo de 31B fica em 10–30.
Eu vi 223–473 entre janelas, chamei de "corpus difícil" e segui — porque a
comparação *relativa* que me interessava parecia sadia.

> Métrica relativa some com erro que afeta os dois lados. O valor **absoluto**
> era o único que denunciava, e ele estava na tela o tempo todo.

### O que muda no processo

1. **Toda métrica com valor absoluto conhecido tem faixa esperada declarada.**
   Perplexidade fora de 10–40 é falha, não observação.
2. **Referência de arquitetura tem de ser externa.** Oráculo próprio valida
   aritmética, não interpretação. A validação real é o `transformers` rodando o
   modelo — que nunca rodou aqui por não caber em 64 GB. Enquanto não houver,
   a geração livre é o único teste que estressa a interpretação.
3. **Teste de geração é obrigatório, não complementar.** Teacher-forcing esconde
   erro que realimenta; foi o `gentest` que expôs isto, e ele só existiu porque
   uma resposta pareceu estranha a olho nu.

### O que ficou inválido

`ref32.json` foi regerado. As medições de perplexidade e a ablação de
[04](04-modelo-alvo.md) — v1 +6,8%, v2 ≈ 0, "atenção explica 100%" — foram
todas feitas com RoPE errado e **precisam ser refeitas**. Os números de
velocidade seguem válidos: o bug não mudava a quantidade de trabalho.

---

## PODE ESPERAR

### D7. Protocolo engine ↔ serviço (M3)
O engine anterior usa protocolo de linha por stdin/stdout com quadros `SUBMIT`. Funciona
e é fácil de depurar. Decidir no M3.

### D8. Detalhes do KV paginado (M1 simples, M3 completo)
[11](11-memoria.md) define blocos compartilhados por hash de prefixo. Tamanho de
bloco, estrutura do índice e política de despejo ficam para quando houver
medição. O M1 pode usar KV contíguo por slot.

### D9. Amostragem
Greedy, temperatura, top-p, top-k. Trivial, mas precisa existir para o CLI.

### D10. Licença e estrutura de repositório
Não definidas.

---

## Ordem sugerida

1. **D1** — 10 minutos, e destrava a conversão de 19 GB
2. **D3** — a especificação, antes de escrever o carregador
3. **D2 + D4** — as duas decisões numéricas, juntas: definem o kernel
4. **D5** — tokenizer, caminho crítico e mecânico
5. **D6** — threading, quando houver um kernel para paralelizar

D7–D10 entram nos marcos correspondentes.
