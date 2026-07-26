# 04 — Modelo alvo: GLM-4.7-Flash

`zai-org/GLM-4.7-Flash` — 31,2B totais, ~3,57B ativos por token, fonte BF16 de
62,4 GB, **19,3 GB no container int4** (medido; a estimativa de 15,6 GB que
circulou antes ignorava escalas, MTP em int8 e embeddings — ver
[Formato de quantização](#formato-de-quantização)).

## Por que este

Levantamos 89 MoE publicados nos últimos 400 dias e classificamos por *fit* real
nesta máquina. Os candidatos que sobraram:

| modelo | data | int4 | SWE-bench | contexto | cabe em 64 GB |
|---|---|---|---|---|---|
| **GLM-4.7-Flash** | 2026-01-19 | **19,3 GB** | **59,2** | **202.752** | sim, com folga |
| Ling-2.6-flash | 2026-04-28 | 53,7 GB† | 61,2 | 131.072 | sim, apertado |
| DeepSeek-V4-Flash | 2026-04-22 | 79,8 GB† | — | 1.048.576 | **não** |
| Qwen3-30B-A3B-2507 | 2025-07-28 | 15,3 GB† | — | 262.144 | sim |

† estimativa a 0,5 B/peso. Só o GLM foi convertido e medido; os demais
cresceriam na mesma proporção (~+12% de escalas, mais o que a política de
precisão preservar) se passassem pelo mesmo conversor. A comparação relativa não
muda — o Ling continua ~2,8× o GLM.

O Ling ganha por 2 pontos em SWE-bench sendo quase **3× maior**. E para uso
agêntico o KV entra na conta junto com os pesos:

```
GLM-4.7-Flash   19,3 (pesos) + 11 (KV a 200k) = 30,3 GB   sobram 34 GB
Ling-2.6-flash  53,7 (pesos) + 4,8 (KV a 131k) = 58,5 GB  sobram 5 GB
```

Cinco GB de folga é a margem em que vimos aquele engine **se recusar a
subir** por risco de OOM. Não é margem, é armadilha.

O Qwen3-30B-A3B tem tamanho parecido e é bom, mas usa GQA sem `kv_lora_rank` —
irrelevante para o TootegaEngine, que não terá a trava do engine anterior, mas relevante
para o M1, onde queremos comparar contra um engine que já funciona.

Ressalva honesta: os números de benchmark são do fabricante comparando-se com
concorrentes. Valem o desconto habitual. O Ling-2.6-flash fica como plano B se a
qualidade real decepcionar.

## Especificação

```
num_hidden_layers        47      (1 densa + 46 MoE)
hidden_size              2048
num_attention_heads      20
n_routed_experts         64
num_experts_per_tok      4
n_shared_experts         1
moe_intermediate_size    1536
intermediate_size        10240   (camada densa)
first_k_dense_replace    1
vocab_size               154880
max_position_embeddings  202752
torch_dtype              bfloat16
```

**Atenção MLA:**

```
q_lora_rank              768
kv_lora_rank             512
qk_nope_head_dim         192
qk_rope_head_dim         64
v_head_dim               256
```

**Roteamento:**

```
n_group                  1
topk_group               1
norm_topk_prob           true
routed_scaling_factor    1.8
```

**Especulação:** `num_nextn_predict_layers = 1` — há cabeça MTP. Testável, e o
regime residente é exatamente onde a experiencia anterior diz que especulação
finalmente paga (perto de residência total, não sob streaming).

**Sem DSA:** não há `index_topk`. Atenção densa; o lightning indexer do GLM-5.2
não se aplica. Menos código a escrever.

## O que o engine tem de implementar

| peça | detalhe |
|---|---|
| Atenção MLA | com q-LoRA (`q_a_proj` → `q_b_proj`), KV comprimido a 576 valores/token/camada |
| RoPE parcial | 64 das 256 dimensões por cabeça têm rotação; 192 não |
| Roteador MoE | top-4 de 64, `norm_topk_prob`, escala 1,8, **sem** limite de grupo |
| Expert compartilhado | 1 por camada MoE, sempre ativo |
| Camada densa | a camada 0 é MLP comum de `intermediate_size=10240` |
| MTP | 1 cabeça, opcional no M1 |
| Tokenizer | `tokenizer.json` no formato HF; obrigatório |

## Formato de quantização

Alvo: **int4 com escalas f32 por grupo de 64** ao longo da dimensão de entrada —
0,5625 byte por peso efetivo. É o formato que o cálculo de teto assume.

**Bloqueio conhecido:** a fonte é BF16, e o conversor do engine anterior
(`c/tools/convert_fp8_to_int4.py`) foi escrito para o checkpoint FP8 do GLM-5.2.
O modelo é compatível com o *engine* do engine anterior, mas o caminho de quantização
precisa aceitar BF16. Isso é código a escrever, não download — e é o primeiro
item do M0 no [roadmap](05-roadmap.md).

Precisão por tensor, seguindo a lição aprendida (cabeça MTP em int4 derruba
aceitação de draft para 0–4%). Contagens **lidas do índice do container já
escrito**, não do `--dry-run`:

| tensor | precisão | tensores | saída |
|---|---|---|---|
| experts, shared, densa, atenção | int4 g64 | 9.161 | 16,452 GB |
| camada MTP (`model.layers.47.*`) + `lm_head` | int8 g64 | 203 | 1,358 GB |
| roteador (`gate`), `e_score_correction_bias`, normas | fp32 | 337 | 0,247 GB |
| `embed_tokens` (do modelo **e** da MTP) | bf16 | 2 | 1,269 GB |
| **total** | | **9.703** | **19,326 GB** |

Arquivo em disco: 19,327 GB. O delta de 1,5 MB é o índice JSON; o padding de
alinhamento a 64 B some no arredondamento.

**São 19,3 GB, não os 15,6 estimados antes.** A diferença: escalas
(+0,0625 B/peso), MTP em int8 e embeddings preservados. Destes, ~18,1 GB são
residentes — os dois `embed_tokens` ficam file-backed, lidos uma linha por token.

> Uma tabela anterior aqui dizia 212 tensores / 1,69 GB em int8 e um só
> `embed_tokens`, fechando em 19,03 GB. Vinha de um `--dry-run` anterior ao
> conserto de ordem da `POLICY` — as normas da MTP eram capturadas pela regra
> int8 antes de chegarem à regra fp32, e a MTP tem o seu próprio `embed_tokens`.

### Custo medido da quantização

Do conversor, erro relativo da **saída de um matmul** (a métrica que prevê
qualidade; erro por peso isolado não prevê). Modelo inteiro, escala `absmax/qmax`
pura (`--clip-grid 1`):

| | por peso | saída | pior saída | n |
|---|---|---|---|---|
| int4 g32 † | — | 10,44% | — | — |
| **int4 g64** | 11,47% | **10,81%** | 14,13% | 9.161 |
| int4 g128 † | — | 12,76% | — | — |
| int8 g64 | 0,65% | **0,62%** | 1,17% | 203 |

† g32 e g128 vêm de conversões parciais anteriores; só g64 rodou no modelo
inteiro.

**Nada está quebrado — isto é o piso do round-to-nearest.** Com passo
`Δ = absmax/qmax` e erro uniforme dentro do passo, `σ_e = Δ/√12`, e como
`per_output = ‖E‖_F/‖W‖_F`, a razão é exatamente `σ_e/σ_w`. A checagem de
sanidade é a razão entre as classes: **10,81 / 0,62 = 17,4**, contra
`127/7 = 18,1` de razão de passo. Bate. Se um dia divergir muito disso, é bug,
não quantização.

Três conclusões que mudam o plano de ablação:

**O erro não diminui com o tamanho da matriz.** Em `y = Wx`, ruído e sinal
crescem ambos com `√cols`, então a razão fica `σ_e/σ_w` — independente de
quantos termos a soma tem. Não há cancelamento.

**Tamanho de grupo é alavanca fraca.** g64 → g32 compra 1,3 ponto percentual e
custa +11% de arquivo (0,5625 → 0,625 B/peso). Troca ruim, e é por isso que o
`--group` deixou de ser descrito como "o principal knob".

**A escala ótima não é o `absmax`.** `absmax/qmax` otimiza o pior peso do grupo,
não o grupo. Encolher a escala clipa os extremos e aperta o passo para todos os
outros pesos. `convert_to_int4.py` agora faz uma busca MSE por grupo sobre uma
grade de frações do `absmax` (`--clip-grid`, default 13 pontos entre 0,70 e
1,00). Medido em 24 tensores reais do shard 4, 512 linhas cada, com o mesmo `x`
por tensor:

| `--clip-grid` | saída int4 g64 | ganho | saída int8 g64 | ganho |
|---|---|---|---|---|
| 1 (absmax puro) | 10,89% | — | 0,60% | — |
| 5 | 9,76% | −10,4% | 0,60% | 0% |
| **13 (default)** | **9,63%** | **−11,6%** | 0,60% | 0% |
| 25 | 9,58% | −12,1% | 0,60% | 0,8% |

De graça: **o formato não muda** (continua uma escala f32 por grupo) e o kernel
não sabe a diferença. Em int8 o ganho é zero — com 255 níveis o `absmax` já é
praticamente ótimo, e a busca fica lá só para não abrir um caso especial.

> **O container em disco ainda não tem isto.** Ele foi escrito antes da mudança,
> com escala `absmax/qmax` pura, e o erro dele é o da primeira tabela: **10,81%**.
> Seu `meta` nem tem a chave `clip_grid`. Os **~9,6%** que a grade de 13 pontos
> deve dar no modelo inteiro são **extrapolação** de 24 tensores de um shard, não
> medição — e só valem depois de reconverter. O custo dessa reconversão também
> não foi medido: a busca faz 13 passadas de quantização por bloco, então esperar
> um múltiplo dos 11,3 min da conversão com grade de 1 ponto.

Efeito colateral: com escala menor que `absmax/7` o código `-8` passa a ser
alcançável. Antes, `rint` nunca o produzia e o int4 usava **15 dos 16 níveis**
que `dequant_grouped` já sabia ler. Conferido desempacotando um container de
teste de 1 shard e batendo o round-trip contra a fonte: 16/16 níveis ocupados,
contra 15/16 antes.

### O que a tabela acima *não* mede

A sonda usa `x ~ N(0,1)` iid. Ativações reais de LLM não são isso: alguns canais
carregam magnitude 10–100× a mediana, e o erro de saída real é dominado pelas
colunas de `W` alinhadas a esses canais. **Os números acima são otimistas.** É a
premissa inteira do AWQ.

Fechar esse furo é o que ainda vale mais que qualquer ajuste de escala:

| caminho | ganho esperado | custo |
|---|---|---|
| **rotação Hadamard** antes de quantizar | ataca o outlier na raiz | kernel precisa aplicar a rotação nas ativações |
| **GPTQ** (compensação via Hessiana) | 2–4× no erro de saída | dados de calibração + horas de conversão; formato e kernel **idênticos** |
| int8 no tensor sensível | ~18× | 2× os bytes daquele tensor |

Se ~9,6% é aceitável, esta seção não responde — só perplexidade ou o oráculo
token-exato respondem, e é isso que o critério de saída do M0 exige.

### Perplexidade medida — o critério de saída do M0

`tools/perplexity.py`, conjunto fixo (os próprios `docs/*.md`), **10 janelas de
384 tokens, 3.830 previsões**, comparação **pareada** janela a janela. O lado
int4 é medido com o `tootega.exe` — o artefato que roda —, e o BF16 é lido dos
safetensors originais por um caminho independente.

```
delta de nll (int4 - BF16):  +0,0513 +- 0,0076
intervalo 95%:               [+0,0365, +0,0661]
perplexidade:                1,0526x  (+5,26%)
por janela: +0,028 +0,027 +0,066 +0,078 +0,017 +0,047 +0,051 +0,093 +0,045 +0,061
```

**int4 g64 custa +5,3% de perplexidade.** As dez janelas deram positivo e o
intervalo não cruza zero: é medida, não impressão.

E é cerca de **metade** do que o erro de saída por matmul sugeriria — o que
confirma a ressalva acima de que erro por tensor não prevê qualidade.

> **Como quase escrevi o número errado.** As duas primeiras tentativas foram
> não pareadas: 191 previsões deram **−4,2%** — int4 "melhor" que BF16, o que é
> impossível — e 381 deram **+2,5%**. O sinal inverteu ao dobrar a amostra.
>
> A causa: a perplexidade varia de **223 a 473 entre janelas** deste corpus, e
> essa variância engole um efeito de 5%. A comparação pareada cancela, porque
> as duas configurações veem exatamente o mesmo texto e o que se mede é a
> diferença por janela — cujo desvio é vinte vezes menor.
>
> Comparar duas médias globais teria produzido um número publicável e errado.

### Ablação — e a tabela de precisão estava errada

`tools/ablate.exe` (C, reusando `src/`), mesmas 10 janelas. Cada grupo é
restaurado ao BF16 original **um de cada vez**, sobre a linha de base int4:

| grupo restaurado | tensores | delta de nll | da degradação |
|---|---|---|---|
| **atenção** (`self_attn.*`) | 193 | **−0,0661** | **100,3%** |
| shared experts | 141 | −0,0132 | 20,0% |
| camada densa 0 | 3 | −0,0019 | 2,8% |
| `lm_head` | 1 | +0,0002 | −0,3% |
| experts roteados | 9.024 | +0,0221 | −33,6% |

**Toda a degradação vem da atenção.** Restaurar 193 tensores — 2% da contagem,
~520 MB — recupera 100,3% do que a quantização custou: nll 5,8309 contra 5,8311
do BF16 completo. Os 9.024 tensores de expert, que são 16,45 GB e 93% do
container, **não são fonte de perda**.

Isso inverte a hipótese que esta seção assumia. O motivo é estrutural, não
acidental: cada token ativa 4 experts de 64, então o erro de um expert entra
diluído e descorrelacionado; a atenção roda **inteira, em todas as 47 camadas,
para todo token**, e o erro dela se compõe ao longo da profundidade.

**Consequência para o conversor: atenção em int8 g64, experts continuam int4.**
Custo estimado: os ~520 MB de atenção viram ~1,04 GB, o container passa de
19,33 para ~19,85 GB — **+2,7% de tamanho por ~100% da qualidade de volta**.

Duas ressalvas honestas, porque os números pedem:

- O intervalo de 95% do custo total é `[+0,0366, +0,0952]` — largo, e o
  `+0,0221` dos experts fica dentro dele. **A ferramenta ainda não reporta o
  intervalo pareado por grupo**, então "experts em int4 melhoram o modelo" não
  está demonstrado; o que está demonstrado é que eles **não degradam**.
- A degradação por janela vai de `−0,000` a `+0,152`. O efeito é real (9 de 10
  janelas positivas) mas heterogêneo, e um corpus maior apertaria o intervalo.

### A previsão foi testada — e cumprida

Container reconvertido com `tools/convert.exe` (atenção em int8, resto igual),
mesmas 10 janelas, mesma ferramenta:

| container | atenção | delta de nll vs BF16 | perplexidade | tamanho |
|---|---|---|---|---|
| v1 | int4 g64 | **+0,0659 ± 0,0149** | +6,81% | 19,33 GB |
| **v2** | **int8 g64** | **−0,0019 ± 0,0078** | **−0,19%** | 19,65 GB |

O intervalo de 95% do v2 é `[−0,0171, +0,0134]` — **cruza zero**. Não é que a
quantização ficou boa: é que **não dá mais para distingui-la do BF16 com esta
medida**. Por 1,7% de tamanho.

Contagens finais do v2: int4 8.973 (15,91 GB), int8 438 (2,44 GB), f32 290
(25,7 MB), bf16 2 (1,27 GB). Residente 18,38 GB.

> **O que quase mudei depois de medir.** O v2 mandou também o
> `kv_a_proj_with_mqa` de f32 para int8 — 47 tensores que a ablação nunca
> avaliou, porque lá já eram exatos. Cheguei a corrigir isso no conversor.
>
> Mas o container **que foi medido** é o que tem esse tensor em int8, e ele
> mediu perda indistinguível de zero. Reverter produziria uma configuração
> *não testada*, 167 MB maior, para comprar uma garantia que a medição já dá.
>
> Fica a regra: **publica-se o que foi medido**, não o que parece mais seguro.

## Uso agêntico

`chat_template.jinja` com suporte a ferramentas (`tools`, `tool_call`,
`tool_calls`), 202.752 tokens de contexto, SWE-bench Verified 59,2 e τ²-Bench
79,5 na tabela do próprio repo — que compara contra Qwen3-30B-A3B-Thinking (22,0)
e GPT-OSS-20B (34,0) no mesmo teste.

Para referência de escala: são 19,3 GB de pesos sustentando 59,2 em SWE-bench.
É a razão qualidade/byte que torna o projeto viável.
