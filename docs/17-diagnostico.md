# 17 — Diagnóstico: a caça ao erro de qualidade

Registro do que foi tentado, do que cada tentativa mediu, e do que ficou de
pé. Existe porque a sequência de conclusões erradas aqui tem valor maior que
o resultado final: cada uma falhou por um motivo diferente e reproduzível.

## O sintoma

Geração livre produzia texto degenerado — `*   *   *` em laço com amostragem
gulosa, salada de palavras com temperatura — enquanto **todos** os testes
internos passavam.

## Caminho 1 — culpar a quantização

**Hipótese:** o container `v2` (atenção em int8) quebrou o modelo, porque o
`v1` respondeu certo e o `v2` não.

**Medição:** um prompt, comparado a olho.

**Resultado: falso.** O `gentest` sobre 12 prompts mostrou que os pesos **BF16
originais degeneram igual** — 52,7% de 4-gramas repetidos, 1 de 12 prompts
parando no EOS.

**Erro de método:** conclusão a partir de N=1, depois de eu mesmo ter escrito
no parágrafo anterior que um prompt não é amostra.

## Caminho 2 — culpar o decode guloso

**Hipótese:** decode guloso trava em laço; é patologia conhecida.

**Medição:** o mesmo prompt com `--temp 0.8`.

**Resultado: falso.** Com temperatura a saída virou salada de palavras —
"Recadesso o a seguint", "corcovamento" — em vez de laço. Português inventado
não é patologia de decode; é assinatura de **logits errados**.

## Caminho 3 — RoPE interleave

**Hipótese:** alguma operação da atenção está errada.

**Medição:** leitura do `configuration_glm4_moe_lite.py`.

**Resultado: VERDADEIRO, e foi um bug real.**

`rope_interleave` é `True` por default e o `config.json` não sobrescreve. A
atenção usa `apply_rotary_pos_emb_interleave`; eu havia implementado o ramo
`else`:

| variante | pareia |
|---|---|
| `rotate_half` — implementada | `x[i]` com `x[i + rot/2]` |
| **interleave** — correta | `x[2i]` com `x[2i+1]` |

Na posição 0 as duas coincidem (`cos = 1`, `sin = 0`), e o erro cresce com a
posição. Consertado em `src/forward.c` e em `ptools/oracle.py`.

**Efeito:** a geração passou de salada para resposta correta com raciocínio
completo e parada no EOS.

## Caminho 4 — binário obsoleto

**Hipótese:** o conserto do RoPE não foi suficiente, porque a perplexidade
seguia em 380, depois 281, depois 273.

**Resultado: falso, e a causa era trivial.**

`Tests/*.exe` linkam `src/` estaticamente. Depois de consertar
`src/forward.c` eu rodei `.\build.ps1` — que compila **só o engine**. As
ferramentas de medição continuaram com o forward quebrado dentro.

Todas as perplexidades pós-conserto vieram de binário obsoleto. Recompilando:

| | antes (binário velho) | depois |
|---|---|---|
| int4 | ppl 273,5 | **ppl 5,66** |
| BF16 | ppl 281,4 | **ppl 8,63** |
| transformers | — | 8,89 |

O bug estava consertado havia horas. O `build-all.ps1` existe exatamente para
isso, foi escrito no mesmo dia, e não foi usado.

> Ferramenta que embute o código sob teste tem de ser reconstruída junto com
> ele. Um único ponto de entrada de build não é conveniência: é o que impede
> medir a versão errada sem perceber.

## Caminho 5 — "ainda há bug, a perplexidade continua em 340"

**Hipótese:** o conserto do RoPE não foi suficiente.

**Medição:** `ablate` sobre `docs/*.md`, depois sobre prosa comum. Deu 380 e
281.

**Resultado: falso — mas levou ao teste que resolveu.** A suspeita motivou
construir `ptools/hf_ref.py`, a **primeira referência externa do projeto**.

## O que a referência externa estabeleceu

Rodar o `transformers` com offload em disco custou **2 minutos**. Os 62 GB não
cabem em RAM, e eu havia tratado isso como "não dá para rodar" desde o início.

| medição | valor |
|---|---|
| transformers, 384 posições de prosa | **ppl 8,89** |
| engine, **as mesmas 384 posições** | **ppl 5,66** |
| `Tests/ablate.exe`, "a mesma" janela | **ppl 273–281** |

O engine e o `transformers` concordam. **O 281 vem da ferramenta de medição.**

Verificações que fecham o cerco:

- **tokenizer:** 1.995 ids idênticos aos da HF no corpus inteiro, round-trip
  exato. Não é ele.
- **bisseção por camada:** embed com erro 0,000%; camadas 0–45 entre 4% e 15%,
  subindo e descendo, sem salto. É ruído de quantização, não bug.
- **logits contra o transformers:** 4,05% de erro, 7 de 8 tokens iguais.

## O que fica de pé

**O engine está correto e validado contra referência externa.**

| medição | engine | transformers |
|---|---|---|
| perplexidade, 384 posições de prosa, BF16 | **8,63** | **8,89** |
| perplexidade, mesmo texto, container int4 | **5,66** | — |

Os 3% entre 8,63 e 8,89 são o que resta: tensores de uma dimensão que a
substituição não troca (já exatos) e diferenças de ordem de acumulação.

Tokenizer, container, kernels, determinismo e velocidade seguem válidos e
foram validados por caminhos independentes.

**O que ainda não fecha:** o int4 dando perplexidade 34% MENOR que o BF16 na
mesma janela. Quantização não melhora modelo. É uma janela só; precisa de
mais amostra antes de virar suspeita ou explicação.

**O que precisa ser refeito:** a ablação por grupo e as tabelas de precisão de
[04](04-modelo-alvo.md). Foram medidas com RoPE errado, com binário obsoleto,
ou ambos.

## Os erros de método

1. **Conclusão de N=1** (caminho 1).
2. **Explicar o número em vez de persegui-lo** — perplexidade 340 virou "é o
   corpus", 281 virou "é o corpus de novo".
3. **Validar interpretação contra interpretação** — o oráculo e o engine são
   duas escritas da mesma leitura minha do modelo. Concordaram 32/32 inclusive
   com o RoPE errado.
4. **Medir com binário obsoleto** (caminho 4).

O terceiro é a raiz dos dois primeiros: na falta de referência externa,
qualquer história que feche parece evidência. O quarto é de outra natureza —
disciplina de build — e custou mais horas que os outros três somados.

## A regra que faltava

> **Antes de escrever a primeira linha de um forward, rodar o modelo de
> referência uma vez e guardar os logits.**

Não como validação final — como **ponto de partida**. Custa minutos e teria
evitado todo este documento.

E a segunda, que decorre dela:

> **Métrica com valor absoluto conhecido tem faixa declarada antes de medir.**
> Perplexidade fora de 5–40 é falha, não observação.
