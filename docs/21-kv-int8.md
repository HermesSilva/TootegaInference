# 21 — KV em int8: medido, e não adotado

A pergunta: `--kv-prec int8` cabe o contexto máximo do modelo — **202.752
posições em 5,83 GB** contra 22,0 GB do f32 — e está implementado desde o M1.
Por que não é o default?

A resposta curta era "reprovou no 32/32". Mas esse portão mede **diferença**,
não degradação: falhar nele quer dizer "a saída mudou", não "a saída piorou".
Ninguém tinha medido a segunda coisa. Este documento mede.

## O que custa cada precisão

Por posição, e no teto do modelo (202.752):

| | por posição | KV total | com os 18,54 GB de pesos |
|---|---|---|---|
| f32 (default) | 105,8 KB | 21,96 GB | 40,5 GB |
| bf16 | 52,9 KB | 10,98 GB | 29,5 GB |
| **int8 g64** | **28,1 KB** | **5,83 GB** | **24,4 GB** |

O int8 carrega uma escala f32 por grupo de 64 — os 0,5 KB a mais por posição.

## Critério, fixado antes de medir

> perplexidade abaixo de 1% · `match@ref` sem piora relevante · `rep4` e taxa
> de parada iguais

## Triagem — 5 janelas de 384, `corpus_prosa.txt`

| configuração | f32 | int8 | delta |
|---|---|---|---|
| int4 g64 (o que roda) | ppl 6,481 | ppl 6,503 | **+0,34%** |
| BF16 (referência) | ppl 6,857 | ppl 6,825 | **−0,47%** |

Duas medidas independentes, **sinais opostos, mesma magnitude**. Perturbação
sem direção. Passou.

## O teste que decide — 4 janelas de 4096, `corpus.txt`

| configuração | f32 | int8 | delta |
|---|---|---|---|
| int4 g64 | nll 4,8042 · ppl 122,0 | nll 4,8286 · ppl 125,0 | **+2,47%** |
| BF16 | nll 4,9641 · ppl 143,2 | nll 4,9546 · ppl 141,8 | **−0,94%** |

**+2,47% estoura o critério.** E a análise pareada, que é a regra desde
[04](04-modelo-alvo.md), mostra por quê:

```
int4 g64            BF16 (referência)
janela 1  -0,0006   janela 1  +0,0024
janela 2  +0,0829   janela 2  -0,0511     <- domina as duas médias
janela 3  +0,0080   janela 3  +0,0084
janela 4  +0,0074   janela 4  +0,0023

média +0,0244        média -0,0095
IC95 [-0,014, +0,063]  IC95 [-0,037, +0,018]
```

**Os dois intervalos cruzam zero.** A janela 2 sozinha vira as duas médias, em
direções opostas — e ela tem `nll 11,5`, perplexidade de ~100.000.

Fui ver o que é: `corpus.txt` são os próprios documentos concatenados, e a
janela 2 caiu numa região de **tabelas markdown com números**. O modelo não tem
como prever células de tabela; aquela janela não mede modelagem de linguagem,
mede o comportamento diante de texto imprevisível.

Erro de desenho meu: [17](17-diagnostico.md) registra que o corpus de
calibração é o `corpus_prosa.txt`. Usei o outro só por ser longo o bastante
para 4 janelas de 4096, e paguei com uma janela degenerada.

Nas três janelas de prosa, int8 custa **+0,49% de perplexidade**, na mesma
direção nas três. É um custo pequeno, consistente e plausível.

## Geração — 12 prompts, 96 tokens, guloso

| | match@ref | rep4 | parou | tok/s |
|---|---|---|---|---|
| f32 | 32,4 | 5,3% | 0/12 | 9,13 |
| int8 | **37,3** | 5,4% | 0/12 | 8,79 |

Não piorou. O `match@ref` até subiu — o int8 acompanhou a referência BF16 por
**mais** tokens antes de divergir. Com 12 prompts isso não é evidência de que
seja melhor; é evidência de que não é pior.

## Veredito: não adotado

O critério foi declarado antes e a estatística principal não o cumpriu.
Ajustar o critério depois de ver o número seria justamente o que
[17](17-diagnostico.md) existe para impedir.

E o resultado honesto não é "int8 degrada" — é **inconclusivo**: os dois
intervalos de confiança cruzam zero, os dois sinais discordam, e a geração não
mudou. Inconclusivo não é aprovado.

`--kv-prec int8` continua disponível em `serve`, `ablate` e `gentest`. Quem
precisar de 200k de contexto agora tem como, com este documento ao lado
dizendo o que se sabe e o que não se sabe.

## O que tornaria a decisão possível

1. **Corpus de prosa longo.** O `corpus_prosa.txt` tem 1.995 tokens — não dá
   nem uma janela de 4096. Falta um corpus de prosa com ~32k tokens, e com ele
   a medição de 4 janelas de 4096 vale.
2. **Mais janelas.** Com 4, um outlier vira a média. Oito ou dez dariam um
   intervalo que não cruza zero.
3. **Geração em contexto longo.** O `gentest` tem `ctx` fixo em 1024 e prompts
   curtos: ele não testa a atenção sobre milhares de posições, que é justamente
   onde a quantização do KV teria mais chance de aparecer. Não existe
   ferramenta para isso hoje.

Custo estimado: uma noite de máquina, como esta.
