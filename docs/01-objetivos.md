# 01 — Objetivos e escopo

## O que é

Engine de inferência para modelos **Mixture-of-Experts que cabem na memória
rápida da máquina**, com um CLI agêntico em cima.

Duas metas mensuráveis:

1. **≥ 15 tok/s** single-stream com GLM-4.7-Flash em int4, contexto de 4k, só na
   CPU — e **≥ 25 tok/s** com a partição GPU/CPU, se ela se pagar.
   Baseline a bater: 0,93 tok/s do engine anterior com GLM-5.2.
   Roofline: 27 tok/s CPU, 44 split ([08](08-teto-de-desempenho.md)).
2. **Throughput agregado escalando quase linear** com sessões concorrentes até
   saturar a banda de memória. Baseline a bater: `N^0,27` medido sob streaming.

As metas são ~55% do roofline, não 80%. É deliberado: primeira versão de um
engine não atinge a eficiência de banda de um maduro, e prometer 80% no papel é
como o número de 1,12 GB ativos que tivemos de corrigir.

A segunda meta é a que justifica o engine para uso agêntico: um CLI abre várias
sessões, e com pesos residentes o batch **amortiza** a leitura de peso entre
sequências em vez de multiplicá-la.

## O que explicitamente não é

**Não é um engine de streaming de disco.** Todo o maquinário que o streaming
construiu para esconder disco — PILOT, prefetch cross-layer, roteamento
dual-SSD, ajuste de `O_DIRECT`, pin set aprendido, tier LRU por camada — é a
maior parte do orçamento de complexidade dele e **não serve para nada** quando o
modelo é residente. Fica de fora. Se o modelo não cabe, este não é o engine certo.

**Não é multi-GPU.** Uma GPU, um host. Sharding entre placas é problema de outra
escala.

**Não é um servidor de produção.** Sem multi-tenancy, sem autenticação além de
loopback, sem cotas. É engine + CLI local.

**Não é model-agnostic no dia 1.** Ver a seção seguinte.

## Princípios

**Colocação decidida por banda medida, não por capacidade.** O
[modelo de hardware](02-hardware.md) mostra que o PCIe desta máquina é mais lento
que o disco. Um planner que assuma "VRAM é rápida, encha-a" produz a configuração
que medimos em 0,054 tok/s. O planner resolve um custo por token contra as bandas
**medidas na inicialização**, incluindo os links de transferência.

**Validação token-exata, inegociável.** Todo forward é comparado contra um
oráculo `transformers` com teacher-forcing. É o que separa engine de gerador de
plausibilidade, e o engine anterior acertou nisso — mantém-se.

**Arquitetura como configuração, não como pressuposto compilado.** O engine anterior
crava `n_group == 1` e MLA obrigatório no parser e sai
com erro fatal fora dessa janela. Foi isso que reduziu o universo de modelos
compatíveis à família GLM-5.x e transformou "testar outro modelo" numa
investigação que terminou sem opção. Aqui: config → IR, com atenção (MHA / GQA /
MLA) e roteamento (com ou sem limite de grupo) como variantes declaradas.

Concretamente, o dia 1 implementa **só o caminho MLA + `n_group=1`** — o do
modelo alvo. A diferença é que os pontos de variação existem como despacho
explícito desde o começo, então adicionar GQA é escrever uma variante, não
reescrever o parser.

**Cabe num orçamento declarado, ou recusa.** O engine roda numa estação que a
pessoa está usando. RAM, VRAM e threads são tetos duros passados por parâmetro;
se o plano não cabe, o engine sai com erro nomeando o que reduzir — nunca degrada
em silêncio. Medimos o custo de acomodar em vez de recusar: 13×. Detalhe em
[09-limites-de-recursos.md](09-limites-de-recursos.md).

**Medir antes de otimizar, e publicar o número.** Toda otimização entra com o
antes/depois na mesma máquina. O engine anterior faz isso e é por isso que o código dele
é audível; a diferença é que aqui os números vão para
[07-baseline.md](07-baseline.md) com a config completa, porque descobrimos que
comparar duas medições com `RAM_GB` diferente não significa nada.

## Não-metas de desempenho

- **Latência de primeiro token não é meta no M1–M3.** Prefill é otimizável
  depois; decode é o que define a experiência agêntica.
- **Quantização própria não é meta.** int4 com escalas por grupo é o formato
  alvo. Pesquisar esquemas melhores é outro projeto.
- **Treino, fine-tuning, LoRA: fora.** Só inferência.

## Critérios de fracasso

O projeto está errado se:

- o CPU-only não passar de **10 tok/s** a 4k no M1 — abaixo de ~37% do roofline
  significa que a conta de banda está errada e a premissa inteira cai;
- o split GPU/CPU do M2 render menos de 20% sobre CPU-only — a complexidade de
  dois backends não se paga e o M2 é cortado;
- a saída divergir do oráculo em qualquer configuração de colocação — é bug de
  correção, não de desempenho, e bloqueia tudo.
