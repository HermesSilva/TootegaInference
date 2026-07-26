# TootegaEngine — para leigos

> **Documento irmão:** [`TootegaEngine-ForDummies.en.md`](TootegaEngine-ForDummies.en.md) —
> mesmo conteúdo, em inglês. As duas versões são mantidas em paralelo; se uma
> mudar, a outra muda junto.

Este é o documento de entrada do projeto. Ele existe porque o resto da
documentação (`docs/01` a `docs/16`) é escrito para quem já sabe o vocabulário —
e boa parte desse vocabulário ou é jargão da área, ou é jargão **deste projeto**,
ou é uma palavra comum que aqui significa outra coisa.

A ordem é deliberada: **primeiro as palavras, depois o sistema.**

---

## Sumário

| # | seção | o que você leva |
|---|---|---|
| 0 | [Como ler este documento](#0--como-ler-este-documento) | convenções, público, o que assumo |
| 1 | [O projeto em uma página](#1--o-projeto-em-uma-página) | a tese, a máquina, o estado atual |
| 2 | [Cinco ideias que explicam quase tudo](#2--cinco-ideias-que-explicam-quase-tudo) | o mínimo para entender qualquer decisão |
| 3 | [Glossário A — palavras que aqui significam outra coisa](#3--glossário-a--palavras-que-aqui-significam-outra-coisa) | os outros sentidos |
| 4 | [Glossário B — neologismos do projeto](#4--glossário-b--neologismos-e-apelidos-do-projeto) | o dialeto interno |
| 5 | [Glossário C — vocabulário da área](#5--glossário-c--vocabulário-da-área-uma-linha-cada) | referência rápida |
| 6 | [Do texto ao texto: o passeio completo](#6--do-texto-ao-texto-o-passeio-completo) | como um token nasce |
| 7 | [Por que é lento: a física do decode](#7--por-que-é-lento-a-física-do-decode) | prefill, decode, roofline |
| 8 | [O hardware desta máquina e a inversão](#8--o-hardware-desta-máquina-e-a-inversão) | por que a GPU não salva |
| 9 | [Residência: a decisão que vale 20×](#9--residência-a-decisão-que-vale-20) | a tese, com números |
| 10 | [Como o modelo é guardado](#10--como-o-modelo-é-guardado) | int4, ablação, container |
| 11 | [Memória: uma cópia, nenhuma a mais](#11--memória-uma-cópia-nenhuma-a-mais) | mmap, arena, os três caches |
| 12 | [Como se sabe que está certo](#12--como-se-sabe-que-está-certo) | oráculo, 32/32, medição pareada |
| 13 | [O servidor, o protocolo e o agente](#13--o-servidor-o-protocolo-e-o-agente) | como o token vira produto |
| 14 | [Como se mede a qualidade](#14--como-se-mede-a-qualidade) | perplexidade, entropia, o que engana |
| 15 | [Onde estamos e o que vem](#15--onde-estamos-e-o-que-vem) | roadmap com critério de corte |
| 16 | [Mãos à obra](#16--mãos-à-obra) | build e comandos, com leitura da saída |
| 17 | [Mapa do repositório](#17--mapa-do-repositório) | onde cada coisa mora |
| 18 | [Mal-entendidos comuns](#18--mal-entendidos-comuns) | FAQ |
| 19 | [Índice PT ↔ EN](#19--índice-de-termos-pt--en) | para ler as duas versões em paralelo |

---

## 0 — Como ler este documento

### Para quem é

Para quem programa, ou pelo menos lê código, e **nunca abriu um engine de
inferência por dentro**. Não é preciso saber nada de aprendizado de máquina.

### O que eu assumo

- Você sabe o que é um arquivo, um ponteiro, um byte e um vetor.
- Você sabe que multiplicar matrizes é caro.
- Você **não** sabe (e não precisa saber) o que é um transformer, o que é
  atenção, o que é quantização, nem o que é um "expert".

### O que eu não faço

Não ensino a treinar modelos. Este projeto **não treina nada** — ele só executa
um modelo já treinado, que alguém publicou.

### Convenções

**Números.** Este documento usa a vírgula decimal (`0,93`) e o ponto de milhar
(`154.880`), como o resto da documentação em português. A versão em inglês usa a
convenção inversa. Os valores são os mesmos.

**Duas marcas que aparecem o tempo todo:**

| marca | significa |
|---|---|
| **[medido]** | o número saiu de uma execução nesta máquina, com a configuração registrada |
| **[estimativa]** | o número saiu de uma conta, e ainda não foi confirmado por execução |

Essa distinção é levada a sério a ponto de estar escrita nos documentos
originais. Um número estimado que vira "fato" por repetição é o defeito que este
projeto mais tenta evitar — e já custou uma correção pública de 60% em uma conta
central (ver [§12](#12--como-se-sabe-que-está-certo)).

**Referências.** Sempre que uma afirmação vier de um documento normativo, ela
aparece assim: → [`docs/08`](../docs/08-teto-de-desempenho.md). Este documento é
uma **porta de entrada**, não a fonte da verdade; onde houver divergência, o
documento numerado ganha.

**"Esta máquina"** significa sempre o computador de referência descrito em
[§1](#a-máquina-de-referência). Nenhum número deste projeto é genérico.

---

## 1 — O projeto em uma página

### O que é

O TootegaEngine é um **programa que faz um modelo de linguagem responder**,
escrito do zero em C, sem bibliotecas, para um caso específico: modelos do tipo
**MoE** que **cabem inteiros na memória RAM da máquina**.

Três palavras já precisam de definição, e elas vêm nas seções seguintes: *modelo*,
*MoE*, *cabe inteiro*.

### A tese, em um parágrafo

Existe uma classe de engine que resolve o problema oposto: **engines de
streaming**, que rodam um modelo grande demais para a RAM lendo pedaços do disco
conforme precisam. Funciona — e entrega **0,93 token por segundo** [medido]
nesta máquina, com um modelo de 744 bilhões de parâmetros. É lento demais para
uso interativo, e é lento pelo motivo certo: o disco é ~370× mais lento que a
RAM.

A observação que originou o TootegaEngine é que **o gargalo não é o hardware**:
é a razão entre o tamanho do modelo e a memória rápida disponível. Comprar mais
RAM melhora pouco (mede-se `N^0,25`: quadruplicar a RAM dá ~1,4×). Trocar por um
modelo **que caiba inteiro na RAM** muda de patamar: o disco sai do caminho e o
teto passa a ser a banda da memória — **13,1 tokens por segundo** [medido],
contra 0,93. Catorze vezes, sem trocar uma peça da máquina.

O TootegaEngine é o engine desenhado para **esse regime**, e explicitamente
**não** para esconder disco.

### A máquina de referência

Todo número deste projeto foi medido aqui, salvo onde marcado [estimativa]:

```
CPU    Intel i9-14900K, 32 threads, AVX2+FMA
RAM    64 GB DDR5-4800 dual channel      68,7 GB/s medidos em runtime
GPU    RTX 2060 SUPER, 8 GB, sm_75       448 GB/s de VRAM
PCIe   gen 3 x16 (limite da GPU)         ~12 GB/s   <-- o elo mais lento
Disco  RAID0 NVMe                        18,4 GB/s sequencial
SO     Windows 10 Pro 19045
```

Guarde a linha do PCIe. Ela é a origem de metade das decisões do projeto
([§8](#8--o-hardware-desta-máquina-e-a-inversão)).

### O modelo alvo

**GLM-4.7-Flash**: 31,2 bilhões de parâmetros guardados, ~3,57 bilhões usados por
token, 19,65 GB no formato do projeto, 202.752 tokens de contexto máximo.
→ [`docs/04`](../docs/04-modelo-alvo.md)

### Estado atual

**M1 e M4 completos, em CPU.** O engine já:

- reproduz o resultado da implementação de referência — **32 de 32 tokens
  idênticos** [medido];
- produz saída **byte-idêntica** entre 1 e 24 threads [medido];
- tokeniza com **36 de 36 casos idênticos** aos da biblioteca de referência
  [medido];
- tem perplexidade **8,63** contra **8,89** da referência externa, no mesmo
  texto [medido];
- conversa em múltiplos turnos reaproveitando o cache — **99% do prefill** num
  turno típico [medido];
- serve o protocolo OpenAI, com ferramentas e streaming, para qualquer cliente
  ([§13](#13--o-servidor-o-protocolo-e-o-agente));
- traz um agente com seis ferramentas locais, sandbox e aprovação;
- recusa configurações que não cabem, dizendo o que reduzir.

Velocidade [medido]: **71 ms por token no decode quente — 13,1 tok/s**, contra
um teto teórico de 24,0 tok/s calculado da banda medida. São **55% do teto**;
os ~28 ms que faltam estão na atenção e nas barreiras, não nos matmuls, que já
rodam a 90% da banda.

O que **não** existe: vários pedidos ao mesmo tempo (M3), estado persistente,
especulação, build no Linux compilado, e GPU — esta última foi escrita, medida
e recusada ([§14.7](#147-a-medição-que-decidiu-não-usar-a-gpu)).

---

## 2 — Cinco ideias que explicam quase tudo

Se você só ler esta seção, já consegue acompanhar 90% das discussões do projeto.

### Ideia 1 — o modelo não escreve texto, ele chuta a próxima peça

O modelo recebe uma lista de números inteiros (**tokens**) e devolve, para cada
uma das 154.880 peças possíveis do vocabulário, uma nota de "quão provável é que
esta seja a próxima". Escolhe-se uma, ela vira o próximo número da lista, e tudo
recomeça.

**Consequência:** gerar 60 tokens significa rodar o modelo inteiro 60 vezes.
O custo é **por token**, sempre.

### Ideia 2 — "parâmetro guardado" e "parâmetro ativo" são coisas diferentes

O arquivo do modelo tem 31,2 bilhões de números (os **pesos**). Mas a arquitetura
MoE liga só uma parte deles a cada token: **3,57 bilhões**.

```
guardado no arquivo   31,2 B parâmetros  ->  19,65 GB em disco
lido por token         3,57 B parâmetros ->   2,01 GB de leitura
```

**Consequência:** o tamanho do arquivo não determina a velocidade. O que
determina é quanto se **lê por token**.

### Ideia 3 — o gargalo é a banda de memória, não a conta

Para gerar um token o processador precisa **ler 2,01 GB** de pesos e fazer
~7,1 GFLOP de aritmética. Nesta máquina:

```
ler   2,01 GB a 60 GB/s   = ~33 ms
contar 7,1 GFLOP           = ~4 a 7 ms
```

A leitura custa cinco vezes mais que a conta. Diz-se que o problema é
**memory-bound** (limitado por banda), e não **compute-bound** (limitado por
aritmética).

**Consequência prática, que surpreende:** trocar precisão por velocidade de
cálculo quase não ajuda, porque a CPU passa a maior parte do tempo esperando a
memória. Foi por isso que o projeto escolheu o kernel **mais preciso** e mais
"lento" em aritmética — sobra folga de compute. → [`docs/15`](../docs/15-kernels-e-numerica.md)

### Ideia 4 — residência: ou o modelo cabe, ou você está em outro jogo

Se o modelo cabe na RAM, cada token lê da RAM (60 GB/s). Se não cabe, cada token
lê pedaços do disco (18,4 GB/s no melhor caso, e na prática muito pior pela
latência de acesso aleatório).

Não é uma diferença de 3×. É uma diferença de **regime**, medida entre 11× e 34×.

### Ideia 5 — nada entra sem passar no oráculo

Toda mudança que possa alterar a saída é comparada contra uma implementação de
referência independente (em Python, com a biblioteca `transformers`), e só é
aceita se produzir **32 de 32 tokens idênticos**.

Três otimizações boas já foram **reprovadas** por esse critério e ficaram como
opção desligada. → [§12](#12--como-se-sabe-que-está-certo)

---

## 3 — Glossário A — palavras que aqui significam outra coisa

Esta é a seção que o resto da documentação assume que você já leu.

### 3.1 A tabela dos outros sentidos

| palavra | o que costuma significar | o que significa **aqui** |
|---|---|---|
| modelo | o "cérebro", algo abstrato | Um arquivo de números. Aqui é literalmente `model.tootega`, 19,65 GB de pesos. Não é um programa nem um serviço. |
| expert | um especialista em um assunto | Um par de matrizes escolhido por uma nota, por token e por camada. Não tem assunto: não existe "o expert de matemática". |
| cache | uma coisa só, memória rápida genérica | Aqui são três coisas diferentes com donos diferentes: cache de pesos, cache de KV e cache de prefixo. Misturá-las é a origem da maior parte da confusão de projeto. |
| banda | banda de internet, medida em Mbps | Quantos gigabytes por segundo a memória entrega. Nada a ver com internet, e **diferente de latência**: banda é quantos bytes por segundo, latência é quanto tempo até o primeiro byte. Quase toda conta deste projeto é de banda, porque no decode a leitura é grande, previsível e sequencial. Nesta máquina são **68,7 GB/s**, medidos em execução — não lidos da caixa. |
| residente | que está instalado na máquina | Que está inteiro na RAM e não sai de lá durante a execução. Não é o mesmo que "instalado" nem que "carregado". |
| quantização | discretizar qualquer sinal | Guardar cada peso em 4 ou 8 bits em vez de 16. Só os pesos: as contas continuam em 32 bits, e o kernel *desempacota* antes de multiplicar. Isso tem nome aqui — **W4A32**, pesos em 4 bits e ativações em 32. Custa perplexidade: medido, +5,26% com tudo em int4 e **−0,19%** (indistinguível de zero) deixando a atenção em int8. |
| LoRA | técnica de ajuste fino de modelos | Aqui (`q_lora_rank`, `kv_lora_rank`) é só projeção de posto baixo, parte da arquitetura original. Não tem relação com a técnica de ajuste fino de mesmo nome. |
| token | um símbolo, ou uma credencial de acesso | Um pedaço de texto, ~4 caracteres em média. O modelo só enxerga o número inteiro que representa esse pedaço. |
| contexto | o assunto de que se está falando | O número máximo de tokens que cabem na janela da conversa — tudo o que o modelo consegue "ter em vista" de uma vez. Não é memória de longo prazo: o que sai da janela deixa de existir para ele. E custa RAM, linearmente: **105,8 KB por posição** no padrão f32, *por conversa*. No teto deste modelo, 202.752 posições, são 21,96 GB — mais do que os 18,54 GB dos pesos. |
| grupo | agrupamento genérico | Dois homônimos independentes: `group=64` (quantos pesos compartilham uma escala) e `n_group` (limite de roteamento entre experts). |
| página | página de um documento ou da web | Bloco de 4 KB (ou 2 MB) com que o sistema operacional gerencia memória. Nada de páginas de documento. |
| plano | uma intenção ou um cronograma | A decisão de onde cada peso vai morar (RAM ou VRAM), calculada na inicialização a partir das bandas medidas. |
| oráculo | um adivinho, ou um banco de dados | A implementação de referência em Python contra a qual o engine é comparado token a token. Ela é lenta e não importa: existe para ser *externa* — escrita por outra gente, a partir de outra leitura do modelo. Foi uma referência externa que pegou um RoPE na variante errada depois de três testes internos verdes, todos cegos ao mesmo engano. |
| ablação | um procedimento médico | Desligar (ou restaurar) uma parte de cada vez para medir quanto ela contribui para o resultado. É emprestado da biologia experimental, e a lógica é a mesma: você não descobre para que serve uma peça olhando para ela, e sim tirando-a. Aqui é o que mostrou que a atenção em int8 não custa nada e que o resto do modelo em int4 custa 5%. |
| slot | um encaixe físico, um horário na agenda | Um contexto de conversa independente dentro do engine, com o seu próprio cache KV. É o que permitiria atender várias pessoas ao mesmo tempo sem misturar as conversas. **Ainda não existe**: hoje é um pedido por vez, e o segundo espera calado na fila do TCP. É o marco M3. |
| shard | fragmento genérico, partição de banco de dados | Um dos 48 arquivos em que o modelo original vem partido, por limite de tamanho de cada arquivo. Não têm significado próprio: um tensor pode começar num e terminar no seguinte. O conversor os lê todos e escreve um container único. |
| kernel | o núcleo do sistema operacional | A rotina que faz a conta pesada — `matmul_int4_g`, por exemplo. Não é o núcleo do sistema operacional. |
| draft | rascunho de um texto | Um palpite barato de token, gerado para ser confirmado ou descartado em bloco (especulação). A ideia: um modelo pequeno chuta vários tokens de uma vez e o grande confere todos numa passada só, pagando uma leitura de pesos em vez de N. Aqui é o marco M5, e ainda não existe. |
| temperatura | calor | Um número que controla o quanto a escolha do próximo token é aleatória. 0 significa sempre o mais provável; acima de 1 as opções improváveis ganham chance. Não é um botão de "criatividade": é o quanto de risco você aceita na próxima peça. Em medição fica sempre em 0, senão mede-se o sorteio junto. |
| engine | motor de carro, ou motor de jogo | O programa que executa o modelo: carrega os pesos, faz as contas, gerencia o cache. Não é o modelo, e não é a interface. |
| agente | uma pessoa que representa outra | Um programa que usa o modelo em ciclo: pergunta, recebe uma resposta, executa ferramentas, realimenta o resultado. |
| ferramenta | utensílio, ou programa de linha de comando | No sentido de agente: uma função que o modelo pode pedir para executar (ler arquivo, rodar comando). O esquema dela entra no prompt. |

### 3.2 As quatro que mais confundem, com exemplo

**"Expert" não é um especialista.** A intuição natural — "o expert de matemática",
"o expert de código" — está errada. Um expert é um par de matrizes. O roteador
dá uma nota para cada um dos 64 e escolhe os 4 maiores, **por token e por
camada**. O mesmo token pode ativar os experts 5, 18, 27 e 52 na camada 3 e os
experts 1, 9, 40 e 63 na camada 4. Não há tema, não há significado atribuível, e
ninguém programou aquilo — a divisão emergiu do treino.

**"Quantização" aqui é só sobre os pesos.** O engine guarda os pesos em 4 bits,
mas **desempacota para 32 bits antes de multiplicar**. As ativações (os números
que fluem entre camadas) nunca são quantizadas. Isso tem nome no projeto:
**W4A32** — *weights 4 bits, activations 32 bits*. A alternativa mais rápida em
aritmética (W4A8) foi rejeitada porque a aritmética não é o gargalo e ela
introduziria erro novo. → [`docs/15`](../docs/15-kernels-e-numerica.md)

**"LoRA" aqui não é ajuste fino.** `q_lora_rank: 768` no arquivo de configuração
significa: em vez de uma matriz 2048×5120, o modelo usa duas — 2048×768 e
768×5120. É uma compressão embutida na arquitetura, presente desde o treino.
A técnica popular de ajuste fino chamada LoRA usa a mesma ideia matemática e
nome parecido, mas **não está envolvida aqui de nenhuma forma**.

**"Banda" não é latência.** Banda é *quantos bytes por segundo*. Latência é
*quanto tempo até o primeiro byte*. Um disco NVMe tem banda excelente e latência
ruim; a RAM tem as duas boas. Quase todas as contas deste projeto são de banda,
porque no decode a leitura é grande, previsível e sequencial.

---

## 4 — Glossário B — neologismos e apelidos do projeto

Termos que você **não vai encontrar em outro lugar**. Eles foram cunhados aqui,
geralmente logo depois de um erro custar tempo.

| termo | significado |
|---|---|
| a inversão | A descoberta de que o PCIe desta máquina (12 GB/s) é mais lento que o disco (18,4 GB/s), invertendo a hierarquia que quase todo engine assume. |
| lei de colocação | "Coloque na GPU apenas o que couber lá permanentemente; nunca transmita pesos por um link mais lento que o tier de origem." |
| partição por camada | Dividir o modelo entre GPU e CPU por camadas, não por tensores. Os pesos ficam fixos onde a camada é calculada; entre camadas cruzam só 8 KB. |
| fração de residência | Que porcentagem do modelo está na memória rápida. É a variável que governa a velocidade, mais que qualquer especificação de hardware. |
| a armadilha desta máquina | A configuração `CUDA_DENSE=1` do engine anterior, que não recusou o impossível e degradou 13× em silêncio: 0,684 → 0,054 tok/s. |
| recusar em vez de degradar | Se o plano não cabe no orçamento declarado, o engine sai com erro dizendo o que reduzir. Nunca aceita e fica lento. |
| 32/32 | O critério de aceitação: 32 posições de teste, todas com o mesmo token da referência. Binário, e bloqueia merge. |
| medição pareada | Comparar duas configurações janela a janela sobre o mesmo texto, nunca por média global. A média global já produziu um número invertido. |
| determinismo por estado | Duas execuções no mesmo estado dão bytes idênticos; estados diferentes podem divergir legitimamente. Não se assume que "greedy basta". |
| absorção | Reescrever a atenção para operar no espaço comprimido, projetando a query uma vez em vez de expandir cada posição do cache. |
| desabsorvida | A formulação original da atenção, que expande o cache antes de usá-lo. É a que o oráculo usa. |
| metade/metade | A convenção de empacotamento dos nibbles: o byte `j` guarda os pesos `j` e `j+32`, não `2j` e `2j+1`. |
| arena de scratch | Um bloco de memória alocado uma vez na inicialização e fatiado para todos os buffers temporários. Zero `malloc` no laço de token. |
| estável primeiro, volátil por último | A regra de ordenação do prompt que faz o cache funcionar: system → ferramentas → contexto longo → histórico → turno atual. |
| critério de saída | O número que fecha um marco do roadmap. Sem número, o marco não fecha — e "está funcionando" não é número. É o que impede a conversa de virar discussão de opinião: ou o valor medido passou do limiar declarado *antes*, ou não passou. |
| critério de corte | O número abaixo do qual a funcionalidade é cancelada, não adiada. O M2 tem um: ganho abaixo de 20% mata a partição GPU/CPU. |
| file-backed | Memória que veio de um arquivo mapeado: o sistema operacional pode descartá-la e reler do disco. Por isso não conta no orçamento de RAM. |
| memória anônima | Memória que só existe na RAM, sem arquivo por trás. Se for despejada, o conteúdo se perde — por isso conta no orçamento. |
| publica-se o que foi medido | Quando a intuição diz uma coisa e a medição diz outra, publica-se a medição — mesmo que a alternativa pareça mais segura. |
| teto | O limite superior de velocidade obtido dividindo bytes por banda, ignorando compute e overhead. Também chamado de roofline. |
| engine de streaming | Engine que roda um modelo maior que a RAM lendo pedaços do disco conforme precisa. Funciona, e é lento pelo motivo certo: o disco é ~370× mais lento que a RAM. Medimos **0,93 tok/s** nesse regime, contra os 13,5 do TootegaEngine com o modelo inteiro na memória. |
| container .tootega | O formato de arquivo do projeto: cabeçalho de 128 bytes, blobs alinhados a 64 bytes, índice JSON no fim. É lido direto, sem conversão. |
| marco | Uma etapa do projeto com **critério de saída mensurável**, não com data. M0 converte e valida o modelo; M1 é o forward em CPU token-exato; M2 era a GPU; M3 é atender vários pedidos ao mesmo tempo; M4 é o agente; M5 é especulação. Um marco fecha quando o número aparece, não quando o código compila. |
| decisão numerada | Cada escolha de projeto que teve alternativas reais ganha um número — D1 a D14 — e uma linha no placar com o estado: aberta, fechada, ou fechada com o número que a fechou. Serve para que ninguém precise reabrir a discussão por não saber que ela já aconteceu. |
| licença MIT | Uma licença de software livre curta e permissiva: qualquer um pode usar, copiar, alterar e vender, desde que mantenha o aviso de copyright, e sem garantia nenhuma. É a licença deste engine. **Não cobre o modelo** — pesos e tokenizer têm licença própria, de quem os treinou. |

### O apelido que mais aparece: **engine de streaming**

Um `engine de streaming` resolve o problema **oposto** ao deste projeto: modelos
grandes demais para a RAM, lidos do disco conforme a execução precisa. A
expressão aparece o tempo todo aqui por dois motivos, e nenhum deles é elogio ou
crítica — é contraste.

O primeiro é que ele fornece a **linha de base**: 0,93 tok/s com 744 bilhões de
parâmetros nesta mesma máquina, que é o número contra o qual os 13,1 tok/s deste
engine significam alguma coisa. Sem uma referência medida, "rápido" é adjetivo.

O segundo é que ele mostra **o que não fazer quando o modelo cabe**. Todo o
maquinário de esconder disco — prefetch entre camadas, mapa de calor de quais
pesos manter, política de despejo, roteamento entre dois SSDs — é a maior parte
do orçamento de complexidade de um engine desses, e **não serve para nada** com
o modelo residente. Jogar isso fora é a principal vantagem estrutural do
TootegaEngine; não é a velocidade, é o tamanho.

**Regra dura do projeto:** nada de código ou de toolchain de terceiros entra
aqui. As comparações são históricas — referência de medição e lição aprendida,
nunca permissão de reuso. → `CLAUDE.md`

---

## 5 — Glossário C — vocabulário da área, uma linha cada

Referência rápida. Nada aqui é específico do projeto.

### 5.1 O modelo

| termo | uma linha |
|---|---|
| LLM | Large Language Model, modelo de linguagem grande: um programa que prevê o próximo token de um texto. Toda a aparência de conversa vem daí — prever a próxima peça, milhares de vezes, cada vez com o resultado anterior na entrada. Não há plano, não há intenção guardada entre uma previsão e a seguinte. |
| transformer | A arquitetura que praticamente todos os LLMs usam: uma pilha de camadas iguais, cada uma com atenção e um bloco de multiplicações. |
| camada | Um bloco da pilha do transformer. Aqui são 47, executados em sequência — a saída de uma é a entrada da seguinte. |
| peso | Um dos números aprendidos durante o treino. Aqui são 31,2 bilhões deles, e são eles que ocupam os 19,65 GB. |
| parâmetro | Sinônimo de peso quando se fala do tamanho do modelo. "31,2 bilhões de parâmetros" = 31,2 bilhões de números aprendidos. |
| parâmetro ativo | Quantos pesos são realmente lidos para gerar um token. Aqui são 3,57 bilhões de 31,2 — e é esse número que determina a velocidade. |
| tensor | Um array multidimensional de pesos, com nome. O container tem 9.703 deles, e o índice diz onde cada um começa. |
| hidden | O vetor que atravessa a pilha de camadas. Aqui tem 2048 números — 8 KB. Também chamado de estado residual. |
| embedding | A tabela que converte um id de token no vetor inicial de 2048 números. Aqui tem 154.880 linhas e lê-se uma por token. |
| logits | As 154.880 notas de saída, uma por peça do vocabulário, antes de virarem probabilidades. Nota alta = candidato provável. |
| softmax | A função que transforma notas em probabilidades que somam 1. Subtrai-se o máximo antes, para não estourar o expoente. |
| RMSNorm | Normalização que reescala o vetor para uma magnitude previsível antes de cada bloco. Sem ela, 47 camadas empilhadas explodem ou somem. |
| resíduo | Somar a entrada de volta na saída do bloco. Faz a camada corrigir o vetor em vez de substituí-lo, e é o que permite empilhar 47. |
| atenção | O mecanismo pelo qual o token atual olha para todos os anteriores e decide de quais aproveitar. É o que dá ao modelo a noção de contexto — e é o único ponto onde o custo cresce com o tamanho da conversa, porque o novo token precisa consultar tudo o que veio antes. Por isso existe o cache KV: sem ele, cada token recalcularia a conversa inteira. |
| cabeça | Uma das 20 vistas paralelas dentro da atenção. Cada uma pode se especializar em um tipo de relação entre tokens. |
| Q, K, V | Query, key e value: o que eu procuro, o que eu ofereço, o que eu entrego. A atenção compara Q com todos os K e mistura os V. |
| RoPE | Codificação de posição por rotação do vetor. Aqui é parcial: só 64 das 256 dimensões de cada cabeça giram. |
| MLP | O bloco de duas multiplicações que vem depois da atenção. Também chamado de FFN, feed-forward network. |
| MoE | Mixture-of-Experts: em vez de um MLP por camada, muitos — e só alguns rodam por token. É o que torna 31,2 B de pesos custarem 3,57 B por token. |
| roteador | A pequena matriz que dá uma nota a cada expert e decide quais rodam. No código e na configuração aparece como `gate`. |
| top-k | "Pegue os k maiores". No roteador é top-4 de 64; na amostragem é escolher entre os k tokens de maior nota. |
| expert compartilhado | Um expert que roda sempre, além dos escolhidos pelo roteador. Guarda o que é útil para qualquer token. |
| MLA | Multi-head Latent Attention: atenção que guarda o cache comprimido, 576 números por token por camada em vez de 10.240. |
| GQA / MHA | As variantes mais comuns de atenção. Funcionam, mas o cache delas é muito maior — por isso o modelo alvo usa MLA. |
| MTP | Multi-Token Prediction: uma cabeça extra que chuta o token seguinte quase de graça, para acelerar por especulação. |
| lm_head | A matriz final, 2048 × 154.880, que converte o vetor da última camada nas notas de saída. É lida inteira a cada token. |
| vocabulário | As 154.880 peças de texto que o modelo conhece. Só 154.856 existem de fato; o resto é preenchimento. |
| camada densa | Uma camada sem MoE, com um MLP comum. Aqui só a camada 0 é assim, com tamanho interno 10.240. |
| arquitetura | O desenho do modelo: quantas camadas, de que tipo, com que tamanhos. Vem do `config.json`, que o container guarda inteiro. |
| checkpoint | O conjunto de arquivos publicado com os pesos treinados. Aqui vem em 48 shards no formato `safetensors`, 62,4 GB em BF16, e é entrada do conversor — o engine não o lê em execução. A licença dele é de quem o treinou, e não a deste código. |
| safetensors | O formato em que os pesos são publicados: um cabeçalho JSON dizendo nome, forma e offset, e os números crus em seguida. |
| Hugging Face | O repositório público de onde o modelo e o tokenizer são baixados, e de onde vem a biblioteca usada como referência. |
| chamada de ferramenta | Quando o modelo, em vez de responder, pede que um programa faça algo por ele — ler um arquivo, rodar um comando. Ele não executa nada: escreve um pedido, alguém executa, e o resultado volta como uma mensagem nova. Neste modelo o pedido usa `<tool_call>`, que é um **token único** do vocabulário, não um texto a ser reconhecido. |
| diretiva | Uma instrução de como responder, dada antes da conversa começar: "responda no idioma da pergunta", "vá direto ao ponto". Vai numa mensagem de sistema no **início** do prompt — que é onde o cache de prefixo a reaproveita em todos os turnos. Mudá-la no meio invalidaria o cache inteiro. |
| degeneração | Quando o modelo perde o fio e passa a repetir, dobrar palavras ou não parar. Não é um erro de programa — é o texto se enrolando sobre si mesmo. Foi assim que se descobriu que um container estava ruim mesmo com a perplexidade dizendo que estava bom. |
| MHA | *Multi-Head Attention*: a atenção clássica, em que cada cabeça guarda a própria chave e o próprio valor. É a mais simples e a mais cara em memória — o oposto do MLA deste modelo, que comprime os dois num espaço menor e economiza 18× de tráfego. |
| GLM | A família de modelos abertos em que este projeto roda. O alvo é o **GLM-4.7-Flash**: 31 bilhões de parâmetros no total, mas só ~3 bilhões ativos por token, porque é um MoE. É essa razão que o faz caber e ser rápido ao mesmo tempo. |
| rascunho | O texto que o modelo escreve para si mesmo antes de responder, entre `<think>` e `</think>`. Medido neste modelo: com 400 tokens de orçamento ele **não termina 8 de 12 perguntas**, repete 8× mais e custa 4,5× o tempo — para "qual é a capital do Brasil" ele escreve seis etapas em inglês e é cortado antes de responder. |

### 5.2 A execução

| termo | uma linha |
|---|---|
| inferência | Executar um modelo já treinado para obter uma resposta. É tudo o que este projeto faz — não há treino em lugar nenhum. |
| treino | O processo, caríssimo, que produziu os pesos. Acontece uma vez, em outro lugar, e não se repete aqui. |
| fine-tuning | Ajustar um modelo já treinado para uma tarefa específica. Declarado fora de escopo neste projeto. |
| forward | Uma passagem completa pela pilha de camadas, da entrada até os logits. Também é o nome do subcomando que faz isso sobre ids fixos. |
| prefill | Processar o prompt inteiro de uma vez, antes de gerar. Todos os tokens passam juntos, então os pesos são lidos uma vez para muitos tokens. |
| decode | Gerar um token de cada vez, em cadeia. Cada token relê os 2,01 GB de peso inteiros — é aqui que a banda manda. |
| TTFT | Time to first token: quanto tempo até a primeira palavra aparecer na tela. Não é meta deste projeto até o marco M3. |
| tok/s | Tokens por segundo, a métrica principal do projeto. Sempre acompanhada do contexto em que foi medida, porque o número muda com ele: **13,1 tok/s** no decode quente aqui, contra um teto de 24,0 calculado da banda. Citar tok/s sem dizer o contexto, se estava quente e quantas threads é o jeito mais fácil de publicar um número que ninguém consegue reproduzir. |
| cache KV | O estado de atenção já calculado para os tokens anteriores. Guardá-lo evita refazer o prompt inteiro a cada palavra. |
| cache de prefixo | Reaproveitar o KV do começo da conversa entre turnos. Vale até o primeiro token diferente; dali em diante tudo é refeito. |
| teacher forcing | Rodar o modelo sobre um texto conhecido, forçando cada palavra correta, para comparar previsão a previsão com a referência. |
| greedy | Escolher sempre o token de maior nota, sem sorteio. É determinístico, e é o modo usado em toda medição — porque comparar duas configurações com amostragem aleatória mede o gerador de números aleatórios junto. Cuidado com a armadilha: greedy **não** garante saída igual entre máquinas; um empate decidido por um bit diferente separa os caminhos. |
| top-p | Sortear entre os tokens de maior nota que, somados, atingem a probabilidade p. Também chamado de nucleus sampling. |
| amostragem | A etapa que escolhe um token entre os 154.880 candidatos, a partir das notas. Greedy, temperatura, top-k e top-p são modos dela. |
| batching contínuo | Atender várias conversas ao mesmo tempo, aproveitando a mesma leitura de peso para todas. É onde mora a segunda meta do projeto. |
| especulação | Chutar vários tokens de uma vez com um mecanismo barato e verificar todos em bloco. Paga quando o modelo é residente. |
| prompt | O texto de entrada completo: instruções, ferramentas, histórico e a pergunta atual. É ele que o cache de prefixo tenta reaproveitar. |
| system prompt | A parte fixa do prompt, com as instruções permanentes. Vai primeiro justamente para nunca invalidar o cache. |
| chat template | As regras de como montar o texto de uma conversa a partir dos turnos, com os marcadores de quem fala. Vem junto com o modelo. |
| turno | Uma rodada da conversa: o que o usuário disse mais o que o assistente respondeu. |
| tokenizer | O componente que converte texto em ids e de volta. Erro nele não estoura: produz ids ligeiramente diferentes e resposta plausível e errada. |
| BPE | Byte Pair Encoding: junta repetidamente os dois pedaços vizinhos mais baratos, segundo uma tabela de 321.649 regras. |
| token especial | Um dos 36 marcadores como `<\|user\|>`, reconhecidos por texto exato antes do BPE. Delimitam quem fala. |
| futex | O mecanismo pelo qual uma thread dorme esperando um valor mudar, e é **acordada por quem mudou** — em vez de acordar sozinha de tempos em tempos para conferir. A diferença aparece em número: dormir por tempo fez um matmul de 1,6 MB custar 26 ms em vez de 0,026 ms. |
| barreira | O ponto em que todas as threads param e esperam a última chegar. É necessária e é cara: sincronizar 20 threads custa da mesma ordem de grandeza que o trabalho de um expert. Foi por isso que 15 barreiras por camada viraram 2 — fundindo os matmuls num despacho só. |
| processo | Um programa em execução, com a própria memória. O servidor é um; o agente é outro, um por aba da interface. Separar assim tem preço e tem razão: se o agente morre, o servidor e os 18,5 GB que ele carregou continuam de pé. |
| pipe | Um cano entre dois programas: o que um escreve, o outro lê. É por um pipe que a interface manda a pergunta ao agente. Tem uma armadilha que custou horas aqui: o programa pode ter *lido* os bytes para um buffer interno, e quem olha só o cano vê vazio — e conclui, errado, que nada chegou. |

### 5.3 A máquina

| termo | uma linha |
|---|---|
| CPU | O processador principal. Aqui, um i9-14900K com 8 núcleos rápidos, 16 econômicos e 32 threads no total. |
| GPU | A placa de vídeo, usada como acelerador de contas. Aqui é uma RTX 2060 SUPER de 8 GB — rápida por dentro, mal conectada por fora. |
| RAM | A memória principal do computador. Aqui, 64 GB entregando ~60 GB/s. É onde o modelo inteiro precisa caber. |
| VRAM | A memória da placa de vídeo. Rápida (448 GB/s aqui) e pequena (8 GB), e só alcançável pelo PCIe — que nesta máquina entrega ~12 GB/s, menos que o disco. É a combinação dos três números que decide tudo: com o modelo em 18,54 GB, nada cabe permanentemente, e o que não cabe tem de atravessar o elo mais lento a cada camada. |
| PCIe | O barramento que liga a GPU ao resto do computador. Aqui é gen 3: ~12 GB/s, o elo mais lento da máquina. |
| NVMe | O padrão dos SSDs rápidos ligados direto ao barramento. Aqui, dois em RAID0 entregando 18,4 GB/s sequenciais. |
| tier | Um nível da hierarquia de memória: VRAM, RAM, disco. Cada tier tem sua banda, sua capacidade e seu link de acesso. |
| memory-bound | Quando o tempo é dominado por esperar a memória, não por fazer contas. É o regime do decode: ~33 ms de leitura contra 4–7 ms de conta. |
| compute-bound | Quando o tempo é dominado pela aritmética. É o regime do prefill, e é por isso que otimizá-lo é outro problema. |
| núcleo | Uma unidade de execução independente dentro da CPU. Este processador é híbrido: 8 P-cores rápidos e 16 E-cores econômicos. |
| thread | Uma linha de execução. O engine usa um pool próprio de threads, com o trabalho dividido por linhas de saída da matriz. |
| afinidade | Fixar cada thread em núcleos escolhidos. O default é ficar só nos P-cores: espalhar trabalho memory-bound pelos E-cores costuma piorar. |
| SIMD | Uma instrução que opera sobre vários números de uma vez. É como o kernel processa 8 ou 16 pesos por ciclo em vez de um. |
| AVX2 / AVX-VNNI | Os conjuntos de instruções SIMD desta CPU. O engine escolhe qual usar em tempo de execução, para o binário rodar em outras máquinas. |
| FMA | Multiplicar e somar numa instrução só. Rápido, mas arredonda diferente — por isso o build proíbe fundir automaticamente. |
| cache L1/L2/L3 | Memórias minúsculas e rapidíssimas dentro do processador. O L3 desta CPU tem 36 MB, e cabe o cache de uma camada inteira a 4k. |
| mmap | Mapear um arquivo direto no espaço de endereços do processo. O arquivo passa a ser a memória, sem passo de cópia. |
| page cache | O cache que o sistema operacional mantém dos arquivos lidos. Como o modelo é mapeado, esse cache é o cache de pesos. |
| falta de página | O que acontece no primeiro acesso a uma página mapeada: o sistema para tudo e vai buscar do disco. Daí o pre-fault na carga. |
| páginas grandes | Usar páginas de 2 MB em vez de 4 KB. Com 19,3 GB de pesos, isso reduz de 4,7 milhões de páginas para ~9 mil. |
| TLB | A tabelinha do processador que traduz endereços. Tem ~2048 entradas: com páginas de 4 KB cobre 8 MB, com 2 MB cobre 4 GB. |
| RSS | Resident Set Size: quanta memória física o processo realmente ocupa. É o número que o watchdog compara com o orçamento. |
| swap | Quando o sistema operacional escreve memória no disco por falta de RAM. Aqui é considerado falha: o orçamento existe para nunca chegar lá. |
| OOM | Out of memory: quando o sistema mata o processo por falta de RAM. É o que a regra de recusar em vez de degradar previne. |
| orçamento | Um teto duro de RAM, VRAM ou threads passado por parâmetro. Verificado em execução, não só no plano. |
| gargalo | O recurso que limita tudo o mais. Aqui é a banda de memória; otimizar qualquer outra coisa não muda o tempo total. |
| overhead | Custo que não é o trabalho em si: chamar função, sincronizar threads, alocar memória. Invisível a 1 tok/s, percentual a 20. |
| lei de Amdahl | A conta que diz quanto acelerar uma parte acelera o todo. Usada aqui para estimar o ganho de mover 39% das camadas para a GPU. |
| CUDA | A plataforma da NVIDIA para rodar cálculo em placa de vídeo. Um programa CUDA tem duas metades: a que roda na CPU e a que roda na GPU, escritas em arquivos diferentes e compiladas por compiladores diferentes. Aqui ela foi escrita, medida e **não adotada** — vive na branch `GPU`. |
| SMT | *Simultaneous Multithreading* — a Intel chama de Hyper-Threading. Um núcleo físico se apresenta como dois lógicos, que dividem as mesmas unidades de execução. Para trabalho limitado por memória, como o nosso, os dois irmãos brigam pelo mesmo recurso e não ajudam: por isso o pool usa **um lógico por núcleo físico**. |
| P-core e E-core | Processadores modernos da Intel misturam dois tipos de núcleo: **P** de *performance*, grandes e rápidos, e **E** de *efficiency*, pequenos e econômicos. Espalhar um trabalho igual pelos dois faz os rápidos esperarem os lentos, então o engine prende as threads nos P-cores por padrão. |
| DDR5-4800 | A geração e a velocidade da memória RAM. O número é a taxa nominal em milhões de transferências por segundo. Nominal não é real: os **68,7 GB/s** que este projeto usa em todas as contas foram *medidos em execução*, não lidos da caixa. |
| RTX | A linha de placas de vídeo da NVIDIA para consumidor. A desta máquina é uma RTX 2060 SUPER com 8 GB. Os 8 GB são o número que decide: o modelo ocupa 18,5 GB, então ele **não cabe**, e qualquer uso de GPU vira partição — o que foi medido e não compensou. |
| SO | Sistema operacional — Windows, Linux, macOS. É quem entrega memória, arquivos e threads ao programa. O engine toca o SO em **um único arquivo**, `compat.c`: é a fronteira onde as diferenças entre Windows e Linux moram, para que o resto do código não precise saber onde está rodando. |

### 5.4 A numérica

| termo | uma linha |
|---|---|
| bit | A menor unidade de informação: 0 ou 1. Um peso em int4 ocupa 4 bits. |
| byte | Oito bits. Cabem dois pesos int4 num byte — daí o empacotamento por nibbles. |
| GB | Gigabyte, mil milhões de bytes. O modelo tem 19,65 GB; lê-se 2,01 GB por token. |
| float | Um número com casas decimais, guardado em formato de ponto flutuante. As contas do engine são todas em float de 32 bits. |
| fp32 | Ponto flutuante de 32 bits, com 23 bits de mantissa. É a precisão de trabalho do engine, e a do oráculo. |
| bf16 | Ponto flutuante de 16 bits com o expoente do fp32 e só 7 bits de mantissa. É como o modelo é publicado. |
| int4 / int8 | Inteiros de 4 e 8 bits, usados para guardar os pesos. Sempre acompanhados de uma escala que os reconverte. |
| nibble | Meio byte, 4 bits. Exatamente um peso int4 — e por isso a ordem em que os dois nibbles de um byte são lidos importa. |
| escala | O float que reconverte um inteiro no peso real: `peso ≈ (código − 8) × escala`. Uma escala para cada 64 pesos. |
| g64 | "Grupo de 64": uma escala compartilhada a cada 64 pesos vizinhos. Custa 4/64 = 0,0625 byte extra por peso. |
| absmax | O maior valor absoluto de um grupo. Dividido por 7, dá a escala mais simples — mas não a ótima. |
| W4A32 | Pesos em 4 bits, ativações em 32. O kernel desempacota e multiplica em float, sem quantizar as ativações. |
| ativação | Os números que fluem entre camadas — o vetor de 2048 valores. São minúsculos perto dos pesos: 8 KB contra 4,7 MB por expert. |
| associatividade | Somar em ordem diferente dá resultado diferente em ponto flutuante. Por isso a ordem de redução do engine é fixa e independente do número de threads. |
| /fp:precise | A opção do compilador que proíbe reassociar somas e fundir multiplicação com adição. Custa desempenho e compra reprodutibilidade. |
| arredondamento | Escolher o valor representável mais próximo. O erro do int4 é o piso do arredondamento, não um bug. |
| erro relativo | De quanto o resultado se afasta do correto, em porcentagem. Aqui é medido na saída de uma multiplicação, não em cada peso isolado. |
| outlier | Um valor muito fora da faixa dos vizinhos. Em ativações reais alguns canais têm magnitude 10–100× a mediana, e são eles que dominam o erro. |

### 5.5 A medição

| termo | uma linha |
|---|---|
| perplexidade | Quão "surpreso" o modelo fica com um texto. Menor é melhor. É a medida que diz se a quantização degradou de verdade. |
| nll | Negative log-likelihood, o logaritmo da perplexidade. Usa-se porque é somável — dá para calcular médias e intervalos. |
| intervalo de 95% | A faixa dentro da qual o valor real provavelmente está. Se ela cruza zero, não há efeito demonstrado — só ruído. |
| janela | Um trecho fixo de texto usado na medição. Aqui são 10 janelas de 384 tokens, e a comparação é feita janela a janela. |
| fixture | Um arquivo com respostas corretas pré-calculadas, gerado uma vez e comparado a cada execução do teste. |
| selftest | O teste embutido no binário: kernels, parser JSON, pool de threads, tokenizer e sampler contra fixtures. Roda em segundos. |
| teste de regressão | Um teste que existe só para impedir que um bug antigo volte. O caso `CUDA_DENSE` virou um. |
| baseline | O número de referência a ser batido. Aqui é 0,93 tok/s, medido com o engine anterior e o modelo grande. |
| benchmark | Uma medição controlada e repetível. Sem orçamento fixado, aquecimento e hash da saída, ela não vale nada. |
| aquecimento | Uma execução descartada antes de medir, para que o custo de encher os caches não caia toda na primeira medida. |
| reprodutibilidade | A propriedade de obter exatamente o mesmo resultado ao repetir. Aqui é critério de saída, não conveniência. |
| hash | Um número curto calculado a partir de um conteúdo, que muda se qualquer byte mudar. Usado para provar que duas saídas são idênticas. |
| throughput | Quanto trabalho sai por unidade de tempo, somando todas as conversas. Diferente de latência, que é o tempo de uma só. |
| latência | O tempo até a resposta começar. Banda alta com latência ruim é comum em disco; a RAM tem as duas boas. |
| entropia | Quão espalhada está a opinião do modelo sobre o próximo token, medida em *bits*. Entropia baixa: ele tem poucos candidatos. Alta: muitos, quase empatados. Serve junto da perplexidade porque separa dois casos que ela confunde — **seguro e certeiro** de **seguro por falta de opção**, e o segundo é o que antecede repetição. |
| rep4 | A fração de sequências de quatro tokens que já apareceram antes na mesma resposta. É o detector precoce de *degeneração*: o modelo começando a se repetir aparece aqui muito antes de incomodar quem lê. Acima de ~15% é degeneração declarada. |
| match@ref | Quantos tokens iniciais uma geração tem em comum com a geração de referência antes de divergir. É a medida mais direta de "este container gera o que o original geraria". Divergir não é erro: dois caminhos quase idênticos se separam no primeiro empate. |
| intervalo de confiança | A faixa em que o valor verdadeiro provavelmente está, dado o quanto as medidas variaram. Se a faixa **cruza o zero**, a medição não distingue "melhorou" de "piorou" — e a resposta honesta é *inconclusivo*, não "deu quase igual". |

### 5.6 O software e o sistema

| termo | uma linha |
|---|---|
| C11 | A versão do padrão da linguagem C usada no engine. Escolhida por ser auditável e não arrastar dependência nenhuma. |
| dependência | Uma biblioteca externa de que o programa precisa. O engine tem zero: JSON, tokenizer e leitor de arquivos são todos próprios. |
| linkagem estática | Embutir tudo no executável, para ele rodar sem instalar nada. A meta é um binário só, com o backend de GPU carregado à parte. |
| compilador | O programa que transforma código em executável. Aqui é o MSVC no Windows, encontrado dinamicamente, e gcc ou clang no Linux. |
| build id | Um hash das fontes e das opções de compilação, gravado no binário. Serve para invalidar caches persistidos quando o engine muda. |
| CLI | Interface de linha de comando: o programa que se usa digitando comandos no terminal. |
| ponteiro | Um endereço de memória guardado numa variável. Mover um bloco grande aqui é trocar ponteiros, nunca copiar bytes. |
| malloc | A função que pede memória ao sistema durante a execução. O laço de token do engine não a chama nenhuma vez. |
| zero-copy | Usar os dados onde eles já estão, sem passo intermediário de cópia. Aqui, os ponteiros do kernel apontam para dentro do arquivo mapeado. |
| JSON | Um formato de texto para dados estruturados. O índice do container e a configuração do modelo são JSON, lidos por um parser próprio. |
| UTF-8 | A codificação padrão de texto, em que um caractere ocupa de 1 a 4 bytes. O tokenizer trabalha nos bytes, não nos caracteres. |
| Unicode | O catálogo de todos os caracteres de todas as escritas. O pré-tokenizador precisa saber, por caractere, se é letra, número ou espaço. |
| expressão regular | Um padrão que descreve como quebrar um texto. O tokenizer usa uma fixa, a mesma do GPT-4, implementada à mão em vez de com um motor genérico. |
| HTTP | O protocolo da web. O subcomando `serve` abre um servidor local em `127.0.0.1:8080` com uma interface de chat. |
| loopback | O endereço `127.0.0.1`, que só a própria máquina alcança. É o padrão do servidor porque isto não é um serviço de produção. |
| escrita atômica | Escrever num arquivo temporário e renomear no fim. Uma queda de energia no meio não deixa um arquivo pela metade com cara de válido. |
| watchdog | Uma thread que compara o consumo real com o orçamento durante a execução. Sem ela, "orçamento" seria só uma opinião. |
| MSVC | O compilador C e C++ da Microsoft, que vem com o Visual Studio. É o que transforma os arquivos `.c` em `tootega.exe` no Windows. O projeto o localiza com a ferramenta `vswhere` em vez de cravar o caminho: caminho fixo quebra quando a versão muda. |
| ISO | Organização que publica normas internacionais. "C11 ISO" quer dizer que o código segue a norma da linguagem C de 2011 *como escrita*, sem extensões de um compilador específico. É o que faz o mesmo arquivo compilar no MSVC e no gcc sem alteração. |
| SSE | *Server-Sent Events* — o jeito padrão de um servidor HTTP mandar texto em pedaços, conforme produz, sem que o cliente fique perguntando. É como a resposta aparece palavra por palavra na tela em vez de surgir pronta depois de meio minuto. |
| CORS | *Cross-Origin Resource Sharing* — a regra do navegador que impede uma página de um endereço de chamar um servidor de outro, salvo se o servidor autorizar. Sem essa autorização, um cliente que roda *dentro* do navegador nem chega a fazer o pedido. |
| NDJSON | *Newline-Delimited JSON*: um objeto JSON por linha, um evento por linha. É como o agente conversa com a interface — cada linha é completa em si, então quem lê pode processar a primeira sem esperar a última. |
| stdin e stdout | Os dois canais de texto que todo programa recebe ao nascer: **stdin** é por onde entra, **stdout** é por onde sai. O agente e a interface conversam por eles — sem rede, sem porta, sem arquivo. Fechar o stdin é como desligar o telefone: o programa vê o fim e encerra. |
| EOF | *End Of File*, fim do arquivo. O sinal de que não vem mais nada. Quando a interface fecha o canal do agente, ele recebe um EOF e sai na hora — é o mecanismo de desligamento limpo, mais confiável que qualquer temporizador. |
| protocolo OpenAI | O formato de mensagem que a OpenAI publicou para conversar com modelos, e que virou padrão de fato: uma lista de mensagens com `role` e `content`, e a resposta no mesmo molde. Falar esse formato é o que permite ao Copilot, ao Continue e a qualquer outro cliente conectar neste servidor **sem uma linha de código nossa**. |
| endpoint | Um endereço específico dentro de um servidor, com uma função. `/v1/chat/completions` gera texto; `/stats` conta como o servidor está. É o "ramal" do serviço: mesma máquina, mesma porta, propósitos diferentes. |
| JSON Schema | Uma descrição, em JSON, de como outro JSON deve ser: quais campos, de que tipo, quais obrigatórios. O cliente manda o schema das ferramentas, e é por ele que o servidor sabe que `3` deve virar o **número** 3 e não o texto `"3"` — sem isso, toda ferramenta com argumento numérico chega errada. |
| MCP | *Model Context Protocol* — um padrão para um agente descobrir e usar ferramentas oferecidas por outros programas, em vez de só as que ele mesmo tem. É hoje o "protocolo comum de ferramentas". O TootegaEngine **não o implementa**; está registrado como não feito. |
| sandbox | Uma cerca em volta do que um programa pode tocar. No agente é a pasta de trabalho: todo caminho é resolvido contra ela e recusado se escapar — **inclusive com a aprovação automática ligada**. É a única coisa entre um modelo de 31 bilhões de parâmetros e o resto do disco. |
| aprovação | A pergunta que o agente faz antes de executar algo que muda estado — escrever arquivo, rodar comando. Leitura não pergunta, de propósito: pedir confirmação para tudo, a 13 tokens por segundo, transforma a tarefa em meia hora de Enter — e **treina a pessoa a aprovar sem ler**, que é o pior resultado possível. |
| branch | Uma linha paralela de desenvolvimento no controle de versão. Aqui a `master` é o engine em CPU pura e a `GPU` guarda todo o trabalho de placa de vídeo — escrito, medido, e não adotado. Separar assim é o que permite **manter a medição sem carregar o código**. |
| CRLF | Os dois caracteres invisíveis com que o Windows termina cada linha de texto, contra um só no Linux. Invisível não quer dizer inofensivo: um trecho de código copiado com um estilo não casa com o arquivo salvo no outro, e a busca falha sem que nada apareça na tela para explicar. |
| UUID | Um identificador longo o bastante para ser único sem que ninguém coordene quem usa qual. Serve para nomear uma conversa, um bloco, uma sessão — quando duas partes precisam falar do mesmo item sem um cadastro central. |

### 5.7 A matemática

| termo | uma linha |
|---|---|
| vetor | Uma lista de números. Aqui, o estado de um token é um vetor de 2048 floats. |
| matriz | Uma tabela de números, com linhas e colunas. Um peso do modelo é uma matriz; a maior tem 2048 × 154.880. |
| multiplicação de matriz | A operação central: cada linha da saída é a soma dos produtos de uma linha de pesos pela entrada. É onde vai quase todo o tempo. |
| produto escalar | Multiplicar dois vetores elemento a elemento e somar tudo. Uma multiplicação de matriz é um monte de produtos escalares. |
| posto baixo | Substituir uma matriz grande por duas menores multiplicadas. Comprime, e é o que `q_lora_rank` e `kv_lora_rank` fazem. |
| transposta | A matriz com linhas e colunas trocadas. Aqui importa porque ler uma coluna é caro e ler uma linha é barato. |
| FLOP | Uma operação de ponto flutuante. Gerar um token custa ~7,1 GFLOP — pouco, perto dos 2,01 GB de leitura. |
| probabilidade | Um número entre 0 e 1 dizendo quão provável algo é. O softmax converte as notas de saída em probabilidades. |
| desvio padrão | Quanto os valores se espalham em torno da média. É por causa dele que a comparação pareada existe: a variância entre janelas engolia o efeito. |
| expoente | Em `N^0,25`, o 0,25. Diz que quadruplicar N multiplica o resultado por só 1,41 — a lei de escala que este projeto recusa. |

---

## 6 — Do texto ao texto: o passeio completo

![O caminho de um token](img/03-pipeline.pt.svg)

Vamos seguir a pergunta *"Qual é a capital do Brasil?"* até a resposta.

### 6.1 Tokenizer — de texto para números

O modelo não vê letras. Vê inteiros. O **tokenizer** faz a conversão, em quatro
etapas. → [`docs/16`](../docs/16-tokenizer.md)

**Etapa 1 — quebrar em pedaços por regra.** Uma expressão regular fixa (a mesma
do GPT-4) separa palavras, números e pontuação. Não é "quebrar por espaço": a
regra é longa e precisa saber o que é letra e o que é número em **qualquer**
alfabeto — daí o projeto ter tabelas Unicode geradas.

**Etapa 2 — bytes viram símbolos.** Cada byte de 0 a 255 vira um caractere de uma
tabela fixa. É por isso que um espaço aparece como `Ġ` dentro do vocabulário.
Garante que **qualquer** arquivo, inclusive binário quebrado, é representável.

**Etapa 3 — BPE.** Junta repetidamente os dois pedaços vizinhos mais "baratos",
segundo uma tabela de 321.649 regras de fusão. Com uma exceção importante: se o
pedaço inteiro já estiver no vocabulário, usa-se o id direto e **nenhuma fusão é
aplicada** (o sinalizador `ignore_merges`). Ignorar isso produz tokenização
errada justamente nas palavras mais comuns.

**Etapa 4 — tokens especiais.** 36 símbolos como `<|user|>`, `<|assistant|>`,
`[gMASK]` são reconhecidos **antes** de tudo, por casamento exato de texto, e
nunca passam por BPE. São eles que delimitam quem está falando.

Resultado:

```
"[gMASK]<sop><|user|>Qual e a capital do Brasil?<|assistant|>"
   -> [154827, 154823, 154834, 9061, 383, 261, 8471, 495, 12089, 30, 154835]
```

> **Por que isso é tratado como caminho crítico:** um erro aqui **não estoura**.
> Produz ids ligeiramente diferentes, e o modelo responde algo plausível e
> errado. É o tipo de bug que consome dias se o teste não isolar o tokenizer —
> daí o `--check` contra 36 casos gerados pela biblioteca de referência.

### 6.2 Embedding — de número para vetor

Cada id vira uma linha da tabela `embed_tokens`: 154.880 linhas × 2048 números.
Lê-se **uma linha por token**. Como a tabela inteira tem 1,27 GB e só se lê 8 KB
dela por token, ela fica no arquivo mapeado e **não** ocupa orçamento de RAM —
distinção que um carregador ingênuo erra. → [`docs/11`](../docs/11-memoria.md)

A partir daqui, o token é um vetor de 2048 números que os documentos chamam de
`hidden` ou "estado residual".

### 6.3 As 47 camadas

![Por dentro de uma camada](img/04-camada.pt.svg)

O vetor atravessa 47 camadas, uma após a outra. Cada camada faz duas coisas, e
cada uma delas é precedida de normalização e seguida de uma soma:

```
x = x + Atenção(RMSNorm(x))
x = x + MoE(RMSNorm(x))
```

A soma se chama **residual** e é o que permite empilhar 47 camadas sem que o
sinal se degrade: cada camada **corrige** o vetor, não o substitui.

A camada 0 é diferente: em vez de MoE, tem um MLP comum de tamanho 10.240
(`first_k_dense_replace: 1`). As camadas 1 a 46 são MoE.

### 6.4 Atenção e o cache KV

![MLA: guardar comprimido](img/06-mla.pt.svg)

**Atenção** é o mecanismo pelo qual o token atual olha para todos os anteriores.
Para cada posição passada existe uma chave (K) e um valor (V); o token atual
gera uma consulta (Q), compara com todas as chaves, e faz uma média ponderada dos
valores.

**Por que existe um cache:** as chaves e valores das posições passadas não mudam.
Recalculá-las a cada token seria refazer todo o prompt a cada palavra. Então elas
ficam guardadas — é o **KV cache**.

**Por que ele é o problema:** no decode, a atenção **relê o cache inteiro a cada
token**. O tamanho do cache é, literalmente, o custo.

**O que o MLA muda:** em vez de guardar as chaves e valores expandidos
(20 cabeças × (256+256) = 10.240 números por token por camada), guarda-se uma
versão comprimida de **576 números**. A expansão é feita na hora — ou melhor,
**não é feita**: existe uma identidade algébrica que permite projetar a *query*
uma vez, em vez de expandir cada posição do cache. É a **absorção**.

```
KV expandido   47 camadas × 4.096 posições × 10.240 × 4 B = 7,9 GB por token
KV comprimido  47 camadas × 4.096 posições ×    576 × 4 B = 443 MB por token
```

18× menos tráfego [medido: 3,64 → 9,98 tok/s a 2.048 de contexto].

> **A armadilha que a primeira versão da absorção caiu.** A identidade envolve
> uma matriz transposta, e transposta sugere "materialize em fp32 para poder ler
> colunas". Feito assim, a absorção ficou **mais lenta** (5,6 contra 12,3 tok/s):
> materializar custava 863 MB de leitura de peso por token, mais do que os 443 MB
> que a absorção economizava em KV. O conserto foi notar que a transposta pode
> ser calculada acumulando **linhas**, que é o que o formato entrega barato.
>
> A lição do projeto: *trocar trabalho por memória precisa da conta de tráfego
> feita antes.* → [`docs/13`](../docs/13-decisoes-pendentes.md), D11

### 6.5 MoE — 64 guardados, 5 usados

![Mixture-of-Experts](img/05-moe.pt.svg)

Numa camada MoE:

1. Uma matriz pequena (o **roteador**, ou `gate`) dá uma nota a cada um dos 64
   experts.
2. Escolhem-se os **4 maiores**; as notas são renormalizadas
   (`norm_topk_prob: true`) e multiplicadas por 1,8 (`routed_scaling_factor`).
3. Esses 4 experts processam o vetor; um **quinto**, o **expert compartilhado**,
   sempre roda.
4. As cinco saídas são somadas com os pesos das notas.

**A escolha é por token e por camada.** Um token pode ativar experts totalmente
diferentes na camada seguinte. Não há tema, não há especialidade nomeável.

`n_group` e `topk_group` são parâmetros que, em outros modelos, restringem a
escolha a certos blocos de experts. Neste modelo valem 1 — ou seja, **não há
restrição**. É por isso que o projeto implementa só esse caminho no dia 1, mas
deixa a variante declarada em vez de cravada no parser (o erro do engine de streaming, que
reduziu o universo de modelos compatíveis a uma família só).
→ [`docs/06`](../docs/06-licoes-aprendidas.md)

### 6.6 lm_head, logits e amostragem

Depois da camada 46, uma última normalização, e então a matriz `lm_head`
(2048 × 154.880) converte o vetor em **154.880 notas** — os *logits*.

O **sampler** escolhe uma:

| modo | como escolhe |
|---|---|
| **greedy** | a maior nota, sempre. Determinístico |
| **temperature** | divide as notas por T antes do sorteio. T alto = mais aleatório |
| **top-k** | sorteia só entre as k maiores |
| **top-p** | sorteia entre as maiores que somem probabilidade p |

Detalhe fácil de errar: `vocab_size` é 154.880, mas só 154.856 ids existem de
fato. Os 24 restantes são preenchimento; **a máscara de amostragem tem de saber
disso**, senão o engine pode gerar um id inexistente.

O id escolhido volta ao passo 6.2 e tudo recomeça.

### 6.7 O exemplo fim a fim, com números

```powershell
.\tootega.exe chat models\glm47flash-int4\model.tootega `
    --tokenizer models\glm47flash-bf16\tokenizer.json --ctx 8192
```

```
A capital do Brasil é Brasília.

prefill  18 tokens em 4.63 s (3.9 tok/s)
decode    8 tokens em 0.82 s (9.79 tok/s)
```

O que aconteceu: 18 tokens de prompt passaram pelas 47 camadas de uma vez
(prefill), depois 8 tokens foram gerados um a um (decode), cada um relendo
2,01 GB de peso e todo o KV acumulado.

---

## 7 — Por que é lento: a física do decode

![Prefill e decode](img/13-prefill-decode.pt.svg)

### 7.1 Dois regimes com o mesmo modelo

| | prefill | decode |
|---|---|---|
| o que faz | lê o prompt inteiro | gera um token de cada vez |
| paralelismo | todos os tokens juntos | nenhum: cada token depende do anterior |
| leitura de peso | 2,01 GB para **N** tokens | 2,01 GB para **1** token |
| limitado por | aritmética | **banda de memória** |

É o decode que define a experiência de uso, e é ele que a conta de banda governa.

> **Nota honesta:** no estado atual, o prefill desta implementação é **mais
> lento por token** que o decode (3,9 contra 9,79 tok/s [medido]). Não é um
> mistério: prefill ainda não foi otimizado, e o roadmap diz isso em voz alta —
> latência de primeiro token não é meta até o M3. → [`docs/05`](../docs/05-roadmap.md)

### 7.2 A conta

```
tempo por token = (2,01 GB de peso + 52,9 KB × contexto) / banda
```

O primeiro termo é fixo. O segundo cresce com a conversa.

![O teto](img/11-roofline.pt.svg)

| contexto | KV lido | total/token | teto CPU | teto com GPU parcial |
|---|---|---|---|---|
| 1.024 | 0,05 GB | 2,06 GB | **29,1 tok/s** | 44,0 [estimativa] |
| 4.096 | 0,22 GB | 2,23 GB | **26,9** | 40,6 [estimativa] |
| 16.384 | 0,87 GB | 2,88 GB | **20,9** | 31,5 [estimativa] |
| 65.536 | 3,47 GB | 5,48 GB | **11,0** | 16,5 [estimativa] |
| 131.072 | 6,93 GB | 8,94 GB | **6,7** | 10,1 [estimativa] |

### 7.3 O que o teto **não** inclui

Três coisas, todas otimistas:

- **eficiência de banda** — engines bem escritos atingem 60 a 80% do pico, não
  100%;
- **compute não é zero** — são ~7,1 GFLOP por token, uns 20% do tempo, e
  precisam **sobrepor** à leitura, não somar;
- **a cadeia é serial** — 47 camadas em sequência, nada paraleliza com a
  seguinte no decode de uma única conversa.

Aplicando 60–80%, a faixa honesta:

| cenário | teto | **realista** |
|---|---|---|
| CPU, contexto curto (≤4k) | 27–29 | **17–22 tok/s** |
| CPU+GPU, contexto curto | 41–44 | **25–32 tok/s** |
| CPU, contexto de agente (32k) | ~15 | **10–12 tok/s** |

### 7.4 A consequência que muda o desenho

**A partir de ~16k de contexto, o KV domina.** A 64k ele é 63% do tráfego por
token. Num CLI de agente, contexto longo é o caso normal, não a exceção.

Daí duas conclusões que aparecem em vários documentos:

1. otimizar leitura de KV importa tanto quanto otimizar leitura de peso;
2. **reusar prefixo vale mais que qualquer kernel** — um turno que aproveita
   100k tokens de KV já calculado evita o prefill inteiro, e nenhuma otimização
   de multiplicação chega perto desse ganho.

---

## 8 — O hardware desta máquina e a inversão

![As quatro bandas](img/01-bandas.pt.svg)

### 8.1 A medição que inverte a intuição

Quase todo engine de inferência assume: *VRAM é rápida, então encha a VRAM*.
Nesta máquina isso é falso, e o motivo é a linha do PCIe.

```
pcie.link.gen.hostmax = 5     <- a placa-mãe oferece Gen 5 (63 GB/s)
pcie.link.gen.max     = 3     <- a GPU só fala Gen 3
```

A placa-mãe é moderna; a GPU é uma RTX 2060 SUPER de 2019, que só fala PCIe
gen 3. O link negocia pelo lado mais lento: **~12 GB/s**.

**Consequência:** qualquer peso que não *more* na VRAM precisa atravessar um
canal mais lento que o disco de onde veio, e cinco vezes mais lento que a RAM.

### 8.2 A armadilha, medida

no engine de streaming, a opção `CUDA_DENSE=1` coloca os tensores densos na GPU **mesmo
quando eles não cabem**:

```
config              tok/s
cpu-nativo          0,684
cuda-experts4       0,647
cuda-tc-w4a16       0,639
cuda-dense          0,054   <-- 13× mais lento
```

O engine **não recusou**. Aceitou, e passou a trafegar peso pelo PCIe a cada
camada, a cada token.

Foi essa medição que produziu duas regras do TootegaEngine:

> **Lei de colocação.** Coloque na GPU apenas o que couber lá permanentemente.
> Nunca transmita pesos por um link mais lento que o tier de origem.

> **Recusar em vez de degradar.** Se o plano não cabe no orçamento declarado, o
> engine sai com erro dizendo o que reduzir.

```
ERRO: nao cabe em --ram-gb 12
  pesos (int4)          18,1 GB
  KV (ctx 8192 x 1)      0,4 GB
  scratch + runtime      1,2 GB
  ------------------------------
  necessario            19,7 GB   excede em 7,7 GB
sugestao: --ram-gb 21, ou --gpu-layers para mover peso para a VRAM
```

### 8.3 A saída: partição por camada

![Partição por camada](img/08-particao.pt.svg)

Se pesos não podem cruzar o PCIe, o que pode? **Ativações.**

Um vetor de ativação tem 2048 números × 4 bytes = **8 KB**. Um expert tem
~4,7 MB. A diferença é de três ordens de grandeza.

Daí a aposta central da arquitetura:

> **Particione por camada, não por tensor.** Cada camada tem seus pesos
> permanentemente residentes no dispositivo que a computa. Entre camadas cruzam
> 8 KB de ativação.

Com ~7 GB de VRAM utilizável e ~18,1 GB de peso de camada, cerca de **39%** das
camadas caberiam na GPU. Por token, o PCIe carregaria 16 KB — 1,3 microssegundo.
Irrelevante.

**Isso ainda é hipótese.** Está marcado como [estimativa] e o marco M2 existe
para validá-la **ou matá-la**: se o ganho ficar abaixo de 20%, o M2 é
*cancelado*, não adiado, porque manter dois backends é custo permanente.
→ [`docs/03`](../docs/03-arquitetura.md), [`docs/05`](../docs/05-roadmap.md)

### 8.4 O que a medição diz sobre comprar hardware

Um detalhe contraintuitivo: se um dia houver upgrade de GPU, o ganho maior **não
é** mais VRAM nem mais compute — é o **link**. Placas Ampere/Ada falam Gen 4
(31,5 GB/s) e Blackwell fala Gen 5 (63 GB/s). E 63 GB/s **ultrapassa a banda da
RAM**, o que inverteria a hierarquia outra vez e tornaria o streaming para a GPU
um bom negócio pela primeira vez.

---

## 9 — Residência: a decisão que vale 20×

![Duas leis de escala](img/02-residencia.pt.svg)

### 9.1 Os dois eixos medidos

Com o modelo grande demais (744B, lido do disco), variando os recursos:

| eixo | variação | ganho | expoente |
|---|---|---|---|
| RAM (16 → 60 GB) | 3,75× | 1,42× | ~0,26 |
| concorrência (K=1 → K=4) | 4× | 1,45× | 0,27 |

**Ambos dão `N^0,25`.** Quadruplicar qualquer coisa rende ~40%.

### 9.2 Por que

Porque a variável que governa não é hardware, é **fração de residência**. Com
60 GB de orçamento e 324 GB de experts, cacheia-se ~15% do modelo; os outros 85%
vêm do disco a cada token, e nenhuma quantidade razoável de RAM muda isso.

Multiplicar por N só se aproxima de N× no trecho em que N faz **cruzar o limiar
de residência**.

### 9.3 A conclusão prática

> Escolher um modelo que já cabe rende 20 a 50×. Comprar hardware rende `N^0,25`.

É a tese inteira do projeto em uma linha. E é o motivo pelo qual o TootegaEngine
**joga fora** todo o maquinário de esconder disco que o engine de streaming construiu —
prefetch, roteamento entre SSDs, `O_DIRECT`, conjunto de experts fixados,
despejo por camada. Nada disso serve quando o modelo é residente, e tudo isso é a
maior parte do orçamento de complexidade daquele projeto.

**A principal vantagem estrutural do TootegaEngine sobre o engine de streaming não é
velocidade — é tamanho de código.**

---

## 10 — Como o modelo é guardado

### 10.1 Por que quantizar

O modelo publicado usa `bfloat16`: 2 bytes por peso, 62,4 GB no total. Não cabe
nos 64 GB da máquina junto com tudo o mais.

Guardando cada peso em **4 bits** com uma escala compartilhada, o mesmo modelo
ocupa 19,65 GB — e cabe com folga larga.

### 10.2 Como funciona, em concreto

![int4 em disco](img/07-int4.pt.svg)

Pegue 64 pesos vizinhos. Encontre o maior valor absoluto entre eles. Divida por 7.
Esse é o passo (a **escala**). Agora cada peso vira um inteiro de −8 a +7:

```
código = arredonda(peso / escala)          guardado como 0..15 (código + 8)
peso   ≈ (código − 8) × escala
```

64 pesos ocupam 32 bytes (dois por byte) mais 4 bytes de escala = **0,5625 byte
por peso**.

**A ordem dos nibbles importa.** A convenção adotada é **metade/metade**: o byte
`j` guarda o peso `j` no nibble baixo e o peso `j+32` no alto. Assim, ao carregar
32 bytes de uma vez com instruções SIMD, `v & 0x0F` entrega 32 pesos **contíguos**
e `v >> 4` entrega os 32 seguintes — dois vetores prontos, sem embaralhar nada.
A alternativa (par/ímpar) obrigaria a permutar as ativações a cada grupo.
→ [`docs/14`](../docs/14-formato-container.md)

### 10.3 Quanto custa em qualidade — e a surpresa

O erro de quantização foi medido de duas formas.

**Primeira: erro por tensor.** Quanto a saída de uma multiplicação muda:

| formato | erro na saída |
|---|---|
| int4 g64 | 10,81% |
| int8 g64 | 0,62% |

A razão entre os dois (17,4) bate com a razão de passo teórica (`127/7 = 18,1`).
Isso é o **piso do arredondamento**, não um bug.

**Segunda: perplexidade.** Quanto o modelo piora de verdade num texto real:

```
int4 g64 contra bf16:  +5,26% de perplexidade   [medido, 10 janelas, pareado]
```

Metade do que o erro por tensor sugeriria — confirmando que erro por tensor **não
prevê qualidade**.

**A surpresa veio da ablação.** Restaurando um grupo de tensores por vez ao
formato original e medindo:

| grupo restaurado | tensores | quanto da degradação explica |
|---|---|---|
| **atenção** | 193 | **100,3%** |
| experts compartilhados | 141 | 20,0% |
| camada densa 0 | 3 | 2,8% |
| **experts roteados** | **9.024** | **−33,6%** |

**Toda a degradação vinha da atenção.** Os 9.024 tensores de expert — 93% do
arquivo — não eram fonte de perda.

O motivo é estrutural: cada token ativa 4 experts de 64, então o erro de um
expert entra diluído e descorrelacionado. A atenção roda **inteira, nas 47
camadas, para todo token**, e o erro dela se compõe ao longo da profundidade.

**A previsão foi testada e cumprida.** Reconvertendo com a atenção em int8:

| container | atenção | perplexidade vs bf16 | tamanho |
|---|---|---|---|
| v1 | int4 g64 | +6,81% | 19,33 GB |
| **v2** | **int8 g64** | **−0,19%** | **19,65 GB** |

O intervalo de 95% do v2 **cruza zero**: não dá mais para distinguir a versão
quantizada do original com esta medida. Custo: 1,7% de tamanho.
→ [`docs/04`](../docs/04-modelo-alvo.md)

### 10.4 O container `.tootega`

![O arquivo .tootega](img/10-container.pt.svg)

Um arquivo único, mapeado na memória:

```
offset 0        cabeçalho fixo, 128 bytes
offset 128..    blobs de dados, cada um alinhado a 64 bytes
index_off       índice JSON
EOF
```

Três decisões que parecem detalhe e não são:

**Alinhamento a 64 bytes.** É o tamanho de uma linha de cache. Sem isso, ler
direto do arquivo mapeado com SIMD exigiria um caminho de cópia para realinhar —
e aí perde-se o zero-copy inteiro.

**O `config.json` inteiro vai dentro do índice.** A arquitetura vem do container,
não de um arquivo ao lado que pode divergir dele.

**Versão desconhecida é recusada, nunca lida "na esperança".** Magic diferente,
versão diferente → erro.

Conteúdo do v2 [medido]: int4 8.973 tensores (15,91 GB), int8 438 (2,44 GB),
f32 290 (25,7 MB), bf16 2 (1,27 GB). Residente: 18,38 GB.

---

## 11 — Memória: uma cópia, nenhuma a mais

Regra geral do projeto: **um byte de peso existe em um lugar só.**
→ [`docs/11`](../docs/11-memoria.md)

### 11.1 O arquivo mapeado **é** a memória

Não existe passo de "carregar o modelo": o arquivo é mapeado e os ponteiros do
quantizador apontam **para dentro das páginas mapeadas**.

```c
qt->q4 = (uint8_t*)(base + tensor->offset);   /* nao ha memcpy */
qt->s  = (float*)  (base + scales->offset);
```

Três consequências:

- **o page cache do sistema operacional é o cache** — sem buffer duplo, sem
  gerenciar residência à mão;
- **páginas de peso são recuperáveis** — sob pressão o SO as descarta e relê. Por
  isso elas **não contam** no orçamento de RAM: são *file-backed*, não anônimas;
- **dois processos no mesmo modelo custam um conjunto de pesos, não dois.**

### 11.2 Arena de scratch: zero `malloc` por token

Todos os buffers temporários do forward têm tamanho fixo, conhecido depois de ler
a configuração. São fatias de **uma arena dimensionada uma vez na inicialização**.
O laço de token não chama o alocador nenhuma vez.

O engine de streaming faz ~500 `malloc`/`free` por token. A 0,93 tok/s isso é 0,01% e não
importa; a 22 tok/s importa — e a correção é estrutural, não uma
micro-otimização para depois.

### 11.3 Os três caches, e por que confundi-los custa caro

![Três caches](img/15-tres-caches.pt.svg)

| cache | guarda | dono |
|---|---|---|
| **de pesos** | tensores por tier (VRAM / RAM) | o engine, exclusivamente |
| **de KV** | estado de atenção dos tokens já vistos | o engine tem o mecanismo; o agente determina se acerta |
| **de prefixo** | a sequência de tokens que você monta | o agente, exclusivamente |

O engine **vê inteiros**. Ele não sabe o que é um "system prompt". Logo:

> **O engine não tem como consertar uma ordenação ruim de prompt.**

### 11.4 A regra de ordenação, e por que ela é decisão de desempenho

![Cache de prefixo](img/09-prefixo.pt.svg)

O engine casa o prefixo token a token, **para no primeiro divergente e trunca
dali em diante**. Tudo depois do primeiro token alterado é recomputado.

> **Estável primeiro, volátil por último.**
> `system → schemas de ferramentas → contexto longo → histórico → turno atual`

Um relógio, um UUID ou uma listagem de diretório no começo do prompt invalida os
100k tokens seguintes, e o engine não tem como saber que aquilo era irrelevante.

Duas consequências específicas de CLI com ferramentas:

- **ferramentas renderizam na posição 0** — trocar, adicionar ou reordenar uma
  ferramenta invalida tudo. Precisa existir um caminho de *append*, não de troca;
- **instrução que chega no meio não pode editar a frente** — estado tardio ("o
  modo mudou", "o arquivo mudou no disco") entra **no fim** do prefixo.

Medição do mecanismo funcionando: **92 de 100 tokens de prompt vieram do cache no
segundo turno** — 8 tokens de prefill em vez de 100. [medido]

**Endereçamento por hash, e o bug que ele previne.** Um bloco de KV é endereçado
pelo **hash acumulado da sequência exata de tokens que o produziu** — nunca pelo
slot, nunca pela posição. O motivo é concreto: no engine de streaming, uma execução divergiu
das outras porque um KV persistido **de outra conversa** casava parcialmente com o
novo prompt. O engine retomou de um estado que não correspondia àquele texto.
Casamento parcial não existe: ou o hash bate, ou o bloco é ignorado.
→ [`docs/12`](../docs/12-estado-persistente.md)

### 11.5 Orçamentos: o que conta e o que não conta

| região | conta em `--ram-gb`? | por quê |
|---|---|---|
| mapeamento do modelo | **não** | file-backed, o SO recupera |
| páginas travadas com `mlock` | **sim** | anônimas na prática |
| blocos de KV | **sim** | anônimas |
| arena de scratch | **sim** | anônimas |
| buffer fixado da GPU | **sim** | anônimo e não paginável |

E o teto é **verificado em execução**, não só no plano — estimativa de
planejamento erra, e sem verificação "orçamento" vira decoração.
→ [`docs/09`](../docs/09-limites-de-recursos.md)

Defaults para esta máquina:

```
--ram-gb 48          # de 64, deixa 16 para o SO e para o seu trabalho
--vram-gb 6.5        # de 8, reserva 1,5 para driver e desktop
--threads 16         # os 8 P-cores com SMT
--affinity p-cores
--ctx 8192
--kv-slots 1
```

---

## 12 — Como se sabe que está certo

Esta é, provavelmente, a seção mais característica do projeto.

### 12.1 O oráculo e o 32/32

Existe uma implementação de referência em Python (`ptools/oracle.py`) que roda o
mesmo modelo, sobre o mesmo container, usando `numpy` e `transformers`. Ela é
lenta e simples — de propósito.

O engine roda 32 posições em *teacher forcing* (isto é, forçando o texto
conhecido) e compara token a token. **32 de 32** ou não passa.

Por que binário: porque separa *engine* de *gerador de plausibilidade*. Um
resultado "parecido" não é resultado.

### 12.2 Medição pareada — e o erro que originou a regra

![Medição pareada](img/12-pareada.pt.svg)

As duas primeiras medidas de perplexidade do int4 deram:

```
191 previsões   ->  −4,2%   (int4 "melhor" que o original — impossível)
381 previsões   ->  +2,5%
```

**O sinal inverteu ao dobrar a amostra.**

A causa: a perplexidade varia de **223 a 473 entre janelas** desse texto. Essa
variância engole um efeito de 5%. Comparar duas médias globais teria produzido um
número publicável e **errado**.

A correção é comparar **a mesma janela nas duas configurações** e medir a
diferença, cujo desvio é vinte vezes menor:

```
delta de nll (int4 − bf16):  +0,0513 ± 0,0076
intervalo 95%:               [+0,0365, +0,0661]
perplexidade:                +5,26%
```

Dez janelas de dez positivas, intervalo não cruza zero. É medida, não impressão.

> A regra virou doutrina do projeto: **comparação pareada, sempre.**

### 12.3 O que já foi reprovado

Quatro otimizações **boas**, implementadas, medidas, e não adotadas como padrão:

| otimização | o que prometia | resultado |
|---|---|---|
| **KV em bf16** | metade da memória de cache | **29/32.** Três tokens mudaram. Disponível em `--kv bf16`, não é default |
| **KV em int8 g64** | 3,8× menos cache, erro menor que bf16 | **29/32** também. E só 2% mais rápido |
| **softmax online** | metade das leituras de KV | **9% mais lento** a 4k — as leituras cortadas já vinham do cache L3 |
| **absorção materializada** | operar no espaço comprimido | 2× **mais lenta** — trocou tráfego de KV por tráfego de peso |

Duas observações que valem mais que as otimizações:

**O critério detecta "diferente", não "pior".** Dois quantizadores de qualidade
bem diferente (1,5% e 0,59% de erro) reprovaram com o mesmo placar. O motivo não
é a qualidade deles: é que as margens deste prompt são pequenas — 0,0007 na
posição 31. Para decidir se um deles **degrada de fato**, falta a medida de
perplexidade específica. Enquanto ela não existir, o padrão é o que passa no
teste que existe.

**"Menos leituras" não é o mesmo que "mais rápido"** quando as leituras cortadas
já vinham do cache.

### 12.4 As outras disciplinas de medição

Vieram todas de erro cometido:

- **não trocar duas variáveis de uma vez** — mediu-se "nativo + CUDA" contra
  "release CPU" e o resultado não dizia nada;
- **fixar o orçamento de RAM entre configurações** — o valor "auto" flutuava
  entre 47,1 e 60,5 GB conforme o momento, e invalidou uma matriz inteira de
  benchmark;
- **aquecer antes de medir concorrência** — sem aquecimento, o custo do tier frio
  cai todo em K=1 e inventa um ganho de escala que não existe;
- **hash da saída em toda medição** — foi assim que se descobriu que CPU e GPU
  produzem bytes idênticos.

### 12.5 A regra final

> **Publica-se o que foi medido, não o que parece mais seguro.**

O caso que fixou a regra: ao reconverter o container para a v2, um tensor mudou
de precisão sem que a ablação o tivesse avaliado. A tentação foi reverter "por
segurança". Mas o container **que foi medido** era o que tinha aquele tensor
naquela precisão — e mediu perda indistinguível de zero. Reverter produziria uma
configuração *não testada* para comprar uma garantia que a medição já dava.

### 12.6 O teste que roda em um segundo

```powershell
.\tootega.exe selftest
```

Verifica kernels, parser JSON, pool de threads, tokenizer e sampler contra
fixtures pré-calculadas (`src/fixture_kernels.h`, `tokfixture.json`). É o
primeiro comando a rodar depois de qualquer mudança.

---

## 13 — O servidor, o protocolo e o agente

As seções anteriores explicam como um token sai do modelo. Esta explica como
esse token chega até um programa que serve para alguma coisa.

### 13.1 Por que existe um servidor

Carregar o modelo custa **5,3 segundos e 18,54 GB**. Se cada pergunta subisse um
processo novo, cada pergunta pagaria isso de novo — e a máquina não tem RAM para
dois ao mesmo tempo.

Então o desenho é o de sempre em software de servidor, e vale para leigo e para
especialista igual: **um processo caro que fica de pé, e clientes baratos que
vão e voltam.** O servidor sobe, aquece a memória, e fica ocioso consumindo
**0,2% de um núcleo e 0,04 GB**. Quem quiser falar com o modelo abre uma
conexão, pergunta e desliga.

O que ele expõe:

| endereço | o que faz |
|---|---|
| `POST /v1/chat/completions` | a conversa: recebe as mensagens, devolve o texto |
| `GET /v1/models` | diz que modelo está carregado |
| `GET /stats` | quantos tokens, quanto tempo, quanto de cache aproveitado |
| `GET /` | o chat web embutido no próprio executável |

### 13.2 Falar o protocolo que o mundo já fala

Um formato de mensagem virou padrão de fato: o que a OpenAI publicou. Uma lista
de mensagens, cada uma com um **papel** (`system`, `user`, `assistant`,
`tool`) e um **conteúdo**; a resposta vem no mesmo molde.

Não há nada de especial nesse formato. O que há de especial é que **todo mundo
já o implementou**. Copilot, Continue, Open WebUI, bibliotecas em dez
linguagens: nenhum deles sabe que o TootegaEngine existe, e todos funcionam com
ele — basta apontar a URL. Escrever um protocolo próprio, ainda que melhor,
significaria escrever também cada um desses clientes.

**Consequência de projeto, e ela é forte:** o engine não decide nada sobre a
conversa. Ele recebe mensagens e devolve texto. Quem monta o histórico, quem
decide chamar uma ferramenta, quem repete o ciclo — é o cliente. Diz-se que
**o servidor é burro e o cliente é esperto**, e isso é elogio: é o que faz um
programa escrito para outro engine funcionar aqui sem uma linha de código
nossa.

### 13.3 Por que a resposta aparece letra por letra

A 13 tokens por segundo, uma resposta de 300 tokens leva **23 segundos**.
Esperar 23 segundos olhando uma tela parada é insuportável; ver a frase se
formando é confortável — e é exatamente o mesmo tempo.

O mecanismo tem nome, é padrão de navegador e não é WebSocket: **SSE**,
*Server-Sent Events*. O servidor mantém a conexão aberta e manda um pedaço de
texto por vez, no formato `data: {...}`, até mandar `data: [DONE]`. O fluxo é
de mão única — do servidor para o cliente — que é justamente o caso aqui.

### 13.4 Ferramentas: o modelo pede, ele não executa

Um modelo de linguagem só sabe produzir texto. Ele não abre arquivo, não roda
comando, não acessa a internet. O que se convencionou é o seguinte teatro:

1. o cliente descreve, no prompt, as ferramentas disponíveis e a forma dos
   argumentos de cada uma;
2. o modelo, em vez de responder, **escreve um pedido** de execução;
3. o cliente executa o pedido de verdade e devolve o resultado como uma
   mensagem nova;
4. o modelo continua de onde parou, agora sabendo o resultado.

O passo 3 é o único que toca o mundo real, e quem o executa é um programa
comum, com as permissões de quem o rodou. **O modelo nunca executa nada.**

Neste modelo o pedido não é texto solto a ser adivinhado: `<tool_call>`,
`<arg_key>`, `<arg_value>` e `<|observation|>` são **tokens únicos do
vocabulário**, colocados lá no treino. O servidor traduz o formato OpenAI para
esses tokens e traduz a resposta de volta. É tradução, não invenção — e é por
isso que funciona na primeira tentativa em vez de depender de expressão
regular sobre a saída.

**A parte que quase deu errado.** O JSON do protocolo distingue o número `3`
do texto `"3"`; o modelo escreve os dois iguais. Sem consultar o **JSON
Schema** que o cliente mandou junto das ferramentas, todo argumento numérico
chegaria como texto do outro lado, e a ferramenta receberia `"3"` onde espera
`3`. O servidor tipa cada argumento pelo schema declarado — e essa é a
diferença entre uma integração que funciona e uma que quase funciona.

### 13.5 O agente é o laço, não o modelo

O **agente** (`tools/agent.exe`) é o programa que fecha o ciclo dos quatro
passos acima até a tarefa acabar. Ele traz seis ferramentas:

| ferramenta | o que faz | pergunta antes? |
|---|---|---|
| `Read` | lê um arquivo | não |
| `Glob` | acha arquivos por padrão de nome | não |
| `Grep` | procura texto dentro dos arquivos | não |
| `Write` | escreve um arquivo | **sim** |
| `Edit` | troca um trecho por outro | **sim** |
| `Bash` | roda um comando | **sim** |

Duas decisões dentro dessa tabela merecem ser lidas devagar.

**A cerca.** Toda ferramenta resolve o caminho contra a pasta de trabalho e
recusa o que sair dela — inclusive com a aprovação automática ligada. É a
única coisa entre um modelo de 31 bilhões de parâmetros e o resto do disco.

**Ler não pergunta.** Parece descuido e é o contrário. A 13 tokens por segundo,
uma tarefa real faz dezenas de leituras; confirmar cada uma transforma o
trabalho em meia hora de apertar Enter — e, pior, **ensina a pessoa a aprovar
sem ler**. Pedir confirmação só onde algo muda de estado mantém a pergunta
rara o bastante para ser lida.

### 13.6 Como o agente conversa com uma interface

O agente não abre porta de rede. Ele conversa por **stdin e stdout** — os dois
canais de texto que todo programa recebe ao nascer — em **NDJSON**: um objeto
JSON por linha, um evento por linha. Quem lê processa a primeira linha sem
esperar a última.

```
{"type":"system","subtype":"init","session_id":"...","tools":[...]}
{"type":"assistant","message":{...}}
{"type":"user","message":{"content":[{"type":"tool_result",...}]}}
{"type":"result","subtype":"success","duration_ms":8213}
```

Esse é o mesmo contrato de processo que o **Claude Code CLI** usa. A
consequência é prática: uma interface escrita para aquele CLI fala com este
agente **trocando o caminho do executável**, e nada mais. Foi assim que o
Cockpit passou a rodar os dois lado a lado, N abas de um e uma do outro.

Fechar o canal encerra o agente. Não há temporizador, não há sinal, não há
processo órfão: quando a interface fecha o stdin, o agente recebe um **EOF** e
sai. É o desligamento mais confiável que existe, porque é o próprio sistema
operacional que o entrega.

### 13.7 O chat que vem dentro do executável

O servidor serve uma página de chat que **não existe como arquivo**: ela é uma
cadeia de caracteres compilada dentro do `tootega.exe`. Não há pasta `www/`
para esquecer de copiar, não há caminho relativo para errar, não há versão do
HTML diferente da versão do binário.

A tela de entrada pede três coisas, e cada uma tem uma razão:

- **o nome** — é do interlocutor, e vai para a mensagem de sistema;
- **as diretivas** — como o modelo deve responder ("responda no idioma da
  pergunta", "vá direto ao ponto"). Ficam no **começo** do prompt, que é onde o
  cache de prefixo as reaproveita em todos os turnos;
- **raciocinar ou não** — desligado por padrão, pela medição de §14.6.

### 13.8 O cache de prefixo, na prática

O cache de prefixo é o mais mal-entendido dos três caches (§3.1). Ele não
guarda perguntas nem respostas: guarda o **estado interno** correspondente aos
primeiros N tokens do prompt. Se a próxima mensagem começa com exatamente os
mesmos tokens, esse estado é reaproveitado e só o resto é calculado.

"Exatamente os mesmos" é literal — um espaço a mais invalida dali para a
frente. É por isso que a ordem do prompt é regra e não gosto:

```
sistema  ->  ferramentas  ->  contexto longo  ->  historico  ->  turno atual
     estavel                                              volatil
```

O que muda a cada turno tem de ficar **no fim**. Com a ordem certa, um turno
típico reaproveita **99% do prefill** — 1163 de 1172 tokens no exemplo do
console. Com a ordem errada, reaproveita zero, e cada turno paga o prompt
inteiro de novo.

---

## 14 — Como se mede a qualidade

Velocidade é fácil de medir: cronômetro. Qualidade não tem cronômetro, e é
onde quase toda avaliação de modelo vira opinião. Esta seção mostra os quatro
números que este projeto usa, o que cada um vê, e — mais importante — o que
cada um **não** vê.

### 14.1 Perplexidade: quão surpreso o modelo ficou

Dá-se ao modelo um texto que ele nunca escolheu e pergunta-se, posição a
posição: *que probabilidade você dava para a palavra que de fato veio?*
Perplexidade é a média dessas surpresas, transformada em um número com
significado intuitivo: **em quantas opções o modelo estava efetivamente em
dúvida**.

Perplexidade 8 quer dizer "hesitando entre umas 8 possibilidades". Menor é
melhor. É a régua padrão, e é objetiva: não depende de alguém achar a resposta
boa.

O número deste engine, contra a implementação de referência em Python, no
mesmo texto:

| medição | engine | `transformers` |
|---|---|---|
| 384 posições de prosa, pesos BF16 | **8,63** | **8,89** |
| mesmo texto, container int4 | **5,66** | — |

**Cuidado ao ler a segunda linha.** Perplexidade só se compara sobre o *mesmo
texto e o mesmo vocabulário*. Números de perplexidade de dois artigos
diferentes não se comparam — é o erro mais comum com essa métrica, e ele
aparece em toda parte.

### 14.2 Entropia e confiança: o que a perplexidade esconde

Perplexidade mede o acerto *na palavra que veio*. Ela não distingue dois
estados muito diferentes:

- o modelo tinha **um** candidato forte e acertou;
- o modelo tinha **quarenta** candidatos parecidos e um deles calhou de vir.

**Entropia** separa os dois. Medida em *bits*, ela diz o quanto a opinião do
modelo está espalhada: baixa, poucos candidatos; alta, muitos quase empatados.
A **confiança** é o complemento prático — a probabilidade que ele deu ao
primeiro colocado.

O console do servidor imprime os quatro juntos, e é a leitura conjunta que
informa:

```
prompt    perplexidade 3.41  nll 1.2267  entropia 2.18 bits  confianca 47.3%
geracao   perplexidade 1.84  nll 0.6088  entropia 1.02 bits  confianca 71.9%
```

Entropia **muito** baixa por muito tempo não é sinal de excelência: é o
sintoma que antecede o modelo entrar em laço e repetir.

### 14.3 rep4: o detector de repetição

`rep4` é a fração de sequências de **quatro tokens** que já apareceram antes na
mesma resposta. Conta sequências, não palavras — e por isso enxerga o problema
cedo, quando ainda não incomoda quem lê. Acima de ~15% é **degeneração**
declarada: o texto se enrolando sobre si mesmo.

Degeneração não é travamento. O programa está funcionando perfeitamente e
produzindo lixo — o que a torna invisível para qualquer teste que só verifique
se houve saída.

### 14.4 match@ref: quantos tokens antes de divergir

Quantos tokens iniciais uma geração tem em comum com a geração de referência
antes de tomar outro caminho. É a medida mais direta de *"esta configuração
gera o que a original geraria"*.

Divergir não é erro. Duas execuções quase idênticas se separam no primeiro
quase-empate, e daí para a frente escrevem textos diferentes e igualmente
válidos. O que o número diz é **quando** a diferença numérica ficou grande o
bastante para mudar uma escolha.

### 14.5 A medição pareada, e por que ela existe

Aqui está o erro que este projeto cometeu e que virou regra.

Para comparar duas configurações, a coisa natural é medir a perplexidade média
de cada uma e subtrair. Foi o que se fez. As duas primeiras tentativas deram
**−4,2%** (int4 "melhor" que o original, o que é impossível) e depois
**+2,5%**. O sinal **inverteu** ao dobrar a amostra.

A causa: a perplexidade varia de **223 a 473 entre janelas** desse texto. Uma
variação dessas engole um efeito de 5% sem deixar rastro.

A correção é comparar **janela a janela**, sobre o mesmo texto, e medir a
*diferença* — cujo desvio é vinte vezes menor, porque as duas configurações
veem exatamente as mesmas palavras. Chama-se **medição pareada**, e sem ela o
projeto teria publicado um número apresentável e errado.

O acompanhante obrigatório é o **intervalo de confiança**: a faixa onde o
valor verdadeiro provavelmente está. Se a faixa **cruza o zero**, a medição
não distingue "melhorou" de "piorou", e a resposta honesta é *inconclusivo* —
não "deu quase igual".

**Foi exatamente o que aconteceu com o KV em int8.** O critério tinha sido
fixado antes de medir; o resultado ficou com os dois intervalos cruzando o
zero. A economia de memória era real e grande (105,8 KB por posição em f32
contra 28,1 KB em int8), a tentação de adotar era enorme, e mesmo assim ficou
registrado como **inconclusivo, não adotado** — disponível como opção, nunca
como padrão. Mudar o critério depois de ver o número é a forma mais comum de
mentir com estatística, e ela quase sempre é involuntária.

### 14.6 Raciocinar antes de responder ajuda?

O modelo sabe escrever um rascunho para si mesmo, entre `<think>` e
`</think>`, antes de responder. A crença geral é que isso melhora a resposta.
Medimos, com 12 perguntas, orçamento de 400 tokens:

| | rep4 | terminou | tempo |
|---|---|---|---|
| sem raciocínio | 1,7% | 12/12 | 78 s |
| **com raciocínio** | **13,8%** | **4/12** | **354 s** |

Com raciocínio o modelo **não termina 8 dos 12 prompts**, repete 8× mais
4-gramas — 13,8% encosta no limiar de degeneração — e custa **4,5× o tempo**.

O caso que ilustra: para *"Qual é a capital do Brasil?"* ele escreve seis
etapas de raciocínio em inglês, com três rascunhos alternativos de resposta, e
é cortado pelo orçamento **antes de responder**.

A leitura correta não é "raciocinar é ruim". É: **neste modelo, com esse
orçamento, e para perguntas desse tamanho, raciocinar sai caro e responde
pior.** Por isso o padrão é desligado — e a opção continua lá, para quem tiver
perguntas maiores e paciência maior.

### 14.7 A medição que decidiu não usar a GPU

Vale como exemplo final de método, porque o resultado contraria a intuição de
todo mundo.

A GPU desta máquina tem 8 GB; o modelo tem 18,54 GB. Ele **não cabe**. Restaria
partir o trabalho entre GPU e CPU. Escreveu-se o backend CUDA, otimizaram-se os
kernels até ficarem **31× mais rápidos** que a primeira versão e **2,6×** mais
rápidos que a CPU nos mesmos dados (179–190 GB/s contra 68,7).

E o decode completo, medido, ficou **mais lento** — nas três configurações
testadas. O motivo está em §8: o PCIe desta máquina (~12 GB/s) é mais lento que
tudo, e qualquer partição paga esse pedágio a cada camada.

Kernel rápido não é sistema rápido. O trabalho foi guardado numa branch
separada, com a medição junto, e o engine continuou em CPU pura. **Um resultado
negativo bem medido é um resultado** — economiza a próxima pessoa que tiver a
mesma ideia boa.

---

## 15 — Onde estamos e o que vem

![Roadmap](img/14-marcos.pt.svg)

Cada marco tem **critério de saída mensurável**. Sem número, não fecha.
Dois deles têm **critério de corte**: se não pagar, é cancelado, não adiado.

| marco | o que entrega | critério de saída | estado |
|---|---|---|---|
| **M0** | converter e validar o modelo | container int4 alinhado + perplexidade medida | **feito** |
| **M1** | forward só em CPU, token-exato | 32/32 no oráculo; ≥10 tok/s a 4k; RSS estável; `--ram-gb` respeitado | **feito** |
| **M2** | partição GPU/CPU por camada | saída byte-idêntica ao M1; **≥20% mais rápido, senão CANCELA** | **cancelado pela medição** ([§14.7](#147-a-medição-que-decidiu-não-usar-a-gpu)) |
| **M3** | servidor e batching contínuo | K=4 rende ≥2,5× o K=1; latência ≤2× | à frente |
| **M4** | CLI agêntico | ferramentas, sandbox, contrato de processo | **feito** ([§13](#13--o-servidor-o-protocolo-e-o-agente)) |
| **M5** | otimização, com número | cada item com antes/depois na mesma configuração | à frente |

### O que o projeto declara **não** ser

- **não é engine de streaming de disco** — se o modelo não cabe, use outra coisa;
- **não é multi-GPU** — uma GPU, um host;
- **não é servidor de produção** — sem multi-tenancy, sem autenticação além de
  loopback, sem cotas;
- **não treina nada** — sem treino, sem ajuste fino, sem LoRA.

### Critérios de fracasso declarados

O projeto está errado se o CPU-only não passar de 10 tok/s a 4k, se o split
GPU/CPU render menos de 20%, ou se a saída divergir do oráculo em qualquer
colocação. → [`docs/01`](../docs/01-objetivos.md)

---

## 16 — Mãos à obra

### 16.1 Compilar

```powershell
.\build.ps1                    # engine (MSVC, encontrado por vswhere)
.\build.ps1 test               # compila e roda o selftest
.\tools\<nome>\build.ps1       # uma ferramenta específica
```

No Linux, `make`. O compilador é encontrado dinamicamente — **nunca** por caminho
fixo.

Uma opção de compilação merece nota: `/fp:precise`, sem contração para FMA.
Ela impede o compilador de reassociar somas e de fundir multiplicação com adição.
Isso custa um pouco de desempenho e compra **reprodutibilidade**, que é critério
de saída, não luxo. Foi conferida no disassembly, não assumida.
→ [`docs/10`](../docs/10-implementacao.md)

### 16.2 Os comandos

```powershell
.\tootega.exe selftest
.\tootega.exe version
.\tootega.exe inspect  models\glm47flash-int4\model.tootega
.\tootega.exe plan     models\glm47flash-int4\model.tootega --ctx 4096
.\tootega.exe tokenize models\glm47flash-bf16\tokenizer.json --check tokfixture.json
.\tootega.exe forward  models\glm47flash-int4\model.tootega --ids-file ids32.txt
.\tootega.exe generate models\glm47flash-int4\model.tootega --tokenizer ... --prompt "..." -n 60
.\tootega.exe chat     models\glm47flash-int4\model.tootega --tokenizer ... --ctx 8192
.\tootega.exe serve    models\glm47flash-int4\model.tootega
```

| comando | para quê | quando usar |
|---|---|---|
| `selftest` | kernels, JSON, pool, tokenizer, sampler contra fixtures | sempre, depois de qualquer mudança |
| `version` | versão e hardware detectado | ao reportar um resultado |
| `inspect` | cabeçalho, índice e orçamento do container | para saber o que tem dentro do arquivo |
| `plan` | o que cabe: pesos, KV e scratch contra `--ram-gb` | **antes** de rodar, para não descobrir na metade |
| `tokenize` | testa o tokenizer contra a fixture | ao mexer em qualquer coisa de texto |
| `forward` | passe *teacher-forcing* sobre ids fixos | é o comando do oráculo |
| `generate` | tokenizer + forward + amostragem | para ver o modelo escrevendo |
| `chat` | conversa multi-turno com cache de prefixo | uso normal |
| `serve` | servidor HTTP + chat web em `127.0.0.1:8080` | uso normal, do navegador |

Opções que aparecem em quase tudo:

| opção | o que faz |
|---|---|
| `--threads N` | quantas threads de cálculo. Default: P-cores físicos |
| `--affinity p-cores\|all\|none` | onde fixar as threads |
| `--kernel auto\|scalar` | `scalar` desliga SIMD; útil para isolar bug de kernel |
| `--attn absorbed\|expanded` | qual formulação da atenção. Default: `absorbed` |
| `--ctx N` | capacidade do KV |
| `--ram-gb N` | teto duro de memória |
| `-v` | detalha o que está sendo feito |
| `--kv-prec f32\|bf16\|int8` | precisão do cache KV. Default `f32`; as outras são opção medida, não padrão |
| `--think\|--no-think` | liga ou desliga o rascunho antes da resposta. Default: desligado |
| `--verbose` | no `serve`, imprime a forma e a estatística de cada pedido — nunca o conteúdo |

### 16.3 Como fazer sua primeira medição sem errar

Quatro regras, todas compradas com tempo perdido:

1. **fixe `--ram-gb` e `--threads`.** Duas medições com orçamento diferente não
   são comparáveis;
2. **aqueça.** Descarte a primeira execução; o efeito de cache quente já valeu
   +20,7% numa medição do baseline;
3. **mude uma coisa de cada vez.** Se você mudou o kernel e o número de threads,
   o resultado não diz nada;
4. **registre o hash da saída.** É assim que se descobre que uma diferença de
   velocidade veio acompanhada de uma diferença de resultado.

---

## 17 — Mapa do repositório

### 17.1 As pastas e a regra de cada uma

| pasta | o que é | regra |
|---|---|---|
| `src/` | o engine | C11 ISO, **zero dependências**; só `compat.c` tem `#ifdef` de SO |
| `tools/` | ferramentas de **produção** | C ou Rust, uma pasta por ferramenta, o `.exe` na raiz de `tools/` |
| `Tests/` | ferramentas de **medição** | mesma regra; medem o engine e podem usar o checkpoint original |
| `ptools/` | ferramentas em Python | só o que exige `transformers`/`tokenizers` como referência externa |
| `docs/` | a especificação | toda decisão com o porquê e o número que a sustenta |
| `models/` | os containers | não versionado |
| `ForDummies/` | este documento e as figuras | PT e EN em paralelo |
| `LICENSE` | a licença MIT | vale para o código; o modelo tem a dele |

**Ferramenta nova é C ou Rust, nunca Python.** A proibição de dependências vale
para `src/` — o engine precisa ser auditável em uma sessão de leitura e consome
arquivos de terceiros. Ferramenta é outra coisa: pode linkar o que precisar,
desde que não entre em `src/`. E ferramenta em C **deve reusar as fontes de
`src/`**, para medir o engine e não uma reimplementação dele. → `CLAUDE.md`

### 17.2 Os arquivos do engine

| arquivo | responsabilidade |
|---|---|
| `main.c` | despacho de subcomandos e a leitura de opções |
| `model.c` | carregar o container, config → estrutura interna |
| `container.c` | o formato `.tootega`, leitura e escrita |
| `forward.c` | a passagem pelas 47 camadas |
| `kernels.c` | as contas: matmul int4, atenção, rmsnorm, RoPE |
| `tokenizer.c` | texto ↔ ids |
| `sample.c` | greedy, temperatura, top-k, top-p |
| `pool.c` | o pool de threads próprio (sem OpenMP) |
| `budget.c` | orçamentos e a recusa |
| `server.c` + `webui.h` | HTTP e a interface web |
| `json.c` | parser JSON próprio |
| `compat.c` | **o único arquivo com `#ifdef` de sistema operacional** |
| `unicode_tables.h` | tabelas geradas para o pré-tokenizador |
| `fixture_kernels.h` | as respostas corretas do `selftest` |

### 17.3 As ferramentas

| ferramenta | o que faz |
|---|---|
| `tools/convert` | BF16 → container int4/int8, com política de precisão por tensor |
| `tools/agent` | o cliente agêntico: seis ferramentas, sandbox, aprovação, modo `--tty` |
| `Tests/ablate` | restaura grupos de tensores ao original e mede o efeito, pareado |
| `Tests/gentest` | qualidade de geração: `rep4`, taxa de parada, `match@ref` |
| `Tests/bench` | tempo, memória e a conta de roofline |
| `ptools/oracle.py` | a implementação de referência em Python |
| `ptools/perplexity.py` | a medida pareada de perplexidade |
| `ptools/gen_*.py` | geradores de fixture e de tabelas Unicode |

> Alguns documentos ainda citam `tools/oracle.py` e `tools/perplexity.py`. Os
> arquivos em Python vivem hoje em **`ptools/`**.

### 17.4 Os 20 documentos

| # | documento | o que decide |
|---|---|---|
| 01 | [Objetivos e escopo](../docs/01-objetivos.md) | o que é e o que explicitamente não é |
| 02 | [Modelo de hardware](../docs/02-hardware.md) | as bandas medidas e a lei de colocação |
| 03 | [Arquitetura](../docs/03-arquitetura.md) | camadas, contratos, os três caches |
| 04 | [Modelo alvo](../docs/04-modelo-alvo.md) | GLM-4.7-Flash e a quantização |
| 05 | [Roadmap](../docs/05-roadmap.md) | marcos com critério de saída |
| 06 | [Lições aprendidas](../docs/06-licoes-aprendidas.md) | adotar, jogar fora, evitar |
| 07 | [Baseline](../docs/07-baseline.md) | os números a bater |
| 08 | [Teto de desempenho](../docs/08-teto-de-desempenho.md) | roofline |
| 09 | [Limites de recursos](../docs/09-limites-de-recursos.md) | orçamentos e a recusa |
| 10 | [Implementação](../docs/10-implementacao.md) | C11, linkagem, portabilidade |
| 11 | [Memória](../docs/11-memoria.md) | zero-copy, KV paginado |
| 12 | [Estado persistente](../docs/12-estado-persistente.md) | subir quente |
| 13 | [Decisões pendentes](../docs/13-decisoes-pendentes.md) | o placar D1–D14 |
| 14 | [Formato do container](../docs/14-formato-container.md) | o `.tootega` byte a byte |
| 15 | [Kernels e numérica](../docs/15-kernels-e-numerica.md) | W4A32, ordem de acumulação |
| 16 | [Tokenizer](../docs/16-tokenizer.md) | BPE, o que implementar e como testar |
| 17 | [Diagnóstico](../docs/17-diagnostico.md) | os bugs, e por que os testes não os pegaram |
| 19 | [Estado de referência](../docs/19-estado.md) | a fotografia do que existe e do que foi medido |
| 20 | [Agente e protocolos](../docs/20-agente.md) | ferramentas, o contrato do CLI, o custo do raciocínio |
| 21 | [KV em int8](../docs/21-kv-int8.md) | medido, inconclusivo, não adotado |

**Leia o 02 primeiro.** Contém a medição que inverte a intuição e da qual quase
todas as outras decisões derivam. Depois o 08, que fixa a expectativa.

---

## 18 — Mal-entendidos comuns

**"É só usar a GPU."**
A GPU desta máquina fala PCIe gen 3: ~12 GB/s. Qualquer peso que não *more* na
VRAM precisa atravessar esse canal, que é mais lento que o disco e cinco vezes
mais lento que a RAM. Medido: 13× pior. → [§8](#8--o-hardware-desta-máquina-e-a-inversão)

**"Modelo de 31 bilhões de parâmetros deve ser 31 vezes mais lento que um de 1 bilhão."**
Não: o que custa é o parâmetro **ativo** por token, que aqui são 3,57 bilhões. E
o custo é de leitura, não de conta.

**"O arquivo tem 19,65 GB, então lê 19,65 GB por token."**
Lê 2,01 GB. O resto do arquivo fica parado.

**"Quantizar para 4 bits deve estragar o modelo."**
Medido: +5,26% de perplexidade com tudo em int4; **−0,19%** (indistinguível de
zero) com a atenção em int8. → [§10.3](#103-quanto-custa-em-qualidade--e-a-surpresa)

**"Os experts se especializam em assuntos."**
Não. A escolha é por token e por camada, e a divisão emergiu do treino sem
significado atribuível.

**"Aumentar o contexto é de graça."**
São **105,8 KB por posição** de contexto no padrão f32, **por conversa**. A 64k isso é
6,8 GB; no teto do modelo, 202.752 posições, são 21,96 GB — mais do que os
pesos. Há as opções `bf16` (52,9 KB) e `int8` (28,1 KB), medidas e disponíveis,
mas nenhuma virou padrão: nenhuma passou no critério fixado antes de medir.
→ [§14.5](#145-a-medição-pareada-e-por-que-ela-existe)

**"Cache de prompt é detalhe de conforto."**
É a maior otimização disponível. Reaproveitar 100k tokens de KV evita um prefill
inteiro; nenhuma otimização de multiplicação chega perto.

**"Um resultado 'quase igual' ao da referência está bom."**
Não passa. O critério é 32/32, binário. Três otimizações boas já foram reprovadas
por 29/32.

**"Mais threads sempre é melhor."**
O trabalho é memory-bound e a CPU é híbrida (8 P-cores + 16 E-cores). Espalhar um
matmul limitado por banda pelos E-cores costuma piorar; o default é `p-cores`.

**"Se não couber na memória, o engine se vira."**
Ele **recusa**, com uma mensagem dizendo o que reduzir. É decisão de projeto, e a
medição que a motivou é a degradação de 13× em silêncio.


**"O agente executa o que o modelo mandou."**
Meio certo, e a metade errada importa. O **modelo** só escreve um pedido; quem
executa é um programa comum, com as permissões de quem o iniciou, dentro de uma
pasta da qual não pode sair. → [§13.4](#134-ferramentas-o-modelo-pede-ele-não-executa)

**"Se ele raciocinar antes, responde melhor."**
Medido neste modelo, com 400 tokens de orçamento: **não termina 8 de 12
perguntas**, repete 8× mais e custa 4,5× o tempo. Para *"qual é a capital do
Brasil"* ele é cortado antes de responder.
→ [§14.6](#146-raciocinar-antes-de-responder-ajuda)

**"Falar o protocolo da OpenAI exige a OpenAI."**
Não exige nada além de implementar o formato, que é público. É justamente por
falá-lo que Copilot, Continue e qualquer outro cliente conectam aqui sem uma
linha de código nossa. → [§13.2](#132-falar-o-protocolo-que-o-mundo-já-fala)

**"O cache guarda as perguntas e as respostas."**
Guarda o **estado interno** dos primeiros N tokens do prompt. Por isso um espaço
a mais no começo invalida tudo dali para a frente, e por isso a ordem do prompt
é regra. → [§13.8](#138-o-cache-de-prefixo-na-prática)

**"Kernel mais rápido, sistema mais rápido."**
Os kernels da GPU ficaram 31× mais rápidos que a primeira versão e 2,6× mais
rápidos que a CPU — e o decode inteiro ficou **mais lento**, três vezes seguidas.
O gargalo estava no caminho entre os dois, não na conta.
→ [§14.7](#147-a-medição-que-decidiu-não-usar-a-gpu)

**"Perplexidade 8,63 é melhor que a perplexidade 12 daquele outro modelo."**
Só se for exatamente o mesmo texto e o mesmo vocabulário. Comparar perplexidades
de origens diferentes é o erro mais comum com essa métrica.
→ [§14.1](#141-perplexidade-quão-surpreso-o-modelo-ficou)

**"Deu quase igual, então pode adotar."**
Se o intervalo de confiança cruza o zero, a medição não distingue melhora de
piora. A resposta é *inconclusivo* — e foi o que aconteceu com o KV em int8,
com a economia de memória toda a favor.
→ [§14.5](#145-a-medição-pareada-e-por-que-ela-existe)

---

## 19 — Índice de termos PT ↔ EN

Para ler as duas versões deste documento em paralelo, ou para procurar um termo
no código (que é escrito em português, mas usa identificadores em inglês).

| português | inglês | seção |
|---|---|---|
| ablação | ablation | [§10.3](#103-quanto-custa-em-qualidade--e-a-surpresa) |
| absorção (do MLA) | absorption | [§6.4](#64-atenção-e-o-cache-kv) |
| afinidade | affinity | [§16.2](#162-os-comandos) |
| agente (o laço) | agent (the loop) | [§13.5](#135-o-agente-é-o-laço-não-o-modelo) |
| amostragem | sampling | [§6.6](#66-lm_head-logits-e-amostragem) |
| aprovação | approval | [§13.5](#135-o-agente-é-o-laço-não-o-modelo) |
| arena de scratch | scratch arena | [§11.2](#112-arena-de-scratch-zero-malloc-por-token) |
| banda (de memória) | (memory) bandwidth | [§2](#ideia-3--o-gargalo-é-a-banda-de-memória-não-a-conta) |
| barreira | barrier | [§11.2](#112-arena-de-scratch-zero-malloc-por-token) |
| branch | branch | [§14.7](#147-a-medição-que-decidiu-não-usar-a-gpu) |
| cabeça (de atenção) | (attention) head | [§5.1](#51-o-modelo) |
| camada | layer | [§6.3](#63-as-47-camadas) |
| camada densa | dense layer | [§6.3](#63-as-47-camadas) |
| cerca (sandbox) | sandbox | [§13.5](#135-o-agente-é-o-laço-não-o-modelo) |
| chamada de ferramenta | tool call | [§13.4](#134-ferramentas-o-modelo-pede-ele-não-executa) |
| colocação | placement | [§8.2](#82-a-armadilha-medida) |
| confiança | confidence | [§14.2](#142-entropia-e-confiança-o-que-a-perplexidade-esconde) |
| CORS | CORS | [§13.2](#132-falar-o-protocolo-que-o-mundo-já-fala) |
| critério de corte | kill criterion | [§15](#15--onde-estamos-e-o-que-vem) |
| critério de saída | exit criterion | [§15](#15--onde-estamos-e-o-que-vem) |
| CRLF | CRLF | [§5.6](#56-o-software-e-o-sistema) |
| CUDA | CUDA | [§14.7](#147-a-medição-que-decidiu-não-usar-a-gpu) |
| degeneração | degeneration | [§14.3](#143-rep4-o-detector-de-repetição) |
| diretiva | directive | [§13.7](#137-o-chat-que-vem-dentro-do-executável) |
| endpoint | endpoint | [§13.1](#131-por-que-existe-um-servidor) |
| entropia | entropy | [§14.2](#142-entropia-e-confiança-o-que-a-perplexidade-esconde) |
| EOF | EOF | [§13.6](#136-como-o-agente-conversa-com-uma-interface) |
| escala (de quantização) | (quantization) scale | [§10.2](#102-como-funciona-em-concreto) |
| especulação | speculative decoding | [§5.2](#52-a-execução) |
| expert compartilhado | shared expert | [§6.5](#65-moe--64-guardados-5-usados) |
| fração de residência | residency fraction | [§9.2](#92-por-que) |
| futex | futex | [§11.2](#112-arena-de-scratch-zero-malloc-por-token) |
| intervalo de confiança | confidence interval | [§14.5](#145-a-medição-pareada-e-por-que-ela-existe) |
| JSON Schema | JSON Schema | [§13.4](#134-ferramentas-o-modelo-pede-ele-não-executa) |
| lei de colocação | placement law | [§8.2](#82-a-armadilha-medida) |
| marco | milestone |  |
| match@ref | match@ref | [§14.4](#144-matchref-quantos-tokens-antes-de-divergir) |
| MCP | MCP | [§13.4](#134-ferramentas-o-modelo-pede-ele-não-executa) |
| medição pareada | paired measurement | [§12.2](#122-medição-pareada--e-o-erro-que-originou-a-regra) |
| metade/metade | half/half | [§10.2](#102-como-funciona-em-concreto) |
| NDJSON | NDJSON | [§13.6](#136-como-o-agente-conversa-com-uma-interface) |
| ordem de redução | reduction order | [§5.4](#54-a-numérica) |
| oráculo | oracle | [§12.1](#121-o-oráculo-e-o-3232) |
| orçamento | budget | [§11.5](#115-orçamentos-o-que-conta-e-o-que-não-conta) |
| partição por camada | layer partition | [§8.3](#83-a-saída-partição-por-camada) |
| perplexidade | perplexity | [§14.1](#141-perplexidade-quão-surpreso-o-modelo-ficou) |
| perplexidade | perplexity | [§12.2](#122-medição-pareada--e-o-erro-que-originou-a-regra) |
| peso | weight | [§5.1](#51-o-modelo) |
| pipe | pipe | [§13.6](#136-como-o-agente-conversa-com-uma-interface) |
| pool de threads | thread pool | [§17.2](#172-os-arquivos-do-engine) |
| processo | process | [§13.1](#131-por-que-existe-um-servidor) |
| protocolo OpenAI | OpenAI protocol | [§13.2](#132-falar-o-protocolo-que-o-mundo-já-fala) |
| rascunho (think) | draft (think) | [§14.6](#146-raciocinar-antes-de-responder-ajuda) |
| rep4 | rep4 | [§14.3](#143-rep4-o-detector-de-repetição) |
| residente | resident | [§9](#9--residência-a-decisão-que-vale-20) |
| resíduo | residual | [§6.3](#63-as-47-camadas) |
| roteador | router / gate | [§6.5](#65-moe--64-guardados-5-usados) |
| SSE | SSE | [§13.3](#133-por-que-a-resposta-aparece-letra-por-letra) |
| stdin e stdout | stdin and stdout | [§13.6](#136-como-o-agente-conversa-com-uma-interface) |
| teto | roofline / ceiling | [§7.2](#72-a-conta) |
| tokenizador | tokenizer | [§6.1](#61-tokenizer--de-texto-para-números) |
| travessia | crossing | [§8.3](#83-a-saída-partição-por-camada) |

---

## Onde ir depois

1. [`docs/02`](../docs/02-hardware.md) — a medição que inverte a intuição;
2. [`docs/08`](../docs/08-teto-de-desempenho.md) — a expectativa de desempenho;
3. [`docs/13`](../docs/13-decisoes-pendentes.md) — o placar das decisões, com o
   raciocínio de cada uma preservado, inclusive das que deram errado;
4. `src/forward.c` — se você quiser ver as 47 camadas em código.
