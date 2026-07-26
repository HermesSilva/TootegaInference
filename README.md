# TootegaEngine

Engine de inferência em **C11 puro, sem dependências**, para modelos MoE que
cabem inteiros na RAM.

Um servidor HTTP compatível com OpenAI, um chat web embutido, um cliente
agêntico com ferramentas locais, e 20 documentos que explicam **por que** cada
decisão é o que é — com o número que a sustenta ao lado.

```powershell
.\build-all.ps1      # engine + ferramentas, MSVC via vswhere
serve.cmd            # sobe o servidor; a UI fica em http://127.0.0.1:8080/
```

---

## O que ele faz

| | |
|---|---|
| **Servidor** | `/v1/chat/completions` com streaming SSE, `tools`/`tool_calls`, CORS. Qualquer cliente OpenAI conecta |
| **Chat web** | embutido no binário, sem arquivo solto: nome, diretivas do modelo e raciocínio escolhidos na entrada |
| **Agente** | `tools/agent.exe` — `Read`, `Write`, `Edit`, `Glob`, `Grep`, `Bash`, com sandbox e aprovação |
| **Cockpit** | o agente fala o contrato de processo do Claude Code CLI: uma UI escrita para aquele CLI funciona trocando o caminho do executável |
| **Terminal** | `agent.exe --tty` — sessão de linha de comando, sem UI nenhuma |
| **Medição** | perplexidade, ablação pareada, qualidade de geração, roofline |

Tudo isso em **13 arquivos `.c`**, ~420 KB de executável, zero DLL de runtime.

## Estado

**M1 completo, em CPU.** Medido nesta máquina, não estimado:

```
banda de memoria    68,7 GB/s         medida em runtime, nao especificada
trafego por token    2,87 GB          o que um decode realmente toca
teto do roofline    24,0 tok/s        banda / trafego

decode quente       71 ms/token       = 13,1 tok/s
ttft medio          2,25 s
prefill             ~11 tok/s
aquecimento         18,54 GB em 5,3 s
RSS                 16,8 GB de pico   35,9 GB livres
servidor ocioso     0,2% de um nucleo, 0,04 GB
```

Dito com o mesmo rigor do resto: o decode está a **55% do teto**. Os ~28 ms que
faltam por token estão na atenção e nas barreiras — não nos matmuls, que já
rodam a 90% da banda.

### O que está provado

| propriedade | resultado |
|---|---|
| forward contra oráculo `transformers` | **32/32 tokens idênticos** |
| determinismo | logits **byte-idênticos** de 1 a 24 threads |
| tokenizer contra o `tokenizers` da HuggingFace | **36/36 casos**, round-trip exato |
| kernels contra a quantização do numpy | **bit a bit** |
| perplexidade contra referência externa | 8,63 contra 8,89 do `transformers` |

### O que não existe

- **Slots e batching** (M3). Um pedido por vez; a segunda aba espera no backlog
  do TCP, em silêncio.
- **Estado persistente** e **especulação MTP**.
- **Build no Linux** — o `Makefile` existe e nunca foi compilado.
- **GPU.** Foi escrita, medida, e **não se pagou**: vive na branch `GPU`.

## A tese, em um parágrafo

Um MoE de 744 B transmitido do disco roda a **0,93 tok/s** nesta máquina, e
escalar hardware nesse regime rende `N^0,25` — quadruplicar RAM dá 1,4×. O
gargalo não é o hardware, é a razão entre o tamanho do modelo e a memória
rápida. Um MoE moderno e enxuto (GLM-4.7-Flash, 19,3 GB em int4) **cabe inteiro
na RAM**, o disco sai do caminho, e o teto passa a ser banda de memória: os
13,1 tok/s medidos, contra 0,93. O TootegaEngine é o engine desenhado para esse
regime — não para esconder disco.

A conta completa está em [08 — Teto de desempenho](docs/08-teto-de-desempenho.md).

## Começando

### 1. Compilar

```powershell
.\build-all.ps1              # tudo, com carimbo de build e detecção de binário obsoleto
.\build-all.ps1 test         # + selftest
.\tootega.exe selftest       # kernels, JSON, pool, tokenizer, sampler
```

Windows é MSVC, achado por `vswhere` — nunca caminho fixo. Linux é o
`Makefile`, com gcc ou clang.

### 2. Obter e converter o modelo

```powershell
python ptools\download_model.py                    # 62,4 GB em BF16, retomável
.\tools\convert.exe models\glm47flash-bf16 `
    --out models\glm47flash-int4\model.tootega     # ~19,3 GB em int4
```

### 3. Usar

```powershell
serve.cmd                                    # servidor + UI web
.\tools\agent.exe --tty                      # agente no terminal
.\tootega.exe chat models\...\model.tootega `
    --tokenizer models\glm47flash-bf16\tokenizer.json
```

O servidor mostra no console o que entra e sai — **forma e estatística, nunca o
conteúdo das mensagens**:

```
[2026-07-26 09:14:02] 127.0.0.1  POST /v1/chat/completions  (3386 bytes)

--- pedido -----------------------------------------------
  conversa    8 mensagens  (papel x quantidade x caracteres)
              sistema 1/1204   usuario 3/71   assistente 3/842   ferramenta 1/2317
  ferramentas 6 declaradas: Read, Write, Edit, Glob, Grep, Bash
  contexto    1172 tokens montados, sobram 148827 posicoes (0.8% da janela)
  prefill     1163 de 1172 do cache (99%), 9 recomputados em 0.74 s
  prompt      perplexidade 3.41  nll 1.2267  entropia 2.18 bits  confianca 47.3%
  chamada     Read -- 1 argumento, 9 caracteres
  geracao     perplexidade 1.84  nll 0.6088  entropia 1.02 bits  confianca 71.9%
  fim         tool_calls -- 11 tokens em 0.8 s (13.1 tok/s)
```

## Decisões que valem ser conhecidas

**MLA absorvido.** O KV guarda 576 valores por token por camada em vez de
10.240 — 18× menos tráfego. Sem isso, 150 mil posições de contexto custariam
254 GB; custam 16,2.

**Ferramentas são tradução, não invenção.** O modelo já traz `<tool_call>`,
`<arg_key>` e `<|observation|>` como **tokens únicos** no vocabulário. O
servidor traduz o protocolo OpenAI para esse formato — e os argumentos são
tipados pelo JSON Schema que o cliente mandou, senão todo argumento numérico
chega como texto do outro lado.

**O servidor é burro, o cliente é esperto.** O laço de ferramentas mora no
agente, não no engine. É o que faz o Copilot e o Continue funcionarem sem uma
linha nossa.

**Raciocínio desligado por padrão.** Medido: com 400 tokens de orçamento, o
rascunho **não termina 8 de 12 prompts**, repete 8× mais 4-gramas, e custa 4,5×
o tempo. Para *"Qual é a capital do Brasil?"* ele escreve seis etapas em inglês
com três rascunhos alternativos — e é cortado antes de responder.

**Nada entra sem medição.** Absorção do MLA, precisão do KV, offload para GPU:
cada um teve critério declarado **antes** de medir. O que foi recusado está
documentado com o número, não com opinião.

## Documentos

Escritos para serem lidos na ordem. Comece pelo **02**: contém a medição que
inverte a intuição usual, e quase todas as outras decisões derivam dela.

| # | documento | o que decide |
|---|---|---|
| 01 | [Objetivos e escopo](docs/01-objetivos.md) | o que é e o que explicitamente não é |
| 02 | [Modelo de hardware](docs/02-hardware.md) | as bandas medidas e a lei de colocação |
| 03 | [Arquitetura](docs/03-arquitetura.md) | camadas, contratos, os três caches |
| 04 | [Modelo alvo](docs/04-modelo-alvo.md) | GLM-4.7-Flash: o que implementar |
| 05 | [Roadmap](docs/05-roadmap.md) | marcos com critério de saída mensurável |
| 06 | [Lições aprendidas](docs/06-licoes-aprendidas.md) | o que adotar, o que jogar fora, o que evitar |
| 07 | [Baseline](docs/07-baseline.md) | os números a bater, medidos aqui |
| 08 | [Teto de desempenho](docs/08-teto-de-desempenho.md) | roofline: quanto dá para extrair |
| 09 | [Limites de recursos](docs/09-limites-de-recursos.md) | orçamentos, e recusar em vez de degradar |
| 10 | [Implementação](docs/10-implementacao.md) | C11, linkagem estática, portabilidade |
| 11 | [Memória](docs/11-memoria.md) | zero-copy, KV paginado, páginas grandes |
| 12 | [Estado persistente](docs/12-estado-persistente.md) | subir quente |
| 13 | [Decisões pendentes](docs/13-decisoes-pendentes.md) | o placar das decisões |
| 14 | [Formato do container](docs/14-formato-container.md) | o `.tootega`, byte a byte |
| 15 | [Kernels e numérica](docs/15-kernels-e-numerica.md) | W4A32, ordem de acumulação |
| 16 | [Tokenizer](docs/16-tokenizer.md) | BPE tiktoken |
| 17 | [Diagnóstico](docs/17-diagnostico.md) | os bugs, e por que os testes não os pegaram |
| 19 | [Estado de referência](docs/19-estado.md) | a fotografia do que existe e do que foi medido |
| 20 | [Agente e protocolos](docs/20-agente.md) | ferramentas, o contrato do CLI, o custo do raciocínio |
| 21 | [KV em int8](docs/21-kv-int8.md) | medido, inconclusivo, não adotado |

**Para leigos:** [ForDummies](ForDummies/) traz o projeto inteiro explicado do
zero, em português e inglês, com um glossário de mais de 200 termos — cada um
com o que a palavra significa **aqui** e o que ela costuma significar por aí.

## Como o projeto trabalha

Três regras que vieram de erro cometido, não de teoria:

**Medição pareada.** Comparar duas configurações pela média global engole o
efeito na variância entre janelas. A primeira medida de perplexidade deu o
**sinal invertido**.

**Mudança de saída não entra em silêncio.** Qualquer alteração algebricamente
equivalente só é adotada com **32/32 tokens idênticos** contra a referência.

**Referência tem de ser externa.** Três testes verdes nascidos da mesma leitura
do problema foram cegos ao mesmo erro — um RoPE na variante errada. A
referência externa custou 2 minutos e resolveu o que dias de teste interno não
resolveram. Está em [17](docs/17-diagnostico.md).

## Máquina de referência

Todos os números destes documentos foram medidos aqui — salvo onde marcado
**[estimativa]**.

```
CPU    Intel i9-14900K, 32 threads, AVX2+FMA
RAM    64 GB DDR5-4800 dual channel      68,7 GB/s efetivos
GPU    RTX 2060 SUPER, 8 GB, sm_75       448 GB/s de VRAM
PCIe   gen 3 x16                         ~12 GB/s   <-- o elo mais lento
Disco  RAID0 NVMe                        18,4 GB/s sequencial
SO     Windows 10 Pro 19045
```

## Estrutura

```
src/         o engine -- C11 ISO, zero dependencias, so compat.c tem #ifdef de SO
tools/       ferramentas de producao: convert, agent
Tests/       ferramentas de medicao: ablate, gentest, bench
ptools/      Python que depende de transformers/numpy como referencia externa
docs/        a especificacao, com o porque e o numero de cada decisao
ForDummies/  o projeto explicado para quem nao e da area, PT e EN
```

## Licença

MIT — veja [LICENSE](LICENSE).

O modelo, os pesos e o tokenizer têm licença própria, do autor deles, e não são
cobertos por esta.
