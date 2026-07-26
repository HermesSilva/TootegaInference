# 20 — Agente, ferramentas e protocolos

O M4 de [05](05-roadmap.md) pedia um "CLI agêntico". O desenho mudou antes da
primeira linha, e a mudança é a coisa mais importante deste documento.

## A decisão: o servidor é burro, o cliente é esperto

O plano original punha o laço de ferramentas **dentro do engine**. Está errado
para quem quer interoperar: um cliente de mercado — Copilot, Continue, Cline —
não chama laço nosso. Ele manda `tools` no pedido, espera `tool_calls` de
volta, **executa ele mesmo** e devolve `role:"tool"` no turno seguinte.

```
Copilot / Cline / qualquer cliente OpenAI
        |  tools, tool_calls, role:tool
        v
  tootega.exe serve        <-- traduz para o template do modelo, e so
        ^
        |  o MESMO protocolo
  tools/agent.exe          <-- o laço mora aqui
        ^
        |  stream-json no stdin/stdout
     Cockpit (VS Code)
```

O servidor fica com uma responsabilidade só: **traduzir o protocolo aberto para
o template deste modelo**. Todo cliente que fala OpenAI funciona sem uma linha
nossa. O nosso agente vira só mais um desses clientes.

## O que o modelo já sabe fazer

Nada foi inventado. `chat_template.jinja` e o vocabulário já trazem o formato:

```
154829 <|observation|>   154843 <tool_call>      154844 </tool_call>
154845 <tool_response>   154846 </tool_response> 154847 <arg_key>
154848 </arg_key>        154849 <arg_value>
```

São **tokens únicos**. Detectar chamada é comparar id, não casar string — imune
a espaço, acento e ao modelo escrever `<tool_call>` no meio de uma frase.

### A tradução, campo a campo

| OpenAI | prompt do modelo |
|---|---|
| `tools: [...]` | bloco `<|system|># Tools … <tools>{json}</tools>` no **início** do prefixo |
| `tool_calls` do assistente | `<tool_call>nome<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` |
| `role:"tool"` | `<|observation|><tool_response>…</tool_response>` |
| `finish_reason:"tool_calls"` | geração parou no token `</tool_call>` |

Duas decisões dentro dessa tradução:

**O bloco de ferramentas vai primeiro.** É a parte que não muda entre turnos, e
a regra de ordenação de [03](03-arquitetura.md) — estável primeiro — é o que
faz o cache de prefixo valer alguma coisa numa sessão de agente.

**`</tool_call>` interrompe a geração.** Esperar o EOS depois da chamada é
queimar tokens a 13 tok/s para receber texto que o cliente descarta.

**Os argumentos são tipados pelo schema do cliente.** O modelo emite
`<arg_value>3</arg_value>` sem dizer se é o número 3 ou a string "3"; OpenAI
exige `arguments` como JSON tipado. Sem consultar `parameters.properties`, toda
ferramenta com argumento numérico ou booleano chega errada do outro lado.

Medido de ponta a ponta, ida e volta completa:

```
turno 1   finish_reason=tool_calls   content=""
          {"name":"get_weather","arguments":"{\"city\":\"Curitiba\",\"days\":3}"}
turno 2   16,5 s  167 tokens  cache 185/252  resposta em tabela, correta
```

`days` chegou como número, `city` como string — o schema decidiu.

## Portão do M4 — o modelo aguenta um laço?

Antes de escrever ferramenta de verdade, a mesma disciplina do portão da GPU:
critério declarado **antes** de medir (`Tests/toolgate.ps1`).

> ≥ 8/10 segue · 5–7 segue com ressalva · ≤ 4 o agente é fachada

Dez pedidos, três ferramentas falsas, raciocínio desligado, `temperature 0`:

```
resultado: 9/10 (90%)   PASSOU

 # esperado    chamou      ok  tokens   seg argumentos
 1 read_file   read_file   sim     13 22,70 {"path":"src/pool.c"}
 2 read_file   read_file   sim     18 22,90 {"path":"README.md","max_lines":40}
 3 search_code search_code sim     29 23,10 {"pattern":"tt_futex_wait"}
 4 search_code search_code sim     24 23,20 {"pattern":"TODO","glob":"*.c","case_sensitive":false}
 5 run_command run_command sim     11 21,60 {"command":"git status"}
 6 run_command run_command sim     47 24,60 {"command":"powershell -File build.ps1","timeout_ms":300000}
 7 run_command read_file   NAO     25 23,30 {"path":"docs/03-arquitetura.md","max_lines":1000}
 8 read_file   read_file   sim     13 21,90 {"path":"CLAUDE.md"}
 9 search_code search_code sim     24 22,40 {"pattern":"malloc"}
10 (nenhuma)   (nenhuma)   sim     60 25,70
```

**Zero chamadas mal formadas.** Todo `arguments` parseou e trouxe o argumento
obrigatório. O caso 10 é o mais informativo: com ferramentas disponíveis e uma
pergunta que não pede nenhuma, o modelo **não chamou** — não há gatilho fácil.

O único erro (caso 7, "quantas linhas tem o arquivo?") escolheu `read_file` em
vez de `run_command`. É defensável: ler o arquivo responde a pergunta.

Ressalva honesta: são ferramentas falsas e um turno só. Isto mede **formato e
escolha**, não a capacidade de sustentar oito turnos com resultado real
voltando. Esse número só existe depois da fase B.

## O contrato de processo do Claude Code CLI

A UI alvo é o [Cockpit](../../Cockpit) — extensão do VS Code, autoria do
Hermes. Ela não fala HTTP com o motor: fala **NDJSON pelo stdin/stdout** de um
processo que ela mesma dispara.

A especificação **não é a nossa leitura do protocolo**: é o parser dela,
congelado contra fixtures reais em `shared/events.ts` e
`src/cli/CliProcessManager.ts`. Escrever contra referência externa é a lição de
[17](17-diagnostico.md) — três testes verdes nascidos da mesma leitura foram
cegos ao mesmo erro.

Escolha registrada: **o nosso executável finge ser o `claude`**, a ponto de
bastar trocar o caminho nas configurações. Mais duro para nós, mas mantém o
princípio fundador da UI intacto — ela não sabe que trocou de motor.

### Estado por evento

| evento | fase A | observação |
|---|---|---|
| `--version` com exit 0 | **ok** | `detect()` reprova o executável inteiro sem isso |
| `system/init` | **ok** | session_id, cwd, model, permissionMode |
| `stream_event` `message_start`/`stop` | **ok** | |
| `content_block_delta` `text_delta` | **ok** | |
| `content_block_delta` `thinking_delta` | **ok** | fronteira detectada no `</think>` |
| `assistant` completo | **ok** | blocos `thinking` + `text` |
| `result` com usage | **ok** | `total_cost_usd: 0` — verdade, e é o argumento |
| `control_response` do `initialize` | **ok** | silêncio deixaria a UI no timeout |
| `tool_use` / `tool_result` | **ok** | fase B |
| `can_use_tool` (aprovação) | **ok** | bloqueia em stdin; seguro porque a geração já parou |
| `interrupt` durante a geração | não | exigiria uma thread lendo stdin |
| `--resume`, sessões em disco | fase C | depende de [12](12-estado-persistente.md) |
| `rate_limit_event` | nunca | não há cota de assinatura |

Medido com `tools/agent/smoke.ps1` contra o engine real:

```
54 eventos em 4,2 s
  system 1 · stream_event 50 · assistant 1 · result 1 · control_response 1
  message_start 1 · content_block_start 1 · text_delta 45 · content_block_stop 1
  message_delta 1 · message_stop 1
contrato ok
```

O smoke test não valida a resposta do modelo — valida o **envelope**. A UI só
renderiza o que reconhece, e campo com nome errado não dá erro: dá tela vazia.

## Fase B — as mãos

Seis ferramentas, com os **nomes do Claude Code** de propósito: a UI renderiza
diff, saída de shell e cartão de busca *por nome*. Nome nosso cairia no
renderizador genérico e a UI pareceria quebrada sem estar.

| ferramenta | muda estado | observação |
|---|---|---|
| `Read` | não | linhas numeradas, `offset`/`limit` |
| `Glob` | não | `*` não atravessa `/`, `**` atravessa |
| `Grep` | não | **texto literal, não regex** — e o schema diz isso ao modelo |
| `Write` | sim | pede aprovação |
| `Edit` | sim | trecho exato; **ambíguo é recusado**, não aplicado no primeiro |
| `Bash` | sim | `CreateProcess` com limite de tempo e `TerminateProcess` |

Três regras que não são cortesia:

**Sandbox acima de tudo.** Todo caminho é resolvido contra `--cwd` e recusado
se escapar — **inclusive com `--yes`**. É a única coisa entre um modelo de 31 B
e o resto do disco.

**Aprovação só para o que muda estado.** Pedir confirmação para tudo, a
13 tok/s, transforma a tarefa em meia hora de Enter — e treina o usuário a
aprovar sem ler, que é o pior resultado possível.

**Teto de saída por ferramenta (12 kB).** Um `Read` sem limite ou um build
inteiro despejam 50 k tokens e o contexto acaba no segundo passo.

### Medido, ponta a ponta

Leitura (nenhuma aprovação pedida, correto):

```
Read {"file_path":"docs/03-arquitetura.md"}
result: success  passos=2  in=3425 out=18   246 s
resposta: 126
```

Comando com aprovação (`perm_1` pedido, respondido `allow`, executado):

```
Bash {"command":"git status --short"}
aprovacao: Bash perm_1
result: success  passos=2  in=1269 out=60   53 s
resposta: "...listando 16 itens: 11 arquivos modificados e 5 nao rastreados."
```

**O mecanismo funciona e a resposta está certa** — o arquivo tem mesmo 126
linhas.

Fica registrado o que aconteceu na primeira conferência, porque a lição é
minha e não do modelo: dei a resposta como errada comparando com
`Measure-Object -Line`, que **não conta linhas em branco** e devolveu 97. O
modelo acertou; a régua é que estava torta. É a mesma armadilha de
[17](17-diagnostico.md) invertida — lá o teste concordava com o erro, aqui o
teste acusou um erro que não existia.

E o custo de contexto aparece inteiro: ler um arquivo de 3,4 k tokens custou
**246 s**, quase tudo em prefill a ~11 tok/s. Ferramenta que devolve muito é o
gargalo real deste agente, não a geração.

## O prompt de sistema, e o que ele vale

O agente mandava o schema das ferramentas e mais nada: o modelo não sabia que
era um agente, em que pasta estava, nem que o shell é o `cmd.exe`. Um prompt de
sistema entrou no início do histórico — parte estável, portanto dentro do cache
de prefixo.

Mesma pergunta, antes e depois:

| | passos | contexto | tempo |
|---|---|---|---|
| sem prompt de sistema | 4 | 5.961 tokens | 291 s |
| com prompt de sistema | **2** | **1.870** | **73 s** |

**4× mais rápido, e a mesma resposta certa.** Sem o prompt o modelo tentou
`wc -l` (não existe no `cmd.exe`), errou, tentou de novo, e acabou lendo o
arquivo inteiro — 3,4 k tokens de contexto para produzir um número. Com o
prompt foi direto a `powershell -Command "(Get-Content ...).Count"`.

As regras que estão lá saem todas de medição, não de estilo: preferir `Grep` a
ler arquivo inteiro, usar `Bash` para contar, e a advertência sobre o shell do
Windows.

## Modo terminal

`--tty` troca o NDJSON por uma sessão de gente: resposta em streaming, cartão
por ferramenta, aprovação `[s/N]` no console.

```
> Quantas linhas tem o arquivo docs/03-arquitetura.md?

  * Bash {"command":"powershell -Command \"(Get-Content 'docs/...').Count\""}

  ! Bash quer executar:
    {"command":"powershell -Command \"(Get-Content 'docs/...').Count\""}
    permitir? [s/N] s
    -> 2 linhas  126

O arquivo `docs/03-arquitetura.md` tem **126 linhas**.

  2 passos  48 tokens gerados  1870 de contexto  73.4 s
```

O rascunho não aparece e a saída das ferramentas vira uma linha de resumo: o
que interessa na tela é a resposta, e despejar 12 kB de `Grep` enterraria ela.

### O que a UI vai mostrar vazio

O Cockpit tem um lado inteiro que é *Claude*, não *CLI*: `StatuslineInstaller`,
`UsageApi` (OAuth), `ModelPricing`, `PluginManager`, `SkillBodyIndex`,
`McpInventory`, `OtelReceiver`. Contra o nosso executável esses painéis ficam
vazios, e o §4.4 do catálogo dela — custo e limites, "coração do produto" —
mostra zero.

Não é falha: rodar local não custa nem tem cota. O medidor que continua fazendo
sentido é o de **contexto**, e esse alimentamos bem.

## O custo de desligar o raciocínio — medido

`</think>` fechado antes de começar não é truque: é a mesma construção que o
template usa para turnos de assistente já concluídos, então o modelo a viu no
treino.

`Tests/gentest.exe --think` / `--no-think`, 12 prompts, amostragem gulosa:

| orçamento | | rep4 | **parou** | tempo |
|---|---|---|---|---|
| 96 tokens | com raciocínio | 5,3% | **0/12** | — |
| | sem | 0,9% | 9/12 | — |
| 400 tokens | com raciocínio | **13,8%** | **4/12** | 354 s |
| | sem | 1,7% | **12/12** | 78 s |

**Com 400 tokens o raciocínio ainda não termina 8 dos 12 prompts**, repete 8×
mais 4-gramas — 13,8% encosta no limiar de degeneração de 15% — e custa 4,5×
o tempo.

O texto explica o número. Para *"Qual é a capital do Brasil? Responda em uma
frase"*:

```
1. **Analyze the Request:** ... Constraint: "Responda em uma frase."
2. **Identify the Fact:** The capital of Brazil is Brasília.
3. **Draft the Response (Internal Monologue/Drafting):**
    *Draft 1:* A capital do Brasil é Brasília.
    *Draft 2:* Brasília é a capital do Brasil.
    *Draft 3:* A cidade de Brasília é a capital do Brasil.
4. **Select the Best Option:** ...
5. **Final Polish (Portuguese):** ...
6. **Check Constraints:** Is it one sente          <- cortado aqui
```

Seis etapas em inglês e três rascunhos alternativos para um fato de uma
palavra, e cortado antes de responder. O rascunho deste modelo é
performático — enumera, escreve versões, se autoavalia — em vez de calcular.

Duas ressalvas: os 12 prompts são curtos e diretos, que é o regime em que
raciocinar é conhecido por não ajudar; e isto mede a *implementação* de
raciocínio deste modelo, não a ideia. Falta o teste oposto — problema de
vários passos com estado intermediário — e escolher esse conjunto agora, depois
de ver o resultado, seria escolher o teste pela resposta.

Para chat e agente, o default **desligado** está certo, e agora com número.

Uma nota de método que caiu junto: até esta medição o `gentest` tinha `<think>`
**fixo no código**. Todos os números de qualidade de geração do projeto eram
com raciocínio, enquanto o chat e o agente rodam sem — medindo um regime e
entregando outro.

## O que ainda não existe

- **`interrupt` durante a geração.** O botão de parar só age entre passos; para
  abortar no meio do decode falta uma thread lendo stdin.
- **`tool_calls` no streaming em pedaços** — o `input_json_delta` do protocolo
  exige JSON parcial válido, e o modelo emite `<arg_key>` sem tipo. Hoje a
  chamada sai inteira no quadro final; o card aparece completo em vez de
  crescer.
- **Sessões e `--resume`** (fase C), presas ao estado persistente de
  [12](12-estado-persistente.md).
- **MCP.** "Protocolo comum de ferramentas" hoje é MCP, e ele não está aqui.
- **Meta do M4**: acerto de cache de prefixo ≥ 90% em sessão multi-turno. Só
  mensurável com o laço real rodando.
