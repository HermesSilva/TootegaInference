# 20 — Agent, tools and protocols

M4 in [05](05-roadmap.md) asked for an "agentic CLI". The design changed before
the first line was written, and that change is the most important thing in this
document.

## The decision: the server is dumb, the client is smart

The original plan put the tool loop **inside the engine**. That is wrong for
anyone who wants to interoperate: an off-the-shelf client — Copilot, Continue,
Cline — does not call a loop of ours. It sends `tools` in the request, expects
`tool_calls` back, **executes them itself** and returns `role:"tool"` on the next
turn.

```
Copilot / Cline / any OpenAI client
        |  tools, tool_calls, role:tool
        v
  tootega.exe serve        <-- translates to the model's template, and that is all
        ^
        |  the SAME protocol
  tools/agent.exe          <-- the loop lives here
        ^
        |  stream-json over stdin/stdout
     Cockpit (VS Code)
```

The server keeps a single responsibility: **translating the open protocol into
this model's template**. Every client that speaks OpenAI works with no code of
ours. Our own agent becomes just one more of those clients.

## What the model already knows how to do

Nothing was invented. `chat_template.jinja` and the vocabulary already carry the
format:

```
154829 <|observation|>   154843 <tool_call>      154844 </tool_call>
154845 <tool_response>   154846 </tool_response> 154847 <arg_key>
154848 </arg_key>        154849 <arg_value>
```

They are **single tokens**. Detecting a call is comparing an id, not matching a
string — immune to whitespace, accents, and to the model writing `<tool_call>` in
the middle of a sentence.

### The translation, field by field

| OpenAI | model prompt |
|---|---|
| `tools: [...]` | a `<|system|># Tools … <tools>{json}</tools>` block at the **start** of the prefix |
| the assistant's `tool_calls` | `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` |
| `role:"tool"` | `<|observation|><tool_response>…</tool_response>` |
| `finish_reason:"tool_calls"` | generation stopped at the `</tool_call>` token |

Two decisions inside that translation:

**The tools block goes first.** It is the part that does not change between
turns, and the ordering rule from [03](03-architecture.md) — stable first — is
what makes the prefix cache worth anything in an agent session.

**`</tool_call>` stops generation.** Waiting for EOS after the call is burning
tokens at 13 tok/s to receive text the client throws away.

**Arguments are typed by the client's schema.** The model emits
`<arg_value>3</arg_value>` without saying whether it is the number 3 or the
string "3"; OpenAI requires `arguments` as typed JSON. Without consulting
`parameters.properties`, every tool with a numeric or boolean argument arrives
wrong on the other side.

Measured end to end, full round trip:

```
turn 1    finish_reason=tool_calls   content=""
          {"name":"get_weather","arguments":"{\"city\":\"Curitiba\",\"days\":3}"}
turn 2    16.5 s  167 tokens  cache 185/252  answer in a table, correct
```

`days` arrived as a number, `city` as a string — the schema decided.

## The M4 gate — can the model sustain a loop?

Before writing real tools, the same discipline as the GPU gate: criterion
declared **before** measuring (`Tests/toolgate.ps1`).

> ≥ 8/10 proceed · 5–7 proceed with caveats · ≤ 4 the agent is a façade

Ten requests, three fake tools, reasoning off, `temperature 0`:

```
result: 9/10 (90%)   PASSED

 # expected     called      ok  tokens   sec arguments
 1 read_file    read_file   yes     13 22.70 {"path":"src/pool.c"}
 2 read_file    read_file   yes     18 22.90 {"path":"README.md","max_lines":40}
 3 search_code  search_code yes     29 23.10 {"pattern":"tt_futex_wait"}
 4 search_code  search_code yes     24 23.20 {"pattern":"TODO","glob":"*.c","case_sensitive":false}
 5 run_command  run_command yes     11 21.60 {"command":"git status"}
 6 run_command  run_command yes     47 24.60 {"command":"powershell -File build.ps1","timeout_ms":300000}
 7 run_command  read_file   NO      25 23.30 {"path":"docs/03-architecture.md","max_lines":1000}
 8 read_file    read_file   yes     13 21.90 {"path":"CLAUDE.md"}
 9 search_code  search_code yes     24 22.40 {"pattern":"malloc"}
10 (none)       (none)      yes     60 25.70
```

**Zero malformed calls.** Every `arguments` parsed and carried the required
argument. Case 10 is the most informative: with tools available and a question
that needs none, the model **did not call** — there is no easy trigger.

The single error (case 7, "how many lines does the file have?") chose `read_file`
instead of `run_command`. That is defensible: reading the file answers the
question.

An honest caveat: these are fake tools and a single turn. This measures **format
and choice**, not the ability to sustain eight turns with real results coming
back. That number only exists after phase B.

## The Claude Code CLI process contract

The target UI is Cockpit — a VS Code extension, written by Hermes. It does not
speak HTTP to the engine: it speaks **NDJSON over the stdin/stdout** of a process
it launches itself.

The specification is **not our reading of the protocol**: it is that
extension's parser, frozen against real fixtures. The two files that define it,
`shared/events.ts` and `src/cli/CliProcessManager.ts`, live **in Cockpit's
repository, not in this one** — nothing here is built from them, and no path
below refers to a file in this tree. Writing against an external reference is the
lesson of [17](17-diagnosis.md) — three green tests born of the same reading were
blind to the same error.

A recorded choice: **our executable pretends to be `claude`**, to the point where
changing the path in the settings is enough. Harder for us, but it keeps the UI's
founding principle intact — it does not know it changed engines.

### State per event

| event | phase A | note |
|---|---|---|
| `--version` with exit 0 | **ok** | `detect()` rejects the whole executable without it |
| `system/init` | **ok** | session_id, cwd, model, permissionMode |
| `stream_event` `message_start`/`stop` | **ok** | |
| `content_block_delta` `text_delta` | **ok** | |
| `content_block_delta` `thinking_delta` | **ok** | boundary detected at `</think>` |
| full `assistant` | **ok** | `thinking` + `text` blocks |
| `result` with usage | **ok** | `total_cost_usd: 0` — true, and that is the point |
| `control_response` to `initialize` | **ok** | silence would leave the UI at a timeout |
| `tool_use` / `tool_result` | **ok** | phase B |
| `can_use_tool` (approval) | **ok** | blocks on stdin; safe because generation has already stopped |
| `interrupt` during generation | no | it would need a thread reading stdin |
| `--resume`, sessions on disk | phase C | depends on [12](12-persistent-state.md) |
| `rate_limit_event` | never | there is no subscription quota |

Measured with `tools/agent/smoke.ps1` against the real engine:

```
54 events in 4.2 s
  system 1 · stream_event 50 · assistant 1 · result 1 · control_response 1
  message_start 1 · content_block_start 1 · text_delta 45 · content_block_stop 1
  message_delta 1 · message_stop 1
contract ok
```

The smoke test does not validate the model's answer — it validates the
**envelope**. The UI only renders what it recognises, and a field with the wrong
name does not raise an error: it produces a blank screen.

## Phase B — the hands

Six tools, deliberately with **Claude Code's names**: the UI renders diffs, shell
output and search cards *by name*. A name of our own would fall through to the
generic renderer and the UI would look broken without being broken.

| tool | changes state | note |
|---|---|---|
| `Read` | no | numbered lines, `offset`/`limit` |
| `Glob` | no | `*` does not cross `/`, `**` does |
| `Grep` | no | **literal text, not regex** — and the schema says so to the model |
| `Write` | yes | asks for approval |
| `Edit` | yes | exact snippet; **an ambiguous match is refused**, not applied to the first |
| `Bash` | yes | `CreateProcess` with a time limit and `TerminateProcess` |

Three rules that are not courtesy:

**Sandbox above everything.** Every path is resolved against `--cwd` and refused
if it escapes — **including with `--yes`**. It is the only thing between a 31 B
model and the rest of the disk.

**Approval only for what changes state.** Asking for confirmation on everything,
at 13 tok/s, turns the task into half an hour of pressing Enter — and trains the
user to approve without reading, which is the worst possible outcome.

**A per-tool output ceiling (12 kB).** An unbounded `Read` or a whole build dumps
50 k tokens and the context is gone by the second step.

### Measured, end to end

A read (no approval requested, correct):

```
Read {"file_path":"docs/03-architecture.md"}
result: success  steps=2  in=3425 out=18   246 s
answer: 126
```

A command with approval (`perm_1` requested, answered `allow`, executed):

```
Bash {"command":"git status --short"}
approval: Bash perm_1
result: success  steps=2  in=1269 out=60   53 s
answer: "...listing 16 items: 11 modified files and 5 untracked."
```

**The mechanism works and the answer is right** — the file really does have 126
lines.

What happened on the first check is on record, because the lesson is mine and not
the model's: I marked the answer wrong by comparing against `Measure-Object
-Line`, which **does not count blank lines** and returned 97. The model was
right; the ruler was bent. It is the same trap as [17](17-diagnosis.md)
inverted — there the test agreed with the error, here the test accused an error
that did not exist.

And the context cost shows up in full: reading a 3.4 k-token file cost **246 s**,
almost all of it prefill at ~11 tok/s. A tool that returns a lot is this agent's
real bottleneck, not generation.

## The system prompt, and what it is worth

The agent used to send the tool schema and nothing else: the model did not know
it was an agent, which folder it was in, or that the shell is `cmd.exe`. A system
prompt went in at the start of the history — a stable part, therefore inside the
prefix cache.

Same question, before and after:

| | steps | context | time |
|---|---|---|---|
| without a system prompt | 4 | 5,961 tokens | 291 s |
| with a system prompt | **2** | **1,870** | **73 s** |

**4× faster, and the same correct answer.** Without the prompt the model tried
`wc -l` (which does not exist in `cmd.exe`), failed, tried again, and ended up
reading the whole file — 3.4 k tokens of context to produce one number. With the
prompt it went straight to `powershell -Command "(Get-Content ...).Count"`.

The rules in there all come from measurement, not from style: prefer `Grep` over
reading a whole file, use `Bash` to count, and the warning about the Windows
shell.

## Terminal mode

`--tty` swaps NDJSON for a human session: streamed answer, one card per tool,
`[y/N]` approval at the console.

```
> How many lines does docs/03-architecture.md have?

  * Bash {"command":"powershell -Command \"(Get-Content 'docs/...').Count\""}

  ! Bash wants to run:
    {"command":"powershell -Command \"(Get-Content 'docs/...').Count\""}
    allow? [y/N] y
    -> 2 lines  126

The file `docs/03-architecture.md` has **126 lines**.

  2 steps  48 tokens generated  1870 of context  73.4 s
```

The draft does not appear and tool output becomes a one-line summary: what
matters on screen is the answer, and dumping 12 kB of `Grep` would bury it.

### What the UI will show empty

Cockpit has a whole side that is *Claude*, not *CLI*: `StatuslineInstaller`,
`UsageApi` (OAuth), `ModelPricing`, `PluginManager`, `SkillBodyIndex`,
`McpInventory`, `OtelReceiver`. Against our executable those panels stay empty,
and §4.4 of its catalogue — cost and limits, "the heart of the product" — shows
zero.

That is not a failure: running locally costs nothing and has no quota. The meter
that still makes sense is **context**, and we feed that one well.

## The cost of turning reasoning off — measured

A `</think>` closed before starting is not a trick: it is the same construction
the template uses for already-completed assistant turns, so the model saw it in
training.

`Tests/gentest.exe --think` / `--no-think`, 12 prompts, greedy sampling:

| budget | | rep4 | **stopped** | time |
|---|---|---|---|---|
| 96 tokens | with reasoning | 5.3% | **0/12** | — |
| | without | 0.9% | 9/12 | — |
| 400 tokens | with reasoning | **13.8%** | **4/12** | 354 s |
| | without | 1.7% | **12/12** | 78 s |

**With 400 tokens, reasoning still fails to finish 8 of the 12 prompts**, repeats
8× more 4-grams — 13.8% brushes the 15% degeneration threshold — and costs 4.5×
the time.

The text explains the number. For *"What is the capital of Brazil? Answer in one
sentence"*:

```
1. **Analyze the Request:** ... Constraint: "Answer in one sentence."
2. **Identify the Fact:** The capital of Brazil is Brasília.
3. **Draft the Response (Internal Monologue/Drafting):**
    *Draft 1:* The capital of Brazil is Brasília.
    *Draft 2:* Brasília is the capital of Brazil.
    *Draft 3:* The city of Brasília is the capital of Brazil.
4. **Select the Best Option:** ...
5. **Final Polish:** ...
6. **Check Constraints:** Is it one sente          <- cut off here
```

Six steps and three alternative drafts for a one-word fact, and cut off before
answering. This model's draft is performative — it enumerates, writes versions,
grades itself — instead of computing.

Two caveats: the 12 prompts are short and direct, which is the regime where
reasoning is known not to help; and this measures this model's *implementation*
of reasoning, not the idea. The opposite test is missing — a multi-step problem
with intermediate state — and choosing that set now, after seeing the result,
would be choosing the test by its answer.

For chat and for the agent, the **off** default is right, and now it has a number
behind it.

A method note that fell out of this: until this measurement, `gentest` had
`<think>` **hard-coded**. Every generation-quality number in the project was
measured with reasoning on, while chat and the agent run with it off — measuring
one regime and shipping another.

## What still does not exist

- **`interrupt` during generation.** The stop button only acts between steps; to
  abort mid-decode we need a thread reading stdin.
- **Chunked `tool_calls` in streaming** — the protocol's `input_json_delta`
  requires valid partial JSON, and the model emits `<arg_key>` untyped. Today the
  call comes out whole in the final frame; the card appears complete instead of
  growing.
- **Sessions and `--resume`** (phase C), tied to the persistent state of
  [12](12-persistent-state.md).
- **MCP.** "The common tool protocol" today is MCP, and it is not here.
- **The M4 target**: prefix cache hit rate ≥ 90% in a multi-turn session. Only
  measurable with the real loop running.
