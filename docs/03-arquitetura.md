# 03 — Arquitetura

## Camadas

```
┌─ CLI agêntico ────────────────────────────────────────────┐
│  monta o prompt, chama ferramentas, ordena o contexto     │
│  ÚNICO que conhece semântica (system / tools / histórico) │
└────────────────────────┬──────────────────────────────────┘
                         │  tokens + slot
┌─ Camada de serviço ────┴──────────────────────────────────┐
│  HTTP, sessões, admissão, qual sessão ocupa qual slot     │
└────────────────────────┬──────────────────────────────────┘
                         │  SUBMIT(slot, tokens, params)
┌─ Engine ───────────────┴──────────────────────────────────┐
│  planner de colocação · forward · KV · kernels            │
│  vê inteiros. Não sabe o que é um "system prompt".        │
└───────────────────────────────────────────────────────────┘
```

O contrato entre serviço e engine é deliberadamente estreito: **dê-me tokens; eu
reuso o maior prefixo idêntico e não recomputo o que já sei.** Nada mais.

## Os três caches e seus donos

"Cache" são três coisas diferentes neste sistema. Misturá-las é a origem da
confusão de projeto.

| cache | guarda | dono |
|---|---|---|
| **pesos** | tensores por tier (VRAM / RAM) | **engine**, exclusivamente |
| **KV** | estado de atenção por token já processado | engine tem o **mecanismo**; agente determina se **acerta** |
| **prefixo do prompt** | a sequência que você monta | **agente**, exclusivamente |

O cache de pesos é invisível ao agente e deve continuar assim.

O de prefixo é 100% do agente: o engine vê inteiros, não sabe o que é um system
prompt. **O engine não tem como consertar uma ordenação ruim.**

O KV é onde a fronteira importa. O engine oferece casamento de prefixo token a
token que para no primeiro divergente e trunca dali em diante. A consequência é
dura: **tudo depois do primeiro token alterado é recomputado**. Um timestamp no
começo do system prompt invalida os 100k tokens seguintes, e o engine não tem
como saber que aquilo era irrelevante.

Daí a regra que o CLI tem de seguir, e que é o desenho todo:

> **Estável primeiro, volátil por último.**
> `system → schemas de ferramentas → contexto longo → histórico → turno atual`

Nunca um relógio, um UUID ou uma listagem de diretório na frente.

Duas decisões que vêm da API da Anthropic e valem copiar:

- **Ferramentas renderizam na posição 0.** Trocar, adicionar ou reordenar uma
  ferramenta invalida tudo — o pior assassino de cache num CLI agêntico com
  ferramentas dinâmicas. O engine precisa de um caminho de *append*, não de
  troca.
- **Instrução que chega no meio não pode editar a frente.** Um canal para
  injetar estado tardio (modo mudou, arquivo mudou no disco) **no fim** do
  prefixo, nunca no começo.

## Módulos do engine

| módulo | responsabilidade | notas |
|---|---|---|
| `model/` | carregar safetensors, config → IR | mmap único, zero cópia; sem `n_group` cravado |
| `plan/` | medir bandas, resolver colocação **dentro dos orçamentos** | roda na init; ver [02](02-hardware.md) e [09](09-limites-de-recursos.md) |
| `budget/` | tetos de RAM/VRAM/threads, contabilidade, watchdog | teto duro verificado em execução, não só no plano |
| `kernel/cpu/` | matmul int4, atenção MLA, rmsnorm | AVX-VNNI (`dp4a` equivalente), OpenMP |
| `kernel/gpu/` | os mesmos, sm_75 | opcional; DLL carregada em runtime |
| `sched/` | slots de KV, batching contínuo | onde vive a meta de throughput agregado |
| `kv/` | alocação, casamento de prefixo, truncamento | MLA comprimido: 576 valores/token/camada |
| `serve/` | HTTP, protocolo | fino de propósito |

## A aposta central: partição por camada

Do [modelo de hardware](02-hardware.md): pesos são grandes, ativações são
minúsculas (8 KB com `hidden=2048`). Portanto **cada camada tem seus pesos
permanentemente residentes no device que a computa**, e entre camadas cruzam
8 KB de ativação.

```
camadas 0..k     -> pesos residentes em VRAM, computadas na GPU
    [fronteira: 8 KB de ativação atravessam o PCIe]
camadas k+1..46  -> pesos residentes em RAM, computadas na CPU
```

`k` sai do planner: o maior `k` cujos pesos cabem na VRAM utilizável, menos a
reserva de workspace e KV.

Isso é o oposto do que o colibrì faz (mover *pesos* para a GPU sob demanda) e é
consequência direta do PCIe gen 3. **Marcado como hipótese**: o M2 existe para
validá-la contra o CPU-only do M1, e o critério de corte está em
[01-objetivos.md](01-objetivos.md).

## Estruturas de dados: as regras

Derivadas do que medimos no colibrì (ver [06](06-licoes-colibri.md)):

**Movimento de bloco grande é troca de ponteiro, nunca cópia.** O colibrì acerta
isso com swap de struct (`colibri.c:3493`) e slab reciclado por capacidade
(`:1527`). Manter.

**Scratch é arena persistente, não `malloc` por token.** O colibrì faz ~500
`malloc`/`free` por token em buffers de tamanho fixo conhecidos na init. A 1 tok/s
não custa nada; a 50 tok/s vira percentual. Todos os buffers de forward saem de
uma arena dimensionada uma vez.

**Env lido uma vez, na init.** O colibrì chama `getenv` dentro do laço de camadas
(`colibri.c:3505`, `:4158`) — ~300 varreduras de `environ` por token. Config vira
struct imutável na init.

**Busca de expert é O(1).** Varredura linear sobre o pin set custa ~150k
comparações por token no colibrì. Com residência total o problema quase some,
mas o índice é `layer × expert → slot` direto, sem busca.

## Correção

- **Oráculo `transformers`, teacher-forcing, token-exato.** Bloqueia merge.
- **A colocação não pode alterar a saída.** Rodar o mesmo prompt em CPU-only,
  GPU-parcial e configurações de tier diferentes tem de dar bytes idênticos.
  Verificamos isso no colibrì e ele passou — CPU e CUDA deram o mesmo hash.
- **Determinismo é por estado, não global.** Duas execuções no mesmo estado dão
  saída idêntica; estados diferentes podem divergir legitimamente. O teste tem
  de fixar o estado, não assumir que greedy basta.
