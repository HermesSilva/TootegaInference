# 12 — Estado persistente: subir quente

Reabrir o engine não deve refazer trabalho que já foi feito. Quatro artefatos vão
para o disco, ao lado do modelo, e são validados na carga.

```
<modelo>/
  .tootega/
    index          indice de tensores       ~2 MB
    plan           bandas medidas + colocacao  ~4 KB
    tokenizer      merges pre-compilados    ~8 MB
    kv/
      <hash>.blk   blocos de KV por prefixo  ~1 GB por prefixo de 20k
```

## 1. Índice de tensores

O modelo alvo tem ~9.500 tensores espalhados por 48 shards. Abrir significa ler
o cabeçalho JSON de cada shard, parsear e construir o mapa nome → (arquivo,
offset, forma, quantização).

O engine anterior aprendeu isso na dor: a varredura linear sobre ~120k tensores custava
*"decine di secondi per token"* até trocarem por hash aberto (outro módulo dele). Aqui o
problema é menor — 9,5k entradas — mas o parse dos cabeçalhos ainda é trabalho
repetido a cada abertura.

O índice persistido é um arquivo **plano, mapeável, já ordenado por hash**. Carga
= `mmap` + validar cabeçalho. Sem parse, sem alocação, sem construção de tabela.

## 2. Plano de colocação e bandas medidas

Medir banda de RAM, VRAM, disco e PCIe na inicialização custa segundos. O
resultado só muda se o hardware mudar.

Persistido: as bandas medidas e o plano resolvido, **chaveado por**
`(fingerprint de hardware, orçamentos, hash do modelo, build id)`. Qualquer
divergência invalida e remede.

O `--plan` continua recalculando sempre; o cache serve ao caminho normal.

## 3. Tokenizer pré-compilado

`tokenizer.json` é JSON com dezenas de milhares de merges. Parsear e montar as
estruturas de BPE a cada abertura é puro desperdício. Persistir a forma
compilada — tabelas já no layout que o encoder usa — reduz a abertura a um
`mmap`.

## 4. KV de prefixo: o que realmente importa

Este é o item que muda a experiência de um CLI agêntico.

Toda sessão começa com o mesmo prefixo — system prompt, schemas de ferramentas,
instruções do projeto. Tipicamente 10–20k tokens. **Esse prefill é recomputado a
cada abertura do processo**, e é o custo dominante do primeiro turno.

A conta, com os números de [08](08-teto-de-desempenho.md):

```
prefixo de 20k tokens
  KV correspondente              20.000 x 52,9 KB = 1,06 GB
  recomputar (prefill ~400 tok/s)                 ~50 s
  ler do RAID0 (18,4 GB/s)                        ~58 ms
```

Três ordens de grandeza. Os blocos de KV do prefixo estável ([11](11-memoria.md))
são gravados uma vez e mapeados de volta na abertura seguinte.

Medimos isso funcionando no engine anterior:

```
[KV] resumed conversation from disk: 106 tokens in 0.0s (no re-prefill)
```

### E medimos também como isso quebra

Na bateria de determinismo, uma execução divergiu das demais. A causa **não** era
a colocação — CPU e CUDA deram bytes idênticos. Era um KV persistido de
**outra conversa**, cujo prefixo casava parcialmente com o novo prompt. O engine
retomou de um estado que não correspondia àquele texto.

Daí a regra dura:

> **Um bloco de KV é endereçado pelo hash acumulado da sequência exata de tokens
> que o produziu.** Não pelo slot, não pelo nome da sessão, não por posição.

Um bloco só é reutilizável se o hash do prefixo até ele bate exatamente. Casamento
parcial não existe: ou o bloco corresponde àquela sequência, ou é ignorado. Isso
torna impossível a classe de bug que observamos.

## Estatísticas: honestidade sobre o que vale

O engine anterior persiste um mapa de calor persistido — mapa de calor de quais experts o seu workload
roteia — e usa para fixar os mais quentes. **Nesse regime faz todo sentido**: com
15% de residência, escolher os 15% certos é a diferença entre 0,65 e 0,93 tok/s.

Com o modelo **inteiro residente, isso deixa de ter função.** Não há expert a
promover; todos já estão na memória. Registrar calor de expert para decidir
colocação seria copiar maquinário sem copiar o problema que ele resolve.

O que continua valendo a pena registrar:

| estatística | para quê |
|---|---|
| taxa de acerto do cache de prefixo | mede se o CLI ordena o prompt certo — critério de saída do M4 |
| tok/s por turno, TTFT, tokens de prefill vs decode | telemetria; base de comparação entre versões |
| aceitação de draft do MTP | decide se a especulação paga, por workload |
| ocupação de slots de KV | dimensionar `--kv-slots` |

Nenhuma dessas realimenta a colocação. São observabilidade, não política — e a
distinção é deliberada, porque foi assumir o contrário que produziu metade da
complexidade do engine anterior.

## Regras de integridade

Todo artefato persistido carrega cabeçalho com:

- **magic + versão de formato** — versão desconhecida é descartada, nunca lida
  "na esperança";
- **hash de proveniência** — do que ele deriva (arquivo de modelo, tokenizer,
  build id do engine, orçamentos no caso do plano);
- **checksum do conteúdo**.

Divergência em qualquer campo → o artefato é **descartado e regenerado**, com uma
linha no log dizendo qual campo divergiu. Nunca migração silenciosa.

Escrita é **atômica**: arquivo temporário, `fsync`, `rename`. O engine anterior acerta
isso (`telemetry.h:168`, escreve `.tmp` e renomeia) e é o mínimo — um artefato
truncado por queda de energia não pode virar um índice corrompido que o engine
aceita.

## O que não é persistido

- **Pesos desquantizados ou reordenados.** O container em disco já é o formato de
  execução ([11](11-memoria.md)); qualquer transformação na carga seria uma cópia
  a evitar, não um cache a criar.
- **KV de prefixo divergente.** Só o prefixo estável e compartilhável vale a
  gravação; a cauda específica de cada conversa vive em memória enquanto a sessão
  existe.
- **Nada fora do diretório do modelo.** Um `rm -rf .tootega/` devolve o estado
  original, e a próxima abertura só é mais lenta.

## Custo de abertura, alvo

| etapa | frio | quente |
|---|---|---|
| índice de tensores | ~1–3 s | ~10 ms (mmap) |
| tokenizer | ~1 s | ~10 ms (mmap) |
| bandas + plano | ~3–5 s | ~1 ms (leitura) |
| pre-fault dos pesos | ~1–3 s (18,4 GB/s) | idem, o SO decide |
| prefill do prefixo de 20k | ~50 s | ~60 ms (mmap dos blocos) |
| **total** | **~60 s** | **~2 s** |

**[estimativa]** — vira critério de saída medido no M4.
