# 16 — Tokenizer

Resolve **D5** de [13](13-decisoes-pendentes.md). Fatos extraídos do
`tokenizer.json` do modelo, não de suposição.

## O que é

BPE no esquema **GPT-4 / tiktoken**, embalado no formato do `tokenizers` da
HuggingFace:

```
model.type                  BPE
model.vocab                 154.820 entradas
model.merges                321.649 regras
model.byte_fallback         false
model.ignore_merges         true
model.unk_token             null
normalizer                  null
pre_tokenizer               Sequence[ Split(regex, Isolated), ByteLevel ]
decoder / post_processor    ByteLevel
added_tokens                36 especiais (154820..154855)
vocab_size (config)         154.880   <- ha folga no fim
```

Sem normalizador e sem `unk`: todo byte de entrada é representável.

## As quatro etapas

### 1. Pré-tokenização por regex

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

É o padrão do `cl100k_base`. Exige classes Unicode `\p{L}` e `\p{N}`, grupo
case-insensitive e lookahead negativo.

**Não implementar um motor de regex.** A alternativa correta é uma máquina de
estados escrita à mão para *este* padrão, com tabelas Unicode geradas — que é
exatamente o que o engine anterior faz em `c/tok_unicode.h` e `c/tok_unicode_o200k.h`,
geradas por `c/tools/gen_unicode.py`. Reaproveitar a abordagem (e possivelmente
o gerador) poupa a parte mais chata e mais sujeita a erro sutil.

O que as tabelas precisam responder, por code point: *é letra?* (`\p{L}`), *é
número?* (`\p{N}`), *é espaço?* (`\s`). Intervalos comprimidos, busca binária.

Erro aqui não estoura: produz uma divisão ligeiramente diferente, ids diferentes
e saída diferente do oráculo — com o forward perfeitamente correto. É o tipo de
bug que consome dias se o teste não isolar o tokenizer.

### 2. ByteLevel

Cada byte `0..255` vira um code point pela tabela do GPT-2: imprimíveis mapeiam
para si, o resto para `0x100 + n`. É como um espaço vira `Ġ` no vocabulário.

Tabela fixa de 256 entradas, gerada uma vez. `add_prefix_space=false` — **não**
inserir espaço no início.

### 3. BPE com `ignore_merges = true`

Este flag muda o algoritmo e é fácil de não notar:

> Se a string do pré-token estiver **inteira** no vocabulário, use esse id
> direto e **não aplique merge nenhum**.

Só quando não está é que roda o BPE clássico: pares adjacentes, aplicar
repetidamente o de menor rank.

Ignorar o flag produz tokenização diferente para justamente as palavras mais
comuns — as que estão no vocabulário inteiras.

### 4. Tokens especiais

36 entradas em `added_tokens`, ids 154820..154855: `<|endoftext|>`, `[MASK]`,
`[gMASK]`, `[sMASK]`, `<sop>`, `<eop>`, `<|system|>`, `<|user|>`, e os demais.

São reconhecidos **antes** do regex, por casamento exato, e nunca passam por
BPE. O template de chat os emite; um CLI agêntico depende deles para delimitar
turnos e chamadas de ferramenta.

`vocab_size` é 154.880 contra 154.856 ids em uso: a folga é padding do
`lm_head`. Ids nessa faixa nunca devem ser amostrados — a máscara de amostragem
precisa saber disso.

## Estruturas em C

| estrutura | conteúdo | tamanho |
|---|---|---|
| `vocab` | id → bytes | ~2 MB |
| `lookup` | bytes → id, hash aberto | ~2 MB |
| `merges` | (id, id) → rank, hash aberto | ~5 MB, 321.649 entradas |
| `unicode` | intervalos de `\p{L}`, `\p{N}`, `\s` | ~20 KB |
| `byte_map` | 256 entradas, ida e volta | trivial |

Tudo pré-compilado e persistido conforme [12](12-estado-persistente.md): parsear
321.649 merges de JSON a cada abertura é desperdício puro. Carga = `mmap`.

A busca de merge é o laço quente do encode. Hash aberto com chave de 64 bits
`(id_esquerdo << 32) | id_direito`; o engine anterior aprendeu na dor que varredura
linear aqui custa *"decine di secondi"*.

## Teste

Sem isso não há como confiar:

1. **Round-trip**: `decode(encode(s)) == s` para um corpus de teste com CJK,
   emoji, sequências de espaços, quebras de linha e bytes inválidos em UTF-8.
2. **Igualdade contra a referência**: mesmos ids que o `tokenizers` da HF para
   o mesmo corpus. É o teste que importa, e é barato — gerar os ids esperados
   uma vez em Python e comparar em C.
3. **Tokens especiais**: cada um dos 36 volta com o id certo, e não é dividido.
4. **Fronteira de vocabulário**: ids 154856..154879 nunca são gerados.

O teste 2 tem de rodar **antes** do primeiro teste de forward. Um tokenizer
errado faz o oráculo do M1 falhar com o engine perfeitamente correto, e a
depuração começa no lugar errado.
