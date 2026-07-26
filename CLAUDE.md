# TootegaEngine — diretivas

Engine de inferência para MoE residentes, escrito do zero. O projeto inteiro
está especificado em `docs/`; leia `docs/02-hardware.md` primeiro e
`docs/13-decisoes-pendentes.md` para o placar das decisões.

## Independência

**Nada de dependência do projeto colibrì (`F:\colibri`)** — nem código, nem
toolchain. Os documentos citam o colibrì como referência de medições e lições
aprendidas; isso é histórico, não permissão de reuso. Onde um documento sugerir
reaproveitar código de lá (`json.h`, `tok_unicode.h`, `gen_unicode.py`),
escrever do zero.

## Onde cada coisa mora

| pasta | o que é | regra |
|---|---|---|
| `src/` | o engine | C11 ISO, **zero dependências**, só `compat.c` tem `#ifdef` de SO |
| `Tests/` | ferramentas de **medição** | uma pasta por ferramenta, **o `.exe` na raiz de `Tests/`** |
| `tools/` | ferramentas de **produção** | mesma regra; `tools/shared/` é a biblioteca comum |
| `ptools/` | ferramentas em Python | legado e referências que exigem `transformers`/`numpy` |
| `docs/` | especificação | toda decisão com o porquê e o número que a sustenta |

### Ferramentas novas: C ou Rust, nunca Python

Ferramenta nova vai para `Tests/<nome>/` ou `tools/<nome>/`, e o executável é
gerado na raiz da árvore correspondente. Python só continua em `ptools/` para o
que depende de `transformers` ou `tokenizers` como referência externa.

O `build.ps1` de cada ferramenta só delega para `tools/shared/build-tool.ps1`,
passando o próprio caminho — por isso o mesmo script serve às duas árvores.

Hoje: `Tests/` tem `ablate` (perplexidade e ablação), `gentest` (qualidade de
geração) e `bench` (tempo, memória, roofline). `tools/` tem `convert` e `agent`
(cliente agêntico).

**A master é CPU pura.** O backend CUDA, a colocação em VRAM e o portão do M2
existem e vivem na branch `GPU` — foram medidos e a conclusão foi que a
granularidade está errada (`docs/18` daquela branch). Nada de GPU entra aqui
sem que o `run_layer` mostre ganho no decode completo.

### O que separa Tests de tools

`Tests/` mede o engine e pode depender do checkpoint BF16 original. `tools/`
produz artefatos que o engine consome. Um conversor não é teste; um medidor de
perplexidade não é produção.

**Ferramenta nativa pode usar biblioteca.** A proibição de dependências vale
para `src/` — o engine é auditável em uma sessão de leitura e consome arquivos
de terceiros. Ferramenta é outra coisa: pode linkar o que o Python linkaria
(BLAS, mmap, o que for), desde que não entre em `src/`.

Ferramenta em C pode — e deve — **reusar as fontes de `src/`**. É o caminho que
garante que a ferramenta mede o engine, e não uma reimplementação dele.

## Build

```powershell
.\build.ps1                  # engine, MSVC via vswhere; nunca caminho fixo
.\build.ps1 test             # + selftest
.\tools\<nome>\build.ps1     # uma ferramenta
```

Windows é MSVC (Visual Studio 2026). Linux é o `Makefile` com gcc ou clang.
`/fp:precise` sem contração para FMA — verificado no disassembly, e é o que a
reprodutibilidade de `docs/15` exige.

## Regras que vieram de erro cometido

- **Medição pareada.** Comparar duas configurações pela média global engole o
  efeito na variância entre janelas. Ver o aviso em `docs/04`: a primeira
  medida de perplexidade deu o **sinal invertido**.
- **Mudança de saída não entra em silêncio.** Qualquer alteração
  algebricamente equivalente (absorção do MLA, precisão de KV) só é adotada com
  **32/32 tokens idênticos** contra `ref32.json`. bf16 e int8 no KV reprovaram
  e ficaram como opção, não como default.
- **Trocar trabalho por memória precisa da conta de tráfego antes.** A primeira
  versão da absorção materializou `kv_b` em fp32 e ficou 2× mais lenta.
- **Se Hermes disse que vai rodar algo no console dele, não disparar em
  paralelo.**
