# ForDummies

Documento de entrada do TootegaEngine, para quem ainda não tem o vocabulário.
Duas versões, **mesmo conteúdo**.

The TootegaEngine front door, for readers who do not have the vocabulary yet.
Two versions, **same content**.

## Comece aqui / Start here

**Abra [`index.html`](index.html)** — é a versão para leitura, com sumário
lateral, busca, tema claro/escuro, figuras ampliáveis e **uma dica em cada termo
técnico**: passe o mouse (ou use Tab) sobre qualquer palavra sublinhada e ela
explica o que significa *neste* contexto.

Open [`index.html`](index.html) — the reading version, with a sidebar outline,
search, light/dark theme, zoomable figures and **a hint on every technical
term**: hover (or press Tab) over any underlined word and it explains what it
means *in this* context.

| formato / format | 🇧🇷 Português | 🇬🇧 English |
|---|---|---|
| **HTML** (recomendado / recommended) | [TootegaEngine-ForDummies.pt.html](TootegaEngine-ForDummies.pt.html) | [TootegaEngine-ForDummies.en.html](TootegaEngine-ForDummies.en.html) |
| Markdown (fonte / source) | [TootegaEngine-ForDummies.pt.md](TootegaEngine-ForDummies.pt.md) | [TootegaEngine-ForDummies.en.md](TootegaEngine-ForDummies.en.md) |

## Espelho da especificação / Specification mirror

[`docs-html/`](docs-html/index.html) tem os 16 documentos de `docs/` em HTML, com
o mesmo visual, as mesmas dicas de glossário e a mesma busca. Existe por um
motivo só: **navegador não renderiza `.md`**, então um link dentro do HTML tem de
abrir HTML. A fonte normativa continua sendo o Markdown em `docs/`; onde
divergirem, o Markdown ganha.

[`docs-html/`](docs-html/index.html) holds the 16 documents from `docs/` as HTML,
same look, same glossary hints, same search. It exists for one reason: **browsers
do not render `.md`**, so a link inside the HTML must open HTML. The normative
source is still the Markdown in `docs/`; where they disagree, the Markdown wins.

A busca abre uma lista de resultados agrupada em **Glossário**, **Seções** e
**No texto**, com o trecho encontrado destacado e a seção de origem embaixo.
Ignora acentos — `ablacao` acha `ablação`. Setas navegam, Enter abre e rola até
o ponto, piscando o trecho.

Search opens a grouped result list — **Glossary**, **Sections** and **In the
text** — with the match highlighted and its section underneath. It ignores
accents. Arrows navigate, Enter jumps to the spot and flashes it.

Atalhos de teclado no HTML: `/` foca a busca, `g` abre o glossário, `Esc` fecha.

Keyboard shortcuts in the HTML: `/` focuses search, `g` opens the glossary,
`Esc` closes.

## Glossário / Glossary

[`glossary.json`](glossary.json) é a **fonte única** dos termos: 186 entradas,
cada uma com termo, definição e apelidos em PT e EN, e — no grupo *outro
sentido* — o que a palavra costuma significar fora daqui. As páginas HTML
embutem esses dados,
então os balões funcionam abrindo o arquivo direto, sem servidor.

[`glossary.json`](glossary.json) is the **single source** for the terms: 186
entries, each with term, definition and aliases in PT and EN, and — in the
*other meaning* group — what the word usually means elsewhere. The HTML pages
embed that data,
so the hints work when opening the file directly, with no server.

| categoria / category | o que reúne / what it holds |
|---|---|
| **Outro sentido / Other meaning** | palavras comuns que aqui significam outra coisa (`expert`, `cache`, `banda`, `LoRA`, `plano`) |
| **Do projeto / Project** | neologismos cunhados aqui (`a inversão`, `lei de colocação`, `32/32`, `medição pareada`) |
| **Vocabulário / Vocabulary** | o resto do campo, em sete blocos: modelo, execução, máquina, numérica, medição, software, matemática |

## Figuras / Figures

As 15 figuras ficam em [`img/`](img), em uma versão por idioma
(`<nome>.pt.svg` e `<nome>.en.svg`). São SVG escritos à mão, sem dependência de
fonte externa, e trocam de paleta sozinhos em tema claro e escuro
(`prefers-color-scheme`).

The 15 figures live in [`img/`](img), in one version per language
(`<name>.pt.svg` and `<name>.en.svg`). They are hand-written SVG with no external
font dependency, and switch palette on their own between light and dark themes
(`prefers-color-scheme`).

| # | figura / figure | assunto / topic |
|---|---|---|
| 01 | `01-bandas` | as quatro bandas medidas e a inversão do PCIe / the four measured bandwidths and the PCIe inversion |
| 02 | `02-residencia` | `N^0,25` contra mudar de regime / `N^0.25` versus changing regime |
| 03 | `03-pipeline` | o caminho de um token, do texto ao texto / a token's journey, text to text |
| 04 | `04-camada` | por dentro de uma camada / inside one layer |
| 05 | `05-moe` | 64 experts guardados, 5 usados / 64 experts stored, 5 used |
| 06 | `06-mla` | KV comprimido e a absorção / compressed KV and absorption |
| 07 | `07-int4` | empacotamento metade/metade / half-half nibble packing |
| 08 | `08-particao` | partição por camada GPU/CPU / GPU/CPU layer partition |
| 09 | `09-prefixo` | cache de prefixo e a ordem do prompt / prefix cache and prompt ordering |
| 10 | `10-container` | o arquivo `.tootega` byte a byte / the `.tootega` file byte by byte |
| 11 | `11-roofline` | o teto de desempenho / the performance ceiling |
| 12 | `12-pareada` | por que a medição pareada / why paired measurement |
| 13 | `13-prefill-decode` | os dois regimes / the two regimes |
| 14 | `14-marcos` | o roadmap com critério de saída / the roadmap with exit criteria |
| 15 | `15-tres-caches` | os três caches e seus donos / the three caches and their owners |

## Regra de manutenção / Maintenance rule

Os dois documentos são **espelhos**: qualquer mudança de conteúdo entra nos dois,
e figura nova entra nas duas línguas. Termo novo entra em `glossary.json`, nas
tabelas de glossário dos `.md` e nos cartões do HTML — os três vêm dos mesmos
dados. Onde este documento divergir de `docs/`, o documento numerado ganha.

The two documents are **mirrors**: any content change lands in both, and a new
figure lands in both languages. A new term lands in `glossary.json`, in the `.md`
glossary tables and in the HTML cards — all three come from the same data. Where
this document disagrees with `docs/`, the numbered document wins.

Os `.html` — inclusive tudo em `docs-html/` — são **artefatos gerados** a partir
dos `.md` mais o `glossary.json`.
Editar o HTML à mão funciona (ele é legível e autocontido), mas a próxima
geração sobrescreve; o lugar certo de mudar conteúdo é o `.md`, e de mudar termo
é o `.json`.

The `.html` files — including everything under `docs-html/` — are **generated
artifacts** built from the `.md` files plus `glossary.json`. Hand-editing the HTML works (it is readable and self-contained),
but the next generation overwrites it; the right place to change content is the
`.md`, and to change a term, the `.json`.
