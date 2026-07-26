#!/usr/bin/env python3
"""Gera src/unicode_tables.h -- as tres perguntas que o pre-tokenizador faz.

O padrao do cl100k (docs/16) precisa de `\\p{L}`, `\\p{N}` e `\\s`.  Nao vamos
embutir um motor de regex nem uma biblioteca Unicode inteira: o pre-tokenizador
e uma maquina de estados escrita a mao, e tudo o que ela precisa saber sobre um
code point sao esses tres predicados.  Aqui eles viram intervalos ordenados,
consultados por busca binaria.

    python tools/gen_unicode.py
"""

import sys
import unicodedata
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "unicode_tables.h"
MAX_CP = 0x110000

# Unicode White_Space (PropList.txt).  Curto e estavel; nao vale derivar de
# str.isspace(), que inclui os separadores de arquivo/grupo do ASCII (0x1C-0x1F)
# e portanto discorda do `\s` do regex.
WHITESPACE = [
    (0x0009, 0x000D), (0x0020, 0x0020), (0x0085, 0x0085), (0x00A0, 0x00A0),
    (0x1680, 0x1680), (0x2000, 0x200A), (0x2028, 0x2029), (0x202F, 0x202F),
    (0x205F, 0x205F), (0x3000, 0x3000),
]


def ranges_for(predicate):
    out, start = [], None
    for cp in range(MAX_CP):
        if predicate(cp):
            if start is None:
                start = cp
        elif start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, MAX_CP - 1))
    return out


def emit(f, name, ranges):
    f.write(f"static const tt_cp_range {name}[] = {{\n")
    for i in range(0, len(ranges), 4):
        chunk = ", ".join(f"{{0x{a:05X},0x{b:05X}}}" for a, b in ranges[i:i + 4])
        f.write(f"    {chunk},\n")
    f.write("};\n\n")


def main():
    cat = {}
    for cp in range(MAX_CP):
        try:
            cat[cp] = unicodedata.category(chr(cp))
        except ValueError:
            cat[cp] = "Cn"

    letters = ranges_for(lambda cp: cat[cp][0] == "L")
    numbers = ranges_for(lambda cp: cat[cp][0] == "N")

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"""/* unicode_tables.h -- GERADO por tools/gen_unicode.py.  Nao edite.
 *
 * Unicode {unicodedata.unidata_version}.  Intervalos ordenados, busca binaria.
 * So os tres predicados que o padrao do cl100k usa (docs/16).
 */
#ifndef TOOTEGA_UNICODE_TABLES_H
#define TOOTEGA_UNICODE_TABLES_H

#include <stdint.h>

typedef struct {{ uint32_t lo, hi; }} tt_cp_range;

""")
        emit(f, "tt_range_letter", letters)
        emit(f, "tt_range_number", numbers)
        emit(f, "tt_range_space", WHITESPACE)

        f.write("""static int tt_cp_in(const tt_cp_range *r, size_t n, uint32_t cp)
{
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < r[mid].lo)      hi = mid;
        else if (cp > r[mid].hi) lo = mid + 1;
        else                     return 1;
    }
    return 0;
}

#define TT_N(a) (sizeof(a) / sizeof((a)[0]))

/* ASCII primeiro: e o caminho comum e evita a busca binaria em quase todo
 * texto real. */
static inline int tt_is_letter(uint32_t cp)
{
    if (cp < 128) return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    return tt_cp_in(tt_range_letter, TT_N(tt_range_letter), cp);
}

static inline int tt_is_number(uint32_t cp)
{
    if (cp < 128) return cp >= '0' && cp <= '9';
    return tt_cp_in(tt_range_number, TT_N(tt_range_number), cp);
}

static inline int tt_is_space(uint32_t cp)
{
    if (cp < 128) return cp == ' ' || (cp >= 0x09 && cp <= 0x0D);
    return tt_cp_in(tt_range_space, TT_N(tt_range_space), cp);
}

#undef TT_N

#endif /* TOOTEGA_UNICODE_TABLES_H */
""")

    print(f"{OUT} escrito")
    print(f"  Unicode {unicodedata.unidata_version}")
    print(f"  \\p{{L}}  {len(letters)} intervalos")
    print(f"  \\p{{N}}  {len(numbers)} intervalos")
    print(f"  \\s     {len(WHITESPACE)} intervalos")


if __name__ == "__main__":
    sys.exit(main())
