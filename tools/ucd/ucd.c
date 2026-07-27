/* ucd -- generates `src/unicode_tables.h` from the Unicode Character Database.
 *
 * The cl100k pattern (docs/16) needs `\p{L}`, `\p{N}` and `\s`.  We are not
 * embedding a regex engine or a whole Unicode library: the pre-tokeniser is a
 * hand-written state machine, and everything it needs to know about a code point
 * is those three predicates.  Here they become sorted ranges, looked up by
 * binary search.
 *
 * The input is `UnicodeData.txt` of the version you want -- one file, published
 * by the Unicode Consortium, plain text.  It is not vendored: the table it
 * produces IS vendored (`src/unicode_tables.h` is committed), and this tool
 * exists so that the table can be rebuilt and, above all, CHECKED.
 *
 *     curl -O https://www.unicode.org/Public/13.0.0/ucd/UnicodeData.txt
 *     tools\ucd.exe UnicodeData.txt --version 13.0.0 --check src\unicode_tables.h
 *     tools\ucd.exe UnicodeData.txt --version 13.0.0 --out   src\unicode_tables.h
 *
 * `--check` is the mode that matters day to day: it says whether the committed
 * header still is what the database says, without touching the file.
 */
#include "common.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CP 0x110000

/* Unicode White_Space, from PropList.txt.  Short and stable, so it is spelled
 * out here instead of costing a second input file.  Deriving it from "is this
 * blank" would be wrong: the ASCII file/group separators (0x1C-0x1F) count as
 * space for some definitions and are NOT White_Space, so `\s` would disagree
 * with the regex the tokeniser is imitating. */
static const uint32_t whitespace[][2] = {
    {0x0009, 0x000D}, {0x0020, 0x0020}, {0x0085, 0x0085}, {0x00A0, 0x00A0},
    {0x1680, 0x1680}, {0x2000, 0x200A}, {0x2028, 0x2029}, {0x202F, 0x202F},
    {0x205F, 0x205F}, {0x3000, 0x3000},
};

/* --------------------------------------------------------------- buffer */

typedef struct {
    char  *p;
    size_t len, cap;
} obuf;

static void ob_init(obuf *b)
{
    b->cap = 1u << 20;
    b->len = 0;
    b->p = (char *)malloc(b->cap);
    if (!b->p) { fprintf(stderr, "sem memoria\n"); exit(1); }
}

static void ob_put(obuf *b, const char *s)
{
    const size_t n = strlen(s);
    if (b->len + n + 1 > b->cap) { fprintf(stderr, "buffer de saida estourou\n"); exit(1); }
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}

static void ob_printf(obuf *b, const char *fmt, ...)
{
    char line[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    ob_put(b, line);
}

/* ---------------------------------------------------------- UnicodeData */

/* One field of a `;`-separated line.  Returns the start and writes the length;
 * the line is not modified, so the caller can keep scanning. */
static const char *field(const char *line, int want, size_t *len)
{
    int f = 0;
    const char *s = line;
    for (;;) {
        const char *e = s;
        while (*e && *e != ';' && *e != '\n' && *e != '\r') e++;
        if (f == want) { *len = (size_t)(e - s); return s; }
        if (*e != ';') { *len = 0; return NULL; }
        s = e + 1;
        f++;
    }
}

static int ends_with(const char *s, size_t n, const char *suffix)
{
    const size_t m = strlen(suffix);
    return n >= m && memcmp(s + n - m, suffix, m) == 0;
}

/* `cat[cp]` is the FIRST letter of the General_Category -- 'L', 'N', 'C'...
 * That is all the three predicates need, and it keeps the table at one byte per
 * code point. */
static int load_ucd(const char *path, unsigned char *cat)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "nao consegui abrir '%s'\n", path); return -1; }

    memset(cat, 'C', MAX_CP);          /* Cn -- unassigned */

    char line[1024];
    uint32_t range_first = 0;
    int in_range = 0, lines = 0;

    while (fgets(line, sizeof(line), f)) {
        size_t n;
        const char *s = field(line, 0, &n);
        if (!s || !n) continue;
        const uint32_t cp = (uint32_t)strtoul(s, NULL, 16);
        if (cp >= MAX_CP) continue;

        size_t name_len, cat_len;
        const char *name = field(line, 1, &name_len);
        const char *c    = field(line, 2, &cat_len);
        if (!name || !c || !cat_len) continue;

        /* Big blocks (CJK, Hangul, planes 15-16) do not get one line per code
         * point: they come as a `<..., First>` / `<..., Last>` pair, and
         * everything between the two inherits the category.  Missing this is
         * how a generator silently drops 90k letters. */
        if (ends_with(name, name_len, ", First>")) {
            range_first = cp;
            in_range = 1;
        } else if (in_range && ends_with(name, name_len, ", Last>")) {
            for (uint32_t k = range_first; k <= cp && k < MAX_CP; k++)
                cat[k] = (unsigned char)c[0];
            in_range = 0;
        } else {
            cat[cp] = (unsigned char)c[0];
            in_range = 0;
        }
        lines++;
    }

    fclose(f);
    if (lines < 1000) {
        fprintf(stderr, "'%s' tem so %d linhas uteis -- nao parece UnicodeData.txt\n",
                path, lines);
        return -1;
    }
    return 0;
}

/* --------------------------------------------------------------- emitir */

static size_t emit_ranges(obuf *b, const char *name, const unsigned char *cat,
                          unsigned char want)
{
    uint32_t (*r)[2] = (uint32_t (*)[2])malloc(sizeof(uint32_t) * 2 * 4096);
    size_t n = 0;
    int64_t start = -1;

    for (uint32_t cp = 0; cp < MAX_CP; cp++) {
        if (cat[cp] == want) {
            if (start < 0) start = (int64_t)cp;
        } else if (start >= 0) {
            r[n][0] = (uint32_t)start; r[n][1] = cp - 1; n++;
            start = -1;
        }
        if (n >= 4096) { fprintf(stderr, "intervalos demais\n"); exit(1); }
    }
    if (start >= 0) { r[n][0] = (uint32_t)start; r[n][1] = MAX_CP - 1; n++; }

    ob_printf(b, "static const tt_cp_range %s[] = {\n", name);
    for (size_t i = 0; i < n; i += 4) {
        ob_put(b, "    ");
        for (size_t k = i; k < i + 4 && k < n; k++)
            ob_printf(b, "%s{0x%05X,0x%05X}", k > i ? ", " : "", r[k][0], r[k][1]);
        ob_put(b, ",\n");
    }
    ob_put(b, "};\n\n");

    free(r);
    return n;
}

static void emit_fixed(obuf *b, const char *name,
                       const uint32_t (*ranges)[2], size_t n)
{
    ob_printf(b, "static const tt_cp_range %s[] = {\n", name);
    for (size_t i = 0; i < n; i += 4) {
        ob_put(b, "    ");
        for (size_t k = i; k < i + 4 && k < n; k++)
            ob_printf(b, "%s{0x%05X,0x%05X}", k > i ? ", " : "",
                      ranges[k][0], ranges[k][1]);
        ob_put(b, ",\n");
    }
    ob_put(b, "};\n\n");
}

/* ----------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    const char *ucd = NULL, *out = NULL, *check = NULL, *version = NULL;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const int has = (i + 1 < argc);
        if      (!strcmp(a, "--out") && has)     out = argv[++i];
        else if (!strcmp(a, "--check") && has)   check = argv[++i];
        else if (!strcmp(a, "--version") && has) version = argv[++i];
        else if (a[0] == '-') { fprintf(stderr, "opcao desconhecida: %s\n", a); return 2; }
        else if (!ucd) ucd = a;
    }

    if (!ucd || !version || (!out && !check)) {
        fprintf(stderr,
            "uso: ucd <UnicodeData.txt> --version <X.Y.Z> [--out <file.h>]\n"
            "                                             [--check <file.h>]\n\n"
            "  o arquivo vem de https://www.unicode.org/Public/<versao>/ucd/\n"
            "  --check  compara com o header commitado e nao escreve nada\n");
        return 2;
    }

    unsigned char *cat = (unsigned char *)malloc(MAX_CP);
    if (!cat) { fprintf(stderr, "sem memoria\n"); return 1; }
    if (load_ucd(ucd, cat) != 0) return 1;

    obuf b;
    ob_init(&b);
    ob_printf(&b,
        "/* unicode_tables.h -- GENERATED by tools/ucd.exe.  Do not edit.\n"
        " *\n"
        " * Unicode %s.  Sorted ranges, binary search.\n"
        " * Only the three predicates the cl100k pattern uses (docs/16).\n"
        " */\n"
        "#ifndef TOOTEGA_UNICODE_TABLES_H\n"
        "#define TOOTEGA_UNICODE_TABLES_H\n"
        "\n"
        "#include <stdint.h>\n"
        "\n"
        "typedef struct { uint32_t lo, hi; } tt_cp_range;\n"
        "\n", version);

    const size_t n_letter = emit_ranges(&b, "tt_range_letter", cat, 'L');
    const size_t n_number = emit_ranges(&b, "tt_range_number", cat, 'N');
    const size_t n_space  = sizeof(whitespace) / sizeof(whitespace[0]);
    emit_fixed(&b, "tt_range_space", whitespace, n_space);

    ob_put(&b,
        "static int tt_cp_in(const tt_cp_range *r, size_t n, uint32_t cp)\n"
        "{\n"
        "    size_t lo = 0, hi = n;\n"
        "    while (lo < hi) {\n"
        "        size_t mid = lo + (hi - lo) / 2;\n"
        "        if (cp < r[mid].lo)      hi = mid;\n"
        "        else if (cp > r[mid].hi) lo = mid + 1;\n"
        "        else                     return 1;\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
        "\n"
        "#define TT_N(a) (sizeof(a) / sizeof((a)[0]))\n"
        "\n"
        "/* ASCII first: it is the common path and it skips the binary search on\n"
        " * almost all real text. */\n"
        "static inline int tt_is_letter(uint32_t cp)\n"
        "{\n"
        "    if (cp < 128) return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');\n"
        "    return tt_cp_in(tt_range_letter, TT_N(tt_range_letter), cp);\n"
        "}\n"
        "\n"
        "static inline int tt_is_number(uint32_t cp)\n"
        "{\n"
        "    if (cp < 128) return cp >= '0' && cp <= '9';\n"
        "    return tt_cp_in(tt_range_number, TT_N(tt_range_number), cp);\n"
        "}\n"
        "\n"
        "static inline int tt_is_space(uint32_t cp)\n"
        "{\n"
        "    if (cp < 128) return cp == ' ' || (cp >= 0x09 && cp <= 0x0D);\n"
        "    return tt_cp_in(tt_range_space, TT_N(tt_range_space), cp);\n"
        "}\n"
        "\n"
        "#undef TT_N\n"
        "\n"
        "#endif /* TOOTEGA_UNICODE_TABLES_H */\n");

    printf("Unicode %s\n", version);
    printf("  \\p{L}  %zu intervalos\n", n_letter);
    printf("  \\p{N}  %zu intervalos\n", n_number);
    printf("  \\s     %zu intervalos\n", n_space);

    int rc = 0;

    if (check) {
        FILE *f = fopen(check, "rb");
        if (!f) { fprintf(stderr, "nao consegui abrir '%s'\n", check); return 1; }
        char *cur = (char *)malloc(b.cap + 1);
        if (!cur) { fclose(f); fprintf(stderr, "sem memoria\n"); return 1; }
        const size_t got = fread(cur, 1, b.cap, f);
        cur[got] = '\0';
        fclose(f);

        /* The banner carries the version and the generator's name, so it is not
         * part of what is being checked -- the TABLES are.  Comparison starts at
         * the include guard. */
        const char *mark = "#ifndef TOOTEGA_UNICODE_TABLES_H";
        const char *a = strstr(b.p, mark);
        const char *c = strstr(cur, mark);

        if (!a || !c) {
            printf("\n  %s: nao achei o include guard\n", check);
            rc = 1;
        } else {
            const size_t alen = b.len - (size_t)(a - b.p);
            const size_t clen = got - (size_t)(c - cur);
            if (alen == clen && memcmp(a, c, alen) == 0) {
                printf("\n  %s confere com a base Unicode %s\n", check, version);
            } else {
                size_t i = 0;
                while (i < alen && i < clen && a[i] == c[i]) i++;
                int line = 1;
                for (size_t k = 0; k < i; k++) if (a[k] == '\n') line++;
                printf("\n  %s DIVERGE a partir da linha %d do include guard\n",
                       check, line);
                rc = 1;
            }
        }
        free(cur);
    }

    if (out) {
        FILE *f = fopen(out, "wb");
        if (!f) { fprintf(stderr, "nao consegui escrever '%s'\n", out); return 1; }
        fwrite(b.p, 1, b.len, f);
        fclose(f);
        printf("\n  %s escrito (%zu bytes)\n", out, b.len);
    }

    free(cat);
    free(b.p);
    return rc;
}
