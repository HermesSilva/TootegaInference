/* index.c -- turns a corpus of markdown books into a `.ttindex`.
 *
 *   tools\index.exe build "models\Versao Completa" --out models\v5-sft\cosmologia.ttindex
 *   tools\index.exe query models\v5-sft\cosmologia.ttindex "o que e a Arena" -k 5
 *   tools\index.exe inspect models\v5-sft\cosmologia.ttindex
 *
 * The unit is the SECTION -- a `## IV. <heading>` block -- because that is one
 * argument, and it is what a citation points at.  Bodies under 400 characters
 * are dropped: they are transitions and epigraphs, and retrieving one costs a
 * passage slot without answering anything.
 *
 * It does NOT tokenise.  `tt_tokenize` from `src/retrieve.c` does, the same
 * function the engine calls at query time, and that is the whole reason this is
 * a C tool linked against `src/` rather than a script: a builder that folded
 * "acao" differently from the searcher would produce an index that silently
 * retrieves nothing, and the symptom -- a model answering from memory -- looks
 * exactly like a bad fine-tune.  It is the failure docs/22 measured.
 *
 * `query` exists to be compared against `finetune/scripts/retrieve.py` on the
 * same corpus.  Same ranking, same scores, or one of the two is wrong.
 */
#include "common.h"
#include "compat.h"
#include "retrieve.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <windows.h>
#  define TT_SEP '\\'
#else
#  include <dirent.h>
#  include <sys/stat.h>
#  define TT_SEP '/'
#endif

#define MAX_PATH_LEN 1024
#define MIN_BODY_CHARS 400

/* ------------------------------------------------------------ utf-8 paths */

/* Windows hands `main` its argv and `FindFirstFileA` its names in the ANSI code
 * page, not UTF-8.  The corpus lives under "Versao Completa" with an accent, so
 * the bug is not hypothetical -- and it is silent, because every A-suffixed call
 * agrees with itself; it only surfaces when those bytes are written into the
 * index and the titles come back mangled.
 *
 * The same three edges are converted in `tools/finetune/finetune.c`.  Two copies
 * of twenty lines, and the day a third tool needs them they belong in
 * `tools/shared/`. */
#if defined(_WIN32)

static int wide_to_u8(const wchar_t *w, char *out, size_t max)
{
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, out, (int)max, NULL, NULL);
    return n > 0 ? n - 1 : -1;
}

static int u8_to_wide(const char *s, wchar_t *out, size_t max)
{
    const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, out, (int)max);
    return n > 0 ? n - 1 : -1;
}

static FILE *open_u8(const char *path, const char *mode)
{
    wchar_t wp[MAX_PATH_LEN], wm[8];
    if (u8_to_wide(path, wp, MAX_PATH_LEN) < 0) return NULL;
    if (u8_to_wide(mode, wm, 8) < 0) return NULL;
    return _wfopen(wp, wm);
}

#else

static FILE *open_u8(const char *path, const char *mode) { return fopen(path, mode); }

#endif

/* ------------------------------------------------------------------ util */

static void put_u32(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)v;         p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24);
}

static void put_u64(unsigned char *p, uint64_t v)
{
    put_u32(p, (uint32_t)v);
    put_u32(p + 4, (uint32_t)(v >> 32));
}

typedef struct {
    unsigned char *p;
    size_t         n, cap;
} buf;

static int buf_grow(buf *b, size_t need)
{
    if (b->n + need <= b->cap) return 0;
    size_t cap = b->cap ? b->cap : 4096;
    while (cap < b->n + need) cap *= 2;
    unsigned char *q = (unsigned char *)realloc(b->p, cap);
    if (!q) return tt_fail("out of memory");
    b->p = q;
    b->cap = cap;
    return 0;
}

static int buf_put(buf *b, const void *d, size_t n)
{
    if (buf_grow(b, n) != 0) return -1;
    memcpy(b->p + b->n, d, n);
    b->n += n;
    return 0;
}

/* Appends a NUL-terminated string and returns where it starts. */
static long buf_str(buf *b, const char *s, size_t n)
{
    const long at = (long)b->n;
    if (buf_put(b, s, n) != 0) return -1;
    if (buf_put(b, "", 1) != 0) return -1;
    return at;
}

static int is_space(int c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

/* Python's `len()` counts characters.  A 400-byte cut would drop sections that
 * are 400 characters of accented Portuguese, which is most of them. */
static size_t count_chars(const char *s, size_t n)
{
    size_t k = 0;
    for (size_t i = 0; i < n; i++)
        if (((unsigned char)s[i] & 0xC0) != 0x80) k++;
    return k;
}

static char *read_file(const char *path, size_t *len)
{
    FILE *f = open_u8(path, "rb");
    if (!f) { tt_fail("cannot open %s", path); return NULL; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); tt_fail("cannot size %s", path); return NULL; }
    char *p = (char *)malloc((size_t)n + 1);
    if (!p) { fclose(f); tt_fail("out of memory"); return NULL; }
    if (n && fread(p, 1, (size_t)n, f) != (size_t)n) {
        free(p); fclose(f); tt_fail("short read on %s", path); return NULL;
    }
    fclose(f);
    p[n] = '\0';
    *len = (size_t)n;
    return p;
}

/* The manifest of the fine-tuning pipeline checked this and it earned its keep:
 * the corpus travelled through a Windows path with accented folder names, and a
 * mojibake title would have been indexed without a word of complaint. */
static int valid_utf8(const char *s, size_t n)
{
    for (size_t i = 0; i < n; ) {
        const unsigned char c = (unsigned char)s[i];
        size_t k;
        if (c < 0x80)             k = 1;
        else if ((c & 0xE0) == 0xC0) k = 2;
        else if ((c & 0xF0) == 0xE0) k = 3;
        else if ((c & 0xF8) == 0xF0) k = 4;
        else return 0;
        if (i + k > n) return 0;
        for (size_t j = 1; j < k; j++)
            if (((unsigned char)s[i + j] & 0xC0) != 0x80) return 0;
        i += k;
    }
    return 1;
}

/* -------------------------------------------------------------- listing */

typedef struct {
    char (*name)[MAX_PATH_LEN];
    int   n, cap;
} names;

static int names_add(names *v, const char *s)
{
    if (v->n == v->cap) {
        const int cap = v->cap ? v->cap * 2 : 64;
        void *q = realloc(v->name, (size_t)cap * MAX_PATH_LEN);
        if (!q) return tt_fail("out of memory");
        v->name = (char (*)[MAX_PATH_LEN])q;
        v->cap = cap;
    }
    snprintf(v->name[v->n], MAX_PATH_LEN, "%s", s);
    v->n++;
    return 0;
}

static int cmp_name(const void *a, const void *b)
{
    return strcmp((const char *)a, (const char *)b);
}

/* Entries of `dir`, sorted the way Python's `sorted(os.listdir(...))` sorts
 * them -- byte order, which for these names is also reading order.  `want_dir`
 * selects directories over files. */
static int list_dir(const char *dir, int want_dir, const char *suffix, names *out)
{
    out->name = NULL; out->n = out->cap = 0;

#if defined(_WIN32)
    char pattern[MAX_PATH_LEN];
    snprintf(pattern, sizeof(pattern), "%s\\*", dir);
    wchar_t wp[MAX_PATH_LEN];
    if (u8_to_wide(pattern, wp, MAX_PATH_LEN) < 0)
        return tt_fail("path is not valid UTF-8: %s", dir);

    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(wp, &fd);
    if (h == INVALID_HANDLE_VALUE) return tt_fail("cannot list %s", dir);
    do {
        const int isdir = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        if (isdir != want_dir) continue;
        char nm[MAX_PATH_LEN];
        if (wide_to_u8(fd.cFileName, nm, sizeof(nm)) < 0) continue;
        if (!strcmp(nm, ".") || !strcmp(nm, "..")) continue;
        if (suffix) {
            const size_t ln = strlen(nm), ls = strlen(suffix);
            if (ln < ls || strcmp(nm + ln - ls, suffix)) continue;
        }
        if (names_add(out, nm) != 0) { FindClose(h); return -1; }
    } while (FindNextFileW(h, &fd));
    FindClose(h);
#else
    DIR *d = opendir(dir);
    if (!d) return tt_fail("cannot list %s", dir);
    struct dirent *e;
    while ((e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        char full[MAX_PATH_LEN];
        snprintf(full, sizeof(full), "%s/%s", dir, e->d_name);
        struct stat st;
        if (stat(full, &st) != 0) continue;
        const int isdir = S_ISDIR(st.st_mode) ? 1 : 0;
        if (isdir != want_dir) continue;
        if (suffix) {
            const size_t ln = strlen(e->d_name), ls = strlen(suffix);
            if (ln < ls || strcmp(e->d_name + ln - ls, suffix)) continue;
        }
        if (names_add(out, e->d_name) != 0) { closedir(d); return -1; }
    }
    closedir(d);
#endif
    if (out->n > 1) qsort(out->name, (size_t)out->n, MAX_PATH_LEN, cmp_name);
    return 0;
}

/* -------------------------------------------------------------- parsing */

/* Start of the next line beginning with `##` followed by whitespace.  `###` is
 * not a section: three hashes then a hash, and a hash is not whitespace. */
static const char *next_heading(const char *p, const char *end)
{
    while (p < end) {
        const char *nl = (const char *)memchr(p, '\n', (size_t)(end - p));
        const char *line = p;
        if (line + 2 < end && line[0] == '#' && line[1] == '#' && is_space(line[2]))
            return line;
        if (!nl) break;
        p = nl + 1;
    }
    return NULL;
}

/* `^#\s+(.+?)\s*$`, first match.  The document's own title. */
static int find_h1(const char *text, size_t len, const char **out, size_t *out_n)
{
    const char *p = text, *end = text + len;
    while (p < end) {
        const char *nl = (const char *)memchr(p, '\n', (size_t)(end - p));
        const char *stop = nl ? nl : end;
        if (p + 1 < stop && p[0] == '#' && is_space(p[1])) {
            const char *s = p + 1;
            while (s < stop && is_space(*s)) s++;
            const char *e = stop;
            while (e > s && is_space(e[-1])) e--;
            if (e > s) { *out = s; *out_n = (size_t)(e - s); return 1; }
        }
        if (!nl) break;
        p = nl + 1;
    }
    return 0;
}

static int roman_len(const char *s, const char *end)
{
    int n = 0;
    while (s + n < end) {
        const char c = s[n];
        if (c != 'I' && c != 'V' && c != 'X' && c != 'L' && c != 'C') break;
        n++;
    }
    return n;
}

/* `^\*(Livro\s+\w+)[^\n]*?Ensaio\s+([IVXLC]+)\s*\*$` -- the essay's roman
 * numeral, taken from the subtitle line the books already print. */
static int find_essay(const char *text, size_t len, char *out, size_t cap)
{
    const char *p = text, *end = text + len;
    while (p < end) {
        const char *nl = (const char *)memchr(p, '\n', (size_t)(end - p));
        const char *stop = nl ? nl : end;
        if (p + 6 < stop && p[0] == '*' && !strncmp(p + 1, "Livro", 5)) {
            for (const char *q = p; q + 6 < stop; q++) {
                if (strncmp(q, "Ensaio", 6)) continue;
                const char *s = q + 6;
                while (s < stop && is_space(*s)) s++;
                const int rn = roman_len(s, stop);
                if (rn > 0 && (size_t)rn < cap) {
                    memcpy(out, s, (size_t)rn);
                    out[rn] = '\0';
                    return 1;
                }
            }
        }
        if (!nl) break;
        p = nl + 1;
    }
    return 0;
}

/* The book title, from the folder name.  "I - A Estrutura da Existencia" names
 * the book and orders it at the same time; the ordering prefix is for the file
 * system and does not belong in a citation. */
static void book_title(const char *dirname, char *out, size_t cap)
{
    const char *s = dirname;
    const int rn = roman_len(s, s + strlen(s));
    if (rn > 0) {
        const char *q = s + rn;
        while (*q == ' ') q++;
        if (*q == '-' || *q == '.') {
            q++;
            while (*q == ' ') q++;
            s = q;
        }
    }
    snprintf(out, cap, "%s", s);
}

/* --------------------------------------------------------------- corpus */

typedef struct {
    long     title_off, sect_off, cite_off;
    uint32_t text_off, text_len;
    uint32_t n_terms;
} doc_rec;

typedef struct {
    doc_rec *v;
    size_t   n, cap;
} docs;

static doc_rec *docs_push(docs *d)
{
    if (d->n == d->cap) {
        const size_t cap = d->cap ? d->cap * 2 : 256;
        doc_rec *q = (doc_rec *)realloc(d->v, cap * sizeof(*q));
        if (!q) { tt_fail("out of memory"); return NULL; }
        d->v = q; d->cap = cap;
    }
    return &d->v[d->n++];
}

typedef struct {
    docs  doc;
    buf   strings;
    buf   text;
    int   n_files, n_skipped;
} corpus;

/* One document: split it into sections and record the ones worth retrieving. */
static int add_document(corpus *c, const char *path, const char *btitle)
{
    size_t len = 0;
    char *text = read_file(path, &len);
    if (!text) return -1;

    if (!valid_utf8(text, len)) {
        free(text);
        return tt_fail("%s is not valid UTF-8 -- indexing it would store "
                       "mojibake in the titles", path);
    }

    const char *title = NULL;
    size_t title_n = 0;
    if (!find_h1(text, len, &title, &title_n)) { title = path; title_n = strlen(path); }

    char essay[16];
    const int has_essay = find_essay(text, len, essay, sizeof(essay));

    const char *end = text + len;
    const char *h = next_heading(text, end);
    while (h) {
        const char *body = h + 2;
        while (body < end && is_space(*body)) body++;     /* the `\s+` of `^##\s+` */
        const char *nl = (const char *)memchr(body, '\n', (size_t)(end - body));
        const char *head_end = nl ? nl : end;
        while (head_end > body && is_space(head_end[-1])) head_end--;

        /* `## IV. <heading>` -> roman "IV", section "<heading>".  A heading with
         * no numeral keeps its whole text as the section name. */
        char roman[16];
        roman[0] = '\0';
        const char *sect = body;
        size_t sect_n = (size_t)(head_end - body);
        const int rn = roman_len(body, head_end);
        if (rn > 0 && (size_t)rn < sizeof(roman)) {
            const char *q = body + rn;
            while (q < head_end && (*q == ' ' || *q == '\t')) q++;
            if (q < head_end && (*q == '.' || *q == ')' || *q == '-')) {
                q++;
                while (q < head_end && (*q == ' ' || *q == '\t')) q++;
                memcpy(roman, body, (size_t)rn);
                roman[rn] = '\0';
                sect = q;
                sect_n = (size_t)(head_end - q);
            }
        }

        const char *next = nl ? next_heading(nl + 1, end) : NULL;
        const char *stop = next ? next : end;
        const char *bs = nl ? nl + 1 : head_end;
        while (bs < stop && is_space(*bs)) bs++;
        const char *be = stop;
        while (be > bs && is_space(be[-1])) be--;

        if (count_chars(bs, (size_t)(be - bs)) >= MIN_BODY_CHARS) {
            char cite[512];
            if (has_essay) snprintf(cite, sizeof(cite), "%s, Ensaio %s", btitle, essay);
            else           snprintf(cite, sizeof(cite), "%s, %.*s", btitle,
                                    (int)title_n, title);
            if (roman[0]) {
                char full[600];
                /* U+00A7 spelled out: a literal section sign next to a roman
                 * numeral parses as one long hex escape, because C is greedy and
                 * `C` is a hex digit. */
                snprintf(full, sizeof(full), "%s, \xc2\xa7%s", cite, roman);
                snprintf(cite, sizeof(cite), "%s", full);
            }

            doc_rec *d = docs_push(&c->doc);
            if (!d) { free(text); return -1; }
            d->title_off = buf_str(&c->strings, title, title_n);
            d->sect_off  = buf_str(&c->strings, sect, sect_n);
            d->cite_off  = buf_str(&c->strings, cite, strlen(cite));
            d->text_off  = (uint32_t)c->text.n;
            d->text_len  = (uint32_t)(be - bs);
            d->n_terms   = 0;
            if (d->title_off < 0 || d->sect_off < 0 || d->cite_off < 0 ||
                buf_str(&c->text, bs, (size_t)(be - bs)) < 0) {
                free(text);
                return -1;
            }
        } else {
            c->n_skipped++;
        }
        h = next;
    }

    free(text);
    c->n_files++;
    return 0;
}

static int load_corpus(corpus *c, const char *root)
{
    memset(c, 0, sizeof(*c));

    names books;
    if (list_dir(root, 1, NULL, &books) != 0) return -1;
    if (!books.n) { free(books.name); return tt_fail("%s has no book folders", root); }

    for (int i = 0; i < books.n; i++) {
        char bdir[MAX_PATH_LEN], btitle[256];
        snprintf(bdir, sizeof(bdir), "%s%c%s", root, TT_SEP, books.name[i]);
        book_title(books.name[i], btitle, sizeof(btitle));

        names files;
        if (list_dir(bdir, 0, ".md", &files) != 0) { free(books.name); return -1; }
        for (int j = 0; j < files.n; j++) {
            char path[MAX_PATH_LEN];
            snprintf(path, sizeof(path), "%s%c%s", bdir, TT_SEP, files.name[j]);
            if (add_document(c, path, btitle) != 0) {
                free(files.name); free(books.name);
                return -1;
            }
        }
        printf("  %-34s %3d files\n", btitle, files.n);
        free(files.name);
    }
    free(books.name);
    return 0;
}

/* -------------------------------------------------------- term dictionary */

typedef struct {
    uint32_t *slot;        /* id + 1, 0 empty */
    size_t    mask;
    uint32_t *off;         /* id -> offset into the string pool */
    uint32_t  n, cap;
    buf      *pool;
} dict;

static uint32_t hash_bytes(const char *s, size_t n)
{
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 16777619u; }
    return h;
}

static int dict_init(dict *d, buf *pool)
{
    memset(d, 0, sizeof(*d));
    d->pool = pool;
    d->mask = 4095;
    d->slot = (uint32_t *)calloc(d->mask + 1, sizeof(uint32_t));
    return d->slot ? 0 : tt_fail("out of memory");
}

static int dict_rehash(dict *d)
{
    const size_t mask = d->mask * 2 + 1;
    uint32_t *slot = (uint32_t *)calloc(mask + 1, sizeof(uint32_t));
    if (!slot) return tt_fail("out of memory");
    for (uint32_t i = 0; i < d->n; i++) {
        const char *s = (const char *)d->pool->p + d->off[i];
        size_t k = hash_bytes(s, strlen(s)) & mask;
        while (slot[k]) k = (k + 1) & mask;
        slot[k] = i + 1;
    }
    free(d->slot);
    d->slot = slot;
    d->mask = mask;
    return 0;
}

/* The term's id, adding it if new.  UINT32_MAX on failure. */
static uint32_t dict_intern(dict *d, const char *s, size_t n)
{
    if ((size_t)d->n * 2 >= d->mask && dict_rehash(d) != 0) return UINT32_MAX;

    size_t k = hash_bytes(s, n) & d->mask;
    while (d->slot[k]) {
        const char *have = (const char *)d->pool->p + d->off[d->slot[k] - 1];
        if (strlen(have) == n && !memcmp(have, s, n)) return d->slot[k] - 1;
        k = (k + 1) & d->mask;
    }
    if (d->n == d->cap) {
        const uint32_t cap = d->cap ? d->cap * 2 : 1024;
        uint32_t *q = (uint32_t *)realloc(d->off, (size_t)cap * sizeof(uint32_t));
        if (!q) { tt_fail("out of memory"); return UINT32_MAX; }
        d->off = q; d->cap = cap;
    }
    const long at = buf_str(d->pool, s, n);
    if (at < 0) return UINT32_MAX;
    d->off[d->n] = (uint32_t)at;
    d->slot[k] = d->n + 1;
    return d->n++;
}

/* ----------------------------------------------------------- inversion */

typedef struct { uint32_t term, doc, tf; } triple;

typedef struct {
    dict     *d;
    uint32_t *ids;         /* term ids of the document being read, in order */
    size_t    n, cap;
    int       failed;
} collect;

static void on_term(const char *t, size_t n, void *ud)
{
    collect *c = (collect *)ud;
    if (c->failed) return;
    const uint32_t id = dict_intern(c->d, t, n);
    if (id == UINT32_MAX) { c->failed = 1; return; }
    if (c->n == c->cap) {
        const size_t cap = c->cap ? c->cap * 2 : 1024;
        uint32_t *q = (uint32_t *)realloc(c->ids, cap * sizeof(uint32_t));
        if (!q) { tt_fail("out of memory"); c->failed = 1; return; }
        c->ids = q; c->cap = cap;
    }
    c->ids[c->n++] = id;
}

static int cmp_u32(const void *a, const void *b)
{
    const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static int cmp_triple(const void *a, const void *b)
{
    const triple *x = (const triple *)a, *y = (const triple *)b;
    if (x->term != y->term) return x->term < y->term ? -1 : 1;
    return x->doc < y->doc ? -1 : (x->doc > y->doc ? 1 : 0);
}

/* Sorting the dictionary by the term bytes, so the engine can binary-search it.
 * `qsort` gets the pool through file-scope pointers because C11 has no `qsort_r`
 * that both toolchains agree on.
 *
 * TWO of them, and that is the point: what is being sorted is an array of term
 * IDS, and the id has to go through `off` to become an offset into the pool.
 * Using the id directly compiles, runs, and produces a table sorted by nothing
 * at all -- the binary search then misses most terms, the ranking quietly
 * degrades, and only `index terms` makes it visible. */
static const buf      *g_pool;
static const uint32_t *g_off;

static int cmp_term_id(const void *a, const void *b)
{
    const char *x = (const char *)g_pool->p + g_off[*(const uint32_t *)a];
    const char *y = (const char *)g_pool->p + g_off[*(const uint32_t *)b];
    return strcmp(x, y);
}

/* --------------------------------------------------------------- build */

static int pad_to(FILE *f, uint64_t *at)
{
    static const unsigned char zero[TT_INDEX_ALIGN] = {0};
    const uint64_t want = (*at + TT_INDEX_ALIGN - 1) & ~(uint64_t)(TT_INDEX_ALIGN - 1);
    const size_t n = (size_t)(want - *at);
    if (n && fwrite(zero, 1, n, f) != n) return tt_fail("write failed");
    *at = want;
    return 0;
}

static int write_section(FILE *f, uint64_t *at, const void *p, size_t n,
                         unsigned char *hdr, size_t field)
{
    if (pad_to(f, at) != 0) return -1;
    put_u64(hdr + field, *at);
    put_u64(hdr + field + 8, n);
    if (n && fwrite(p, 1, n, f) != n) return tt_fail("write failed");
    *at += n;
    return 0;
}

static int cmd_build(int argc, char **argv)
{
    const char *root = NULL, *out = NULL;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (argv[i][0] == '-') return tt_fail("build: unknown option '%s'", argv[i]);
        else if (!root) root = argv[i];
        else return tt_fail("build: extra argument '%s'", argv[i]);
    }
    if (!root || !out)
        return tt_fail("usage: index build <corpus-dir> --out <file.ttindex>");

    printf("corpus %s\n", root);
    corpus c;
    if (load_corpus(&c, root) != 0) return -1;
    printf("  %zu sections from %d files (%d under %d characters, dropped)\n",
           c.doc.n, c.n_files, c.n_skipped, MIN_BODY_CHARS);
    if (!c.doc.n) return tt_fail("no section survived -- is this the right folder?");

    /* Terms go into the SAME string pool as the titles: one blob, one kind of
     * offset, and no way for the two to disagree about which is which. */
    dict d;
    if (dict_init(&d, &c.strings) != 0) return -1;

    collect col;
    memset(&col, 0, sizeof(col));
    col.d = &d;

    triple *tri = NULL;
    size_t n_tri = 0, cap_tri = 0;
    uint64_t total_len = 0;

    for (size_t i = 0; i < c.doc.n; i++) {
        doc_rec *doc = &c.doc.v[i];
        col.n = 0;

        /* The heading is COPIED out of the pool first.  Interning a term appends
         * to that same pool, which reallocs it, and the second call would then
         * be reading freed memory -- a corruption that survives to the index and
         * shows up as one section with garbage terms. */
        char sect[512];
        snprintf(sect, sizeof(sect), "%s", (const char *)c.strings.p + doc->sect_off);

        /* Body and heading, which is what the reference indexes: the heading is
         * often the only place a term of art appears in the section.  Two calls
         * instead of one concatenation -- the tokeniser flushes at the end of
         * each, so the result is identical to joining them with a space. */
        tt_tokenize((const char *)c.text.p + doc->text_off, doc->text_len,
                    on_term, &col);
        tt_tokenize(sect, strlen(sect), on_term, &col);
        if (col.failed) return -1;

        doc->n_terms = (uint32_t)col.n;      /* |D| counts repeats, as BM25 does */
        total_len += col.n;

        if (col.n > 1) qsort(col.ids, col.n, sizeof(uint32_t), cmp_u32);
        for (size_t k = 0; k < col.n; ) {
            size_t j = k;
            while (j < col.n && col.ids[j] == col.ids[k]) j++;
            if (n_tri == cap_tri) {
                cap_tri = cap_tri ? cap_tri * 2 : 65536;
                triple *q = (triple *)realloc(tri, cap_tri * sizeof(*q));
                if (!q) return tt_fail("out of memory");
                tri = q;
            }
            tri[n_tri].term = col.ids[k];
            tri[n_tri].doc  = (uint32_t)i;
            tri[n_tri].tf   = (uint32_t)(j - k);
            n_tri++;
            k = j;
        }
    }

    /* The dictionary is built in first-seen order; the engine needs it in byte
     * order.  Sort the ids, invert the permutation, and renumber the postings. */
    uint32_t *order = (uint32_t *)malloc((size_t)d.n * sizeof(uint32_t));
    uint32_t *rank  = (uint32_t *)malloc((size_t)d.n * sizeof(uint32_t));
    if (!order || !rank) return tt_fail("out of memory");
    for (uint32_t i = 0; i < d.n; i++) order[i] = i;
    g_pool = &c.strings;
    g_off  = d.off;
    qsort(order, d.n, sizeof(uint32_t), cmp_term_id);
    for (uint32_t i = 0; i < d.n; i++) rank[order[i]] = i;
    for (size_t i = 0; i < n_tri; i++) tri[i].term = rank[tri[i].term];
    qsort(tri, n_tri, sizeof(triple), cmp_triple);

    tt_ix_term *terms = (tt_ix_term *)calloc(d.n, sizeof(tt_ix_term));
    tt_ix_post *post  = (tt_ix_post *)malloc(n_tri * sizeof(tt_ix_post));
    tt_ix_doc  *dv    = (tt_ix_doc *)malloc(c.doc.n * sizeof(tt_ix_doc));
    if (!terms || !post || !dv) return tt_fail("out of memory");

    for (size_t i = 0; i < n_tri; i++) {
        post[i].doc = tri[i].doc;
        post[i].tf  = tri[i].tf;
        tt_ix_term *t = &terms[tri[i].term];
        if (!t->df) {
            t->str_off  = d.off[order[tri[i].term]];
            t->post_off = (uint32_t)i;
        }
        t->df++;
        t->post_n = t->df;
    }
    for (size_t i = 0; i < c.doc.n; i++) {
        dv[i].title_off = (uint32_t)c.doc.v[i].title_off;
        dv[i].sect_off  = (uint32_t)c.doc.v[i].sect_off;
        dv[i].cite_off  = (uint32_t)c.doc.v[i].cite_off;
        dv[i].text_off  = c.doc.v[i].text_off;
        dv[i].text_len  = c.doc.v[i].text_len;
        dv[i].n_terms   = c.doc.v[i].n_terms;
    }

    unsigned char hdr[TT_INDEX_HEADER];
    memset(hdr, 0, sizeof(hdr));
    memcpy(hdr, TT_INDEX_MAGIC, 8);
    put_u32(hdr + 8, TT_INDEX_VERSION);
    put_u32(hdr + 12, (uint32_t)c.doc.n);
    put_u32(hdr + 16, d.n);
    put_u64(hdr + 24, total_len);

    FILE *f = open_u8(out, "wb");
    if (!f) return tt_fail("cannot write %s", out);
    if (fwrite(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        fclose(f); return tt_fail("write failed");
    }
    uint64_t at = TT_INDEX_HEADER;
    const int bad =
        write_section(f, &at, terms, (size_t)d.n * sizeof(*terms), hdr, 32) != 0 ||
        write_section(f, &at, c.strings.p, c.strings.n,            hdr, 48) != 0 ||
        write_section(f, &at, post, n_tri * sizeof(*post),         hdr, 64) != 0 ||
        write_section(f, &at, dv, c.doc.n * sizeof(*dv),           hdr, 80) != 0 ||
        write_section(f, &at, c.text.p, c.text.n,                  hdr, 96) != 0;
    if (!bad) {
        /* The offsets are only known now, so the header is rewritten in place. */
        if (fseek(f, 0, SEEK_SET) != 0 ||
            fwrite(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
            fclose(f); return tt_fail("cannot rewrite the header of %s", out);
        }
    }
    fclose(f);
    if (bad) return -1;

    char b1[32];
    printf("\n  %u terms, %zu postings, %llu tokens (avg %.0f per section)\n",
           d.n, n_tri, (unsigned long long)total_len,
           (double)total_len / (double)c.doc.n);
    printf("  -> %s (%s)\n", out, tt_bytes(b1, sizeof(b1), at));

    /* Reading it back is not ceremony: the writer and the reader validate
     * different things, and an index that only the writer accepts is an index
     * that fails at the first chat instead of here. */
    tt_index ix;
    if (tt_index_open(&ix, out) != 0) return -1;
    printf("  verified: %u docs, %u terms, avgdl %.1f\n",
           ix.n_docs, ix.n_terms, ix.avgdl);
    tt_index_close(&ix);
    return 0;
}

/* --------------------------------------------------------- query/inspect */

static int cmd_query(int argc, char **argv)
{
    const char *path = NULL, *q = NULL;
    int k = 5, show = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-k") && i + 1 < argc) k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--text")) show = 1;
        else if (argv[i][0] == '-' && argv[i][1] && !path)
            return tt_fail("query: unknown option '%s'", argv[i]);
        else if (!path) path = argv[i];
        else if (!q) q = argv[i];
        else return tt_fail("query: extra argument '%s'", argv[i]);
    }
    if (!path || !q) return tt_fail("usage: index query <file.ttindex> \"...\" [-k 5]");
    if (k < 1) k = 1;

    tt_index ix;
    if (tt_index_open(&ix, path) != 0) return -1;

    tt_hit *hits = (tt_hit *)malloc((size_t)k * sizeof(tt_hit));
    if (!hits) { tt_index_close(&ix); return tt_fail("out of memory"); }

    const double t0 = tt_now();
    const int n = tt_index_search(&ix, q, hits, k);
    const double dt = tt_now() - t0;

    printf("%u sections, vocabulary %u\n\nquery: %s\n\n", ix.n_docs, ix.n_terms, q);
    for (int i = 0; i < n; i++) {
        printf("  %6.2f  %s\n          %s\n", hits[i].score,
               tt_index_cite(&ix, hits[i].doc), tt_index_section(&ix, hits[i].doc));
        if (show) printf("          %.240s...\n", tt_index_text(&ix, hits[i].doc, NULL));
    }
    if (!n) printf("  nothing -- the question shares no term with the corpus\n");
    printf("\n  %.1f ms\n", dt * 1000.0);

    free(hits);
    tt_index_close(&ix);
    return 0;
}

static int cmd_inspect(int argc, char **argv)
{
    if (argc < 1) return tt_fail("usage: index inspect <file.ttindex>");
    tt_index ix;
    if (tt_index_open(&ix, argv[0]) != 0) return -1;

    char b1[32];
    printf("%s\n", argv[0]);
    printf("  %s on disk\n", tt_bytes(b1, sizeof(b1), ix.map.size));
    printf("  %u sections, %u terms, %llu postings\n",
           ix.n_docs, ix.n_terms, (unsigned long long)ix.n_post);
    printf("  avgdl %.1f terms, text %s\n", ix.avgdl,
           tt_bytes(b1, sizeof(b1), ix.text_len));

    uint32_t widest = 0;
    for (uint32_t i = 1; i < ix.n_terms; i++)
        if (ix.terms[i].df > ix.terms[widest].df) widest = i;
    printf("  commonest term \"%s\" in %u of %u sections\n",
           ix.strings + ix.terms[widest].str_off, ix.terms[widest].df, ix.n_docs);

    printf("\n  first sections:\n");
    for (uint32_t i = 0; i < ix.n_docs && i < 5; i++)
        printf("    %-46.46s %5u terms  %s\n", tt_index_section(&ix, i),
               ix.docs[i].n_terms, tt_index_cite(&ix, i));

    tt_index_close(&ix);
    return 0;
}

/* Every term with its document frequency, one per line, in index order.
 *
 * It exists to be diffed against the reference tokeniser.  The two produced
 * vocabularies of 13469 and 13465 the first time they were compared, and four
 * terms is not a rounding difference -- it is a rule one side applies and the
 * other does not.  Without a way to list them the only visible symptom would
 * have been a ranking that is right almost always. */
static int cmd_terms(int argc, char **argv)
{
    if (argc < 1) return tt_fail("usage: index terms <file.ttindex>");
    tt_index ix;
    if (tt_index_open(&ix, argv[0]) != 0) return -1;
    for (uint32_t i = 0; i < ix.n_terms; i++)
        printf("%s\t%u\n", ix.strings + ix.terms[i].str_off, ix.terms[i].df);
    tt_index_close(&ix);
    return 0;
}

static int run(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
          "index -- BM25 over a corpus of markdown books (docs/13 D15)\n\n"
          "  index build <corpus-dir> --out <file.ttindex>\n"
          "  index query <file.ttindex> \"...\" [-k 5] [--text]\n"
          "  index inspect <file.ttindex>\n"
          "  index terms <file.ttindex>        every term and its df\n");
        return 1;
    }
    const char *cmd = argv[1];
    int rc;
    if      (!strcmp(cmd, "build"))   rc = cmd_build(argc - 2, argv + 2);
    else if (!strcmp(cmd, "query"))   rc = cmd_query(argc - 2, argv + 2);
    else if (!strcmp(cmd, "inspect")) rc = cmd_inspect(argc - 2, argv + 2);
    else if (!strcmp(cmd, "terms"))   rc = cmd_terms(argc - 2, argv + 2);
    else rc = tt_fail("unknown command '%s'", cmd);

    if (rc != 0 && tt_error()[0]) fprintf(stderr, "\nerror: %s\n", tt_error());
    return rc == 0 ? 0 : 1;
}

#if defined(_WIN32)

/* `wmain` and not `main`: the CRT hands the arguments over in UTF-16, and the
 * corpus folder is "Versao Completa" with an accent.  Through plain `main` that
 * name arrives in the ANSI code page and the directory cannot be opened -- which
 * is exactly what happened the first time this ran.  `tools/finetune` has the
 * same entry point for the same reason. */
int wmain(int argc, wchar_t **wargv)
{
    SetConsoleOutputCP(CP_UTF8);

    char **argv = (char **)calloc((size_t)argc + 1, sizeof(char *));
    if (!argv) { fprintf(stderr, "error: out of memory\n"); return 1; }
    for (int i = 0; i < argc; i++) {
        int need = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
        if (need <= 0) need = 1;
        argv[i] = (char *)malloc((size_t)need);
        if (!argv[i]) { fprintf(stderr, "error: out of memory\n"); return 1; }
        if (wide_to_u8(wargv[i], argv[i], (size_t)need) < 0) argv[i][0] = '\0';
    }
    return run(argc, argv);
}

#else

int main(int argc, char **argv) { return run(argc, argv); }

#endif
