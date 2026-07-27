/* finetune -- corpus packer and, later, the LoRA trainer.
 *
 * This file implements F-1 of the fine-tuning plan: turn a folder of Markdown
 * into a deterministic pack of token ids the trainer reads.  It exists before
 * any backward pass because every estimate in the plan depends on a number this
 * tool produces -- how many tokens the corpus actually has.
 *
 * Three decisions are baked in, and each came from a measurement:
 *
 *   sections    Windows are cut at `##` boundaries whenever a section fits.
 *               An essay is 5-14k tokens, larger than any window; cutting at a
 *               fixed stride splits sentences and arguments, and a text that
 *               argues in a chain teaches the model to start from nothing.
 *
 *   header      Every window is prefixed with the document title.  It costs a
 *               few tokens and it is the hook the prompt uses later: the model
 *               learns that the header conditions the register.
 *
 *   holdout     The split is BY DOCUMENT, never by window.  Random windows of
 *               the same essay leak -- the model saw the neighbouring
 *               paragraphs.  docs/04 records what unpaired comparison did to
 *               the first perplexity measurement; this is the same trap one
 *               level up.
 *
 * Round-trip is verified by default: every chunk is decoded back and compared
 * against the bytes that were encoded.  A tokenizer error does not blow up, it
 * silently trains on different text (docs/16).
 *
 *   tools\finetune.exe pack <dir> --tokenizer <tokenizer.json> --out <file.ttpack>
 *   tools\finetune.exe inspect <file.ttpack>
 */
#include "common.h"
#include "compat.h"
#include "json.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dirent.h>
#  include <sys/stat.h>
#endif

#define PACK_MAGIC   "TTPACK\0\1"
#define PACK_VERSION 1
#define PACK_HEADER  128
#define PACK_ALIGN   64

#define MAX_PATH_LEN 1024
#define MAX_DOCS     4096
#define MAX_PATTERNS 16

/* ------------------------------------------------------------ utf-8 paths */

/* Windows hands `main` its argv, and `FindFirstFileA` its names, in the ANSI
 * code page -- not UTF-8.  Files still open, because every A-suffixed call
 * speaks the same page, so the bug is silent: it only shows up when those bytes
 * are written into the JSON manifest, which then is not valid UTF-8.  The
 * corpus this was built for lives under "Versão Completa", so it showed up on
 * the first run.
 *
 * The tool therefore speaks UTF-8 everywhere and converts at the three edges:
 * argv, directory listing and fopen.  `src/` has the same limitation and is not
 * touched here -- that is a project decision, not a detail of this tool. */

#if defined(_WIN32)

static int wide_to_u8(const wchar_t *w, char *out, size_t max)
{
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, out, (int)max, NULL, NULL);
    return n > 0 ? n - 1 : -1;
}

static int u8_to_wide(const char *s, wchar_t *out, size_t max)
{
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, out, (int)max);
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

static uint32_t get_u32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t get_u64(const unsigned char *p)
{
    return (uint64_t)get_u32(p) | ((uint64_t)get_u32(p + 4) << 32);
}

static int lower(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

/* `*` and `?`, case-insensitive.  Matched against the file NAME, not the whole
 * path -- a pattern that had to know the folder layout would be useless the
 * first time the corpus moved. */
static int glob_match(const char *pat, const char *s)
{
    const char *star = NULL, *back = NULL;
    while (*s) {
        if (*pat == '?' || lower(*pat) == lower((unsigned char)*s)) { pat++; s++; }
        else if (*pat == '*') { star = pat++; back = s; }
        else if (star) { pat = star + 1; s = ++back; }
        else return 0;
    }
    while (*pat == '*') pat++;
    return *pat == '\0';
}

/* Copies at most `cols` CHARACTERS -- not bytes -- and reports how many it
 * wrote.  `printf("%-58s")` counts bytes, so a title with accents came out
 * short by one column per accent, and the table lost its alignment on exactly
 * the corpus this tool was written for. */
static int u8_clip(char *dst, size_t max, const char *s, size_t len, int cols)
{
    size_t at = 0;
    int n = 0;
    while (at < len && n < cols) {
        size_t w = 1;
        unsigned char c = (unsigned char)s[at];
        if      (c >= 0xF0) w = 4;
        else if (c >= 0xE0) w = 3;
        else if (c >= 0xC0) w = 2;
        if (at + w > len || at + w >= max) break;
        memcpy(dst + at, s + at, w);
        at += w;
        n++;
    }
    dst[at] = '\0';
    return n;
}

static const char *base_name(const char *path)
{
    const char *b = path;
    for (const char *p = path; *p; p++)
        if (*p == '/' || *p == '\\') b = p + 1;
    return b;
}

/* Whole file into a NUL-terminated buffer.  A UTF-8 BOM is dropped: the
 * tokenizer would encode it as content, and it only ever appears because some
 * editor put it there. */
static char *read_file(const char *path, size_t *out_len)
{
    FILE *f = open_u8(path, "rb");
    if (!f) { tt_fail("cannot open %s", path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); tt_fail("seek %s", path); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); tt_fail("tell %s", path); return NULL; }
    rewind(f);

    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); tt_fail("out of memory for %s", path); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';

    size_t off = 0;
    if (got >= 3 && (unsigned char)buf[0] == 0xEF
                 && (unsigned char)buf[1] == 0xBB
                 && (unsigned char)buf[2] == 0xBF) off = 3;
    if (off) memmove(buf, buf + off, got - off + 1);

    *out_len = got - off;
    return buf;
}

/* ------------------------------------------------------------- documents */

typedef struct {
    char     path[MAX_PATH_LEN];
    char     title[256];
    int32_t *hdr;              /* encoded header, repeated in every window */
    int64_t  n_hdr;
    int32_t *ids;              /* the body, sections already concatenated */
    int64_t  n_ids;
    int64_t *sec_end;          /* end offset of each section inside `ids` */
    int      n_sec;
    int      holdout;
} doc_t;

typedef struct {
    uint32_t doc, off, len, flags;
} win_t;

#define WIN_HOLDOUT 1u

/* --------------------------------------------------------------- walking */

typedef struct {
    char (*paths)[MAX_PATH_LEN];
    int    n, cap;
    const char *inc[MAX_PATTERNS]; int n_inc;
    const char *exc[MAX_PATTERNS]; int n_exc;
} walk_t;

static int walk_keep(const walk_t *w, const char *name)
{
    int ok = (w->n_inc == 0);
    for (int i = 0; i < w->n_inc; i++) if (glob_match(w->inc[i], name)) { ok = 1; break; }
    if (!ok) return 0;
    for (int i = 0; i < w->n_exc; i++) if (glob_match(w->exc[i], name)) return 0;
    return 1;
}

static int walk_push(walk_t *w, const char *path)
{
    if (w->n >= w->cap) return tt_fail("more than %d files; raise MAX_DOCS", w->cap);
    snprintf(w->paths[w->n], MAX_PATH_LEN, "%s", path);
    w->n++;
    return 0;
}

static int walk_dir(walk_t *w, const char *dir)
{
    char child[MAX_PATH_LEN];
#if defined(_WIN32)
    /* Wide listing, converted to UTF-8 on the way out: `FindFirstFileA` would
     * hand back ANSI, and a name outside the code page comes back as '?'. */
    wchar_t wpat[MAX_PATH_LEN];
    char pat[MAX_PATH_LEN];
    snprintf(pat, sizeof(pat), "%s\\*", dir);
    if (u8_to_wide(pat, wpat, MAX_PATH_LEN) < 0) return tt_fail("path too long: %s", dir);

    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(wpat, &fd);
    if (h == INVALID_HANDLE_VALUE) return tt_fail("cannot list %s", dir);
    do {
        char name[MAX_PATH_LEN];
        if (fd.cFileName[0] == L'.') continue;      /* . .. and hidden */
        if (wide_to_u8(fd.cFileName, name, sizeof(name)) < 0) continue;
        snprintf(child, sizeof(child), "%s\\%s", dir, name);
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            if (walk_dir(w, child) != 0) { FindClose(h); return -1; }
        } else if (walk_keep(w, name)) {
            if (walk_push(w, child) != 0) { FindClose(h); return -1; }
        }
    } while (FindNextFileW(h, &fd));
    FindClose(h);
#else
    DIR *d = opendir(dir);
    if (!d) return tt_fail("cannot list %s", dir);
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        snprintf(child, sizeof(child), "%s/%s", dir, e->d_name);
        struct stat st;
        if (stat(child, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            if (walk_dir(w, child) != 0) { closedir(d); return -1; }
        } else if (walk_keep(w, e->d_name)) {
            if (walk_push(w, child) != 0) { closedir(d); return -1; }
        }
    }
    closedir(d);
#endif
    return 0;
}

/* Sorting is not cosmetic: the holdout split is chosen by stride over this
 * list, so the order decides which essays are held out.  Directory order is not
 * stable across machines, and an unstable split makes two runs incomparable --
 * the same lesson as the floating RAM budget in docs/09. */
static int path_cmp(const void *a, const void *b)
{
    return strcmp((const char *)a, (const char *)b);
}

/* --------------------------------------------------------------- parsing */

/* First `# ` line.  The title goes into every window's header, so a document
 * without one falls back to its file name rather than to nothing. */
static void extract_title(const char *text, const char *path, char *out, size_t max)
{
    const char *p = text;
    while (*p) {
        if (p[0] == '#' && p[1] == ' ') {
            const char *e = strchr(p, '\n');
            size_t n = e ? (size_t)(e - p - 2) : strlen(p + 2);
            while (n && (p[2 + n - 1] == '\r' || p[2 + n - 1] == ' ')) n--;
            if (n >= max) n = max - 1;
            memcpy(out, p + 2, n);
            out[n] = '\0';
            return;
        }
        const char *e = strchr(p, '\n');
        if (!e) break;
        p = e + 1;
    }
    snprintf(out, max, "%s", base_name(path));
}

/* Offsets of every `^## ` in the text, plus 0 and the end.  Returns how many
 * SECTIONS that yields (cuts - 1). */
static int find_sections(const char *text, size_t len, size_t *cut, int max)
{
    int n = 0;
    cut[n++] = 0;
    for (size_t i = 0; i + 3 < len && n < max - 1; i++) {
        if (i && text[i - 1] != '\n') continue;
        if (text[i] == '#' && text[i + 1] == '#' && text[i + 2] == ' ') {
            if (cut[n - 1] != i) cut[n++] = i;
        }
    }
    cut[n++] = len;
    return n - 1;
}

/* ---------------------------------------------------------------- packing */

typedef struct {
    int64_t window;
    int     min_section;       /* sections shorter than this are folded ahead */
    int     verify;
} pack_opt;

/* Encodes `text` and, when asked, decodes it back and compares.  A wrong
 * tokenizer produces a perfectly valid pack over different text; this is the
 * only place where that is cheap to catch. */
static int64_t encode_checked(const tt_tokenizer *tk, const char *text, size_t len,
                              int32_t *out, int64_t max, int verify,
                              const char *where)
{
    int64_t n = tt_encode(tk, text, len, out, max);
    if (n < 0) return tt_fail("%s: text does not fit in %lld ids", where, (long long)max);
    if (!verify) return n;

    char *back = (char *)malloc(len + 64);
    if (!back) return tt_fail("out of memory verifying %s", where);
    int64_t m = tt_decode(tk, out, n, back, len + 64);
    int bad = (m != (int64_t)len) || (m > 0 && memcmp(back, text, (size_t)m) != 0);
    if (bad) {
        int64_t at = 0;
        while (at < m && at < (int64_t)len && back[at] == text[at]) at++;
        tt_fail("%s: round-trip differs at byte %lld (%lld in, %lld out)",
                where, (long long)at, (long long)len, (long long)m);
    }
    free(back);
    return bad ? -1 : n;
}

static int doc_load(doc_t *d, const tt_tokenizer *tk, const pack_opt *o)
{
    size_t len = 0;
    char *text = read_file(d->path, &len);
    if (!text) return -1;

    extract_title(text, d->path, d->title, sizeof(d->title));

    /* The header is encoded once and copied into every window. */
    char hdr[512];
    int hn = snprintf(hdr, sizeof(hdr), "# %s\n\n", d->title);
    int32_t hbuf[256];
    d->n_hdr = encode_checked(tk, hdr, (size_t)hn, hbuf, 256, o->verify, d->path);
    if (d->n_hdr < 0) { free(text); return -1; }
    d->hdr = (int32_t *)malloc(sizeof(int32_t) * (size_t)d->n_hdr);
    if (!d->hdr) { free(text); return tt_fail("out of memory"); }
    memcpy(d->hdr, hbuf, sizeof(int32_t) * (size_t)d->n_hdr);

    size_t *cut = (size_t *)malloc(sizeof(size_t) * (MAX_DOCS + 2));
    if (!cut) { free(text); return tt_fail("out of memory"); }
    int n_sec = find_sections(text, len, cut, MAX_DOCS + 2);

    /* One id per byte is a safe ceiling: the worst case in this tokenizer is a
     * byte that stands alone. */
    int32_t *ids = (int32_t *)malloc(sizeof(int32_t) * (len + 16));
    int64_t *ends = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n_sec + 1));
    if (!ids || !ends) { free(text); free(cut); free(ids); free(ends);
                         return tt_fail("out of memory for %s", d->path); }

    int64_t at = 0;
    int kept = 0;
    for (int s = 0; s < n_sec; s++) {
        size_t a = cut[s], b = cut[s + 1];
        while (b > a && (text[b - 1] == '\n' || text[b - 1] == '\r')) b--;
        if (b <= a) continue;
        int64_t n = encode_checked(tk, text + a, b - a, ids + at,
                                   (int64_t)(len + 16) - at, o->verify, d->path);
        if (n < 0) { free(text); free(cut); free(ids); free(ends); return -1; }
        /* A stub shorter than `min_section` is not a section: it is a heading
         * with nothing under it.  Folding it into the previous one keeps it in
         * the corpus without producing a window that is almost all header. */
        if (kept && n < o->min_section) ends[kept - 1] = at + n;
        else ends[kept++] = at + n;
        at += n;
    }

    d->ids = ids;
    d->n_ids = at;
    d->sec_end = ends;
    d->n_sec = kept;
    free(cut);
    free(text);
    return 0;
}

/* Greedy fill respecting section boundaries: take whole sections while they
 * fit; a section larger than the window is split, because there is no
 * alternative. */
static int doc_windows(const doc_t *d, int idoc, int64_t window,
                       int32_t *ids_out, int64_t *ids_at,
                       win_t *wins, int *n_wins, int max_wins)
{
    const int64_t budget = window - d->n_hdr;
    if (budget < 32) return tt_fail("window %lld too small for the header of %s",
                                    (long long)window, d->path);
    int64_t s0 = 0;
    int sec = 0;
    while (s0 < d->n_ids) {
        int64_t take = 0;
        /* whole sections while they fit */
        while (sec < d->n_sec && d->sec_end[sec] - s0 <= budget) {
            take = d->sec_end[sec] - s0;
            sec++;
        }
        if (take == 0) {                 /* one section is bigger than a window */
            take = budget;
            if (s0 + take > d->n_ids) take = d->n_ids - s0;
        }
        if (*n_wins >= max_wins) return tt_fail("too many windows");
        wins[*n_wins].doc = (uint32_t)idoc;
        wins[*n_wins].off = (uint32_t)*ids_at;
        wins[*n_wins].len = (uint32_t)(d->n_hdr + take);
        wins[*n_wins].flags = d->holdout ? WIN_HOLDOUT : 0u;
        (*n_wins)++;

        memcpy(ids_out + *ids_at, d->hdr, sizeof(int32_t) * (size_t)d->n_hdr);
        *ids_at += d->n_hdr;
        memcpy(ids_out + *ids_at, d->ids + s0, sizeof(int32_t) * (size_t)take);
        *ids_at += take;
        s0 += take;
        while (sec < d->n_sec && d->sec_end[sec] <= s0) sec++;
    }
    return 0;
}

/* ----------------------------------------------------------------- writer */

static int w_all(FILE *f, const void *p, size_t n)
{
    return (n && fwrite(p, 1, n, f) != n) ? tt_fail("write failed") : 0;
}

static int w_pad(FILE *f, uint64_t *pos)
{
    static const unsigned char zero[PACK_ALIGN] = { 0 };
    size_t pad = (size_t)(tt_align_up((size_t)*pos, PACK_ALIGN) - *pos);
    if (w_all(f, zero, pad) != 0) return -1;
    *pos += pad;
    return 0;
}

/* JSON strings here are file paths and titles: they can carry a backslash, a
 * quote or a control character, and an index that does not escape them is an
 * index the reader rejects. */
static int json_str(char *out, size_t max, const char *s)
{
    size_t at = 0;
    if (at + 1 >= max) return -1;
    out[at++] = '"';
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        char esc[8];
        int n;
        if      (*p == '"')  n = snprintf(esc, sizeof(esc), "\\\"");
        else if (*p == '\\') n = snprintf(esc, sizeof(esc), "\\\\");
        else if (*p == '\n') n = snprintf(esc, sizeof(esc), "\\n");
        else if (*p == '\r') n = snprintf(esc, sizeof(esc), "\\r");
        else if (*p == '\t') n = snprintf(esc, sizeof(esc), "\\t");
        else if (*p < 0x20)  n = snprintf(esc, sizeof(esc), "\\u%04x", *p);
        else { esc[0] = (char)*p; esc[1] = '\0'; n = 1; }
        if (at + (size_t)n + 2 >= max) return -1;
        memcpy(out + at, esc, (size_t)n);
        at += (size_t)n;
    }
    out[at++] = '"';
    out[at] = '\0';
    return (int)at;
}

/* ------------------------------------------------------------------ pack */

static void usage(void)
{
    fprintf(stderr,
      "finetune " TT_VERSION " -- corpus packer for LoRA training\n"
      "\n"
      "usage:\n"
      "  finetune pack <dir> --tokenizer <tokenizer.json> --out <file.ttpack>\n"
      "  finetune inspect <file.ttpack>\n"
      "\n"
      "pack:\n"
      "  --tokenizer <path>     tokenizer.json, HuggingFace format (required)\n"
      "  --out <path>           output pack (required)\n"
      "  --window N             tokens per window (default: 1024)\n"
      "  --include <glob>       file name pattern, repeatable (default: *.md)\n"
      "  --exclude <glob>       excluded pattern, repeatable\n"
      "  --holdout N            hold out N whole documents, by even stride\n"
      "  --holdout-match <glob> hold out every document matching the pattern\n"
      "  --min-section N        fold sections shorter than N ids (default: 48)\n"
      "  --vocab N              vocabulary size (default: 154880)\n"
      "  --no-verify            skip the decode round-trip (not recommended)\n"
      "\n"
      "  -v, --verbose          spell out what is being done\n");
}

static int cmd_pack(int argc, char **argv)
{
    const char *dir = NULL, *tokpath = NULL, *out = NULL, *hmatch = NULL;
    int32_t vocab = 154880;
    int holdout_n = 0;
    pack_opt opt = { 1024, 48, 1 };

    walk_t w;
    memset(&w, 0, sizeof(w));

    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        int has = (i + 1 < argc);
        if      (!strcmp(a, "--tokenizer") && has) tokpath = argv[++i];
        else if (!strcmp(a, "--out") && has)       out = argv[++i];
        else if (!strcmp(a, "--window") && has)    opt.window = atoll(argv[++i]);
        else if (!strcmp(a, "--min-section") && has) opt.min_section = atoi(argv[++i]);
        else if (!strcmp(a, "--vocab") && has)     vocab = (int32_t)atoi(argv[++i]);
        else if (!strcmp(a, "--holdout") && has)   holdout_n = atoi(argv[++i]);
        else if (!strcmp(a, "--holdout-match") && has) hmatch = argv[++i];
        else if (!strcmp(a, "--no-verify"))        opt.verify = 0;
        else if (!strcmp(a, "--include") && has) {
            if (w.n_inc >= MAX_PATTERNS) return tt_fail("too many --include");
            w.inc[w.n_inc++] = argv[++i];
        } else if (!strcmp(a, "--exclude") && has) {
            if (w.n_exc >= MAX_PATTERNS) return tt_fail("too many --exclude");
            w.exc[w.n_exc++] = argv[++i];
        } else if (a[0] == '-') return tt_fail("pack: unknown option '%s'", a);
        else if (!dir) dir = a;
        else return tt_fail("pack: extra argument '%s'", a);
    }

    if (!dir || !tokpath || !out) { usage(); return 2; }
    if (opt.window < 64) return tt_fail("--window must be at least 64");
    if (w.n_inc == 0) w.inc[w.n_inc++] = "*.md";

    /* ---- files */
    w.cap = MAX_DOCS;
    w.paths = (char (*)[MAX_PATH_LEN])malloc((size_t)w.cap * MAX_PATH_LEN);
    if (!w.paths) return tt_fail("out of memory");
    if (walk_dir(&w, dir) != 0) { free(w.paths); return -1; }
    if (w.n == 0) { free(w.paths); return tt_fail("no file matched under %s", dir); }
    qsort(w.paths, (size_t)w.n, MAX_PATH_LEN, path_cmp);
    tt_log("%d file(s) under %s", w.n, dir);

    /* ---- tokenizer */
    double t0 = tt_now();
    tt_tokenizer tk;
    if (tt_tokenizer_load(&tk, tokpath, vocab) != 0) { free(w.paths); return -1; }
    tt_dbg("tokenizer: %d tokens, %d merges, %.2f s",
           tk.n_tokens, tk.n_merges, tt_now() - t0);

    /* ---- documents */
    doc_t *docs = (doc_t *)calloc((size_t)w.n, sizeof(doc_t));
    if (!docs) { tt_tokenizer_free(&tk); free(w.paths); return tt_fail("out of memory"); }

    int64_t total_ids = 0;
    for (int i = 0; i < w.n; i++) {
        snprintf(docs[i].path, MAX_PATH_LEN, "%s", w.paths[i]);
        if (doc_load(&docs[i], &tk, &opt) != 0) goto fail;
        total_ids += docs[i].n_ids;
        tt_dbg("  %-64s %6lld ids, %3d sections",
               base_name(docs[i].path), (long long)docs[i].n_ids, docs[i].n_sec);
    }

    /* ---- holdout, by whole document */
    int n_hold = 0;
    if (hmatch) {
        for (int i = 0; i < w.n; i++)
            if (glob_match(hmatch, base_name(docs[i].path))) { docs[i].holdout = 1; n_hold++; }
    }
    if (holdout_n > 0) {
        if (holdout_n >= w.n) { tt_fail("--holdout %d leaves no training document", holdout_n);
                                goto fail; }
        /* Even stride, starting off-zero: it spreads the split over the whole
         * corpus instead of taking the first N, which on a sorted list would
         * mean "all of book one". */
        for (int k = 0; k < holdout_n; k++) {
            int i = (int)(((int64_t)k * w.n + w.n / 2) / holdout_n);
            if (i >= w.n) i = w.n - 1;
            while (i < w.n && docs[i].holdout) i++;
            if (i < w.n) { docs[i].holdout = 1; n_hold++; }
        }
    }

    /* ---- windows */
    int max_wins = (int)(total_ids / (opt.window / 2) + w.n + 16);
    win_t *wins = (win_t *)malloc(sizeof(win_t) * (size_t)max_wins);
    /* Every window repeats the header, so the id stream is longer than the
     * corpus.  The ceiling below is generous on purpose: running out here
     * would be a sizing bug, and a sizing bug should not be a corrupt pack. */
    int64_t ids_cap = total_ids + (int64_t)max_wins * 64 + 1024;
    int32_t *ids = (int32_t *)malloc(sizeof(int32_t) * (size_t)ids_cap);
    if (!wins || !ids) { tt_fail("out of memory for the pack"); goto fail; }

    int n_wins = 0;
    int64_t ids_at = 0;
    for (int i = 0; i < w.n; i++)
        if (doc_windows(&docs[i], i, opt.window, ids, &ids_at, wins, &n_wins, max_wins) != 0)
            goto fail;

    int64_t train_ids = 0, hold_ids = 0;
    int n_wtrain = 0, n_whold = 0;
    for (int i = 0; i < n_wins; i++) {
        if (wins[i].flags & WIN_HOLDOUT) { hold_ids += wins[i].len; n_whold++; }
        else                             { train_ids += wins[i].len; n_wtrain++; }
    }

    /* ---- write */
    FILE *f = open_u8(out, "wb");
    if (!f) { tt_fail("cannot create %s", out); goto fail; }

    unsigned char hdr[PACK_HEADER];
    memset(hdr, 0, sizeof(hdr));
    memcpy(hdr, PACK_MAGIC, 8);
    put_u32(hdr + 8, PACK_VERSION);
    put_u32(hdr + 12, 0);
    put_u64(hdr + 16, (uint64_t)w.n);
    put_u64(hdr + 24, (uint64_t)n_wins);
    put_u64(hdr + 32, (uint64_t)ids_at);
    if (w_all(f, hdr, sizeof(hdr)) != 0) { fclose(f); goto fail; }

    uint64_t pos = PACK_HEADER;
    if (w_pad(f, &pos) != 0) { fclose(f); goto fail; }
    uint64_t ids_off = pos;
    if (w_all(f, ids, sizeof(int32_t) * (size_t)ids_at) != 0) { fclose(f); goto fail; }
    pos += sizeof(int32_t) * (uint64_t)ids_at;

    if (w_pad(f, &pos) != 0) { fclose(f); goto fail; }
    uint64_t win_off = pos;
    for (int i = 0; i < n_wins; i++) {
        unsigned char rec[16];
        put_u32(rec, wins[i].doc);   put_u32(rec + 4, wins[i].off);
        put_u32(rec + 8, wins[i].len); put_u32(rec + 12, wins[i].flags);
        if (w_all(f, rec, sizeof(rec)) != 0) { fclose(f); goto fail; }
    }
    pos += (uint64_t)n_wins * 16;

    if (w_pad(f, &pos) != 0) { fclose(f); goto fail; }
    uint64_t index_off = pos;

    size_t icap = (size_t)w.n * 1024 + 4096;
    char *index = (char *)malloc(icap);
    if (!index) { fclose(f); tt_fail("out of memory for the index"); goto fail; }
    size_t iat = 0;
    char tokq[MAX_PATH_LEN * 2], dirq[MAX_PATH_LEN * 2];
    json_str(tokq, sizeof(tokq), tokpath);
    json_str(dirq, sizeof(dirq), dir);
    iat += (size_t)snprintf(index + iat, icap - iat,
        "{\"meta\":{\"tool\":\"finetune\",\"version\":\"%s\",\"build\":\"%s\","
        "\"corpus\":%s,\"tokenizer\":%s,\"vocab\":%d,\"window\":%lld,"
        "\"min_section\":%d,\"verified\":%s,\"n_docs\":%d,\"n_windows\":%d,"
        "\"n_ids\":%lld,\"n_holdout_docs\":%d,\"train_ids\":%lld,"
        "\"holdout_ids\":%lld},\"docs\":[",
        TT_VERSION, tt_build_id(), dirq, tokq, vocab, (long long)opt.window,
        opt.min_section, opt.verify ? "true" : "false", w.n, n_wins,
        (long long)ids_at, n_hold, (long long)train_ids, (long long)hold_ids);

    for (int i = 0; i < w.n; i++) {
        char pq[MAX_PATH_LEN * 2], tq[1024];
        json_str(pq, sizeof(pq), docs[i].path);
        json_str(tq, sizeof(tq), docs[i].title);
        int n = snprintf(index + iat, icap - iat,
            "%s{\"path\":%s,\"title\":%s,\"sections\":%d,\"ids\":%lld,"
            "\"holdout\":%s}", i ? "," : "", pq, tq, docs[i].n_sec,
            (long long)docs[i].n_ids, docs[i].holdout ? "true" : "false");
        if (n < 0 || iat + (size_t)n + 4 >= icap) {
            fclose(f); free(index); tt_fail("index overflowed its buffer"); goto fail;
        }
        iat += (size_t)n;
    }
    iat += (size_t)snprintf(index + iat, icap - iat, "]}");

    if (w_all(f, index, iat) != 0) { fclose(f); free(index); goto fail; }

    /* The offsets are only known now, so the header is rewritten in place. */
    put_u64(hdr + 40, ids_off);
    put_u64(hdr + 48, win_off);
    put_u64(hdr + 56, index_off);
    put_u64(hdr + 64, (uint64_t)iat);
    if (fseek(f, 0, SEEK_SET) != 0 || w_all(f, hdr, sizeof(hdr)) != 0) {
        fclose(f); free(index); tt_fail("cannot rewrite the header"); goto fail;
    }
    fclose(f);
    free(index);

    int n_sec_total = 0;
    for (int i = 0; i < w.n; i++) n_sec_total += docs[i].n_sec;

    tt_log("");
    tt_log("  documents      %d  (%d held out)", w.n, n_hold);
    tt_log("  sections       %d", n_sec_total);
    tt_log("  corpus tokens  %lld", (long long)total_ids);
    tt_log("  windows        %d of %lld  (%d train, %d holdout)",
           n_wins, (long long)opt.window, n_wtrain, n_whold);
    tt_log("  packed tokens  %lld  (train %lld, holdout %lld)",
           (long long)ids_at, (long long)train_ids, (long long)hold_ids);
    tt_log("  round-trip     %s", opt.verify ? "verified" : "SKIPPED");
    tt_log("  -> %s", out);

    free(ids); free(wins);
    for (int i = 0; i < w.n; i++) { free(docs[i].ids); free(docs[i].hdr); free(docs[i].sec_end); }
    free(docs);
    tt_tokenizer_free(&tk);
    free(w.paths);
    return 0;

fail:
    for (int i = 0; i < w.n; i++) { free(docs[i].ids); free(docs[i].hdr); free(docs[i].sec_end); }
    free(docs);
    tt_tokenizer_free(&tk);
    free(w.paths);
    return -1;
}

/* --------------------------------------------------------------- inspect */

static int cmd_inspect(int argc, char **argv)
{
    const char *path = NULL;
    int list = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--docs")) list = 1;
        else if (argv[i][0] == '-') return tt_fail("inspect: unknown option '%s'", argv[i]);
        else if (!path) path = argv[i];
        else return tt_fail("inspect: extra argument '%s'", argv[i]);
    }
    if (!path) { usage(); return 2; }

    /* Read rather than map: `tt_map_open` goes through `CreateFileA`, and this
     * tool speaks UTF-8 paths.  Only the header and the index are read -- the
     * ids are not needed to describe a pack. */
    FILE *f = open_u8(path, "rb");
    if (!f) return tt_fail("cannot open %s", path);

    unsigned char p[PACK_HEADER];
    if (fread(p, 1, sizeof(p), f) != sizeof(p) || memcmp(p, PACK_MAGIC, 8) != 0) {
        fclose(f);
        return tt_fail("%s is not a .ttpack (bad magic or truncated)", path);
    }
    uint32_t ver = get_u32(p + 8);
    if (ver != PACK_VERSION) {
        fclose(f);
        return tt_fail("unknown pack version %u; this build reads %d", ver, PACK_VERSION);
    }

    uint64_t n_docs = get_u64(p + 16), n_wins = get_u64(p + 24), n_ids = get_u64(p + 32);
    uint64_t idx_off = get_u64(p + 56), idx_len = get_u64(p + 64);

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return tt_fail("seek failed"); }
    uint64_t fsize = (uint64_t)ftell(f);
    if (idx_off + idx_len > fsize) {
        fclose(f);
        return tt_fail("pack is truncated: the index falls outside the file");
    }

    char *idx = (char *)malloc((size_t)idx_len + 1);
    if (!idx) { fclose(f); return tt_fail("out of memory"); }
    if (fseek(f, (long)idx_off, SEEK_SET) != 0
        || fread(idx, 1, (size_t)idx_len, f) != (size_t)idx_len) {
        free(idx); fclose(f); return tt_fail("cannot read the index");
    }
    idx[idx_len] = '\0';
    fclose(f);

    printf("%s\n", path);
    printf("  version %u, %llu docs, %llu windows, %llu ids (%.2f MB)\n",
           ver, (unsigned long long)n_docs, (unsigned long long)n_wins,
           (unsigned long long)n_ids, (double)(n_ids * 4) / (1024.0 * 1024.0));

    tt_arena a;
    if (tt_arena_init(&a, tt_json_bound(idx, (size_t)idx_len) + 4096, "index") != 0) {
        free(idx); return -1;
    }
    tt_json *root = tt_json_parse(idx, (size_t)idx_len, &a);
    if (!root) { tt_arena_free(&a); free(idx); return -1; }

    const tt_json *meta = tt_json_get(root, "meta");
    if (meta) {
        static const char *keys[] = { "corpus", "tokenizer", "window", "vocab",
                                      "min_section", "verified", "n_ids",
                                      "train_ids", "holdout_ids",
                                      "n_holdout_docs", "build", NULL };
        for (int k = 0; keys[k]; k++) {
            const tt_json *v = tt_json_get(meta, keys[k]);
            if (!v) continue;
            if (v->type == TT_JSON_STRING)
                printf("  %-15s %.*s\n", keys[k], (int)v->v.str.len, v->v.str.ptr);
            else if (v->type == TT_JSON_INT)
                printf("  %-15s %lld\n", keys[k], (long long)v->v.i);
            else if (v->type == TT_JSON_BOOL)
                printf("  %-15s %s\n", keys[k], v->v.boolean ? "true" : "false");
        }
    }

    if (list) {
        const tt_json *dl = tt_json_get(root, "docs");
        printf("\n  %-58s %7s %5s %s\n", "document", "ids", "secs", "split");
        for (const tt_json *d = dl ? dl->child : NULL; d; d = d->next) {
            size_t tl = 0;
            const char *t = tt_json_str(tt_json_get(d, "title"), &tl);
            char cut[256];
            int used = u8_clip(cut, sizeof(cut), t, tl, 58);
            printf("  %s%*s %7lld %5lld %s\n", cut, 58 - used, "",
                   (long long)tt_json_int(tt_json_get(d, "ids"), 0),
                   (long long)tt_json_int(tt_json_get(d, "sections"), 0),
                   tt_json_bool(tt_json_get(d, "holdout"), 0) ? "HOLDOUT" : "train");
        }
    }

    tt_arena_free(&a);
    free(idx);
    return 0;
}

/* ------------------------------------------------------------------ main */

static int run(int argc, char **argv)
{
    int i = 1;
    const char *cmd = NULL;
    char **sub = argv + 1;
    int subc = argc - 1;

    for (; i < argc; i++) {
        if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose")) {
            tt_verbose(1);
            sub = argv + i + 1; subc = argc - i - 1;
            continue;
        }
        cmd = argv[i];
        sub = argv + i + 1; subc = argc - i - 1;
        break;
    }
    if (!cmd) { usage(); return 2; }

    int rc;
    if      (!strcmp(cmd, "pack"))    rc = cmd_pack(subc, sub);
    else if (!strcmp(cmd, "inspect")) rc = cmd_inspect(subc, sub);
    else if (!strcmp(cmd, "help"))    { usage(); return 0; }
    else { tt_fail("unknown command '%s'", cmd); usage(); rc = -1; }

    if (rc != 0 && tt_error()[0]) fprintf(stderr, "error: %s\n", tt_error());
    return rc == 0 ? 0 : 1;
}

#if defined(_WIN32)

/* `wmain` rather than `main` plus CommandLineToArgvW: the CRT already hands the
 * arguments over in UTF-16, and going through the shell helper would drag in
 * shell32 for nothing.  Everything past this point is UTF-8. */
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
