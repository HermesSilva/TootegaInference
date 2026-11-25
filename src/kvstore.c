/* kvstore.c -- see kvstore.h for what this is and the rule that governs it.
 *
 * The file, little-endian, version 1:
 *
 *   header      128 bytes, below
 *   ids         int32[len]
 *   planes      for each plane, for each layer: len * per_pos bytes
 *
 * A "plane" is one array of the state's cache.  f32 and bf16 have one (the
 * values); int8 has two (the codes, then the per-group scales).  In memory a
 * plane is [layer][ctx][per_pos], sized for the whole context; on disk it is
 * [layer][len][per_pos], only what was computed -- which is why a file written
 * with one `--ctx` loads under another.
 *
 * There is no padding between sections and no alignment promise: the file is
 * READ into the cache, never mapped over it.  The cache is written by every
 * token that follows, and a mapping would either dirty the file or need a
 * private copy made by page faults, 4 KB at a time.
 */
#include "kvstore.h"

#include "common.h"
#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KV_VERSION 1u
#define KV_SEED    0x54544B5631ULL          /* "TTKV1" */

static const char KV_MAGIC[8] = { 'T', 'T', 'K', 'V', 0, 0, 0, 1 };

typedef struct {
    char     magic[8];
    uint32_t version;
    uint32_t attn;             /* tt_attn_mode */
    uint32_t kv_prec;          /* tt_kv_prec */
    uint32_t layers;
    uint32_t width;            /* values per position per layer */
    uint32_t reserved;
    uint64_t len;              /* positions stored */
    uint64_t model_fp;         /* tt_model_fingerprint */
    uint64_t ids_hash;         /* of the token sequence, see ids_hash() */
    uint64_t checksum;         /* of everything after the header */
    char     build[16];        /* tt_build_id, NUL padded */
    uint8_t  pad[48];
} kv_header;

typedef char kv_header_is_128_bytes[(sizeof(kv_header) == 128) ? 1 : -1];

/* The selftest provokes every refusal on purpose; it must not fill the log
 * with them. */
static int g_quiet = 0;

/* ----------------------------------------------------------------- hash */

uint64_t tt_hash64(uint64_t seed, const void *data, size_t bytes)
{
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = seed ^ ((uint64_t)bytes * 0xD6E8FEB86659FD93ULL);

    while (bytes >= 8) {
        uint64_t v;
        memcpy(&v, p, 8);
        h = (h ^ v) * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 29;
        p += 8;
        bytes -= 8;
    }
    if (bytes) {
        uint64_t v = 0;
        memcpy(&v, p, bytes);
        h = (h ^ v) * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 29;
    }
    h ^= h >> 32;
    h *= 0xD6E8FEB86659FD93ULL;
    h ^= h >> 32;
    return h;
}

static uint64_t ids_hash(const int32_t *ids, int64_t n)
{
    return tt_hash64(KV_SEED, ids, (size_t)n * sizeof(int32_t));
}

uint64_t tt_model_fingerprint(const tt_model *m)
{
    const tt_container *c = &m->c;
    uint64_t size = (uint64_t)c->map.size;
    uint64_t h = tt_hash64(KV_SEED, &size, sizeof(size));

    if (c->map.base && c->index_len && c->index_off + c->index_len <= c->map.size)
        h = tt_hash64(h, (const char *)c->map.base + c->index_off,
                      (size_t)c->index_len);

    for (size_t i = 0; i < c->n_tensors; i++) {
        const tt_tensor *t = &c->tensors[i];
        if (t->data && t->data_bytes) {
            /* The middle, not the start: the first bytes of a row are as likely
             * as any to survive a small edit, and the middle of the blob is the
             * cheapest place that is not special. */
            size_t off = 0, len = (size_t)t->data_bytes;
            if (len >= 512) { off = (len / 2) & ~(size_t)63; len = 256; }
            h = tt_hash64(h, (const char *)t->data + off, len);
        }
        if (t->scales && t->scales_bytes)
            h = tt_hash64(h, t->scales,
                          t->scales_bytes < 64 ? (size_t)t->scales_bytes : 64);
    }
    return h;
}

void tt_kv_path(char *out, size_t cap, const char *dir,
                const int32_t *ids, int64_t n)
{
    snprintf(out, cap, "%s/%016llx.ttkv", dir,
             (unsigned long long)ids_hash(ids, n));
}

/* --------------------------------------------------------------- planes */

typedef struct {
    unsigned char *base;       /* [layer][ctx][per_pos] */
    size_t         per_pos;    /* bytes one position takes in one layer */
} kv_plane;

static int kv_planes(const tt_state *s, kv_plane p[2], uint32_t *width)
{
    const tt_config *c = &s->m->cfg;
    const size_t W = (size_t)(c->kv_lora_rank + c->qk_rope_head_dim);
    *width = (uint32_t)W;

    if (s->kvc) {
        p[0].base = (unsigned char *)s->kvc;   p[0].per_pos = W * sizeof(float);
        return 1;
    }
    if (s->kvc16) {
        p[0].base = (unsigned char *)s->kvc16; p[0].per_pos = W * sizeof(uint16_t);
        return 1;
    }
    if (s->kvq) {
        p[0].base = (unsigned char *)s->kvq;   p[0].per_pos = W;
        p[1].base = (unsigned char *)s->kvq_scales;
        p[1].per_pos = (W / TT_KV_GROUP) * sizeof(float);
        return 2;
    }
    return 0;      /* the expanded formulation: not stored, see kvstore.h */
}

static unsigned char *plane_layer(const tt_state *s, const kv_plane *p, int64_t li)
{
    return p->base + (size_t)li * (size_t)s->ctx * p->per_pos;
}

/* ----------------------------------------------------------------- save */

int tt_kv_save(const tt_state *s, const char *path, uint64_t fingerprint,
               int64_t n)
{
    kv_plane pl[2];
    uint32_t width = 0;
    const int np = kv_planes(s, pl, &width);
    if (!np)
        return tt_fail("kv save: only the absorbed formulation is stored");
    if (n <= 0 || n > s->len)
        return tt_fail("kv save: asked for %lld positions, the state holds %lld",
                       (long long)n, (long long)s->len);

    const int64_t L = s->m->cfg.num_hidden_layers;

    kv_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, KV_MAGIC, sizeof(h.magic));
    h.version  = KV_VERSION;
    h.attn     = (uint32_t)s->m->attn;
    h.kv_prec  = (uint32_t)s->m->kv_prec;
    h.layers   = (uint32_t)L;
    h.width    = width;
    h.len      = (uint64_t)n;
    h.model_fp = fingerprint;
    h.ids_hash = ids_hash(s->tokens, n);
    snprintf(h.build, sizeof(h.build), "%s", tt_build_id());

    /* The checksum goes in the header, which is written first, so it is taken
     * in a pass of its own.  It is one read of data that is about to be read
     * again by the write -- cheap next to the prefill that produced it. */
    uint64_t sum = tt_hash64(KV_SEED, s->tokens, (size_t)n * sizeof(int32_t));
    for (int p = 0; p < np; p++)
        for (int64_t li = 0; li < L; li++)
            sum = tt_hash64(sum, plane_layer(s, &pl[p], li),
                            (size_t)n * pl[p].per_pos);
    h.checksum = sum;

    const size_t plen = strlen(path);
    char *tmp = (char *)malloc(plen + 8);
    if (!tmp) return tt_fail("kv save: out of memory");
    snprintf(tmp, plen + 8, "%s.tmp", path);

    FILE *f = fopen(tmp, "wb");
    if (!f) { free(tmp); return tt_fail("kv save: cannot create %s.tmp", path); }

    int ok = fwrite(&h, sizeof(h), 1, f) == 1
          && fwrite(s->tokens, sizeof(int32_t), (size_t)n, f) == (size_t)n;
    for (int p = 0; ok && p < np; p++)
        for (int64_t li = 0; ok && li < L; li++) {
            const size_t bytes = (size_t)n * pl[p].per_pos;
            ok = fwrite(plane_layer(s, &pl[p], li), 1, bytes, f) == bytes;
        }

    /* On the device BEFORE the rename: otherwise the name can reach the disk
     * ahead of the bytes it names. */
    if (ok) ok = tt_file_commit(f) == 0;
    if (fclose(f) != 0) ok = 0;

    if (ok) ok = tt_file_replace(tmp, path) == 0;
    if (!ok) remove(tmp);
    free(tmp);

    return ok ? 0 : tt_fail("kv save: could not write %s (disk full?)", path);
}

/* ----------------------------------------------------------------- load */

static int64_t refuse(FILE *f, const char *path, const char *why)
{
    if (f) fclose(f);
    if (!g_quiet) tt_log("  kv cache  ignored %s: %s", path, why);
    return 0;
}

int64_t tt_kv_load(tt_state *s, const char *path, uint64_t fingerprint,
                   const int32_t *ids, int64_t n)
{
    kv_plane pl[2];
    uint32_t width = 0;
    const int np = kv_planes(s, pl, &width);
    if (!np || n <= 0) return 0;

    FILE *f = fopen(path, "rb");
    if (!f) {
        tt_dbg("kv cache: no file for this prefix (%s)", path);
        return 0;
    }

    const int64_t L = s->m->cfg.num_hidden_layers;

    kv_header h;
    if (fread(&h, sizeof(h), 1, f) != 1)
        return refuse(f, path, "shorter than a header");
    if (memcmp(h.magic, KV_MAGIC, sizeof(h.magic)) != 0)
        return refuse(f, path, "not a KV file");
    if (h.version != KV_VERSION)
        return refuse(f, path, "format version is not this engine's");

    char build[16];
    memset(build, 0, sizeof(build));
    snprintf(build, sizeof(build), "%s", tt_build_id());
    if (memcmp(h.build, build, sizeof(build)) != 0)
        return refuse(f, path, "written by another build of the engine");
    if (h.model_fp != fingerprint)
        return refuse(f, path, "computed with another container");
    if (h.attn != (uint32_t)s->m->attn || h.kv_prec != (uint32_t)s->m->kv_prec)
        return refuse(f, path, "another attention formulation or KV precision");
    if (h.layers != (uint32_t)L || h.width != width)
        return refuse(f, path, "another KV shape");

    /* No partial match (docs/12): the file stands for one exact sequence. */
    if (h.len != (uint64_t)n)
        return refuse(f, path, "holds a different number of positions");
    if (n > s->ctx)
        return refuse(f, path, "does not fit in this --ctx");
    if (h.ids_hash != ids_hash(ids, n))
        return refuse(f, path, "another token sequence");

    /* The hash named the file and matched; the ids themselves settle it. */
    int32_t *stored = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (!stored) return refuse(f, path, "out of memory");
    const int same = fread(stored, sizeof(int32_t), (size_t)n, f) == (size_t)n
                  && memcmp(stored, ids, (size_t)n * sizeof(int32_t)) == 0;
    free(stored);
    if (!same) return refuse(f, path, "another token sequence (hash collision)");

    /* From here on the cache is being overwritten.  Whatever it held is gone,
     * and until the last check passes it holds nothing. */
    s->len = 0;
    s->logits_len = 0;

    uint64_t sum = tt_hash64(KV_SEED, ids, (size_t)n * sizeof(int32_t));
    for (int p = 0; p < np; p++)
        for (int64_t li = 0; li < L; li++) {
            const size_t bytes = (size_t)n * pl[p].per_pos;
            unsigned char *dst = plane_layer(s, &pl[p], li);
            if (fread(dst, 1, bytes, f) != bytes)
                return refuse(f, path, "truncated");
            sum = tt_hash64(sum, dst, bytes);
        }
    if (sum != h.checksum)
        return refuse(f, path, "checksum does not match (corrupt)");

    fclose(f);
    memcpy(s->tokens, ids, (size_t)n * sizeof(int32_t));
    s->len = n;

    /* Used just now: eviction goes by modification time, and a file that is
     * read on every start must not be the oldest in the directory. */
    tt_file_touch(path);
    return n;
}

/* ------------------------------------------------------------- eviction */

uint64_t tt_kv_file_bytes(const tt_state *s, int64_t n)
{
    kv_plane pl[2];
    uint32_t width = 0;
    const int np = kv_planes(s, pl, &width);
    uint64_t per_pos = sizeof(int32_t);
    for (int p = 0; p < np; p++)
        per_pos += (uint64_t)pl[p].per_pos * (uint64_t)s->m->cfg.num_hidden_layers;
    return sizeof(kv_header) + (uint64_t)n * per_pos;
}

typedef struct {
    char     name[64];
    uint64_t bytes;
    int64_t  mtime;
    int      is_tmp;
} kv_entry;

typedef struct {
    kv_entry *v;
    size_t    n, cap;
    int       failed;
} kv_listing;

static int ends_with(const char *s, const char *suffix)
{
    const size_t a = strlen(s), b = strlen(suffix);
    return a >= b && memcmp(s + a - b, suffix, b) == 0;
}

static void collect(void *arg, const char *name, uint64_t bytes, int64_t mtime)
{
    kv_listing *l = (kv_listing *)arg;

    /* Only what this module wrote.  The directory is the operator's, and a
     * cache that deletes files it does not recognise is a hazard, not a cache. */
    const int tmp = ends_with(name, ".ttkv.tmp");
    if (!tmp && !ends_with(name, ".ttkv")) return;
    if (strlen(name) >= sizeof(l->v[0].name)) return;

    if (l->n == l->cap) {
        const size_t cap = l->cap ? l->cap * 2 : 32;
        kv_entry *v = (kv_entry *)realloc(l->v, cap * sizeof(kv_entry));
        if (!v) { l->failed = 1; return; }
        l->v = v; l->cap = cap;
    }
    kv_entry *e = &l->v[l->n++];
    snprintf(e->name, sizeof(e->name), "%s", name);
    e->bytes = bytes; e->mtime = mtime; e->is_tmp = tmp;
}

static int oldest_first(const void *a, const void *b)
{
    const kv_entry *x = (const kv_entry *)a, *y = (const kv_entry *)b;
    if (x->mtime != y->mtime) return (x->mtime > y->mtime) - (x->mtime < y->mtime);
    return strcmp(x->name, y->name);        /* a total order, so runs repeat */
}

int tt_kv_evict(const char *dir, uint64_t budget, uint64_t incoming, int *removed)
{
    if (removed) *removed = 0;
    if (incoming > budget) return -1;

    kv_listing l;
    memset(&l, 0, sizeof(l));
    if (tt_dir_list(dir, collect, &l) != 0 || l.failed) { free(l.v); return 0; }

    qsort(l.v, l.n, sizeof(kv_entry), oldest_first);

    uint64_t total = 0;
    for (size_t i = 0; i < l.n; i++) if (!l.v[i].is_tmp) total += l.v[i].bytes;

    char full[1024];
    for (size_t i = 0; i < l.n; i++) {
        const kv_entry *e = &l.v[i];
        const int over = !e->is_tmp && total + incoming > budget;
        if (!e->is_tmp && !over) continue;
        if (snprintf(full, sizeof(full), "%s/%s", dir, e->name) >= (int)sizeof(full))
            continue;
        if (remove(full) == 0) {
            if (!e->is_tmp) { total -= e->bytes; if (removed) (*removed)++; }
        }
    }
    free(l.v);
    return 0;
}

/* ------------------------------------------------------------- selftest */

static void fake_model(tt_model *m, tt_kv_prec prec)
{
    memset(m, 0, sizeof(*m));
    tt_config *c = &m->cfg;
    c->num_hidden_layers = 3;   c->hidden_size = 64;      c->vocab_size = 256;
    c->num_attention_heads = 2; c->q_lora_rank = 32;
    c->kv_lora_rank = 128;      c->qk_rope_head_dim = 64; c->qk_nope_head_dim = 32;
    c->qk_head_dim = 96;        c->v_head_dim = 32;
    c->intermediate_size = 128; c->moe_intermediate_size = 64;
    c->n_shared_experts = 1;    c->n_routed_experts = 8;  c->num_experts_per_tok = 2;
    m->attn = TT_ATTN_ABSORBED;
    m->kv_prec = prec;
}

static void fill_state(tt_state *s, int64_t n, uint32_t seed)
{
    kv_plane pl[2];
    uint32_t width = 0;
    const int np = kv_planes(s, pl, &width);
    const int64_t L = s->m->cfg.num_hidden_layers;
    uint32_t x = seed | 1u;

    /* The WHOLE context is filled, not just `n` positions: what lies past `n`
     * must not reach the file, and garbage there is how that is tested. */
    for (int p = 0; p < np; p++)
        for (int64_t li = 0; li < L; li++) {
            unsigned char *d = plane_layer(s, &pl[p], li);
            for (size_t i = 0; i < (size_t)s->ctx * pl[p].per_pos; i++) {
                x = x * 1664525u + 1013904223u;
                d[i] = (unsigned char)(x >> 24);
            }
        }
    for (int64_t i = 0; i < s->ctx; i++) s->tokens[i] = (int32_t)(1000 + 7 * i);
    s->len = n;
}

static int same_prefix(const tt_state *a, const tt_state *b, int64_t n)
{
    kv_plane pa[2], pb[2];
    uint32_t wa = 0, wb = 0;
    const int np = kv_planes(a, pa, &wa);
    if (np != kv_planes(b, pb, &wb) || wa != wb) return 0;
    for (int p = 0; p < np; p++)
        for (int64_t li = 0; li < a->m->cfg.num_hidden_layers; li++)
            if (memcmp(plane_layer(a, &pa[p], li), plane_layer(b, &pb[p], li),
                       (size_t)n * pa[p].per_pos) != 0) return 0;
    return memcmp(a->tokens, b->tokens, (size_t)n * sizeof(int32_t)) == 0;
}

static int selftest_prec(tt_kv_prec prec)
{
    static const char path[] = "tt_kvstore_selftest.ttkv";
    const uint64_t fp = 0x1234567890ABCDEFULL;
    const int64_t n = 10;
    const char *name = tt_kv_prec_name(prec);

    tt_model m;
    fake_model(&m, prec);

    /* Two contexts of different size on purpose: the file must not depend on
     * the `--ctx` it was written under. */
    tt_state a, b;
    if (tt_state_init(&a, &m, 16) != 0) return -1;
    if (tt_state_init(&b, &m, 24) != 0) { tt_state_free(&a); return -1; }

    int rc = 0;
    #define FAIL(msg) do { rc = tt_fail("kvstore %s: %s", name, msg); goto done; } while (0)

    fill_state(&a, n, 12345u);
    fill_state(&b, 5, 999u);
    if (tt_kv_save(&a, path, fp, n) != 0) { rc = -1; goto done; }

    if (tt_kv_load(&b, path, fp, a.tokens, n) != n) FAIL("a good file did not load");
    if (b.len != n || !same_prefix(&a, &b, n))      FAIL("what came back is not what went in");
    if (tt_state_logits(&b) != NULL)                FAIL("logits claimed valid after a load");

    /* No partial match: one position fewer is a different sequence. */
    fill_state(&b, 5, 999u);
    if (tt_kv_load(&b, path, fp, a.tokens, n - 1) != 0) FAIL("a shorter prefix was accepted");
    if (b.len != 5)                                     FAIL("a refused file touched the state");

    int32_t other[16];
    memcpy(other, a.tokens, sizeof(other));
    other[3] ^= 1;
    if (tt_kv_load(&b, path, fp, other, n) != 0)        FAIL("another token sequence was accepted");
    if (tt_kv_load(&b, path, fp + 1, a.tokens, n) != 0) FAIL("another container was accepted");
    if (b.len != 5)                                     FAIL("a refused file touched the state");

    /* A flipped byte in the payload: the header still matches, so the cache IS
     * overwritten before the checksum fails -- and must end up empty. */
    FILE *f = fopen(path, "r+b");
    if (!f) FAIL("cannot reopen the file");
    unsigned char byte = 0;
    int io = fseek(f, -1, SEEK_END) == 0 && fread(&byte, 1, 1, f) == 1;
    byte ^= 0xFF;
    io = io && fseek(f, -1, SEEK_END) == 0 && fwrite(&byte, 1, 1, f) == 1;
    fclose(f);
    if (!io) FAIL("cannot corrupt the file");

    if (tt_kv_load(&b, path, fp, a.tokens, n) != 0) FAIL("a corrupt file was accepted");
    if (b.len != 0)                                 FAIL("a corrupt file left a half-restored state");

    #undef FAIL
done:
    remove(path);
    tt_state_free(&a);
    tt_state_free(&b);
    return rc;
}

static int put_file(const char *path, size_t bytes)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int ok = 1;
    for (size_t i = 0; ok && i < bytes; i++) ok = fputc('k', f) != EOF;
    return (fclose(f) == 0 && ok) ? 0 : -1;
}

static int exists(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f) fclose(f);
    return f != NULL;
}

static int selftest_evict(void)
{
    #define D "tt_kvstore_selftest.dir"
    static const char *all[] = { D "/a.ttkv", D "/b.ttkv", D "/c.ttkv",
                                 D "/half.ttkv.tmp", D "/notours.txt" };
    int rc = 0, removed = -1;
    #define FAIL(msg) do { rc = tt_fail("kvstore eviction: %s", msg); goto done; } while (0)

    if (tt_mkdir(D) != 0) return tt_fail("kvstore eviction: cannot create " D);

    /* Written a, b, c with a pause in between, then `a` is used again: the
     * order of last use is b, c, a. */
    for (int i = 0; i < 3; i++) {
        if (put_file(all[i], 1000) != 0) FAIL("cannot write a test file");
        tt_sleep_ms(30);
    }
    if (put_file(all[3], 10) != 0 || put_file(all[4], 10) != 0) FAIL("cannot write a test file");
    if (tt_file_touch(all[0]) != 0) FAIL("cannot touch a file");

    /* A file larger than the whole budget: refused, and nothing pays for it. */
    if (tt_kv_evict(D, 500, 1000, &removed) != -1 || removed != 0) FAIL("an oversized file was let in");
    if (!exists(all[0]) || !exists(all[1]) || !exists(all[2])) FAIL("a refusal removed files");

    /* 3000 held + 1000 coming under 2500: two must go, the two least recently
     * used -- and the interrupted write goes regardless. */
    if (tt_kv_evict(D, 2500, 1000, &removed) != 0) FAIL("room could not be made");
    if (removed != 2)     FAIL("the wrong number of files went");
    if (!exists(all[0]))  FAIL("the most recently used file was removed");
    if (exists(all[1]) || exists(all[2])) FAIL("an older file survived a newer one");
    if (exists(all[3]))   FAIL("an interrupted write was kept");
    if (!exists(all[4]))  FAIL("a file that is not ours was removed");

    /* Already under budget: nothing more to do. */
    if (tt_kv_evict(D, 2500, 1000, &removed) != 0 || removed != 0) FAIL("evicted with room to spare");

    #undef FAIL
done:
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) remove(all[i]);
    tt_rmdir(D);
    #undef D
    return rc;
}

int tt_kvstore_selftest(void)
{
    static const tt_kv_prec precs[] = { TT_KV_F32, TT_KV_BF16, TT_KV_INT8 };

    /* Chaining: hashing in two calls has to differ from hashing a different
     * split of other bytes, and be stable for the same bytes. */
    const char text[] = "the same bytes, twice";
    if (tt_hash64(1, text, sizeof(text)) != tt_hash64(1, text, sizeof(text)) ||
        tt_hash64(1, text, sizeof(text)) == tt_hash64(2, text, sizeof(text)) ||
        tt_hash64(1, text, sizeof(text)) == tt_hash64(1, text, sizeof(text) - 1))
        return tt_fail("kvstore: the hash does not tell inputs apart");

    g_quiet = 1;
    int rc = 0;
    for (size_t i = 0; rc == 0 && i < sizeof(precs) / sizeof(precs[0]); i++)
        rc = selftest_prec(precs[i]);
    g_quiet = 0;
    if (rc == 0) rc = selftest_evict();
    return rc;
}
