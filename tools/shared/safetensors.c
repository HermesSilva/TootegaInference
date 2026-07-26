#include "safetensors.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int add_shard(st_file *st, const char *folder, const char *shard,
                     size_t slot)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", folder, shard);

    tt_map *m = &st->maps[slot];
    if (tt_map_open(m, path) != 0)
        return tt_fail("nao consegui mapear '%s'", path);
    if (m->size < 8) return tt_fail("'%s' truncado", path);

    /* safetensors: u64 little-endian com o tamanho do cabecalho JSON, depois o
     * JSON, depois os dados. */
    const unsigned char *p = (const unsigned char *)m->base;
    uint64_t hlen = 0;
    for (int i = 7; i >= 0; i--) hlen = (hlen << 8) | p[i];
    if (hlen == 0 || hlen > m->size - 8)
        return tt_fail("'%s': cabecalho de %llu bytes nao cabe", path,
                       (unsigned long long)hlen);

    tt_json *root = tt_json_parse((const char *)m->base + 8, (size_t)hlen,
                                  &st->arena);
    if (!root) return -1;

    const uint64_t base = 8 + hlen;

    for (const tt_json *e = root->child; e; e = e->next) {
        if (e->type != TT_JSON_OBJECT) continue;      /* pula __metadata__ */

        const tt_json *dt = tt_json_get(e, "dtype");
        const tt_json *shape = tt_json_get(e, "shape");
        const tt_json *off = tt_json_get(e, "data_offsets");
        if (!dt || !shape || !off) continue;

        st_entry *x = &st->entries[st->n_entries];
        memset(x, 0, sizeof(*x));

        if (tt_json_str_eq(dt, "BF16"))     x->kind = TT_KIND_BF16;
        else if (tt_json_str_eq(dt, "F32")) x->kind = TT_KIND_F32;
        else {
            size_t n = 0;
            const char *s = tt_json_str(dt, &n);
            return tt_fail("dtype '%.*s' em %.*s nao suportado",
                           (int)n, s ? s : "", (int)e->key_len, e->key);
        }

        int64_t rows = 1;
        int d = 0;
        for (const tt_json *s = shape->child; s && d < TT_MAX_DIMS; s = s->next, d++) {
            const int64_t v = tt_json_int(s, 0);
            x->shape[d] = v;
            if (d + 1 < (int)shape->count) rows *= v;
            x->cols = v;
        }
        x->ndim = d;
        x->rows = rows;

        const int64_t a = tt_json_int(tt_json_at(off, 0), -1);
        const int64_t b = tt_json_int(tt_json_at(off, 1), -1);
        if (a < 0 || b < a || base + (uint64_t)b > m->size)
            return tt_fail("'%s': tensor %.*s fora do arquivo", shard,
                           (int)e->key_len, e->key);

        x->name = e->key;
        x->name_len = e->key_len;
        x->data = (const unsigned char *)m->base + base + a;
        x->bytes = (uint64_t)(b - a);
        x->shard = (int)slot;
        st->n_entries++;
    }
    return 0;
}

int st_open(st_file *st, const char *folder)
{
    memset(st, 0, sizeof(*st));

    char index_path[1024];
    snprintf(index_path, sizeof(index_path), "%s/model.safetensors.index.json",
             folder);

    tt_map idx;
    if (tt_map_open(&idx, index_path) != 0)
        return tt_fail("nao consegui abrir '%s'", index_path);

    /* Os cabecalhos dos 48 shards mais o indice.  Generoso de proposito: a
     * arena morre junto com o objeto e o pico e transitorio. */
    if (tt_arena_init(&st->arena, 512u << 20, "safetensors") != 0) {
        tt_map_close(&idx);
        return -1;
    }

    tt_json *root = tt_json_parse((const char *)idx.base, idx.size, &st->arena);
    const tt_json *wm = root ? tt_json_get(root, "weight_map") : NULL;
    if (!wm || wm->type != TT_JSON_OBJECT) {
        tt_map_close(&idx);
        return tt_fail("'%s' sem weight_map", index_path);
    }

    /* Shards distintos, na ordem de aparicao -- e a ordem dos arquivos, que e
     * a que da leitura sequencial. */
    enum { MAX_SHARDS = 512 };
    char (*names)[128] = (char (*)[128])tt_arena_alloc(&st->arena,
                                MAX_SHARDS * 128, 8);
    size_t n_names = 0;

    for (const tt_json *e = wm->child; e; e = e->next) {
        size_t len = 0;
        const char *s = tt_json_str(e, &len);
        if (!s || len >= 128) continue;

        int seen = 0;
        for (size_t i = 0; i < n_names; i++)
            if (memcmp(names[i], s, len) == 0 && names[i][len] == '\0') { seen = 1; break; }
        if (seen) continue;
        if (n_names >= MAX_SHARDS) { tt_map_close(&idx); return tt_fail("shards demais"); }

        memcpy(names[n_names], s, len);
        names[n_names][len] = '\0';
        n_names++;
    }

    st->maps = (tt_map *)tt_arena_alloc(&st->arena, n_names * sizeof(tt_map), 8);
    st->entries = (st_entry *)tt_arena_alloc(&st->arena,
                        (wm->count + 64) * sizeof(st_entry), 8);
    if (!st->maps || !st->entries) { tt_map_close(&idx); return -1; }
    st->n_maps = n_names;
    tt_map_close(&idx);

    const double t0 = tt_now();
    for (size_t i = 0; i < n_names; i++)
        if (add_shard(st, folder, names[i], i) != 0) return -1;

    tt_dbg("safetensors: %zu shards, %zu tensores em %.2f s",
           st->n_maps, st->n_entries, tt_now() - t0);
    return 0;
}

void st_close(st_file *st)
{
    for (size_t i = 0; i < st->n_maps; i++) tt_map_close(&st->maps[i]);
    tt_arena_free(&st->arena);
    memset(st, 0, sizeof(*st));
}

const st_entry *st_find(const st_file *st, const char *name, size_t len)
{
    for (size_t i = 0; i < st->n_entries; i++)
        if (st->entries[i].name_len == len &&
            memcmp(st->entries[i].name, name, len) == 0)
            return &st->entries[i];
    return NULL;
}

char *st_read_config(const char *folder, size_t *len)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", folder);

    FILE *f = fopen(path, "rb");
    if (!f) { tt_fail("nao consegui abrir '%s'", path); return NULL; }

    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); tt_fail("'%s' vazio", path); return NULL; }

    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); tt_fail("sem memoria"); return NULL; }
    *len = fread(buf, 1, (size_t)n, f);
    buf[*len] = '\0';
    fclose(f);
    return buf;
}
