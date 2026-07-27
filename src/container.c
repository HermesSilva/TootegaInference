#include "container.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* Cabecalho fixo -- docs/14, secao "Cabecalho (128 bytes)". */
#define TT_MAGIC       "TOOTEGA\x01"
#define TT_MAGIC_LEN   8
#define TT_HEADER_SIZE 128
#define TT_ALIGN_BLOB  64
#define TT_FORMAT_MAX  1u

const char *tt_kind_name(tt_kind k)
{
    switch (k) {
    case TT_KIND_INT4: return "int4";
    case TT_KIND_INT8: return "int8";
    case TT_KIND_F32:  return "f32";
    case TT_KIND_BF16: return "bf16";
    default:           return "?";
    }
}

/* Leitura little-endian explicita: o arquivo e LE por spec, e ler por cast de
 * ponteiro assumiria a ordem do hospedeiro e o alinhamento do campo. */
static uint32_t rd_u32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd_u64(const unsigned char *p)
{
    return (uint64_t)rd_u32(p) | ((uint64_t)rd_u32(p + 4) << 32);
}

/* -------------------------------------------------------------- hash */

static uint64_t fnv1a(const char *s, size_t n)
{
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ull;
    }
    return h;
}

static int hash_insert(tt_container *c, size_t tensor_index)
{
    const tt_tensor *t = &c->tensors[tensor_index];
    size_t mask = c->n_buckets - 1;
    size_t i = (size_t)(fnv1a(t->name, t->name_len) & mask);

    for (size_t probe = 0; probe < c->n_buckets; probe++) {
        if (c->buckets[i] == 0) {
            c->buckets[i] = (uint32_t)(tensor_index + 1);
            return 0;
        }
        const tt_tensor *o = &c->tensors[c->buckets[i] - 1];
        if (o->name_len == t->name_len && memcmp(o->name, t->name, t->name_len) == 0)
            return tt_fail("container: tensor duplicado no indice: %.*s",
                           (int)t->name_len, t->name);
        i = (i + 1) & mask;
    }
    return tt_fail("container: tabela hash cheia (bug de dimensionamento)");
}

const tt_tensor *tt_container_find(const tt_container *c, const char *name)
{
    if (!c->buckets) return NULL;
    size_t n = strlen(name);
    size_t mask = c->n_buckets - 1;
    size_t i = (size_t)(fnv1a(name, n) & mask);

    for (size_t probe = 0; probe < c->n_buckets; probe++) {
        uint32_t slot = c->buckets[i];
        if (slot == 0) return NULL;
        const tt_tensor *t = &c->tensors[slot - 1];
        if (t->name_len == n && memcmp(t->name, name, n) == 0) return t;
        i = (i + 1) & mask;
    }
    return NULL;
}

const tt_tensor *tt_container_findf(const tt_container *c, const char *fmt, ...)
{
    char name[256];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= sizeof(name)) return NULL;
    return tt_container_find(c, name);
}

/* ------------------------------------------------------- indice -> IR */

static int kind_from_str(const tt_json *node, tt_kind *out)
{
    if (tt_json_str_eq(node, "int4")) { *out = TT_KIND_INT4; return 0; }
    if (tt_json_str_eq(node, "int8")) { *out = TT_KIND_INT8; return 0; }
    if (tt_json_str_eq(node, "f32"))  { *out = TT_KIND_F32;  return 0; }
    if (tt_json_str_eq(node, "bf16")) { *out = TT_KIND_BF16; return 0; }
    size_t n = 0;
    const char *s = tt_json_str(node, &n);
    return tt_fail("container: kind desconhecido '%.*s'", (int)n, s ? s : "");
}

/* Le [offset, bytes] e valida contra a spec: dentro do arquivo, antes do
 * indice, e alinhado a 64.  Blob desalinhado quebra a leitura SIMD sem dar
 * sintoma imediato, entao a recusa e aqui e nao no kernel. */
static int read_span(const tt_container *c, const tt_json *span, const char *what,
                     const char *tname, size_t tname_len,
                     uint64_t *off, uint64_t *bytes)
{
    if (!span || span->type != TT_JSON_ARRAY || span->count != 2)
        return tt_fail("container: '%s' de %.*s nao e [offset, bytes]",
                       what, (int)tname_len, tname);

    int64_t o = tt_json_int(tt_json_at(span, 0), -1);
    int64_t b = tt_json_int(tt_json_at(span, 1), -1);
    if (o < 0 || b < 0)
        return tt_fail("container: '%s' de %.*s com valor negativo",
                       what, (int)tname_len, tname);

    if ((uint64_t)o % TT_ALIGN_BLOB != 0)
        return tt_fail("container: '%s' de %.*s em %lld, nao multiplo de %d "
                       "(docs/14: todo blob alinhado a 64)",
                       what, (int)tname_len, tname, (long long)o, TT_ALIGN_BLOB);

    if ((uint64_t)o < TT_HEADER_SIZE ||
        (uint64_t)o + (uint64_t)b > c->index_off)
        return tt_fail("container: '%s' de %.*s sai da area de dados "
                       "[%d, %llu)", what, (int)tname_len, tname,
                       TT_HEADER_SIZE, (unsigned long long)c->index_off);

    *off = (uint64_t)o;
    *bytes = (uint64_t)b;
    return 0;
}

static int build_tensor(tt_container *c, const tt_json *entry, tt_tensor *t)
{
    t->name     = entry->key;
    t->name_len = entry->key_len;

    if (entry->type != TT_JSON_OBJECT)
        return tt_fail("container: entrada %.*s nao e objeto",
                       (int)t->name_len, t->name);

    if (kind_from_str(tt_json_get(entry, "kind"), &t->kind) != 0) return -1;

    /* shape */
    const tt_json *shape = tt_json_get(entry, "shape");
    if (!shape || shape->type != TT_JSON_ARRAY || shape->count == 0 ||
        shape->count > TT_MAX_DIMS)
        return tt_fail("container: shape invalido em %.*s",
                       (int)t->name_len, t->name);

    t->ndim = (int)shape->count;
    int64_t rows = 1;
    int d = 0;
    for (const tt_json *e = shape->child; e; e = e->next, d++) {
        int64_t v = tt_json_int(e, -1);
        if (v <= 0)
            return tt_fail("container: dimensao %d de %.*s e %lld",
                           d, (int)t->name_len, t->name, (long long)v);
        t->shape[d] = v;
        if (d + 1 < t->ndim) rows *= v;
    }
    t->cols = t->shape[t->ndim - 1];
    t->rows = rows;

    if (read_span(c, tt_json_get(entry, "data"), "data",
                  t->name, t->name_len, &t->data_off, &t->data_bytes) != 0)
        return -1;
    t->data = (const unsigned char *)c->map.base + t->data_off;

    const tt_json *sc = tt_json_get(entry, "scales");
    int has_scales = sc && sc->type != TT_JSON_NULL;
    if (has_scales) {
        if (read_span(c, sc, "scales", t->name, t->name_len,
                      &t->scales_off, &t->scales_bytes) != 0)
            return -1;
        t->scales = (const float *)(const void *)
                    ((const unsigned char *)c->map.base + t->scales_off);
    } else {
        t->scales = NULL;
        t->scales_off = t->scales_bytes = 0;
    }

    t->group = (int32_t)tt_json_int(tt_json_get(entry, "group"), 0);

    /* Coerencia entre shape, group e tamanho dos blobs.  Se isto casa, o
     * kernel pode indexar sem checar nada. */
    const int64_t O = t->rows, I = t->cols;
    uint64_t want_data, want_scales = 0;

    switch (t->kind) {
    case TT_KIND_INT4:
    case TT_KIND_INT8:
        if (t->group <= 0 || I % t->group != 0)
            return tt_fail("container: %.*s tem I=%lld indivisivel por group=%d",
                           (int)t->name_len, t->name, (long long)I, t->group);
        if (t->kind == TT_KIND_INT4) {
            if (I % 2 != 0)
                return tt_fail("container: %.*s int4 com I impar (%lld)",
                               (int)t->name_len, t->name, (long long)I);
            want_data = (uint64_t)O * (uint64_t)(I / 2);
        } else {
            want_data = (uint64_t)O * (uint64_t)I;
        }
        want_scales = (uint64_t)O * (uint64_t)(I / t->group) * sizeof(float);
        if (!t->scales)
            return tt_fail("container: %.*s e %s e nao tem escalas",
                           (int)t->name_len, t->name, tt_kind_name(t->kind));
        break;

    case TT_KIND_F32:
        want_data = (uint64_t)O * (uint64_t)I * 4;
        if (t->scales)
            return tt_fail("container: %.*s e f32 e tem escalas",
                           (int)t->name_len, t->name);
        break;

    case TT_KIND_BF16:
        want_data = (uint64_t)O * (uint64_t)I * 2;
        if (t->scales)
            return tt_fail("container: %.*s e bf16 e tem escalas",
                           (int)t->name_len, t->name);
        break;

    default:
        return tt_fail("container: kind fora de faixa");
    }

    if (t->data_bytes != want_data)
        return tt_fail("container: %.*s (%s, %lldx%lld) diz %llu bytes de dados, "
                       "a spec exige %llu",
                       (int)t->name_len, t->name, tt_kind_name(t->kind),
                       (long long)O, (long long)I,
                       (unsigned long long)t->data_bytes,
                       (unsigned long long)want_data);

    if (t->scales && t->scales_bytes != want_scales)
        return tt_fail("container: %.*s diz %llu bytes de escalas, a spec exige %llu",
                       (int)t->name_len, t->name,
                       (unsigned long long)t->scales_bytes,
                       (unsigned long long)want_scales);

    c->bytes_by_kind[t->kind] += t->data_bytes + t->scales_bytes;
    c->count_by_kind[t->kind]++;
    return 0;
}

/* -------------------------------------------------------------- abrir */

int tt_container_open(tt_container *c, const char *path)
{
    memset(c, 0, sizeof(*c));

    if (tt_map_open(&c->map, path) != 0)
        return tt_fail("nao consegui mapear '%s'", path);

    if (c->map.size < TT_HEADER_SIZE) {
        tt_container_close(c);
        return tt_fail("'%s' tem %zu bytes -- menor que o cabecalho",
                       path, c->map.size);
    }

    const unsigned char *h = (const unsigned char *)c->map.base;

    if (memcmp(h, TT_MAGIC, TT_MAGIC_LEN) != 0) {
        tt_container_close(c);
        return tt_fail("'%s' nao e um container tootega (magic errado)", path);
    }

    c->format_version = rd_u32(h + 8);
    if (c->format_version == 0 || c->format_version > TT_FORMAT_MAX) {
        uint32_t v = c->format_version;
        tt_container_close(c);
        return tt_fail("formato v%u desconhecido (este engine le ate v%u). "
                       "Recusar e deliberado: ler 'na esperanca' produz lixo "
                       "silencioso", v, TT_FORMAT_MAX);
    }

    uint64_t n_tensors = rd_u64(h + 16);
    c->index_off = rd_u64(h + 24);
    c->index_len = rd_u64(h + 32);

    if (c->index_off < TT_HEADER_SIZE || c->index_len == 0 ||
        c->index_off > c->map.size || c->index_len > c->map.size - c->index_off) {
        tt_container_close(c);
        return tt_fail("indice em [%llu, +%llu) nao cabe no arquivo de %zu bytes",
                       (unsigned long long)c->index_off,
                       (unsigned long long)c->index_len, c->map.size);
    }
    if (c->index_off % TT_ALIGN_BLOB != 0) {
        tt_container_close(c);
        return tt_fail("indice em %llu, nao multiplo de 64",
                       (unsigned long long)c->index_off);
    }
    if (n_tensors == 0 || n_tensors > (1u << 24)) {
        tt_container_close(c);
        return tt_fail("n_tensors = %llu fora de faixa",
                       (unsigned long long)n_tensors);
    }

    const char *json = (const char *)c->map.base + c->index_off;

    /* O indice mora dentro do mapeamento; o DOM sai da arena.  Dimensionamos a
     * arena pelo limite exato do parser mais a tabela de tensores, para que
     * estouro seja impossivel em vez de ser erro de execucao. */
    size_t need = tt_json_bound(json, (size_t)c->index_len);
    size_t n_buckets = 1;
    while (n_buckets < n_tensors * 2) n_buckets <<= 1;

    need += (size_t)n_tensors * sizeof(tt_tensor) + n_buckets * sizeof(uint32_t)
            + 2 * TT_ALIGN;

    if (tt_arena_init(&c->arena, need, "indice") != 0) {
        tt_container_close(c);
        return -1;
    }

    c->root = tt_json_parse(json, (size_t)c->index_len, &c->arena);
    if (!c->root) { tt_container_close(c); return -1; }

    c->meta = tt_json_get(c->root, "meta");
    c->config = tt_json_get(c->meta, "config");

    const tt_json *tensors = tt_json_get(c->root, "tensors");
    if (!tensors || tensors->type != TT_JSON_OBJECT) {
        tt_container_close(c);
        return tt_fail("indice sem objeto 'tensors'");
    }
    if (tensors->count != (size_t)n_tensors) {
        size_t got = tensors->count;
        tt_container_close(c);
        return tt_fail("cabecalho diz %llu tensores, indice tem %zu",
                       (unsigned long long)n_tensors, got);
    }

    c->tensors = tt_arena_new(&c->arena, tt_tensor, n_tensors);
    c->buckets = tt_arena_new(&c->arena, uint32_t, n_buckets);
    if (!c->tensors || !c->buckets) { tt_container_close(c); return -1; }
    memset(c->buckets, 0, n_buckets * sizeof(uint32_t));
    c->n_buckets = n_buckets;

    size_t i = 0;
    for (const tt_json *e = tensors->child; e; e = e->next, i++) {
        if (build_tensor(c, e, &c->tensors[i]) != 0) {
            tt_container_close(c);
            return -1;
        }
        c->n_tensors = i + 1;
        if (hash_insert(c, i) != 0) { tt_container_close(c); return -1; }
    }

    tt_dbg("container: v%u, %zu tensores, indice de %llu bytes",
           c->format_version, c->n_tensors, (unsigned long long)c->index_len);
    return 0;
}

void tt_container_close(tt_container *c)
{
    tt_arena_free(&c->arena);
    tt_map_close(&c->map);
    memset(c, 0, sizeof(*c));
}

uint64_t tt_container_resident_bytes(const tt_container *c)
{
    static const char suffix[] = "embed_tokens.weight";
    const size_t slen = sizeof(suffix) - 1;

    uint64_t total = 0;
    for (size_t i = 0; i < c->n_tensors; i++) {
        const tt_tensor *t = &c->tensors[i];
        /* Os dois `embed_tokens` -- o do modelo e o da MTP -- ficam
         * file-backed: uma linha por token, 1,27 GB que nao precisam de
         * residencia (docs/04). */
        if (t->name_len >= slen &&
            memcmp(t->name + t->name_len - slen, suffix, slen) == 0)
            continue;
        total += t->data_bytes + t->scales_bytes;
    }
    return total;
}

/* ------------------------------------------------------------ inspect */

static void print_meta(const tt_container *c)
{
    char buf[64];

    printf("  formato      v%u\n", c->format_version);
    printf("  arquivo      %s\n", tt_bytes(buf, sizeof(buf), c->map.size));
    printf("  indice       em %llu, %s\n",
           (unsigned long long)c->index_off,
           tt_bytes(buf, sizeof(buf), c->index_len));
    printf("  tensores     %zu\n", c->n_tensors);

    const tt_json *model = tt_json_get(c->meta, "model");
    if (model) {
        size_t n = 0;
        const char *s = tt_json_str(model, &n);
        printf("\n  modelo       %.*s\n", (int)n, s ? s : "?");
    }

    int64_t layers = tt_json_int(tt_json_get(c->meta, "num_hidden_layers"), -1);
    if (layers < 0)
        layers = tt_json_int(tt_json_get(c->config, "num_hidden_layers"), -1);
    int64_t group = tt_json_int(tt_json_get(c->meta, "group"), -1);
    int64_t mtp   = tt_json_int(tt_json_get(c->meta, "mtp_layer"), -1);

    if (layers >= 0) printf("  camadas      %lld\n", (long long)layers);
    if (group > 0)   printf("  grupo        %lld pesos por escala\n", (long long)group);
    if (mtp >= 0)    printf("  mtp          camada %lld\n", (long long)mtp);

    /* A busca de clipping nao muda o formato, entao nada aqui a detecta olhando
     * os dados -- so o `meta` diz se ela rodou.  E a diferenca entre 10,81% e
     * 9,67% de erro de saida (docs/04): container sem a chave e container
     * antigo, escrito com absmax puro. */
    const int64_t clip = tt_json_int(tt_json_get(c->meta, "clip_grid"), -1);
    if (clip > 0)
        printf("  clip-grid    %lld pontos de busca de escala\n", (long long)clip);
    else
        printf("  clip-grid    ausente -- absmax puro, ~12%% mais erro (docs/04)\n");

    if (c->config) {
        printf("  config       %zu campos (a arquitetura vem daqui, nao de "
               "config.json ao lado)\n", c->config->count);
    } else {
        printf("  config       AUSENTE -- o engine nao tem de onde ler a "
               "arquitetura\n");
    }
}

static void print_kinds(const tt_container *c)
{
    char buf[64];
    printf("\n  por precisao      tensores          bytes\n");
    uint64_t total = 0;
    for (int k = 0; k < TT_KIND_COUNT; k++) {
        if (!c->count_by_kind[k]) continue;
        printf("    %-6s %14zu %14s\n", tt_kind_name((tt_kind)k),
               c->count_by_kind[k],
               tt_bytes(buf, sizeof(buf), c->bytes_by_kind[k]));
        total += c->bytes_by_kind[k];
    }
    printf("    %-6s %14zu %14s\n", "total", c->n_tensors,
           tt_bytes(buf, sizeof(buf), total));

    uint64_t resident = tt_container_resident_bytes(c);
    printf("\n  residente    %s   (fora: embed_tokens, lido uma linha por token)\n",
           tt_bytes(buf, sizeof(buf), resident));
    printf("  ram livre    %.1f GB de %.1f GB\n",
           tt_ram_avail_gb(), tt_ram_total_gb());

    double gb = (double)resident / 1e9;
    if (gb > tt_ram_avail_gb())
        printf("  ATENCAO      nao cabe na RAM disponivel agora; o engine vai "
               "recusar em vez de degradar (docs/09)\n");
}

int tt_cmd_inspect(int argc, char **argv)
{
    const char *path = NULL;
    const char *list_prefix = NULL;
    int do_list = 0;

    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--list")) {
            do_list = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') list_prefix = argv[++i];
        } else if (argv[i][0] == '-') {
            return tt_fail("inspect: opcao desconhecida '%s'", argv[i]);
        } else if (!path) {
            path = argv[i];
        } else {
            return tt_fail("inspect: mais de um arquivo");
        }
    }

    if (!path)
        return tt_fail("uso: tootega inspect <arquivo.tootega> [--list [prefixo]]");

    double t0 = tt_now();
    tt_container c;
    if (tt_container_open(&c, path) != 0) return -1;
    double dt = tt_now() - t0;

    printf("\n%s\n\n", path);
    print_meta(&c);
    print_kinds(&c);

    if (do_list) {
        size_t plen = list_prefix ? strlen(list_prefix) : 0;
        size_t shown = 0;
        printf("\n  %-58s %-5s %14s %10s\n", "tensor", "kind", "forma", "MB");
        for (size_t i = 0; i < c.n_tensors; i++) {
            const tt_tensor *t = &c.tensors[i];
            if (plen && (t->name_len < plen || memcmp(t->name, list_prefix, plen) != 0))
                continue;

            char shape[32];
            if (t->ndim == 1)
                snprintf(shape, sizeof(shape), "%lld", (long long)t->shape[0]);
            else
                snprintf(shape, sizeof(shape), "%lldx%lld",
                         (long long)t->rows, (long long)t->cols);

            printf("  %-58.*s %-5s %14s %10.2f\n", (int)t->name_len, t->name,
                   tt_kind_name(t->kind), shape,
                   (double)(t->data_bytes + t->scales_bytes) / 1048576.0);
            shown++;
        }
        printf("\n  %zu tensor(es)%s%s\n", shown,
               plen ? " com prefixo " : "", plen ? list_prefix : "");
    }

    printf("\n  aberto e validado em %.0f ms (indice parseado, alinhamento e "
           "tamanhos conferidos)\n\n", dt * 1000.0);

    tt_container_close(&c);
    return 0;
}
