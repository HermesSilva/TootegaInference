#include "json.h"

#include <stdlib.h>
#include <string.h>

/* Profundidade maxima.  O parser e recursivo e o arquivo vem de terceiros;
 * sem este limite, um `[[[[[...` de 100 KB derruba a pilha. */
#define MAX_DEPTH 128

typedef struct {
    const char *begin;
    const char *p;
    const char *end;
    tt_arena   *arena;
    int         depth;
} parser;

static int fail_at(parser *ps, const char *msg)
{
    size_t off = (size_t)(ps->p - ps->begin);
    if (ps->p < ps->end)
        return tt_fail("json: %s no byte %zu (0x%02X)", msg, off,
                       (unsigned char)*ps->p);
    return tt_fail("json: %s no fim da entrada (byte %zu)", msg, off);
}

/* ------------------------------------------------------------ varredura */

static void skip_ws(parser *ps)
{
    while (ps->p < ps->end) {
        char c = *ps->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ps->p++;
        else break;
    }
}

static tt_json *alloc_node(parser *ps)
{
    tt_json *n = (tt_json *)tt_arena_alloc(ps->arena, sizeof(tt_json), 8);
    if (n) memset(n, 0, sizeof(*n));
    return n;
}

/* ------------------------------------------------------------- strings */

static int hex4(const char *p, unsigned *out)
{
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        v <<= 4;
        if      (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return -1;
    }
    *out = v;
    return 0;
}

static size_t utf8_put(unsigned char *dst, unsigned cp)
{
    if (cp < 0x80)    { dst[0] = (unsigned char)cp; return 1; }
    if (cp < 0x800)   { dst[0] = (unsigned char)(0xC0 | (cp >> 6));
                        dst[1] = (unsigned char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { dst[0] = (unsigned char)(0xE0 | (cp >> 12));
                        dst[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
                        dst[2] = (unsigned char)(0x80 | (cp & 0x3F)); return 3; }
    dst[0] = (unsigned char)(0xF0 | (cp >> 18));
    dst[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
    dst[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
    dst[3] = (unsigned char)(0x80 | (cp & 0x3F));
    return 4;
}

/* Le a string em ps->p (que aponta para a aspa de abertura) e devolve
 * ponteiro/tamanho.  Sem escape, aponta para o proprio buffer -- e este o
 * caminho comum e e onde a copia zero acontece. */
static int parse_string(parser *ps, const char **out, size_t *out_len)
{
    if (ps->p >= ps->end || *ps->p != '"') return fail_at(ps, "esperava string");
    ps->p++;

    const char *start   = ps->p;
    int         escaped = 0;

    while (ps->p < ps->end) {
        unsigned char c = (unsigned char)*ps->p;
        if (c == '"') break;
        if (c == '\\') {
            escaped = 1;
            ps->p++;
            if (ps->p >= ps->end) return fail_at(ps, "escape truncado");
            ps->p++;
            continue;
        }
        if (c < 0x20) return fail_at(ps, "byte de controle cru na string");
        ps->p++;
    }
    if (ps->p >= ps->end) return fail_at(ps, "string sem aspa de fechamento");

    const char  *raw     = start;
    size_t       raw_len = (size_t)(ps->p - start);
    ps->p++;                                   /* consome a aspa final */

    if (!escaped) {
        *out = raw;
        *out_len = raw_len;
        return 0;
    }

    /* Desescapando: o resultado nunca cresce (\uXXXX de 6 bytes vira no maximo
     * 3; um par substituto de 12 vira 4), entao raw_len basta. */
    unsigned char *dst = (unsigned char *)tt_arena_alloc(ps->arena, raw_len ? raw_len : 1, 1);
    if (!dst) return -1;

    size_t w = 0;
    for (size_t i = 0; i < raw_len; ) {
        unsigned char c = (unsigned char)raw[i];
        if (c != '\\') { dst[w++] = c; i++; continue; }

        i++;
        if (i >= raw_len) { ps->p = raw + i; return fail_at(ps, "escape truncado"); }
        char e = raw[i++];
        switch (e) {
        case '"':  dst[w++] = '"';  break;
        case '\\': dst[w++] = '\\'; break;
        case '/':  dst[w++] = '/';  break;
        case 'b':  dst[w++] = '\b'; break;
        case 'f':  dst[w++] = '\f'; break;
        case 'n':  dst[w++] = '\n'; break;
        case 'r':  dst[w++] = '\r'; break;
        case 't':  dst[w++] = '\t'; break;
        case 'u': {
            unsigned cp;
            if (i + 4 > raw_len || hex4(raw + i, &cp) != 0) {
                ps->p = raw + i;
                return fail_at(ps, "\\u com hex invalido");
            }
            i += 4;
            if (cp >= 0xD800 && cp <= 0xDBFF) {          /* substituto alto */
                if (i + 6 <= raw_len && raw[i] == '\\' && raw[i + 1] == 'u') {
                    unsigned lo;
                    if (hex4(raw + i + 2, &lo) == 0 && lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                        i += 6;
                    } else {
                        cp = 0xFFFD;                     /* par quebrado */
                    }
                } else {
                    cp = 0xFFFD;
                }
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {   /* substituto solto */
                cp = 0xFFFD;
            }
            w += utf8_put(dst + w, cp);
            break;
        }
        default:
            ps->p = raw + i - 1;
            return fail_at(ps, "escape desconhecido");
        }
    }

    *out = (const char *)dst;
    *out_len = w;
    return 0;
}

/* ------------------------------------------------------------- numeros */

static int parse_number(parser *ps, tt_json *node)
{
    const char *start = ps->p;
    int is_real = 0;

    if (ps->p < ps->end && (*ps->p == '-' || *ps->p == '+')) ps->p++;
    while (ps->p < ps->end) {
        char c = *ps->p;
        if (c >= '0' && c <= '9') { ps->p++; continue; }
        if (c == '.' || c == 'e' || c == 'E') { is_real = 1; ps->p++; continue; }
        if ((c == '+' || c == '-') &&
            (ps->p[-1] == 'e' || ps->p[-1] == 'E')) { ps->p++; continue; }
        break;
    }

    size_t len = (size_t)(ps->p - start);
    if (!len) return fail_at(ps, "numero vazio");

    char buf[64];
    if (len >= sizeof(buf)) {
        ps->p = start;
        return fail_at(ps, "numero longo demais");
    }
    memcpy(buf, start, len);
    buf[len] = '\0';

    char *endp = NULL;
    if (!is_real) {
        /* Offsets, tamanhos e shapes sao inteiros e tem de sobreviver exatos:
         * um offset de 19 GB cabe em int64 e nao cabe no significando de um
         * float de 32 bits.  Se estourar int64, degrada para real. */
        long long v = strtoll(buf, &endp, 10);
        if (endp == buf + len) {
            node->type = TT_JSON_INT;
            node->v.i  = (int64_t)v;
            return 0;
        }
    }

    double d = strtod(buf, &endp);
    if (endp != buf + len) {
        ps->p = start;
        return fail_at(ps, "numero malformado");
    }
    node->type = TT_JSON_REAL;
    node->v.r  = d;
    return 0;
}

/* -------------------------------------------------------------- valores */

static int parse_value(parser *ps, tt_json *node);

static int parse_object(parser *ps, tt_json *node)
{
    node->type = TT_JSON_OBJECT;
    ps->p++;                                   /* '{' */

    tt_json *tail = NULL;
    skip_ws(ps);
    if (ps->p < ps->end && *ps->p == '}') { ps->p++; return 0; }

    for (;;) {
        skip_ws(ps);

        tt_json *m = alloc_node(ps);
        if (!m) return -1;
        if (parse_string(ps, &m->key, &m->key_len) != 0) return -1;

        skip_ws(ps);
        if (ps->p >= ps->end || *ps->p != ':') return fail_at(ps, "esperava ':'");
        ps->p++;
        skip_ws(ps);

        if (parse_value(ps, m) != 0) return -1;

        if (tail) tail->next = m; else node->child = m;
        tail = m;
        node->count++;

        skip_ws(ps);
        if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
        if (ps->p < ps->end && *ps->p == '}') { ps->p++; return 0; }
        return fail_at(ps, "esperava ',' ou '}'");
    }
}

static int parse_array(parser *ps, tt_json *node)
{
    node->type = TT_JSON_ARRAY;
    ps->p++;                                   /* '[' */

    tt_json *tail = NULL;
    skip_ws(ps);
    if (ps->p < ps->end && *ps->p == ']') { ps->p++; return 0; }

    for (;;) {
        skip_ws(ps);

        tt_json *e = alloc_node(ps);
        if (!e) return -1;
        if (parse_value(ps, e) != 0) return -1;

        if (tail) tail->next = e; else node->child = e;
        tail = e;
        node->count++;

        skip_ws(ps);
        if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
        if (ps->p < ps->end && *ps->p == ']') { ps->p++; return 0; }
        return fail_at(ps, "esperava ',' ou ']'");
    }
}

static int lit(parser *ps, const char *word, size_t n)
{
    if ((size_t)(ps->end - ps->p) < n || memcmp(ps->p, word, n) != 0) return -1;
    ps->p += n;
    return 0;
}

static int parse_value(parser *ps, tt_json *node)
{
    if (++ps->depth > MAX_DEPTH) {
        ps->depth--;
        return fail_at(ps, "aninhamento alem do limite");
    }

    int rc;
    skip_ws(ps);
    if (ps->p >= ps->end) { ps->depth--; return fail_at(ps, "valor esperado"); }

    switch (*ps->p) {
    case '{': rc = parse_object(ps, node); break;
    case '[': rc = parse_array(ps, node);  break;
    case '"':
        node->type = TT_JSON_STRING;
        rc = parse_string(ps, &node->v.str.ptr, &node->v.str.len);
        break;
    case 't':
        rc = lit(ps, "true", 4);
        if (rc == 0) { node->type = TT_JSON_BOOL; node->v.boolean = 1; }
        else rc = fail_at(ps, "literal invalido");
        break;
    case 'f':
        rc = lit(ps, "false", 5);
        if (rc == 0) { node->type = TT_JSON_BOOL; node->v.boolean = 0; }
        else rc = fail_at(ps, "literal invalido");
        break;
    case 'n':
        rc = lit(ps, "null", 4);
        if (rc == 0) node->type = TT_JSON_NULL;
        else rc = fail_at(ps, "literal invalido");
        break;
    default:
        rc = parse_number(ps, node);
        break;
    }

    ps->depth--;
    return rc;
}

/* --------------------------------------------------------------- API */

size_t tt_json_bound(const char *data, size_t len)
{
    /* Conta tokens que podem iniciar um no.  Chaves de objeto entram na conta
     * como se fossem nos -- e superestimativa, que e o lado seguro. */
    size_t nodes = 0;
    int in_num = 0;

    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)data[i];

        if (c == '"') {                        /* pula a string inteira */
            nodes++;
            in_num = 0;
            i++;
            while (i < len) {
                if (data[i] == '\\') { i += 2; continue; }
                if (data[i] == '"') break;
                i++;
            }
            continue;
        }

        if (c == '{' || c == '[') { nodes++; in_num = 0; continue; }
        if (c == 't' || c == 'f' || c == 'n') { nodes++; in_num = 0; continue; }

        if ((c >= '0' && c <= '9') || c == '-') {
            if (!in_num) { nodes++; in_num = 1; }
            continue;
        }
        if (c == '.' || c == 'e' || c == 'E' || c == '+') continue;   /* segue no numero */
        in_num = 0;
    }

    /* Nos + espaco para as strings desescapadas (nunca crescem) + folga. */
    return nodes * sizeof(tt_json) + len + 4096;
}

tt_json *tt_json_parse(const char *data, size_t len, tt_arena *arena)
{
    parser ps = { data, data, data + len, arena, 0 };

    /* BOM UTF-8: alguns arquivos HF tem.  Tolerar custa tres bytes. */
    if (len >= 3 && (unsigned char)data[0] == 0xEF &&
        (unsigned char)data[1] == 0xBB && (unsigned char)data[2] == 0xBF)
        ps.p += 3;

    tt_json *root = alloc_node(&ps);
    if (!root) return NULL;
    if (parse_value(&ps, root) != 0) return NULL;

    skip_ws(&ps);
    if (ps.p != ps.end) {
        fail_at(&ps, "lixo depois do valor raiz");
        return NULL;
    }
    return root;
}

const tt_json *tt_json_get(const tt_json *obj, const char *key)
{
    if (!obj || obj->type != TT_JSON_OBJECT) return NULL;
    size_t n = strlen(key);
    for (const tt_json *m = obj->child; m; m = m->next)
        if (m->key_len == n && memcmp(m->key, key, n) == 0) return m;
    return NULL;
}

const tt_json *tt_json_at(const tt_json *arr, size_t index)
{
    if (!arr || (arr->type != TT_JSON_ARRAY && arr->type != TT_JSON_OBJECT))
        return NULL;
    for (const tt_json *e = arr->child; e; e = e->next)
        if (index-- == 0) return e;
    return NULL;
}

int tt_json_key_eq(const tt_json *node, const char *key)
{
    if (!node || !node->key) return 0;
    size_t n = strlen(key);
    return node->key_len == n && memcmp(node->key, key, n) == 0;
}

int tt_json_str_eq(const tt_json *node, const char *s)
{
    if (!node || node->type != TT_JSON_STRING) return 0;
    size_t n = strlen(s);
    return node->v.str.len == n && memcmp(node->v.str.ptr, s, n) == 0;
}

int64_t tt_json_int(const tt_json *node, int64_t def)
{
    if (!node) return def;
    if (node->type == TT_JSON_INT)  return node->v.i;
    if (node->type == TT_JSON_REAL) return (int64_t)node->v.r;
    if (node->type == TT_JSON_BOOL) return node->v.boolean;
    return def;
}

double tt_json_real(const tt_json *node, double def)
{
    if (!node) return def;
    if (node->type == TT_JSON_REAL) return node->v.r;
    if (node->type == TT_JSON_INT)  return (double)node->v.i;
    if (node->type == TT_JSON_BOOL) return node->v.boolean;
    return def;
}

int tt_json_bool(const tt_json *node, int def)
{
    if (!node) return def;
    if (node->type == TT_JSON_BOOL) return node->v.boolean;
    if (node->type == TT_JSON_INT)  return node->v.i != 0;
    return def;
}

const char *tt_json_str(const tt_json *node, size_t *len)
{
    if (!node || node->type != TT_JSON_STRING) { if (len) *len = 0; return NULL; }
    if (len) *len = node->v.str.len;
    return node->v.str.ptr;
}

int tt_json_strcpy(const tt_json *node, char *dst, size_t dstlen)
{
    if (!node || node->type != TT_JSON_STRING || !dstlen) return -1;
    if (node->v.str.len >= dstlen) return -1;
    memcpy(dst, node->v.str.ptr, node->v.str.len);
    dst[node->v.str.len] = '\0';
    return 0;
}

/* ------------------------------------------------------------ selftest */

#define CHECK(cond, ...) do { if (!(cond)) { tt_arena_free(&a); return tt_fail(__VA_ARGS__); } } while (0)

int tt_json_selftest(void)
{
    static const char doc[] =
        "{\n"
        "  \"meta\": {\"model\":\"GLM-4.7-Flash\",\"group\":64,\"mtp\":true,\"nada\":null},\n"
        "  \"shape\": [1536, 2048],\n"
        "  \"data\":  [1234560, 1572864],\n"
        "  \"esc\":   \"a\\\"b\\\\c\\u00e7\\u0120\\ud83d\\ude00\",\n"
        "  \"real\":  -1.5e3,\n"
        "  \"big\":   19034567890,\n"
        "  \"vazio\": {},\n"
        "  \"varr\":  []\n"
        "}";

    tt_arena a;
    size_t bound = tt_json_bound(doc, sizeof(doc) - 1);
    if (tt_arena_init(&a, bound, "json-selftest") != 0) return -1;

    tt_json *root = tt_json_parse(doc, sizeof(doc) - 1, &a);
    CHECK(root, "parse falhou: %s", tt_error());
    CHECK(root->type == TT_JSON_OBJECT, "raiz nao e objeto");
    CHECK(root->count == 8, "raiz tem %zu membros, esperava 8", root->count);

    const tt_json *meta = tt_json_get(root, "meta");
    CHECK(tt_json_str_eq(tt_json_get(meta, "model"), "GLM-4.7-Flash"), "model errado");
    CHECK(tt_json_int(tt_json_get(meta, "group"), -1) == 64, "group errado");
    CHECK(tt_json_bool(tt_json_get(meta, "mtp"), 0) == 1, "mtp errado");
    CHECK(tt_json_get(meta, "nada")->type == TT_JSON_NULL, "null errado");
    CHECK(tt_json_get(meta, "ausente") == NULL, "chave ausente devolveu no");

    const tt_json *shape = tt_json_get(root, "shape");
    CHECK(shape->count == 2, "shape com %zu elementos", shape->count);
    CHECK(tt_json_int(tt_json_at(shape, 0), 0) == 1536, "shape[0]");
    CHECK(tt_json_int(tt_json_at(shape, 1), 0) == 2048, "shape[1]");
    CHECK(tt_json_at(shape, 2) == NULL, "shape[2] deveria ser NULL");

    /* Offsets grandes exatos: e por isso que inteiro nao vira double. */
    CHECK(tt_json_int(tt_json_get(root, "big"), 0) == 19034567890LL, "int64 perdeu precisao");
    CHECK(tt_json_real(tt_json_get(root, "real"), 0) == -1500.0, "real errado");

    /* Escapes: aspa, barra, cedilha (2 bytes), Gpontuado (2), emoji (par
     * substituto -> 4).  E o que tokenizer.json exige. */
    size_t n = 0;
    const char *esc = tt_json_str(tt_json_get(root, "esc"), &n);
    static const char want[] = "a\"b\\c\xC3\xA7\xC4\xA0\xF0\x9F\x98\x80";
    CHECK(esc && n == sizeof(want) - 1 && memcmp(esc, want, n) == 0,
          "unescape errado (%zu bytes)", n);

    CHECK(tt_json_get(root, "vazio")->count == 0, "objeto vazio");
    CHECK(tt_json_get(root, "varr")->count == 0, "array vazio");

    /* Entrada malformada tem de falhar, nao adivinhar. */
    static const char *bad[] = {
        "{\"a\":1",  "[1,]", "{\"a\" 1}", "\"sem fim", "{}x", "tru", "[1,2]]",
        "{\"a\":\"\\q\"}", "[01,",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        tt_arena_reset(&a);
        tt_json *r = tt_json_parse(bad[i], strlen(bad[i]), &a);
        CHECK(r == NULL, "entrada invalida aceita: %s", bad[i]);
    }

    /* Aninhamento fundo tem de ser recusado antes de estourar a pilha. */
    {
        char deep[MAX_DEPTH * 2 + 32];
        size_t k = 0;
        for (int j = 0; j < MAX_DEPTH + 5; j++) deep[k++] = '[';
        for (int j = 0; j < MAX_DEPTH + 5; j++) deep[k++] = ']';
        tt_arena_reset(&a);
        tt_json *r = tt_json_parse(deep, k, &a);
        CHECK(r == NULL, "aninhamento de %d aceito", MAX_DEPTH + 5);
    }

    tt_arena_free(&a);
    tt_error_clear();
    return 0;
}
