#include "tokenizer.h"
#include "compat.h"
#include "json.h"
#include "unicode_tables.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------- UTF-8 */

/* Decodifica um code point.  Byte invalido vira U+FFFD de UM byte -- o
 * pre-tokenizador precisa avancar sempre, e entrada malformada nao pode
 * travar o laco. */
static size_t u8_next(const uint8_t *p, size_t n, uint32_t *cp)
{
    if (!n) { *cp = 0; return 0; }
    uint8_t c = p[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0 && n >= 2 && (p[1] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(c & 0x1F) << 6) | (p[1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && n >= 3 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(p[1] & 0x3F) << 6) | (p[2] & 0x3F);
        return 3;
    }
    if ((c & 0xF8) == 0xF0 && n >= 4 && (p[1] & 0xC0) == 0x80 &&
        (p[2] & 0xC0) == 0x80 && (p[3] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(p[1] & 0x3F) << 12) |
              ((uint32_t)(p[2] & 0x3F) << 6) | (p[3] & 0x3F);
        return 4;
    }
    *cp = 0xFFFD;
    return 1;
}

static int is_crlf(uint32_t cp) { return cp == '\r' || cp == '\n'; }

/* ------------------------------------------- pre-tokenizador do cl100k
 *
 * (?i:'s|'t|'re|'ve|'m|'ll|'d)
 * | [^\r\n\p{L}\p{N}]?\p{L}+
 * | \p{N}{1,3}
 * |  ?[^\s\p{L}\p{N}]+[\r\n]*
 * | \s*[\r\n]+
 * | \s+(?!\S)
 * | \s+
 *
 * Alternancia ORDENADA (leftmost-first, semantica do fancy-regex que o
 * `tokenizers` usa): a primeira alternativa que casa vence, mesmo que outra
 * casasse mais.  Cada funcao devolve bytes consumidos, ou 0 se nao casa.
 */

static size_t alt_contraction(const uint8_t *p, size_t n)
{
    if (n < 2 || p[0] != '\'') return 0;
    int c = p[1] | 0x20;                       /* (?i:) so afeta ASCII aqui */
    if (c == 's' || c == 't' || c == 'm' || c == 'd') return 2;
    if (n >= 3) {
        int d = p[2] | 0x20;
        if ((c == 'r' && d == 'e') || (c == 'v' && d == 'e') ||
            (c == 'l' && d == 'l'))
            return 3;
    }
    return 0;
}

static size_t alt_word(const uint8_t *p, size_t n)
{
    uint32_t cp;
    size_t i = 0;

    /* O prefixo opcional so vale se vier letra depois -- por isso ele nao
     * pode ser confirmado antes de olhar o proximo code point. */
    size_t k = u8_next(p, n, &cp);
    if (k && !is_crlf(cp) && !tt_is_letter(cp) && !tt_is_number(cp)) i = k;

    size_t start = i, j = i;
    while (j < n) {
        k = u8_next(p + j, n - j, &cp);
        if (!k || !tt_is_letter(cp)) break;
        j += k;
    }
    if (j == start) return 0;                  /* nenhuma letra: nao casa */
    return j;
}

static size_t alt_number(const uint8_t *p, size_t n)
{
    uint32_t cp;
    size_t i = 0;
    for (int count = 0; count < 3 && i < n; count++) {
        size_t k = u8_next(p + i, n - i, &cp);
        if (!k || !tt_is_number(cp)) break;
        i += k;
    }
    return i;
}

static size_t alt_symbols(const uint8_t *p, size_t n)
{
    size_t i = 0;
    if (n && p[0] == ' ') i = 1;               /* ` ?` -- exatamente um espaco */

    size_t start = i;
    uint32_t cp;
    while (i < n) {
        size_t k = u8_next(p + i, n - i, &cp);
        if (!k || tt_is_space(cp) || tt_is_letter(cp) || tt_is_number(cp)) break;
        i += k;
    }
    if (i == start) return 0;

    while (i < n && (p[i] == '\r' || p[i] == '\n')) i++;   /* [\r\n]* */
    return i;
}

/* Extensao do maior bloco de \s a partir de p. */
static size_t ws_run(const uint8_t *p, size_t n)
{
    uint32_t cp;
    size_t i = 0;
    while (i < n) {
        size_t k = u8_next(p + i, n - i, &cp);
        if (!k || !tt_is_space(cp)) break;
        i += k;
    }
    return i;
}

static size_t alt_ws_newline(const uint8_t *p, size_t n)
{
    /* `\s*[\r\n]+`: o `\s*` e guloso mas tem de sobrar ao menos um [\r\n].
     * O maior `\s*` valido termina no ULTIMO [\r\n] do bloco, e o `[\r\n]+`
     * consome dali ate o fim da sequencia -- que e so ele proprio. */
    const size_t w = ws_run(p, n);
    for (size_t i = w; i-- > 0; )
        if (p[i] == '\r' || p[i] == '\n') return i + 1;
    return 0;
}

static size_t alt_ws_trailing(const uint8_t *p, size_t n)
{
    /* `\s+(?!\S)`: espacos que NAO sao seguidos de nao-espaco.  Se o bloco
     * termina no fim da entrada, leva tudo; senao, deixa o ultimo espaco para
     * a proxima alternativa -- e o que gruda o espaco na palavra seguinte. */
    const size_t w = ws_run(p, n);
    if (!w) return 0;
    if (w == n) return w;
    return w > 1 ? w - 1 : 0;
}

static size_t pretok_next(const uint8_t *p, size_t n)
{
    size_t k;
    if ((k = alt_contraction(p, n)))  return k;
    if ((k = alt_word(p, n)))         return k;
    if ((k = alt_number(p, n)))       return k;
    if ((k = alt_symbols(p, n)))      return k;
    if ((k = alt_ws_newline(p, n)))   return k;
    if ((k = alt_ws_trailing(p, n)))  return k;
    if ((k = ws_run(p, n)))           return k;
    return n ? 1 : 0;                          /* nunca travar */
}

/* ------------------------------------------------------------- hashes */

static uint64_t hash_bytes(const uint8_t *s, size_t n)
{
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= s[i]; h *= 1099511628211ull; }
    return h;
}

static uint64_t hash_u64(uint64_t k)
{
    k ^= k >> 33; k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ull;
    return k ^ (k >> 33);
}

static int32_t vocab_find(const tt_tokenizer *t, const uint8_t *s, size_t n)
{
    const size_t mask = t->n_vocab_slots - 1;
    size_t i = (size_t)(hash_bytes(s, n) & mask);
    for (size_t probe = 0; probe < t->n_vocab_slots; probe++) {
        int32_t id = t->vocab_slots[i];
        if (id < 0) return -1;
        if (t->tokens[id].len == n && memcmp(t->tokens[id].ptr, s, n) == 0)
            return id;
        i = (i + 1) & mask;
    }
    return -1;
}

static void vocab_insert(tt_tokenizer *t, int32_t id)
{
    const size_t mask = t->n_vocab_slots - 1;
    size_t i = (size_t)(hash_bytes(t->tokens[id].ptr, t->tokens[id].len) & mask);
    while (t->vocab_slots[i] >= 0) i = (i + 1) & mask;
    t->vocab_slots[i] = id;
}

static int merge_find(const tt_tokenizer *t, int32_t a, int32_t b,
                      int32_t *rank, int32_t *id)
{
    const uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    const size_t mask = t->n_merge_slots - 1;
    size_t i = (size_t)(hash_u64(key) & mask);
    for (size_t probe = 0; probe < t->n_merge_slots; probe++) {
        if (t->merge_rank[i] < 0) return 0;
        if (t->merge_keys[i] == key) { *rank = t->merge_rank[i]; *id = t->merge_id[i]; return 1; }
        i = (i + 1) & mask;
    }
    return 0;
}

static void merge_insert(tt_tokenizer *t, int32_t a, int32_t b,
                         int32_t rank, int32_t id)
{
    const uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    const size_t mask = t->n_merge_slots - 1;
    size_t i = (size_t)(hash_u64(key) & mask);
    while (t->merge_rank[i] >= 0) {
        if (t->merge_keys[i] == key) return;    /* o rank menor ja esta la */
        i = (i + 1) & mask;
    }
    t->merge_keys[i] = key;
    t->merge_rank[i] = rank;
    t->merge_id[i] = id;
}

/* --------------------------------------------------------- ByteLevel */

/* Tabela do GPT-2: imprimiveis mapeiam para si, o resto para 0x100+n.  E como
 * um espaco vira `Ġ` no vocabulario. */
static void byte_level_map(uint32_t enc[256])
{
    int used[256] = {0};
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                        (b >= 0xAE && b <= 0xFF);
        if (printable) { enc[b] = (uint32_t)b; used[b] = 1; }
    }
    for (int b = 0; b < 256; b++)
        if (!used[b]) enc[b] = 256u + (uint32_t)(n++);
}

/* Converte uma string do alfabeto ByteLevel de volta para bytes crus.
 * Retorna o tamanho, ou (size_t)-1 se algum code point nao pertence ao
 * alfabeto -- o que significaria vocabulario incompativel, nao entrada ruim. */
static size_t byte_level_decode(const uint32_t enc[256], const char *s, size_t n,
                                uint8_t *out)
{
    size_t w = 0, i = 0;
    while (i < n) {
        uint32_t cp;
        size_t k = u8_next((const uint8_t *)s + i, n - i, &cp);
        if (!k) break;
        i += k;

        int found = -1;
        for (int b = 0; b < 256; b++) if (enc[b] == cp) { found = b; break; }
        if (found < 0) return (size_t)-1;
        out[w++] = (uint8_t)found;
    }
    return w;
}

/* --------------------------------------------------------------- load */

int tt_tokenizer_load(tt_tokenizer *t, const char *path, int32_t vocab_size)
{
    memset(t, 0, sizeof(*t));
    t->vocab_size = vocab_size;

    tt_map map;
    if (tt_map_open(&map, path) != 0)
        return tt_fail("nao consegui abrir '%s'", path);

    /* O DOM do tokenizer.json e transitorio: sai numa arena propria que morre
     * ao fim desta funcao, deixando so as tabelas compactas. */
    tt_arena dom;
    size_t bound = tt_json_bound((const char *)map.base, map.size);
    if (tt_arena_init(&dom, bound, "tokenizer.json") != 0) {
        tt_map_close(&map);
        return -1;
    }

    tt_json *root = tt_json_parse((const char *)map.base, map.size, &dom);
    if (!root) { tt_arena_free(&dom); tt_map_close(&map); return -1; }

    const tt_json *model = tt_json_get(root, "model");
    const tt_json *vocab = tt_json_get(model, "vocab");
    const tt_json *merges = tt_json_get(model, "merges");
    const tt_json *added = tt_json_get(root, "added_tokens");

    if (!vocab || vocab->type != TT_JSON_OBJECT || !merges ||
        merges->type != TT_JSON_ARRAY) {
        tt_arena_free(&dom); tt_map_close(&map);
        return tt_fail("'%s' nao tem model.vocab e model.merges", path);
    }
    if (!tt_json_bool(tt_json_get(model, "ignore_merges"), 0)) {
        /* Se um dia vier falso, o encode muda: sem o atalho, as palavras mais
         * comuns passam a ser reconstruidas por merge (docs/16). */
        tt_dbg("tokenizer: ignore_merges = false, usando BPE em tudo");
    }

    const size_t n_vocab = vocab->count;
    const size_t n_merge = merges->count;
    const size_t n_added = added ? added->count : 0;

    size_t need = (n_vocab + n_added) * (sizeof(tt_token) + 32)
                + n_merge * (sizeof(uint64_t) + 2 * sizeof(int32_t)) * 3
                + n_vocab * sizeof(int32_t) * 4
                + n_added * sizeof(int32_t)
                + map.size                       /* pior caso dos bytes crus */
                + 64 * 1024;
    if (tt_arena_init(&t->arena, need, "tokenizer") != 0) {
        tt_arena_free(&dom); tt_map_close(&map);
        return -1;
    }

    uint32_t enc[256];
    byte_level_map(enc);

    /* --- vocabulario --- */
    int32_t max_id = -1;
    for (const tt_json *e = vocab->child; e; e = e->next) {
        int32_t id = (int32_t)tt_json_int(e, -1);
        if (id > max_id) max_id = id;
    }
    t->n_tokens = max_id + 1 + (int32_t)n_added;

    t->tokens = (tt_token *)tt_arena_alloc(&t->arena,
                    (size_t)t->n_tokens * sizeof(tt_token), TT_ALIGN);
    if (!t->tokens) goto fail;
    memset(t->tokens, 0, (size_t)t->n_tokens * sizeof(tt_token));

    for (const tt_json *e = vocab->child; e; e = e->next) {
        int32_t id = (int32_t)tt_json_int(e, -1);
        if (id < 0 || id >= t->n_tokens) continue;

        uint8_t *raw = (uint8_t *)tt_arena_alloc(&t->arena, e->key_len + 1, 1);
        if (!raw) goto fail;
        size_t len = byte_level_decode(enc, e->key, e->key_len, raw);
        if (len == (size_t)-1) {
            tt_fail("vocabulario com code point fora do alfabeto ByteLevel "
                    "em '%.*s'", (int)e->key_len, e->key);
            goto fail;
        }
        t->tokens[id].ptr = raw;
        t->tokens[id].len = (uint32_t)len;
    }

    t->n_vocab_slots = 1;
    while (t->n_vocab_slots < (size_t)t->n_tokens * 2) t->n_vocab_slots <<= 1;
    t->vocab_slots = (int32_t *)tt_arena_alloc(&t->arena,
                        t->n_vocab_slots * sizeof(int32_t), TT_ALIGN);
    if (!t->vocab_slots) goto fail;
    for (size_t i = 0; i < t->n_vocab_slots; i++) t->vocab_slots[i] = -1;

    for (int32_t id = 0; id <= max_id; id++)
        if (t->tokens[id].ptr) vocab_insert(t, id);

    for (int b = 0; b < 256; b++) {
        uint8_t one = (uint8_t)b;
        t->byte_id[b] = vocab_find(t, &one, 1);
    }

    /* --- merges --- */
    t->n_merge_slots = 1;
    while (t->n_merge_slots < n_merge * 2) t->n_merge_slots <<= 1;
    t->merge_keys = (uint64_t *)tt_arena_alloc(&t->arena,
                        t->n_merge_slots * sizeof(uint64_t), TT_ALIGN);
    t->merge_rank = (int32_t *)tt_arena_alloc(&t->arena,
                        t->n_merge_slots * sizeof(int32_t), TT_ALIGN);
    t->merge_id = (int32_t *)tt_arena_alloc(&t->arena,
                        t->n_merge_slots * sizeof(int32_t), TT_ALIGN);
    if (!t->merge_keys || !t->merge_rank || !t->merge_id) goto fail;
    for (size_t i = 0; i < t->n_merge_slots; i++) t->merge_rank[i] = -1;

    {
        uint8_t left[512], right[512], both[1024];
        int32_t rank = 0;
        for (const tt_json *e = merges->child; e; e = e->next, rank++) {
            const tt_json *a = tt_json_at(e, 0), *b = tt_json_at(e, 1);
            size_t alen = 0, blen = 0;
            const char *as = tt_json_str(a, &alen), *bs = tt_json_str(b, &blen);
            if (!as || !bs || alen >= sizeof(left) || blen >= sizeof(right))
                continue;

            size_t la = byte_level_decode(enc, as, alen, left);
            size_t lb = byte_level_decode(enc, bs, blen, right);
            if (la == (size_t)-1 || lb == (size_t)-1) continue;

            int32_t ia = vocab_find(t, left, la);
            int32_t ib = vocab_find(t, right, lb);
            if (ia < 0 || ib < 0) continue;

            memcpy(both, left, la);
            memcpy(both + la, right, lb);
            int32_t ic = vocab_find(t, both, la + lb);
            if (ic < 0) continue;

            merge_insert(t, ia, ib, rank, ic);
            t->n_merges++;
        }
    }

    /* --- especiais --- */
    if (n_added) {
        t->special = (tt_token *)tt_arena_alloc(&t->arena,
                        n_added * sizeof(tt_token), TT_ALIGN);
        t->special_id = (int32_t *)tt_arena_alloc(&t->arena,
                        n_added * sizeof(int32_t), TT_ALIGN);
        if (!t->special || !t->special_id) goto fail;

        t->special_min_id = t->n_tokens;
        for (const tt_json *e = added->child; e; e = e->next) {
            int32_t id = (int32_t)tt_json_int(tt_json_get(e, "id"), -1);
            size_t clen = 0;
            const char *content = tt_json_str(tt_json_get(e, "content"), &clen);
            if (id < 0 || !content) continue;

            uint8_t *raw = (uint8_t *)tt_arena_alloc(&t->arena, clen + 1, 1);
            if (!raw) goto fail;
            memcpy(raw, content, clen);

            t->special[t->n_special].ptr = raw;
            t->special[t->n_special].len = (uint32_t)clen;
            t->special_id[t->n_special] = id;
            t->n_special++;
            if (clen) t->special_first[raw[0]] = 1;

            if (id < t->n_tokens) {
                t->tokens[id].ptr = raw;      /* para o decode */
                t->tokens[id].len = (uint32_t)clen;
            }
            if (id < t->special_min_id) t->special_min_id = id;
        }
    }

    tt_dbg("tokenizer: %d tokens, %d merges, %d especiais (ids %d+)",
           t->n_tokens, t->n_merges, t->n_special, t->special_min_id);

    tt_arena_free(&dom);
    tt_map_close(&map);
    return 0;

fail:
    tt_arena_free(&dom);
    tt_map_close(&map);
    tt_tokenizer_free(t);
    return -1;
}

void tt_tokenizer_free(tt_tokenizer *t)
{
    tt_arena_free(&t->arena);
    memset(t, 0, sizeof(*t));
}

int tt_token_is_valid(const tt_tokenizer *t, int32_t id)
{
    if (id < 0 || id >= t->vocab_size) return 0;
    if (id < t->n_tokens && t->tokens[id].ptr) return 1;
    for (int32_t i = 0; i < t->n_special; i++)
        if (t->special_id[i] == id) return 1;
    return 0;                                  /* folga de padding do lm_head */
}

/* ------------------------------------------------------------ encode */

#define TT_MAX_PIECE 1024

static int64_t encode_piece(const tt_tokenizer *t, const uint8_t *p, size_t n,
                            int32_t *out, int64_t max, int64_t at)
{
    /* `ignore_merges`: pre-token inteiro no vocabulario usa o id direto e nao
     * aplica merge nenhum.  Ignorar isto tokeniza errado justamente as
     * palavras mais comuns (docs/16). */
    int32_t whole = vocab_find(t, p, n);
    if (whole >= 0) {
        if (at >= max) return -1;
        out[at++] = whole;
        return at;
    }

    int32_t sym[TT_MAX_PIECE];
    size_t  count = 0;
    if (n > TT_MAX_PIECE) n = TT_MAX_PIECE;
    for (size_t i = 0; i < n; i++) {
        int32_t id = t->byte_id[p[i]];
        if (id >= 0) sym[count++] = id;
    }

    /* BPE classico: aplica repetidamente o par adjacente de menor rank. */
    for (;;) {
        int32_t best_rank = 0x7fffffff, best_id = -1;
        size_t  best_at = 0;
        for (size_t i = 0; i + 1 < count; i++) {
            int32_t rank, id;
            if (merge_find(t, sym[i], sym[i + 1], &rank, &id) && rank < best_rank) {
                best_rank = rank;
                best_id = id;
                best_at = i;
            }
        }
        if (best_id < 0) break;

        sym[best_at] = best_id;
        memmove(&sym[best_at + 1], &sym[best_at + 2],
                (count - best_at - 2) * sizeof(int32_t));
        count--;
    }

    if (at + (int64_t)count > max) return -1;
    for (size_t i = 0; i < count; i++) out[at++] = sym[i];
    return at;
}

/* Maior especial que comeca exatamente em `p`.  0 se nenhum. */
static size_t special_at(const tt_tokenizer *t, const uint8_t *p, size_t avail,
                         int32_t *id_out)
{
    size_t best = 0;
    for (int32_t k = 0; k < t->n_special; k++) {
        const tt_token *s = &t->special[k];
        if (s->len > avail || s->len <= best) continue;
        if (memcmp(p, s->ptr, s->len) == 0) {
            best = s->len;
            *id_out = t->special_id[k];
        }
    }
    return best;
}

int64_t tt_encode(const tt_tokenizer *t, const char *text, size_t len,
                  int32_t *out, int64_t max)
{
    const uint8_t *p = (const uint8_t *)text;
    int64_t at = 0;
    size_t i = 0;

    while (i < len) {
        /* Os especiais FATIAM o texto antes do regex -- nao basta testa-los na
         * posicao corrente.  Em "[MASK] [gMASK]", o pre-token que comeca no
         * espaco engoliria o `[` do proximo especial, e `[gMASK]` sairia
         * quebrado em sete ids.  Entao acha-se o proximo especial primeiro, e
         * o trecho ate ele e pre-tokenizado como um segmento fechado. */
        size_t  j = i;
        size_t  hit_len = 0;
        int32_t hit_id = -1;

        while (j < len) {
            if (t->special_first[p[j]]) {
                hit_len = special_at(t, p + j, len - j, &hit_id);
                if (hit_len) break;
            }
            j++;
        }

        for (size_t pos = i; pos < j; ) {
            size_t take = pretok_next(p + pos, j - pos);
            if (!take) break;
            at = encode_piece(t, p + pos, take, out, max, at);
            if (at < 0) return -1;
            pos += take;
        }

        if (!hit_len) break;                   /* j == len, segmento final */
        if (at >= max) return -1;
        out[at++] = hit_id;
        i = j + hit_len;
    }
    return at;
}

int64_t tt_decode(const tt_tokenizer *t, const int32_t *ids, int64_t n,
                  char *out, size_t max)
{
    size_t w = 0;
    for (int64_t i = 0; i < n; i++) {
        int32_t id = ids[i];
        if (id < 0 || id >= t->n_tokens || !t->tokens[id].ptr) continue;
        const tt_token *tok = &t->tokens[id];
        if (w + tok->len > max) return -1;
        memcpy(out + w, tok->ptr, tok->len);
        w += tok->len;
    }
    return (int64_t)w;
}

/* ----------------------------------------------------------- selftest */

/* O teste que vale e contra ids do `tokenizers` da HF (tools/gen_tokenizer_
 * fixture.py).  Este aqui cobre o que nao depende do arquivo: as invariantes
 * do pre-tokenizador, que sao onde o erro sutil mora. */
int tt_tokenizer_selftest(void)
{
    struct { const char *in; size_t want; } cases[] = {
        { "'s resto",       2 },   /* contracao */
        { "'S resto",       2 },   /* (?i:) */
        { "'re x",          3 },
        { "'z",             2 },   /* nao e contracao: cai em ` ?[^\s\p{L}...]+` */
        { " palavra x",     8 },   /* espaco + letras */
        { "abc123",         3 },   /* letras param no digito */
        { "1234",           3 },   /* \p{N}{1,3} */
        { "!!!x",           3 },   /* simbolos */
        { "  \n\n  x",      4 },   /* \s*[\r\n]+ leva ate o ultimo \n */
        { "   x",           2 },   /* \s+(?!\S) deixa o ultimo espaco */
        { "   ",            3 },   /* fim da entrada: leva tudo */
        { "ção",            5 },   /* nao-ASCII: c(1) + cedilha(2) + ao(2) */
        { "\xE4\xB8\xAD\xE6\x96\x87", 6 },   /* CJK sao letras */
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        size_t got = pretok_next((const uint8_t *)cases[i].in,
                                 strlen(cases[i].in));
        if (got != cases[i].want)
            return tt_fail("pre-tokenizador: '%s' consumiu %zu bytes, "
                           "esperava %zu", cases[i].in, got, cases[i].want);
    }

    /* Byte invalido nao pode travar o laco nem consumir zero. */
    const uint8_t bad[] = { 0xFF, 0xFE, 'a' };
    if (pretok_next(bad, sizeof(bad)) == 0)
        return tt_fail("pre-tokenizador travou em byte invalido");

    /* ByteLevel e uma bijecao: 256 code points distintos. */
    uint32_t enc[256];
    byte_level_map(enc);
    for (int a = 0; a < 256; a++)
        for (int b = a + 1; b < 256; b++)
            if (enc[a] == enc[b])
                return tt_fail("ByteLevel: bytes %d e %d colidem em U+%04X",
                               a, b, enc[a]);
    return 0;
}
