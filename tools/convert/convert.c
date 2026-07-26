/* convert -- BF16 safetensors -> container `.tootega`, em C.
 *
 * Substitui `ptools/convert_to_int4.py`.  O formato de saida e normativo em
 * docs/14 e o leitor em `src/container.c` valida tudo o que ele promete, entao
 * a prova de que esta ferramenta esta certa e `tootega inspect` abrir o
 * resultado sem reclamar -- e o oraculo continuar em 32/32.
 *
 * Politica de precisao, medida pela ablacao pareada de docs/04:
 *
 *   atencao, experts   int4 g64
 *   MTP, lm_head       int8 g64
 *   normas, roteador   f32
 *   kv_a_proj          f32
 *   embeddings         bf16 (copia)
 *
 * Chegou a existir aqui um default de atencao em int8, apoiado numa ablacao
 * que dizia "a atencao explica 100% da degradacao". Aquela medicao foi feita
 * com o RoPE errado e comparando medias globais em vez de janelas pareadas.
 * Refeita, o unico grupo com efeito acima do ruido sao os EXPERTS
 * (-0,0225 +- 0,0076 de nll); a atencao da -0,0007 +- 0,0019 -- ruido.
 *
 * `--attn int8` continua disponivel, e custa 0,3 GB por nada mensuravel.
 *
 * Streaming: um tensor por vez, em blocos de linhas.  Os dois lados nunca
 * estao em memoria juntos -- o pico e o bloco atual mais as escalas do tensor.
 *
 *   tools\convert.exe models\glm47flash-bf16 --out models\novo\model.tootega
 */
#include "safetensors.h"

#include "common.h"
#include "compat.h"
#include "container.h"
#include "json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <direct.h>
#  define tt_mkdir(p) _mkdir(p)
#else
#  include <sys/stat.h>
#  include <sys/types.h>
#  define tt_mkdir(p) mkdir((p), 0777)
#endif

#define HEADER_SIZE 128
#define ALIGN_BLOB  64

/* Cria a pasta do arquivo de saida, componente a componente.  `fopen` nao cria
 * diretorio, e o conversor antigo em Python fazia `mkdir(parents=True)` -- sem
 * isto, o primeiro uso de um caminho novo falha depois de o usuario ja ter
 * digitado tudo. */
static void ensure_parent_dir(const char *path)
{
    char buf[1024];
    size_t n = strlen(path);
    if (n >= sizeof(buf)) return;
    memcpy(buf, path, n + 1);

    /* Corta o nome do arquivo. */
    for (size_t i = n; i-- > 0; )
        if (buf[i] == '/' || buf[i] == '\\') { buf[i] = '\0'; n = i; break; }
    if (!n) return;

    for (size_t i = 1; i <= n; i++) {
        const int end = (i == n);
        if (!end && buf[i] != '/' && buf[i] != '\\') continue;

        const char saved = buf[i];
        buf[i] = '\0';
        /* Ignora o erro: a pasta ja existir e o caso comum, e a falha de
         * verdade aparece no `fopen` logo abaixo, com o caminho na mensagem. */
        tt_mkdir(buf);
        if (!end) buf[i] = saved;
    }
}

/* Um bloco de linhas por vez.  16M floats = 64 MB: grande o bastante para a
 * escrita ser sequencial de verdade, pequeno o bastante para caber em cache de
 * pagina sem competir com o mapeamento do checkpoint. */
#define BLOCK_FLOATS (16u << 20)

/* ------------------------------------------------------------ politica */

static int name_has(const st_entry *e, const char *needle)
{
    const size_t n = strlen(needle);
    if (e->name_len < n) return 0;
    for (size_t i = 0; i + n <= e->name_len; i++)
        if (memcmp(e->name + i, needle, n) == 0) return 1;
    return 0;
}

static int name_ends(const st_entry *e, const char *suffix)
{
    const size_t n = strlen(suffix);
    return e->name_len >= n &&
           memcmp(e->name + e->name_len - n, suffix, n) == 0;
}

static tt_kind classify(const st_entry *e, int mtp_layer, tt_kind attn_kind)
{
    /* Uma dimensao -- normas, vieses, o `e_score_correction_bias` -- fica em
     * f32.  Sao 0,25 GB no total e o roteador decide QUAIS experts rodam: um
     * erro ali troca o expert, nao o ultimo bit. */
    if (e->ndim < 2) return TT_KIND_F32;

    /* Embeddings ficam bf16 e sao copiadas byte a byte: sao lidas uma linha
     * por token e nao precisam de residencia (docs/04). */
    if (name_ends(e, "embed_tokens.weight")) return TT_KIND_BF16;

    /* Roteador em f32 pelo mesmo motivo do bias acima; o engine recusa
     * qualquer outra coisa (model.c). */
    if (name_ends(e, "mlp.gate.weight")) return TT_KIND_F32;

    /* `kv_a_proj_with_mqa` em f32, como o conversor original. Ele alimenta o KV
     * comprimido de toda posicao, e a ablacao nunca o avaliou -- la ele ja era
     * exato. Sao 47 tensores e 167 MB. */
    if (name_ends(e, "kv_a_proj_with_mqa.weight")) return TT_KIND_F32;

    if (name_has(e, "self_attn.")) return attn_kind;

    if (name_ends(e, "lm_head.weight")) return TT_KIND_INT8;

    if (mtp_layer >= 0) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "model.layers.%d.", mtp_layer);
        if (e->name_len >= strlen(prefix) &&
            memcmp(e->name, prefix, strlen(prefix)) == 0)
            return TT_KIND_INT8;
    }

    return TT_KIND_INT4;
}

/* -------------------------------------------------------- quantizacao */

/* Simetrica, por grupo, sem zero-point -- docs/14.
 *
 * `rintf` e nao `floor(x+0.5)`: o conversor em Python usa `np.rint`, que
 * arredonda para o par mais proximo.  Divergir aqui produziria um container
 * ligeiramente diferente do que gerou `ref32.json`, e o oraculo acusaria uma
 * falha que nao existe. */
static void quant_group(const float *src, int group, int qmax,
                        float *scale_out, int8_t *codes)
{
    float absmax = 0.0f;
    for (int j = 0; j < group; j++) {
        const float v = fabsf(src[j]);
        if (v > absmax) absmax = v;
    }

    /* absmax == 0: escala 1 evita divisao por zero e reconstroi exato. */
    const float s = absmax > 0.0f ? absmax / (float)qmax : 1.0f;
    *scale_out = s;

    const float inv = 1.0f / s;
    for (int j = 0; j < group; j++) {
        float q = rintf(src[j] * inv);
        if (q > (float)qmax) q = (float)qmax;
        if (q < -(float)(qmax + 1)) q = -(float)(qmax + 1);
        codes[j] = (int8_t)q;
    }
}

/* Metade/metade dentro do grupo: byte[j] leva o peso j no nibble baixo e
 * j + group/2 no alto.  Ao desempacotar com SIMD isso entrega dois vetores
 * contiguos, sem permutar ativacao (docs/14, D1). */
static void pack_int4(const int8_t *codes, int group, uint8_t *out)
{
    const int half = group / 2;
    for (int j = 0; j < half; j++) {
        const uint8_t lo = (uint8_t)((codes[j] + 8) & 0x0F);
        const uint8_t hi = (uint8_t)((codes[j + half] + 8) & 0x0F);
        out[j] = (uint8_t)(lo | (hi << 4));
    }
}

/* ------------------------------------------------------------ escrita */

typedef struct {
    FILE    *f;
    uint64_t pos;
} writer;

static int w_pad(writer *w)
{
    static const unsigned char zero[ALIGN_BLOB] = {0};
    const uint64_t rem = w->pos % ALIGN_BLOB;
    if (!rem) return 0;
    const size_t pad = (size_t)(ALIGN_BLOB - rem);
    if (fwrite(zero, 1, pad, w->f) != pad) return tt_fail("escrita falhou");
    w->pos += pad;
    return 0;
}

static int w_write(writer *w, const void *data, size_t n)
{
    if (n && fwrite(data, 1, n, w->f) != n) return tt_fail("escrita falhou");
    w->pos += n;
    return 0;
}

/* --------------------------------------------------------- conversao */

/* Le `rows` linhas do tensor a partir de `row0` e devolve em float32. */
static void read_rows(const st_entry *e, int64_t row0, int64_t rows, float *dst)
{
    const int64_t cols = e->cols;
    if (e->kind == TT_KIND_F32) {
        memcpy(dst, (const float *)e->data + row0 * cols,
               (size_t)(rows * cols) * sizeof(float));
        return;
    }
    /* bf16 -> f32 e um shift de 16 bits, nao uma conversao. */
    const uint16_t *p = (const uint16_t *)e->data + row0 * cols;
    const int64_t n = rows * cols;
    for (int64_t i = 0; i < n; i++) {
        union { uint32_t u; float f; } c;
        c.u = (uint32_t)p[i] << 16;
        dst[i] = c.f;
    }
}

typedef struct {
    uint64_t bytes[TT_KIND_COUNT];
    size_t   count[TT_KIND_COUNT];
} tally;

static int convert_tensor(writer *w, const st_entry *e, tt_kind kind, int group,
                          float *fbuf, int64_t block_rows, float *scales,
                          uint8_t *obuf, char *index, size_t *index_at,
                          size_t index_cap, tally *tal)
{
    const int64_t O = e->rows, I = e->cols;

    if (w_pad(w) != 0) return -1;
    const uint64_t data_off = w->pos;
    uint64_t scales_bytes = 0;
    int64_t  ngroups = 0;

    if (kind == TT_KIND_BF16) {
        /* Copia crua: a fonte ja e bf16 e o container guarda os mesmos bits. */
        if (w_write(w, e->data, (size_t)e->bytes) != 0) return -1;
    } else if (kind == TT_KIND_INT4 || kind == TT_KIND_INT8) {
        if (I % group)
            return tt_fail("%.*s tem entrada %lld, indivisivel por %d",
                           (int)e->name_len, e->name, (long long)I, group);
        ngroups = I / group;
        const int qmax = (kind == TT_KIND_INT4) ? 7 : 127;

        for (int64_t r0 = 0; r0 < O; r0 += block_rows) {
            const int64_t nr = (r0 + block_rows <= O) ? block_rows : O - r0;
            read_rows(e, r0, nr, fbuf);

            for (int64_t r = 0; r < nr; r++) {
                const float *row = fbuf + r * I;
                float *srow = scales + (r0 + r) * ngroups;
                int8_t codes[1024];

                for (int64_t g = 0; g < ngroups; g++) {
                    quant_group(row + g * group, group, qmax, &srow[g], codes);
                    if (kind == TT_KIND_INT4)
                        pack_int4(codes, group, obuf + r * (I / 2) + g * (group / 2));
                    else
                        memcpy(obuf + r * I + g * group, codes, (size_t)group);
                }
            }
            const size_t out_bytes = (size_t)(nr * (kind == TT_KIND_INT4 ? I / 2 : I));
            if (w_write(w, obuf, out_bytes) != 0) return -1;
        }
        scales_bytes = (uint64_t)O * (uint64_t)ngroups * sizeof(float);
    } else {  /* f32 */
        for (int64_t r0 = 0; r0 < O; r0 += block_rows) {
            const int64_t nr = (r0 + block_rows <= O) ? block_rows : O - r0;
            read_rows(e, r0, nr, fbuf);
            if (w_write(w, fbuf, (size_t)(nr * I) * sizeof(float)) != 0) return -1;
        }
    }

    const uint64_t data_bytes = w->pos - data_off;

    uint64_t scales_off = 0;
    if (scales_bytes) {
        if (w_pad(w) != 0) return -1;
        scales_off = w->pos;
        if (w_write(w, scales, (size_t)scales_bytes) != 0) return -1;
    }

    /* Entrada do indice.  Nomes de tensor sao ASCII (letras, digitos, ponto e
     * sublinhado), entao nao ha o que escapar. */
    char shape[128];
    int sn = snprintf(shape, sizeof(shape), "[");
    for (int d = 0; d < e->ndim; d++)
        sn += snprintf(shape + sn, sizeof(shape) - sn, "%s%lld",
                       d ? "," : "", (long long)e->shape[d]);
    snprintf(shape + sn, sizeof(shape) - sn, "]");

    /* Separador: virgula so quando ja ha entrada.  Olhar o TAMANHO do buffer
     * nao serve -- ele nunca e 1, porque o cabecalho `{"tensors":{` ja ocupa
     * 12 bytes.  O ultimo caractere escrito distingue os dois casos: `{`
     * significa que a lista acabou de abrir. */
    const char *sep = (index[*index_at - 1] == '{') ? "" : ",";

    char entry[768];
    int n = snprintf(entry, sizeof(entry),
        "%s\"%.*s\":{\"kind\":\"%s\",\"shape\":%s,\"data\":[%llu,%llu],",
        sep, (int)e->name_len, e->name,
        tt_kind_name(kind), shape,
        (unsigned long long)data_off, (unsigned long long)data_bytes);

    if (scales_bytes)
        n += snprintf(entry + n, sizeof(entry) - n,
                      "\"scales\":[%llu,%llu],\"group\":%d}",
                      (unsigned long long)scales_off,
                      (unsigned long long)scales_bytes, group);
    else
        n += snprintf(entry + n, sizeof(entry) - n,
                      "\"scales\":null,\"group\":null}");

    if (*index_at + (size_t)n + 1 >= index_cap)
        return tt_fail("indice estourou o buffer");
    memcpy(index + *index_at, entry, (size_t)n);
    *index_at += (size_t)n;

    tal->bytes[kind] += data_bytes + scales_bytes;
    tal->count[kind]++;
    return 0;
}

/* --------------------------------------------------------------- main */

static void put_u32(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24);
}

static void put_u64(unsigned char *p, uint64_t v)
{
    put_u32(p, (uint32_t)v);
    put_u32(p + 4, (uint32_t)(v >> 32));
}

int main(int argc, char **argv)
{
    const char *src = NULL, *out = NULL, *model_name = "GLM-4.7-Flash";
    int group = 64, limit = 0;
    tt_kind attn_kind = TT_KIND_INT4;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const int has = (i + 1 < argc);
        if      (!strcmp(a, "--out") && has)   out = argv[++i];
        else if (!strcmp(a, "--group") && has) group = atoi(argv[++i]);
        else if (!strcmp(a, "--name") && has)  model_name = argv[++i];
        else if (!strcmp(a, "--limit") && has) limit = atoi(argv[++i]);
        else if (!strcmp(a, "--attn") && has) {
            const char *k = argv[++i];
            if (!strcmp(k, "int8"))      attn_kind = TT_KIND_INT8;
            else if (!strcmp(k, "int4")) attn_kind = TT_KIND_INT4;
            else { fprintf(stderr, "--attn int8|int4\n"); return 2; }
        }
        else if (!strcmp(a, "-v")) tt_verbose(1);
        else if (a[0] == '-') { fprintf(stderr, "opcao desconhecida: %s\n", a); return 2; }
        else if (!src) src = a;
    }

    if (!src || !out) {
        fprintf(stderr,
            "uso: convert <pasta-bf16> --out <arquivo.tootega>\n"
            "     [--group 64] [--attn int8|int4] [--limit N] [-v]\n\n"
            "  --attn int8  (default) atencao em int8: a ablacao de docs/04\n"
            "               mediu que ela responde por 100%% da degradacao\n"
            "  --attn int4  reproduz o container anterior\n");
        return 2;
    }
    if (group < 32 || group > 256 || (group & (group - 1)))
        { fprintf(stderr, "--group tem de ser 32, 64, 128 ou 256\n"); return 2; }

    st_file st;
    if (st_open(&st, src) != 0) { fprintf(stderr, "erro: %s\n", tt_error()); return 1; }

    size_t cfg_len = 0;
    char *cfg = st_read_config(src, &cfg_len);
    if (!cfg) { fprintf(stderr, "erro: %s\n", tt_error()); return 1; }

    /* mtp_layer sai da config: `num_hidden_layers` e o indice da cabeca MTP,
     * quando `num_nextn_predict_layers` > 0. */
    tt_arena cfg_arena;
    tt_arena_init(&cfg_arena, tt_json_bound(cfg, cfg_len), "config");
    tt_json *cfg_root = tt_json_parse(cfg, cfg_len, &cfg_arena);
    const int64_t layers = tt_json_int(tt_json_get(cfg_root, "num_hidden_layers"), 0);
    const int64_t nextn = tt_json_int(tt_json_get(cfg_root, "num_nextn_predict_layers"), 0);
    const int mtp_layer = nextn > 0 ? (int)layers : -1;

    /* Escrita atomica: `.part`, fsync, rename.  Uma queda no meio nao pode
     * deixar container truncado com aparencia de valido (docs/14). */
    char part[1024];
    snprintf(part, sizeof(part), "%s.part", out);

    ensure_parent_dir(part);

    writer w = { fopen(part, "wb"), 0 };
    if (!w.f) {
        fprintf(stderr, "nao consegui escrever '%s' -- verifique o caminho e "
                        "o espaco livre\n", part);
        return 1;
    }

    static const unsigned char zeros[HEADER_SIZE] = {0};
    fwrite(zeros, 1, HEADER_SIZE, w.f);
    w.pos = HEADER_SIZE;

    /* Buffers dimensionados pelo maior tensor, uma vez. */
    int64_t max_cols = 1, max_rows = 1;
    for (size_t i = 0; i < st.n_entries; i++) {
        if (st.entries[i].cols > max_cols) max_cols = st.entries[i].cols;
        if (st.entries[i].rows > max_rows) max_rows = st.entries[i].rows;
    }
    int64_t block_rows = (int64_t)(BLOCK_FLOATS / (uint64_t)max_cols);
    if (block_rows < 1) block_rows = 1;

    float   *fbuf   = (float *)malloc((size_t)(block_rows * max_cols) * sizeof(float));
    uint8_t *obuf   = (uint8_t *)malloc((size_t)(block_rows * max_cols));
    float   *scales = (float *)malloc((size_t)(max_rows * (max_cols / group + 1)) * sizeof(float));

    const size_t index_cap = 8u << 20;
    char *index = (char *)malloc(index_cap);
    if (!fbuf || !obuf || !scales || !index) {
        fprintf(stderr, "sem memoria para os buffers\n");
        return 1;
    }

    size_t index_at = (size_t)snprintf(index, index_cap, "{\"tensors\":{");
    tally tal = {0};
    size_t n_done = 0;

    const double t0 = tt_now();
    printf("\nconvertendo %zu tensores de %s\n", st.n_entries, src);
    printf("  atencao em %s, resto conforme docs/04\n\n",
           tt_kind_name(attn_kind));

    for (size_t i = 0; i < st.n_entries; i++) {
        const st_entry *e = &st.entries[i];
        const tt_kind kind = classify(e, mtp_layer, attn_kind);

        if (convert_tensor(&w, e, kind, group, fbuf, block_rows, scales, obuf,
                           index, &index_at, index_cap, &tal) != 0) {
            fprintf(stderr, "\nerro: %s\n", tt_error());
            fclose(w.f);
            return 1;
        }
        n_done++;

        if (n_done % 250 == 0 || n_done == st.n_entries) {
            char b[32];
            printf("  %5zu/%zu  %s escritos  %.0f s\n", n_done, st.n_entries,
                   tt_bytes(b, sizeof(b), w.pos), tt_now() - t0);
            fflush(stdout);
        }
        if (limit && (int)n_done >= limit) break;
    }

    /* O caminho de origem vai para dentro do JSON, e no Windows ele tem
     * barras invertidas -- que sao escape em JSON.  `models\glm...` viraria
     * `\g`, que nao existe, e o proprio leitor do engine recusaria o container
     * que a ferramenta acabou de escrever. */
    char src_json[1024];
    {
        size_t w = 0;
        for (size_t i = 0; src[i] && w + 2 < sizeof(src_json); i++) {
            if (src[i] == '\\' || src[i] == '"') src_json[w++] = '\\';
            src_json[w++] = src[i];
        }
        src_json[w] = '\0';
    }

    index_at += (size_t)snprintf(index + index_at, index_cap - index_at,
        "},\"meta\":{\"model\":\"%s\",\"source\":\"%s\",\"group\":%d,"
        "\"num_hidden_layers\":%lld,\"mtp_layer\":%d,\"config\":",
        model_name, src_json, group, (long long)layers, mtp_layer);

    memcpy(index + index_at, cfg, cfg_len);
    index_at += cfg_len;
    index_at += (size_t)snprintf(index + index_at, index_cap - index_at, "}}");

    if (w_pad(&w) != 0) { fprintf(stderr, "erro: %s\n", tt_error()); return 1; }
    const uint64_t index_off = w.pos;
    if (w_write(&w, index, index_at) != 0) { fprintf(stderr, "erro: %s\n", tt_error()); return 1; }

    unsigned char head[HEADER_SIZE] = {0};
    memcpy(head, "TOOTEGA\x01", 8);
    put_u32(head + 8, 1);                    /* format_version */
    put_u32(head + 12, 0);                   /* flags */
    put_u64(head + 16, (uint64_t)n_done);
    put_u64(head + 24, index_off);
    put_u64(head + 32, (uint64_t)index_at);

    fseek(w.f, 0, SEEK_SET);
    fwrite(head, 1, HEADER_SIZE, w.f);
    fflush(w.f);
    fclose(w.f);

    remove(out);
    if (rename(part, out) != 0) {
        fprintf(stderr, "nao consegui renomear '%s' -> '%s'\n", part, out);
        return 1;
    }

    char b[32];
    printf("\n  por precisao      tensores          bytes\n");
    uint64_t total = 0;
    for (int k = 0; k < TT_KIND_COUNT; k++) {
        if (!tal.count[k]) continue;
        printf("    %-6s %14zu %14s\n", tt_kind_name((tt_kind)k), tal.count[k],
               tt_bytes(b, sizeof(b), tal.bytes[k]));
        total += tal.bytes[k];
    }
    printf("    %-6s %14zu %14s\n", "total", n_done,
           tt_bytes(b, sizeof(b), total));
    printf("\n  %s em %.0f s\n", out, tt_now() - t0);
    printf("  confira com: tootega inspect %s\n\n", out);

    free(fbuf); free(obuf); free(scales); free(index); free(cfg);
    tt_arena_free(&cfg_arena);
    st_close(&st);
    return 0;
}
