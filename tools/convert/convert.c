/* convert -- BF16 safetensors -> container `.tootega`, em C.
 *
 * O formato de saida e normativo em docs/14 e o leitor em `src/container.c`
 * valida tudo o que ele promete, entao a prova de que esta ferramenta esta
 * certa e `tootega inspect` abrir o resultado sem reclamar -- e o oraculo
 * continuar em 32/32.
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
 *   tools\convert.exe models\glm47flash-bf16 --out m.tootega --dry-run
 *   tools\convert.exe --verify m.tootega
 */
#include "quant.h"
#include "safetensors.h"

#include "common.h"
#include "compat.h"
#include "container.h"
#include "json.h"
#include "pool.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <direct.h>
#  include <windows.h>
#  define tt_mkdir(p) _mkdir(p)
#else
#  include <sys/stat.h>
#  include <sys/statvfs.h>
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

/* Espaco livre no volume de destino, em GB; -1 quando nao da para saber.
 *
 * Nao entra em `src/compat.c`: o engine nunca precisa disto, e cada `#ifdef` la
 * dentro e superficie que alguem tem de auditar (CLAUDE.md).  Uma ferramenta
 * pode falar com o SO. */
static double free_space_of_dir(const char *dir)
{
#if defined(_WIN32)
    ULARGE_INTEGER avail;
    if (!GetDiskFreeSpaceExA(dir, &avail, NULL, NULL)) return -1.0;
    return (double)avail.QuadPart / 1e9;
#else
    struct statvfs s;
    if (statvfs(dir, &s) != 0) return -1.0;
    return (double)s.f_bavail * (double)s.f_frsize / 1e9;
#endif
}

static double free_space_gb(const char *path)
{
    char dir[1024];
    size_t n = strlen(path);
    if (n >= sizeof(dir)) return -1.0;
    memcpy(dir, path, n + 1);

    /* Sobe ate achar uma pasta que ja existe: a de destino em geral ainda nao
     * existe, e o volume e o mesmo. */
    for (size_t i = n; i-- > 0; ) {
        if (dir[i] != '/' && dir[i] != '\\') continue;
        dir[i] = '\0';
        if (!dir[0]) break;                      /* raiz POSIX */
        const double gb = free_space_of_dir(dir);
        if (gb >= 0.0) return gb;
    }
    return free_space_of_dir(".");
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

typedef struct {
    int     mtp_layer;
    tt_kind attn;
    tt_kind lm_head;
    tt_kind mtp;
} policy;

static tt_kind classify(const st_entry *e, const policy *pol)
{
    const int mtp_layer = pol->mtp_layer;

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

    if (name_has(e, "self_attn.")) return pol->attn;

    if (name_ends(e, "lm_head.weight")) return pol->lm_head;

    if (mtp_layer >= 0) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "model.layers.%d.", mtp_layer);
        if (e->name_len >= strlen(prefix) &&
            memcmp(e->name, prefix, strlen(prefix)) == 0)
            return pol->mtp;
    }

    return TT_KIND_INT4;
}

/* Bytes por peso na saida, para o plano de `--dry-run`.  int4 g64 sao 0,5 do
 * codigo mais 4/64 da escala = 0,5625 -- o numero da conta de teto de docs/02. */
static double bytes_per_weight(tt_kind kind, int group)
{
    switch (kind) {
    case TT_KIND_INT4: return 0.5 + 4.0 / (double)group;
    case TT_KIND_INT8: return 1.0 + 4.0 / (double)group;
    case TT_KIND_BF16: return 2.0;
    default:           return 4.0;
    }
}

/* ------------------------------------------------------- sonda de erro */

/* Relative error of a tensor, measured on its first block: cheap, and it is what
 * the M0 exit criterion (docs/05) asks for.
 *
 * Two metrics.  The one that decides is per-OUTPUT -- the error of `y = Wx`, not
 * the error per weight.  docs/04 explains why the two are close and why neither
 * shrinks with the size of the sum.
 *
 * LIMIT OF THIS PROBE: `x` is iid gaussian and real LLM activations are not --
 * a few channels carry 10-100x the median magnitude, and the real error is
 * dominated by the columns of W aligned with them.  The number here is therefore
 * OPTIMISTIC.  Measuring with calibration activations is what would close the
 * gap; that is the premise of AWQ and the reason the Hadamard rotation is in
 * docs/04. */

#define PROBE_ROWS 1024

typedef struct {
    double sum_weight[TT_KIND_COUNT];
    double sum_out[TT_KIND_COUNT];
    double worst_out[TT_KIND_COUNT];
    size_t n[TT_KIND_COUNT];
} errstats;

static uint64_t splitmix64(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* Fixed seed, regenerated per tensor: two tensors of the same width see the
 * SAME `x`, so their numbers are comparable, and so are two runs of the tool. */
static void probe_vector(float *x, int64_t n)
{
    uint64_t s = 0;
    for (int64_t i = 0; i < n; i += 2) {
        double u1;
        do {
            u1 = (double)(splitmix64(&s) >> 11) * (1.0 / 9007199254740992.0);
        } while (u1 <= 0.0);                     /* log(0) */
        const double u2 = (double)(splitmix64(&s) >> 11) * (1.0 / 9007199254740992.0);
        const double r = sqrt(-2.0 * log(u1));
        const double t = 6.283185307179586476925286766559 * u2;
        x[i] = (float)(r * cos(t));
        if (i + 1 < n) x[i + 1] = (float)(r * sin(t));
    }
}

static void probe_tensor(errstats *es, tt_kind kind, const float *src,
                         const uint8_t *obuf, const float *scales,
                         int group, int64_t rows, int64_t I,
                         float *x, float *recon)
{
    if (rows > PROBE_ROWS) rows = PROBE_ROWS;
    probe_vector(x, I);

    const int64_t stride = (kind == TT_KIND_INT4) ? I / 2 : I;
    double abs_err = 0.0, abs_src = 0.0, num = 0.0, den = 0.0;

    for (int64_t r = 0; r < rows; r++) {
        const float *w = src + r * I;
        tt_quant_unpack_row(kind, obuf + r * stride, scales + r * (I / group),
                       group, I, recon);

        double y_ref = 0.0, y_q = 0.0;
        for (int64_t j = 0; j < I; j++) {
            abs_err += fabs((double)recon[j] - (double)w[j]);
            abs_src += fabs((double)w[j]);
            y_ref   += (double)w[j] * (double)x[j];
            y_q     += (double)recon[j] * (double)x[j];
        }
        num += (y_q - y_ref) * (y_q - y_ref);
        den += y_ref * y_ref;
    }

    const double per_weight = abs_src > 0.0 ? abs_err / abs_src : 0.0;
    const double per_output = den > 0.0 ? sqrt(num) / sqrt(den) : 0.0;

    es->sum_weight[kind] += per_weight;
    es->sum_out[kind]    += per_output;
    if (per_output > es->worst_out[kind]) es->worst_out[kind] = per_output;
    es->n[kind]++;
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

/* Um bloco de linhas repartido entre as threads.  Cada thread e dona de linhas
 * inteiras e nao soma parcial de ninguem -- o mesmo contrato de docs/15, que e o
 * que faz o container nao depender de `--threads`. */
typedef struct {
    const float *fbuf;
    float       *scales;
    uint8_t     *obuf;
    int64_t      I, ngroups, row0;
    int          group, qmax;
    tt_kind      kind;
} quant_job;

static void quant_rows(void *arg, int64_t r0, int64_t r1)
{
    quant_job *j = (quant_job *)arg;
    int8_t codes[256];

    for (int64_t r = r0; r < r1; r++) {
        const float *row  = j->fbuf + r * j->I;
        float       *srow = j->scales + (j->row0 + r) * j->ngroups;

        for (int64_t g = 0; g < j->ngroups; g++) {
            tt_quant_group(row + g * j->group, j->group, j->qmax, &srow[g], codes);
            if (j->kind == TT_KIND_INT4)
                tt_pack_int4(codes, j->group,
                             j->obuf + r * (j->I / 2) + g * (j->group / 2));
            else
                memcpy(j->obuf + r * j->I + g * j->group, codes, (size_t)j->group);
        }
    }
}

typedef struct {
    writer  *w;
    tt_pool *pool;
    int      group;
    float   *fbuf;
    int64_t  block_rows;
    float   *scales;
    uint8_t *obuf;
    float   *probe_x, *probe_recon;
    char    *index;
    size_t  *index_at;
    size_t   index_cap;
    tally   *tal;
    errstats *err;
} conv_ctx;

static int convert_tensor(conv_ctx *cx, const st_entry *e, tt_kind kind)
{
    writer       *w      = cx->w;
    const int     group  = cx->group;
    float        *fbuf   = cx->fbuf;
    float        *scales = cx->scales;
    uint8_t      *obuf   = cx->obuf;
    const int64_t block_rows = cx->block_rows;

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

            quant_job job = { fbuf, scales, obuf, I, ngroups, r0,
                              group, qmax, kind };
            tt_pool_run(cx->pool, quant_rows, &job, nr, 4);

            if (r0 == 0 && cx->err)
                probe_tensor(cx->err, kind, fbuf, obuf, scales, group, nr, I,
                             cx->probe_x, cx->probe_recon);

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
    char   *index    = cx->index;
    size_t *index_at = cx->index_at;
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

    if (*index_at + (size_t)n + 1 >= cx->index_cap)
        return tt_fail("indice estourou o buffer");
    memcpy(index + *index_at, entry, (size_t)n);
    *index_at += (size_t)n;

    cx->tal->bytes[kind] += data_bytes + scales_bytes;
    cx->tal->count[kind]++;
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

/* `--verify`: reabre o container escrito e deixa o proprio leitor do engine
 * julgar.  `tt_container_open` valida magic, versao, o alinhamento de 64 de todo
 * blob e a relacao entre shape, group e o tamanho de cada blob -- se ele abre,
 * o arquivo esta consistente com docs/14.  Terminar de escrever nao e prova de
 * nada; abrir e. */
static int verify_container(const char *path)
{
    tt_container c;
    if (tt_container_open(&c, path) != 0) {
        fprintf(stderr, "  FALHOU: %s\n", tt_error());
        return 1;
    }

    char b[32];
    printf("  formato v%u \xc2\xb7 %zu tensores \xc2\xb7 indice em %llu\n",
           c.format_version, c.n_tensors, (unsigned long long)c.index_off);
    for (int k = 0; k < TT_KIND_COUNT; k++)
        if (c.count_by_kind[k])
            printf("    %-6s %8zu %14s\n", tt_kind_name((tt_kind)k),
                   c.count_by_kind[k],
                   tt_bytes(b, sizeof(b), c.bytes_by_kind[k]));
    printf("  OK\n");

    tt_container_close(&c);
    return 0;
}

int main(int argc, char **argv)
{
    const char *src = NULL, *out = NULL, *model_name = "GLM-4.7-Flash";
    const char *verify_only = NULL;
    int group = 64, limit = 0, clip_grid = 13, threads = 0, dry_run = 0;
    int64_t rows_per_chunk = 0;
    policy pol = { -1, TT_KIND_INT4, TT_KIND_INT8, TT_KIND_INT8 };

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const int has = (i + 1 < argc);
        if      (!strcmp(a, "--out") && has)   out = argv[++i];
        else if (!strcmp(a, "--group") && has) group = atoi(argv[++i]);
        else if (!strcmp(a, "--name") && has)  model_name = argv[++i];
        else if (!strcmp(a, "--limit") && has) limit = atoi(argv[++i]);
        else if (!strcmp(a, "--clip-grid") && has) clip_grid = atoi(argv[++i]);
        else if (!strcmp(a, "--threads") && has)   threads = atoi(argv[++i]);
        else if (!strcmp(a, "--rows-per-chunk") && has)
            rows_per_chunk = atoll(argv[++i]);
        else if (!strcmp(a, "--dry-run")) dry_run = 1;
        else if (!strcmp(a, "--verify") && has) verify_only = argv[++i];
        else if (!strcmp(a, "--attn") && has) {
            const char *k = argv[++i];
            if (!strcmp(k, "int8"))      pol.attn = TT_KIND_INT8;
            else if (!strcmp(k, "int4")) pol.attn = TT_KIND_INT4;
            else { fprintf(stderr, "--attn int8|int4\n"); return 2; }
        }
        else if (!strcmp(a, "--lm-head") && has) {
            const char *k = argv[++i];
            if (!strcmp(k, "int8"))      pol.lm_head = TT_KIND_INT8;
            else if (!strcmp(k, "int4")) pol.lm_head = TT_KIND_INT4;
            else { fprintf(stderr, "--lm-head int8|int4\n"); return 2; }
        }
        else if (!strcmp(a, "--mtp") && has) {
            const char *k = argv[++i];
            if (!strcmp(k, "int8"))      pol.mtp = TT_KIND_INT8;
            else if (!strcmp(k, "int4")) pol.mtp = TT_KIND_INT4;
            else { fprintf(stderr, "--mtp int8|int4\n"); return 2; }
        }
        else if (!strcmp(a, "-v")) tt_verbose(1);
        else if (a[0] == '-') { fprintf(stderr, "opcao desconhecida: %s\n", a); return 2; }
        else if (!src) src = a;
    }

    if (verify_only) {
        printf("container %s\n", verify_only);
        return verify_container(verify_only);
    }

    if (!src || !out) {
        fprintf(stderr,
            "uso: convert <pasta-bf16> --out <arquivo.tootega>\n"
            "     [--group 64] [--clip-grid 13] [--attn int4|int8]\n"
            "     [--lm-head int8|int4] [--mtp int8|int4]\n"
            "     [--rows-per-chunk N] [--threads N] [--limit N]\n"
            "     [--dry-run] [-v]\n"
            "     convert --verify <arquivo.tootega>\n\n"
            "  --attn int4      (default) a ablacao pareada de docs/04 nao\n"
            "                   acha efeito acima do ruido na atencao\n"
            "  --clip-grid N    pontos da busca de escala MSE entre 0,70 e 1,00\n"
            "                   do absmax. 1 desliga (absmax puro): mais rapido,\n"
            "                   ~12%% mais erro. Nao muda formato nem kernel\n"
            "  --lm-head int4   8,9%% do trafego por token; economiza 0,16 GB e\n"
            "                   pode custar qualidade (default int8)\n"
            "  --mtp int4       a licao #8 do engine anterior diz int8 (default):\n"
            "                   int4 na cabeca MTP derruba a aceitacao a 0-4%%\n"
            "  --dry-run        so o plano e o tamanho estimado, nada escrito\n"
            "  --verify         reabre um container pronto e valida docs/14\n");
        return 2;
    }
    if (group < 32 || group > 256 || (group & (group - 1)))
        { fprintf(stderr, "--group tem de ser 32, 64, 128 ou 256\n"); return 2; }
    if (clip_grid < 1 || clip_grid > TT_CLIP_MAX)
        { fprintf(stderr, "--clip-grid tem de estar entre 1 e %d\n", TT_CLIP_MAX); return 2; }
    tt_clip_grid_init(clip_grid);

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
    pol.mtp_layer = mtp_layer;

    /* ------------------------------------------------------------- plano
     *
     * Recusar ANTES de escrever, nao no meio.  Um `--group` que nao divide
     * alguma dimensao de entrada so apareceria depois de dezenas de minutos de
     * conversao, com o arquivo parcial ja no disco. */
    {
        double est = 0.0;
        size_t count[TT_KIND_COUNT] = {0};
        double bytes[TT_KIND_COUNT] = {0};
        const st_entry *bad = NULL;

        for (size_t i = 0; i < st.n_entries; i++) {
            const st_entry *e = &st.entries[i];
            const tt_kind kind = classify(e, &pol);
            double n = 1.0;
            for (int d = 0; d < e->ndim; d++) n *= (double)e->shape[d];
            const double sz = n * bytes_per_weight(kind, group);
            count[kind]++;
            bytes[kind] += sz;
            est += sz;
            if (!bad && (kind == TT_KIND_INT4 || kind == TT_KIND_INT8) &&
                e->cols % group)
                bad = e;
        }

        printf("\nfonte   %s\n", src);
        printf("destino %s\n", out);
        printf("camadas %lld + MTP em model.layers.%d\n",
               (long long)layers, mtp_layer);
        printf("politica attn=%s lm_head=%s mtp=%s group=%d clip-grid=%d\n",
               tt_kind_name(pol.attn), tt_kind_name(pol.lm_head),
               tt_kind_name(pol.mtp), group, clip_grid);

        printf("\n%-8s %9s %10s\n", "classe", "tensores", "saida");
        printf("------------------------------\n");
        for (int k = 0; k < TT_KIND_COUNT; k++)
            if (count[k])
                printf("%-8s %9zu %9.2fG\n", tt_kind_name((tt_kind)k),
                       count[k], bytes[k] / 1e9);
        printf("%-8s %9zu %9.2fG\n", "TOTAL", st.n_entries, est / 1e9);

        const double have = free_space_gb(out);
        if (have >= 0.0)
            printf("%-8s %9s %9.2fG livres em %s\n", "disco", "", have, out);

        if (bad) {
            fprintf(stderr,
                "\n--group %d nao divide a entrada de %.*s (%lld).\n",
                group, (int)bad->name_len, bad->name, (long long)bad->cols);
            return 2;
        }

        if (dry_run) {
            printf("\n(dry-run: nada escrito)\n");
            st_close(&st);
            return 0;
        }

        /* Recusar aqui e nao a 95% de uma hora de conversao. */
        const double need = est / 1e9 * 1.1;
        if (have >= 0.0 && have < need) {
            fprintf(stderr, "\nespaco insuficiente para %s: %.1f GB livres, "
                            "precisa de ~%.1f GB\n", out, have, need);
            return 1;
        }
    }

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
    int64_t block_rows = rows_per_chunk > 0
                       ? rows_per_chunk
                       : (int64_t)(BLOCK_FLOATS / (uint64_t)max_cols);
    if (block_rows < 1) block_rows = 1;

    float   *fbuf   = (float *)malloc((size_t)(block_rows * max_cols) * sizeof(float));
    uint8_t *obuf   = (uint8_t *)malloc((size_t)(block_rows * max_cols));
    float   *scales = (float *)malloc((size_t)(max_rows * (max_cols / group + 1)) * sizeof(float));
    float   *probe_x     = (float *)malloc((size_t)max_cols * sizeof(float));
    float   *probe_recon = (float *)malloc((size_t)max_cols * sizeof(float));

    const size_t index_cap = 8u << 20;
    char *index = (char *)malloc(index_cap);
    if (!fbuf || !obuf || !scales || !index || !probe_x || !probe_recon) {
        fprintf(stderr, "sem memoria para os buffers\n");
        return 1;
    }

    /* A busca de clipping multiplica por `--clip-grid` o trabalho de
     * quantizacao, e ele e o custo dominante da conversao.  Por linha de saida,
     * como docs/15 manda: cada thread e dona de linhas inteiras e o container
     * nao muda com `--threads`. */
    tt_pool *pool = tt_pool_create(threads, TT_AFFINITY_PCORES);
    if (!pool) { fprintf(stderr, "erro: %s\n", tt_error()); return 1; }

    size_t index_at = (size_t)snprintf(index, index_cap, "{\"tensors\":{");
    tally tal = {0};
    errstats err = {{0}};
    size_t n_done = 0;

    conv_ctx cx = { &w, pool, group, fbuf, block_rows, scales, obuf,
                    probe_x, probe_recon, index, &index_at, index_cap,
                    &tal, &err };

    const double t0 = tt_now();
    printf("\nconvertendo %zu tensores de %s\n", st.n_entries, src);
    printf("  %d threads, clip-grid %d, resto conforme docs/04\n\n",
           tt_pool_threads(pool), clip_grid);

    for (size_t i = 0; i < st.n_entries; i++) {
        const st_entry *e = &st.entries[i];
        const tt_kind kind = classify(e, &pol);

        if (convert_tensor(&cx, e, kind) != 0) {
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
        size_t k = 0;
        for (size_t i = 0; src[i] && k + 2 < sizeof(src_json); i++) {
            if (src[i] == '\\' || src[i] == '"') src_json[k++] = '\\';
            src_json[k++] = src[i];
        }
        src_json[k] = '\0';
    }

    index_at += (size_t)snprintf(index + index_at, index_cap - index_at,
        "},\"meta\":{\"model\":\"%s\",\"source\":\"%s\",\"group\":%d,"
        "\"clip_grid\":%d,"
        "\"num_hidden_layers\":%lld,\"mtp_layer\":%d,\"config\":",
        model_name, src_json, group, clip_grid, (long long)layers, mtp_layer);

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

    printf("\n  erro de quantizacao (o que decide e a coluna 'saida'):\n");
    printf("    %-7s %10s %14s %12s %7s\n",
           "classe", "por peso", "saida matmul", "pior saida", "n");
    for (int k = 0; k < TT_KIND_COUNT; k++) {
        if (!err.n[k]) continue;
        printf("    %-7s %9.2f%% %13.2f%% %11.2f%% %7zu\n",
               tt_kind_name((tt_kind)k),
               100.0 * err.sum_weight[k] / (double)err.n[k],
               100.0 * err.sum_out[k] / (double)err.n[k],
               100.0 * err.worst_out[k], err.n[k]);
    }
    printf("    grupo=%d \xc2\xb7 clip-grid=%d\n", group, clip_grid);
    printf("    O erro da saida NAO cai com o tamanho da soma: em y=Wx o ruido e\n");
    printf("    o sinal crescem ambos com sqrt(cols), entao a razao fica\n");
    printf("    ~sigma_e/sigma_w. Sanidade: int4/int8 entre 15x e 18x -- 18 e a\n");
    printf("    razao de passo (127/7), e a busca de clipping baixa para ~15\n");
    printf("    porque so o int4 melhora com ela.\n");
    printf("    Alavancas, da mais fraca para a mais forte:\n");
    printf("      --group 32       ~11%% menos erro, +11%% de arquivo. Trade ruim.\n");
    printf("      --clip-grid 13   ~12%% menos erro, de graca (ja e o default).\n");
    printf("      int8 no tensor   ~18x menos erro, 2x os bytes daquele tensor.\n");
    printf("      Hadamard/GPTQ    2-4x menos erro, formato e kernel iguais.\n");
    printf("    E lembre que x da sonda e gaussiano iid: o numero e otimista.\n");

    printf("\ncontainer %s\n", out);
    const int bad = verify_container(out);
    printf("\n");

    tt_pool_destroy(pool);
    free(fbuf); free(obuf); free(scales); free(index); free(cfg);
    free(probe_x); free(probe_recon);
    tt_arena_free(&cfg_arena);
    st_close(&st);
    return bad;
}
