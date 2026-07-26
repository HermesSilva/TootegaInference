/* ablate -- perplexidade e ablação de precisão, em C, sobre o próprio engine.
 *
 * Substitui `ptools/perplexity.py`.  A diferença que importa não é linguagem, é
 * QUEM mede: aqui os dois lados da comparação passam pelo mesmo forward de
 * `src/`, e o que muda entre eles é só de onde vem o peso.
 *
 * O truque é que o engine já sabe ler bf16 -- `TT_KIND_BF16` existe no
 * container e nos kernels.  Então "restaurar um grupo de tensores para a
 * precisão original" é literalmente reescrever a entrada daquele tensor no
 * índice já carregado, apontando para dentro do safetensors mapeado.  Nenhum
 * caminho de peso novo, nenhuma reimplementação do forward.
 *
 *   tools\ablate.exe models\glm47flash-int4\model.tootega ^
 *       --bf16 models\glm47flash-bf16 ^
 *       --tokenizer models\glm47flash-bf16\tokenizer.json ^
 *       --corpus corpus.txt --tokens 384 --chunks 10
 */
#include "budget.h"
#include "common.h"
#include "compat.h"
#include "container.h"
#include "forward.h"
#include "json.h"
#include "kernels.h"
#include "model.h"
#include "pool.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "safetensors.h"

/* Precisao do KV, escolhida por `--kv-prec`.
 *
 * Global de arquivo porque atravessa todas as configuracoes medidas e nao e
 * parametro de nenhuma delas: e o ambiente da medicao.  Existe aqui porque a
 * decisao "int8 no KV degrada?" so pode ser respondida por esta ferramenta, e
 * ate agora ela nem sabia pedir. */
static tt_kv_prec g_kv_prec = TT_KV_F32;

/* ------------------------------------------------------ substituicao */

/* Reescreve, no indice ja carregado, os tensores cujo nome contem `pattern`,
 * apontando-os para o safetensors.  Como `tt_layer` guarda ponteiros para
 * dentro deste array, a troca se propaga sozinha -- nao ha nada mais a
 * religar. */
static int override_group(tt_container *c, const st_file *st,
                          const char *pattern, int *count)
{
    *count = 0;
    const size_t plen = strlen(pattern);

    for (size_t i = 0; i < c->n_tensors; i++) {
        tt_tensor *t = &c->tensors[i];
        if (t->name_len < plen) continue;

        /* Só matrizes são substituídas, por dois motivos que se somam:
         *
         *  - Os tensores de uma dimensão -- normas, vieses, roteador -- já
         *    estão EXATOS no container: o conversor os manteve em f32, e f32
         *    vindo de bf16 é alargamento sem perda. Substituí-los não mudaria
         *    número nenhum.
         *  - E quebraria: o forward lê norma com cast direto para `const
         *    float *` (é f32 por contrato), enquanto no safetensors ela é
         *    bf16. O cast leria 8 KB de um buffer de 4 KB. Foi exatamente o
         *    access violation que a primeira versão desta ferramenta deu. */
        if (t->ndim < 2 || t->kind == TT_KIND_F32) continue;

        int hit = 0;
        for (size_t k = 0; k + plen <= t->name_len; k++)
            if (memcmp(t->name + k, pattern, plen) == 0) { hit = 1; break; }
        if (!hit) continue;

        const st_entry *e = st_find(st, t->name, t->name_len);
        if (!e) continue;                  /* MTP tem tensores que o BF16 nao tem */

        /* Forma tem de bater: substituir peso por outro de forma diferente
         * daria numero errado sem dar erro. */
        if (e->rows != t->rows || e->cols != t->cols)
            return tt_fail("%.*s: container %lldx%lld, safetensors %lldx%lld",
                           (int)t->name_len, t->name,
                           (long long)t->rows, (long long)t->cols,
                           (long long)e->rows, (long long)e->cols);

        t->kind = e->kind;
        t->data = e->data;
        t->data_bytes = e->bytes;
        t->scales = NULL;
        t->scales_bytes = 0;
        t->group = 0;
        (*count)++;
    }
    return 0;
}

/* ------------------------------------------------------------ medida */

/* Log-verossimilhanca negativa em double.  Aqui o objetivo e MEDIR a diferenca
 * entre duas configuracoes, nao reproduzir o que o engine faz -- somar 154.880
 * exponenciais em float introduziria ruido da ordem do efeito. */
static double window_nll(tt_state *s, const int32_t *ids, int64_t n,
                         const tt_config *cfg, int64_t *correct)
{
    double total = 0.0;
    int64_t counted = 0;
    *correct = 0;

    tt_state_reset(s);

    for (int64_t i = 0; i < n; i++) {
        float *logits = NULL;
        if (tt_forward(s, ids[i], &logits) != 0) return NAN;
        if (i + 1 >= n) break;

        const int64_t V = cfg->vocab_size;
        float max = logits[0];
        for (int64_t v = 1; v < V; v++) if (logits[v] > max) max = logits[v];

        double sum = 0.0;
        for (int64_t v = 0; v < V; v++) sum += exp((double)(logits[v] - max));

        const int32_t want = ids[i + 1];
        total += log(sum) - (double)(logits[want] - max);
        counted++;
        if (tt_argmax(logits, V) == want) (*correct)++;
    }
    return counted ? total / (double)counted : NAN;
}

typedef struct {
    const char *label;
    const char *pattern;      /* NULL = sem substituicao (linha de base int4) */
    double      nll[64];
    double      mean;
    int         tensors;
    double      seconds;
} config;

static int run_config(config *cfg, const char *model_path, const st_file *st,
                      int64_t ctx, int32_t *ids, int64_t tokens, int chunks,
                      int threads)
{
    const double t0 = tt_now();

    tt_model m;
    if (tt_model_open(&m, model_path, ctx, TT_ATTN_ABSORBED, g_kv_prec) != 0)
        return -1;

    if (cfg->pattern &&
        override_group(&m.c, st, cfg->pattern, &cfg->tensors) != 0) {
        tt_model_close(&m);
        return -1;
    }

    tt_state s;
    if (tt_state_init(&s, &m, ctx) != 0) { tt_model_close(&m); return -1; }

    double sum = 0.0;
    printf("  %-22s ", cfg->label);
    fflush(stdout);

    for (int w = 0; w < chunks; w++) {
        int64_t correct = 0;
        cfg->nll[w] = window_nll(&s, ids + (int64_t)w * tokens, tokens,
                                 &m.cfg, &correct);
        if (cfg->nll[w] != cfg->nll[w]) {
            tt_state_free(&s); tt_model_close(&m);
            return tt_fail("janela %d falhou", w);
        }
        sum += cfg->nll[w];
        printf(".");
        fflush(stdout);
    }

    cfg->mean = sum / chunks;
    cfg->seconds = tt_now() - t0;

    printf("  nll %7.4f   ppl %9.3f   %6.1f s", cfg->mean, exp(cfg->mean),
           cfg->seconds);
    if (cfg->pattern) printf("   (%d tensores)", cfg->tensors);
    printf("\n");
    fflush(stdout);

    tt_state_free(&s);
    tt_model_close(&m);
    return 0;
}

/* Diferenca PAREADA por janela.  Comparar medias globais joga fora a
 * informacao de que as duas configuracoes viram o mesmo texto, e a variancia
 * entre janelas -- que neste corpus vai de ppl 223 a 473 -- engole o efeito.
 * Ver o aviso em docs/04. */
static void paired(const char *label, const double *a, const double *b, int n)
{
    double d[64], mean = 0.0;
    for (int i = 0; i < n; i++) { d[i] = b[i] - a[i]; mean += d[i]; }
    mean /= n;

    double var = 0.0;
    for (int i = 0; i < n; i++) var += (d[i] - mean) * (d[i] - mean);
    const double se = n > 1 ? sqrt(var / (n - 1)) / sqrt((double)n) : NAN;

    printf("\n  %s\n", label);
    printf("    delta de nll  %+.4f +- %.4f      perplexidade %.4f x (%+.2f%%)\n",
           mean, se, exp(mean), (exp(mean) - 1.0) * 100.0);
    printf("    intervalo 95%% [%+.4f, %+.4f]\n", mean - 1.96 * se, mean + 1.96 * se);
    printf("    por janela   ");
    for (int i = 0; i < n; i++) printf(" %+.3f", d[i]);
    printf("\n");

    if (n > 1 && fabs(mean) < 1.96 * se)
        printf("    INCONCLUSIVO: o intervalo cruza zero.\n");
}

/* --------------------------------------------------------------- main */

static char *read_all(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) { tt_fail("nao consegui abrir '%s'", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); tt_fail("sem memoria"); return NULL; }
    *len = fread(buf, 1, (size_t)n, f);
    buf[*len] = '\0';
    fclose(f);
    return buf;
}

int main(int argc, char **argv)
{
    const char *model = NULL, *bf16 = NULL, *tokpath = NULL, *corpus = NULL;
    int64_t tokens = 384;
    int chunks = 10, threads = 20, full_only = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has = (i + 1 < argc);
        if      (!strcmp(a, "--bf16") && has)      bf16 = argv[++i];
        else if (!strcmp(a, "--tokenizer") && has) tokpath = argv[++i];
        else if (!strcmp(a, "--corpus") && has)    corpus = argv[++i];
        else if (!strcmp(a, "--tokens") && has)    tokens = atoll(argv[++i]);
        else if (!strcmp(a, "--chunks") && has)    chunks = atoi(argv[++i]);
        else if (!strcmp(a, "--kv-prec") && has) {
            if (tt_kv_prec_parse(argv[++i], &g_kv_prec) != 0) return 1;
        }
        else if (!strcmp(a, "--threads") && has)   threads = atoi(argv[++i]);
        else if (!strcmp(a, "--no-ablate"))        full_only = 1;
        else if (!strcmp(a, "-v"))                 tt_verbose(1);
        else if (a[0] == '-') { fprintf(stderr, "opcao desconhecida: %s\n", a); return 2; }
        else if (!model) model = a;
    }

    if (!model || !bf16 || !tokpath || !corpus) {
        fprintf(stderr,
            "uso: ablate <modelo.tootega> --bf16 <pasta> --tokenizer "
            "<tokenizer.json> --corpus <arquivo.txt>\n"
            "     [--tokens 384] [--chunks 10] [--threads 20] [--no-ablate]\n");
        return 2;
    }
    if (chunks > 64) { fprintf(stderr, "maximo 64 janelas\n"); return 2; }

    /* --- corpus --- */
    size_t text_len = 0;
    char *text = read_all(corpus, &text_len);
    if (!text) { fprintf(stderr, "erro: %s\n", tt_error()); return 1; }

    tt_tokenizer tk;
    if (tt_tokenizer_load(&tk, tokpath, 154880) != 0) {
        fprintf(stderr, "erro: %s\n", tt_error());
        return 1;
    }

    const int64_t need = tokens * chunks;

    /* Dimensionado pelo CORPUS, nao pelo que se precisa: `tt_encode` recusa o
     * texto inteiro se o buffer nao couber, em vez de truncar no meio de um
     * pre-token.  Um token nunca sai de menos de um byte, entao `text_len` e
     * limite superior seguro. */
    int32_t *ids = (int32_t *)malloc((size_t)(text_len + 64) * sizeof(int32_t));
    if (!ids) { fprintf(stderr, "sem memoria para os ids\n"); return 1; }
    int64_t n_ids = tt_encode(&tk, text, text_len, ids, (int64_t)text_len + 64);
    free(text);

    if (n_ids < need) {
        fprintf(stderr, "corpus deu %lld tokens, precisa de %lld (%d x %lld)\n",
                (long long)n_ids, (long long)need, chunks, (long long)tokens);
        return 1;
    }
    tt_tokenizer_free(&tk);

    /* --- safetensors --- */
    st_file st;
    if (st_open(&st, bf16) != 0) { fprintf(stderr, "erro: %s\n", tt_error()); return 1; }

    tt_pool *pool = tt_pool_create(threads, TT_AFFINITY_ALL);
    tt_kernels_set_pool(pool);

    printf("\\ncorpus  %s -- %d janelas de %lld tokens, %lld previsoes\\n",
           corpus, chunks, (long long)tokens,
           (long long)(chunks * (tokens - 1)));
    /* A precisao do KV entra no cabecalho porque ela MUDA o numero medido.
     * Duas rodadas com nll diferente e sem dizer o que mudou entre elas nao
     * sao comparaveis depois -- a licao de docs/17 e que numero sem a
     * configuracao ao lado nao significa nada. */
    printf("engine  %s, %d threads, kv %s, build %s\n\n", tt_kernels_impl_name(),
           tt_pool_threads(pool), tt_kv_prec_name(g_kv_prec), tt_build_id());

    /* Os grupos da tabela de precisao de docs/04, um por vez.  `""` casa com
     * todo nome e portanto restaura o modelo inteiro. */
    config configs[] = {
        { "int4 g64 (base)",   NULL,                 {0}, 0, 0, 0 },
        { "BF16 (referencia)", "",                   {0}, 0, 0, 0 },
        { "+ atencao exata",   "self_attn.",         {0}, 0, 0, 0 },
        { "+ experts exatos",  "mlp.experts.",       {0}, 0, 0, 0 },
        { "+ shared exatos",   "shared_experts.",    {0}, 0, 0, 0 },
        { "+ lm_head exato",   "lm_head.",           {0}, 0, 0, 0 },
        { "+ densa exata",     "layers.0.mlp.",      {0}, 0, 0, 0 },
    };
    const int n_cfg = full_only ? 2 : (int)(sizeof(configs) / sizeof(configs[0]));

    const int64_t ctx = tokens;
    const double t_start = tt_now();

    for (int i = 0; i < n_cfg; i++)
        if (run_config(&configs[i], model, &st, ctx, ids, tokens, chunks,
                       threads) != 0) {
            fprintf(stderr, "\nerro: %s\n", tt_error());
            return 1;
        }

    paired("int4 contra BF16 (custo da quantizacao)",
           configs[1].nll, configs[0].nll, chunks);

    if (!full_only) {
        printf("\n  ablacao -- cada grupo restaurado, PAREADO contra a base:\n");
        printf("    %-20s %10s %9s %8s %s\n", "grupo", "delta nll", "+- 1 se",
               "da degr.", "tensores");

        /* Pareado, e nao por media global.  Uma janela atipica desloca TODAS
         * as configuracoes juntas -- na media ela vira ruido que enterra
         * efeitos de 0,02 nll; na diferenca por janela ela se cancela.
         *
         * Foi o que aconteceu na primeira leitura destes dados: dois grupos
         * apareceram com degradacao NEGATIVA, que e fisicamente impossivel. */
        double dtot[64];
        for (int w = 0; w < chunks; w++)
            dtot[w] = configs[0].nll[w] - configs[1].nll[w];
        double total = 0.0;
        for (int w = 0; w < chunks; w++) total += dtot[w];
        total /= chunks;

        for (int i = 2; i < n_cfg; i++) {
            double d[64], mean = 0.0;
            for (int w = 0; w < chunks; w++) {
                d[w] = configs[i].nll[w] - configs[0].nll[w];
                mean += d[w];
            }
            mean /= chunks;

            double var = 0.0;
            for (int w = 0; w < chunks; w++) var += (d[w] - mean) * (d[w] - mean);
            const double se = chunks > 1
                ? sqrt(var / (chunks - 1)) / sqrt((double)chunks) : NAN;

            const int resolvido = (chunks > 1 && fabs(mean) > 1.96 * se);
            printf("    %-20s %+10.4f %9.4f %7.1f%% %8d%s\n",
                   configs[i].label, mean, se,
                   total != 0.0 ? -mean / total * 100.0 : 0.0,
                   configs[i].tensors, resolvido ? "" : "   (ruido)");
        }

        printf("\n    marcado (ruido): o efeito nao passa do proprio erro\n"
               "    padrao -- com esta amostra nao da para afirmar nada dele\n");
    }

    /* Os nll por janela vao para disco: reanalisar nao pode exigir 26 minutos
     * de CPU de novo. */
    {
        FILE *f = fopen("ablate_raw.json", "w");
        if (f) {
            /* `kv` e `corpus` no arquivo pela mesma razao do cabecalho: a
             * segunda rodada sobrescreve a primeira, e sem a configuracao
             * dentro nao da para saber qual das duas ficou. */
            fprintf(f, "{\"chunks\":%d,\"tokens\":%lld,\"kv\":\"%s\","
                       "\"corpus\":\"%s\",\"build\":\"%s\",\"configs\":[",
                    chunks, (long long)tokens, tt_kv_prec_name(g_kv_prec),
                    corpus, tt_build_id());
            for (int i = 0; i < n_cfg; i++) {
                fprintf(f, "%s{\"label\":\"%s\",\"tensors\":%d,\"nll\":[",
                        i ? "," : "", configs[i].label, configs[i].tensors);
                for (int w = 0; w < chunks; w++)
                    fprintf(f, "%s%.6f", w ? "," : "", configs[i].nll[w]);
                fprintf(f, "]}");
            }
            fprintf(f, "]}\n");
            fclose(f);
            printf("\n  ablate_raw.json escrito (nll por janela, para reanalise)\n");
        }
    }

    printf("\n  total %.1f s\n\n", tt_now() - t_start);

    tt_kernels_set_pool(NULL);
    tt_pool_destroy(pool);
    st_close(&st);
    free(ids);
    return 0;
}
