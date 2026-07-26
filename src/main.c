/* main.c -- entrada e despacho de subcomandos.
 *
 * Um binario, varios verbos.  A CLI completa (orcamentos, --plan) esta em
 * docs/09-resource-limits.md e entra junto com o forward; por ora existem
 * os verbos que valem sem modelo carregado.
 */
#include "budget.h"
#include "common.h"
#include "compat.h"
#include "container.h"
#include "forward.h"
#include "kernels.h"
#include "model.h"
#include "pool.h"
#include "sample.h"
#include "server.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmd_version(int argc, char **argv)
{
    (void)argc; (void)argv;
    printf("tootega %s  (build %s)\n", TT_VERSION, tt_build_id());
    printf("  cpu    %d threads logicas\n", tt_cpu_threads());
    printf("  ram    %.1f GB totais, %.1f GB disponiveis\n",
           tt_ram_total_gb(), tt_ram_avail_gb());
    return 0;
}

static int cmd_selftest(int argc, char **argv);
static int cmd_forward(int argc, char **argv);
static int cmd_tokenize(int argc, char **argv);
static int cmd_generate(int argc, char **argv);
static int cmd_chat(int argc, char **argv);

static void usage(void)
{
    fprintf(stderr,
        "tootega " TT_VERSION " -- engine de inferencia para MoE residente\n"
        "\n"
        "uso: tootega <comando> [opcoes]\n"
        "\n"
        "comandos:\n"
        "  inspect <arquivo.tootega>   cabecalho, indice e orcamento do container\n"
        "  plan <arquivo.tootega>      o que cabe: pesos, KV e scratch contra --ram-gb\n"
        "  serve <arquivo.tootega>     servidor HTTP + chat web em http://127.0.0.1:8080\n"
        "  chat <arquivo.tootega>      conversa multi-turno com cache de prefixo\n"
        "  generate <arquivo.tootega>  gera texto: tokenizer + forward + amostragem\n"
        "  forward <arquivo.tootega>   passe teacher-forcing sobre ids, para o oraculo\n"
        "  selftest                    testa kernels e estruturas contra a referencia\n"
        "  version                     versao e hardware visto\n"
        "\n"
        "forward:\n"
        "  --ids 1,2,3                 ids de entrada\n"
        "  --ids-file <arquivo>        idem, de um arquivo de texto\n"
        "  --ctx N                     capacidade do KV (default: numero de ids)\n"
        "  --attn absorbed|expanded    formulacao do MLA (default: absorbed)\n"
        "  --threads N                 default: P-cores fisicos\n"
        "  --affinity p-cores|all|none default: p-cores\n"
        "  --kernel auto|scalar        default: auto (AVX2+FMA)\n"
        "  --top N                     quantos candidatos mostrar por posicao\n"
        "  --dump-logits <arquivo>     logits f32 crus, [posicoes][vocab]\n"
        "  --dump-hidden <arquivo>     hidden final f32, [posicoes][hidden]\n"
        "  --dump-layers <arquivo>     hidden por camada, [posicoes][camadas+1][hidden]\n"
        "\n"
        "opcoes globais:\n"
        "  -v, --verbose               detalha o que esta sendo feito\n");
}

int main(int argc, char **argv)
{
    /* Opcoes globais antes do verbo, para que `-v` valha em qualquer um. */
    int i = 1;
    while (i < argc && argv[i][0] == '-') {
        if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose")) {
            tt_verbose(1);
            i++;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage();
            return 0;
        } else {
            break;
        }
    }

    if (i >= argc) { usage(); return 2; }

    const char *cmd = argv[i];
    int   sub_argc  = argc - i - 1;
    char **sub_argv = argv + i + 1;

    int rc;
    if      (!strcmp(cmd, "inspect"))  rc = tt_cmd_inspect(sub_argc, sub_argv);
    else if (!strcmp(cmd, "forward"))  rc = cmd_forward(sub_argc, sub_argv);
    else if (!strcmp(cmd, "plan"))     rc = tt_cmd_plan(sub_argc, sub_argv);
    else if (!strcmp(cmd, "tokenize")) rc = cmd_tokenize(sub_argc, sub_argv);
    else if (!strcmp(cmd, "generate")) rc = cmd_generate(sub_argc, sub_argv);
    else if (!strcmp(cmd, "chat"))     rc = cmd_chat(sub_argc, sub_argv);
    else if (!strcmp(cmd, "serve"))    rc = tt_cmd_serve(sub_argc, sub_argv);
    else if (!strcmp(cmd, "selftest")) rc = cmd_selftest(sub_argc, sub_argv);
    else if (!strcmp(cmd, "version"))  rc = cmd_version(sub_argc, sub_argv);
    else if (!strcmp(cmd, "help"))     { usage(); return 0; }
    else {
        fprintf(stderr, "comando desconhecido: %s\n\n", cmd);
        usage();
        return 2;
    }

    if (rc != 0 && tt_error()[0])
        fprintf(stderr, "erro: %s\n", tt_error());
    return rc == 0 ? 0 : 1;
}

/* ------------------------------------------------------------ forward */

/* Ids separados por qualquer coisa que nao seja digito.  Aceita "1,2,3",
 * "[1, 2, 3]" e uma coluna por linha -- todos aparecem em arquivo de teste. */
static int parse_ids(const char *text, int32_t **out, int64_t *n_out)
{
    int64_t cap = 64, n = 0;
    int32_t *ids = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    if (!ids) return tt_fail("sem memoria para os ids");

    for (const char *p = text; *p; ) {
        if (*p < '0' || *p > '9') { p++; continue; }
        long long v = 0;
        while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
        if (n == cap) {
            cap *= 2;
            int32_t *bigger = (int32_t *)realloc(ids, (size_t)cap * sizeof(int32_t));
            if (!bigger) { free(ids); return tt_fail("sem memoria para os ids"); }
            ids = bigger;
        }
        ids[n++] = (int32_t)v;
    }

    if (!n) { free(ids); return tt_fail("nenhum id encontrado na entrada"); }
    *out = ids;
    *n_out = n;
    return 0;
}

static char *read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { tt_fail("nao consegui abrir '%s'", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len < 0) { fclose(f); tt_fail("'%s' ilegivel", path); return NULL; }

    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) { fclose(f); tt_fail("sem memoria para '%s'", path); return NULL; }
    size_t got = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[got] = '\0';
    return buf;
}

static int cmd_forward(int argc, char **argv)
{
    const char *path = NULL, *ids_text = NULL, *ids_file = NULL;
    const char *dump_logits = NULL, *dump_hidden = NULL, *dump_layers = NULL;
    int64_t ctx = 0;
    int topn = 3, threads = 0;
    double ram_gb = 0;
    tt_affinity affinity = TT_AFFINITY_PCORES;
    tt_attn_mode attn = TT_ATTN_ABSORBED;
    tt_kv_prec kv_prec = TT_KV_F32;

    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        int has_next = (i + 1 < argc);
        if (!strcmp(a, "--ids") && has_next)              ids_text = argv[++i];
        else if (!strcmp(a, "--ids-file") && has_next)    ids_file = argv[++i];
        else if (!strcmp(a, "--ctx") && has_next)         ctx = atoll(argv[++i]);
        else if (!strcmp(a, "--top") && has_next)         topn = atoi(argv[++i]);
        else if (!strcmp(a, "--threads") && has_next)     threads = atoi(argv[++i]);
        else if (!strcmp(a, "--affinity") && has_next) {
            if (tt_affinity_parse(argv[++i], &affinity) != 0) return -1;
        }
        else if (!strcmp(a, "--attn") && has_next) {
            if (tt_attn_parse(argv[++i], &attn) != 0) return -1;
        }
        else if (!strcmp(a, "--ram-gb") && has_next)      ram_gb = atof(argv[++i]);
        else if ((!strcmp(a, "--kv") || !strcmp(a, "--kv-prec")) && has_next) {
            if (tt_kv_prec_parse(argv[++i], &kv_prec) != 0) return -1;
        }
        else if (!strcmp(a, "--kernel") && has_next) {
            const char *k = argv[++i];
            if (!strcmp(k, "scalar"))    tt_kernels_set_impl(TT_IMPL_SCALAR);
            else if (!strcmp(k, "auto")) tt_kernels_set_impl(TT_IMPL_AUTO);
            else return tt_fail("--kernel '%s' desconhecido (auto | scalar)", k);
        }
        else if (!strcmp(a, "--dump-logits") && has_next) dump_logits = argv[++i];
        else if (!strcmp(a, "--dump-hidden") && has_next) dump_hidden = argv[++i];
        else if (!strcmp(a, "--dump-layers") && has_next) dump_layers = argv[++i];
        else if (a[0] == '-') return tt_fail("forward: opcao desconhecida '%s'", a);
        else if (!path) path = a;
        else return tt_fail("forward: argumento extra '%s'", a);
    }

    if (!path) return tt_fail("uso: tootega forward <arquivo.tootega> --ids 1,2,3");
    if (!ids_text && !ids_file) return tt_fail("forward: faltou --ids ou --ids-file");

    char *file_text = NULL;
    if (ids_file) {
        file_text = read_file(ids_file);
        if (!file_text) return -1;
        ids_text = file_text;
    }

    int32_t *ids = NULL;
    int64_t  n_ids = 0;
    if (parse_ids(ids_text, &ids, &n_ids) != 0) { free(file_text); return -1; }
    free(file_text);

    if (ctx <= 0) ctx = n_ids;
    if (ctx < n_ids) { free(ids); return tt_fail("--ctx %lld < %lld ids",
                                                 (long long)ctx, (long long)n_ids); }

    char buf[64];
    double t0 = tt_now();
    tt_model m;
    if (tt_model_open(&m, path, ctx, attn, kv_prec) != 0) { free(ids); return -1; }
    double t_open = tt_now() - t0;

    printf("\nmodelo   %s\n", path);
    printf("pesos    %s residentes\n",
           tt_bytes(buf, sizeof(buf), tt_model_weight_bytes(&m)));
    printf("atencao  %s\n", attn == TT_ATTN_ABSORBED ? "absorvida (KV comprimido)"
                                                     : "expandida (formulacao do oraculo)");
    char buf2[64];
    printf("kv       %s para %lld posicoes (%s por token)\n",
           tt_bytes(buf, sizeof(buf), tt_kv_bytes(&m, ctx)), (long long)ctx,
           tt_bytes(buf2, sizeof(buf2), tt_kv_bytes(&m, 1)));

    /* O orcamento e conferido ANTES de alocar: recusar custa segundos, e
     * descobrir que nao cabia depois de alocar 20 GB custa a maquina inteira
     * paginando (docs/09). */
    tt_plan plan;
    tt_plan_compute(&plan, &m, ctx, ram_gb);
    if (tt_plan_check(&plan, ctx) != 0) { tt_model_close(&m); free(ids); return -1; }
    printf("orcamento %.1f GB, plano usa %.1f GB\n",
           plan.budget_gb, (double)plan.total / 1e9);
    printf("abertura %.2f s (indice + IR; os pesos entram sob demanda)\n\n", t_open);

    tt_pool *pool = tt_pool_create(threads, affinity);
    if (!pool) { tt_model_close(&m); free(ids); return -1; }
    tt_kernels_set_pool(pool);
    printf("kernels  %s, %d thread(s)\n\n", tt_kernels_impl_name(),
           tt_pool_threads(pool));

    tt_state s;
    if (tt_state_init(&s, &m, ctx) != 0) {
        tt_kernels_set_pool(NULL); tt_pool_destroy(pool);
        tt_model_close(&m); free(ids); return -1;
    }

    /* Traco por camada: [posicoes][camadas+1][hidden].  So com --dump-layers,
     * porque o custo e uma copia de 8 KB por camada por token. */
    FILE *fl = NULL, *fh = NULL, *ftr = NULL;
    if (dump_layers) {
        ftr = fopen(dump_layers, "wb");
        if (!ftr) { tt_state_free(&s); tt_kernels_set_pool(NULL);
                    tt_pool_destroy(pool); tt_model_close(&m); free(ids);
                    return tt_fail("nao consegui escrever '%s'", dump_layers); }
        s.trace = (float *)malloc((size_t)(m.cfg.num_hidden_layers + 1) *
                                  (size_t)m.cfg.hidden_size * sizeof(float));
        if (!s.trace) { fclose(ftr); return tt_fail("sem memoria para o traco"); }
    }
    if (dump_logits && !(fl = fopen(dump_logits, "wb"))) {
        tt_state_free(&s); tt_model_close(&m); free(ids);
        return tt_fail("nao consegui escrever '%s'", dump_logits);
    }
    if (dump_hidden && !(fh = fopen(dump_hidden, "wb"))) {
        if (fl) fclose(fl);
        tt_state_free(&s); tt_model_close(&m); free(ids);
        return tt_fail("nao consegui escrever '%s'", dump_hidden);
    }

    printf("  pos    entrada  ->  previsto   logit    margem   candidatos\n");

    int rc = 0;
    for (int64_t i = 0; i < n_ids; i++) {
        float *logits = NULL;
        double tt0 = tt_now();
        if (tt_forward(&s, ids[i], &logits) != 0) { rc = -1; break; }
        double dt = tt_now() - tt0;

        if (fl) fwrite(logits, sizeof(float), (size_t)m.cfg.vocab_size, fl);
        if (fh) fwrite(s.hn, sizeof(float), (size_t)m.cfg.hidden_size, fh);
        if (ftr) fwrite(s.trace, sizeof(float),
                        (size_t)(m.cfg.num_hidden_layers + 1) * (size_t)m.cfg.hidden_size, ftr);

        /* top-N por selecao, sem ordenar 154.880 logits. */
        int32_t idx[8];
        float   val[8];
        int k = topn < 1 ? 1 : (topn > 8 ? 8 : topn);
        tt_topk(logits, m.cfg.vocab_size, k, idx, val);

        printf("  %3lld %10d  ->  %8d  %7.3f  %7.3f   ",
               (long long)i, ids[i], idx[0], (double)val[0],
               (double)(val[0] - val[1]));
        for (int j = 0; j < k; j++)
            printf("%d(%.2f)%s", idx[j], (double)val[j], j + 1 < k ? " " : "");
        printf("   %.2f s\n", dt);
        fflush(stdout);
    }

    if (fl) fclose(fl);
    if (fh) fclose(fh);
    if (ftr) { fclose(ftr); free(s.trace); s.trace = NULL; }

    /* Prometido x medido.  Sem esta linha, `--ram-gb` seria uma estimativa
     * simpatica em vez de um limite. */
    if (rc == 0 && tt_budget_verify(&plan) != 0) rc = -1;

    if (rc == 0 && s.tokens_processed) {
        printf("\n  %llu tokens em %.2f s -- %.2f tok/s, %s de RSS\n\n",
               (unsigned long long)s.tokens_processed, s.seconds_in_forward,
               (double)s.tokens_processed / s.seconds_in_forward,
               tt_bytes(buf, sizeof(buf), (uint64_t)(tt_rss_gb() * 1e9)));
    }

    tt_state_free(&s);
    tt_kernels_set_pool(NULL);
    tt_pool_destroy(pool);
    tt_model_close(&m);
    free(ids);
    return rc;
}

/* ----------------------------------------------------------- generate */

static int cmd_generate(int argc, char **argv)
{
    const char *path = NULL, *tok_path = NULL, *prompt = NULL;
    int64_t ctx = 0, n_new = 64;
    int threads = 20, top_k = 40;
    float temp = 0.8f, top_p = 0.95f;
    double ram_gb = 0;
    uint64_t seed = 1;
    tt_affinity affinity = TT_AFFINITY_ALL;

    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        int has_next = (i + 1 < argc);
        if (!strcmp(a, "--tokenizer") && has_next)    tok_path = argv[++i];
        else if (!strcmp(a, "--prompt") && has_next)  prompt = argv[++i];
        else if (!strcmp(a, "-n") && has_next)        n_new = atoll(argv[++i]);
        else if (!strcmp(a, "--ctx") && has_next)     ctx = atoll(argv[++i]);
        else if (!strcmp(a, "--temp") && has_next)    temp = (float)atof(argv[++i]);
        else if (!strcmp(a, "--top-p") && has_next)   top_p = (float)atof(argv[++i]);
        else if (!strcmp(a, "--top-k") && has_next)   top_k = atoi(argv[++i]);
        else if (!strcmp(a, "--seed") && has_next)    seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--threads") && has_next) threads = atoi(argv[++i]);
        else if (!strcmp(a, "--ram-gb") && has_next)  ram_gb = atof(argv[++i]);
        else if (!strcmp(a, "--affinity") && has_next) {
            if (tt_affinity_parse(argv[++i], &affinity) != 0) return -1;
        }
        else if (a[0] == '-') return tt_fail("generate: opcao desconhecida '%s'", a);
        else if (!path) path = a;
        else return tt_fail("generate: argumento extra '%s'", a);
    }

    if (!path || !tok_path || !prompt)
        return tt_fail("uso: tootega generate <modelo.tootega> --tokenizer "
                       "<tokenizer.json> --prompt \"...\" [-n 64]");

    tt_tokenizer tk;
    if (tt_tokenizer_load(&tk, tok_path, 154880) != 0) return -1;

    int32_t *ids = (int32_t *)malloc(65536 * sizeof(int32_t));
    if (!ids) { tt_tokenizer_free(&tk); return tt_fail("sem memoria"); }

    int64_t n_prompt = tt_encode(&tk, prompt, strlen(prompt), ids, 65536);
    if (n_prompt < 0) { free(ids); tt_tokenizer_free(&tk); return -1; }

    if (ctx <= 0) ctx = n_prompt + n_new + 8;
    if (n_prompt + n_new > ctx) {
        free(ids); tt_tokenizer_free(&tk);
        return tt_fail("prompt (%lld) + geracao (%lld) passa de --ctx %lld",
                       (long long)n_prompt, (long long)n_new, (long long)ctx);
    }

    tt_model m;
    if (tt_model_open(&m, path, ctx, TT_ATTN_ABSORBED, TT_KV_F32) != 0) {
        free(ids); tt_tokenizer_free(&tk);
        return -1;
    }

    tt_plan plan;
    tt_plan_compute(&plan, &m, ctx, ram_gb);
    if (tt_plan_check(&plan, ctx) != 0) {
        tt_model_close(&m); free(ids); tt_tokenizer_free(&tk);
        return -1;
    }

    /* Os ids de fim vem da config do container, nao de constante no codigo:
     * sao tres neste modelo e mudam com o modelo. */
    int32_t eos[8];
    int n_eos = 0;
    const tt_json *e = tt_json_get(m.c.config, "eos_token_id");
    if (e && e->type == TT_JSON_ARRAY) {
        for (const tt_json *x = e->child; x && n_eos < 8; x = x->next)
            eos[n_eos++] = (int32_t)tt_json_int(x, -1);
    } else if (e) {
        eos[n_eos++] = (int32_t)tt_json_int(e, -1);
    }

    tt_pool *pool = tt_pool_create(threads, affinity);
    if (!pool) { tt_model_close(&m); free(ids); tt_tokenizer_free(&tk); return -1; }
    tt_kernels_set_pool(pool);

    tt_state s;
    if (tt_state_init(&s, &m, ctx) != 0) {
        tt_kernels_set_pool(NULL); tt_pool_destroy(pool);
        tt_model_close(&m); free(ids); tt_tokenizer_free(&tk);
        return -1;
    }

    tt_sampler sampler;
    tt_sampler_init(&sampler, temp, top_p, top_k, seed);

    fprintf(stderr, "\n%lld tokens de prompt, %d threads, %s, temp %.2f\n\n",
            (long long)n_prompt, tt_pool_threads(pool), tt_kernels_impl_name(),
            (double)temp);
    fputs(prompt, stdout);
    fflush(stdout);

    int rc = 0;
    double t_prefill = 0, t_decode = 0;
    int64_t n_decoded = 0;
    char piece[512];

    for (int64_t i = 0; i < n_prompt + n_new; i++) {
        float *logits = NULL;
        const double t0 = tt_now();
        if (tt_forward(&s, ids[i], &logits) != 0) { rc = -1; break; }
        const double dt = tt_now() - t0;
        if (i < n_prompt) t_prefill += dt; else t_decode += dt;

        /* A ULTIMA posicao do prompt tambem amostra -- e dela que sai o
         * primeiro token gerado.  Enquanto ainda ha prompt pela frente, o
         * proximo id ja e conhecido e nao se amostra nada. */
        if (i + 1 < n_prompt) continue;
        if (i + 1 >= n_prompt + n_new) break;

        /* `n_tokens` e o primeiro id inexistente: acima dele so ha padding do
         * lm_head, que nunca pode ser amostrado (docs/16). */
        int32_t next = tt_sample(&sampler, logits, tk.n_tokens);

        int done = 0;
        for (int k = 0; k < n_eos; k++) if (next == eos[k]) done = 1;
        if (done) break;

        int64_t len = tt_decode(&tk, &next, 1, piece, sizeof(piece) - 1);
        if (len > 0) { piece[len] = '\0'; fputs(piece, stdout); fflush(stdout); }

        ids[i + 1] = next;
        n_decoded++;
        if (i + 1 >= ctx) break;
    }

    /* A ultima posicao do prompt tambem produz um token; o laco acima o
     * consome ao entrar na fase de decode. */
    if (n_prompt > 0 && t_prefill > 0)
        fprintf(stderr, "\n\nprefill  %lld tokens em %.2f s (%.1f tok/s)\n",
                (long long)n_prompt, t_prefill, (double)n_prompt / t_prefill);
    if (n_decoded > 0)
        fprintf(stderr, "decode   %lld tokens em %.2f s (%.2f tok/s)\n",
                (long long)n_decoded, t_decode, (double)n_decoded / t_decode);

    char b[64];
    fprintf(stderr, "rss      %s de %.1f GB de orcamento\n\n",
            tt_bytes(b, sizeof(b), (uint64_t)(tt_rss_gb() * 1e9)), plan.budget_gb);

    if (rc == 0 && tt_budget_verify(&plan) != 0) rc = -1;

    tt_state_free(&s);
    tt_kernels_set_pool(NULL);
    tt_pool_destroy(pool);
    tt_model_close(&m);
    tt_tokenizer_free(&tk);
    free(ids);
    return rc;
}

/* --------------------------------------------------------------- chat */

/* O template de `chat_template.jinja`, sem Jinja.  Conferido contra o render
 * de `apply_chat_template` do transformers:
 *
 *   [gMASK]<sop><|system|>SYS<|user|>U1<|assistant|></think>A1<|user|>U2<|assistant|><think>
 *
 * Os marcadores sao tokens especiais e o tokenizer ja os reconhece por
 * casamento exato antes do regex -- entao basta concatenar texto.  O turno
 * aberto termina em `<|assistant|><think>`: este modelo comeca pensando. */
#define TT_CHAT_PREFIX  "[gMASK]<sop>"
#define TT_CHAT_SYSTEM  "<|system|>"
#define TT_CHAT_USER    "<|user|>"
#define TT_CHAT_ASSIST  "<|assistant|>"
#define TT_CHAT_THINK   "<think>"

/* Acrescenta ids ao fim de `conv`.  Devolve o novo tamanho, ou -1. */
static int64_t chat_append(const tt_tokenizer *tk, const char *text,
                           int32_t *conv, int64_t at, int64_t cap)
{
    int64_t n = tt_encode(tk, text, strlen(text), conv + at, cap - at);
    return n < 0 ? -1 : at + n;
}

static int cmd_chat(int argc, char **argv)
{
    const char *path = NULL, *tok_path = NULL, *system = NULL;
    int64_t ctx = 8192, n_new = 512;
    int threads = 20, top_k = 40;
    float temp = 0.8f, top_p = 0.95f;
    double ram_gb = 0;
    uint64_t seed = 1;

    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        int has_next = (i + 1 < argc);
        if (!strcmp(a, "--tokenizer") && has_next)    tok_path = argv[++i];
        else if (!strcmp(a, "--system") && has_next)  system = argv[++i];
        else if (!strcmp(a, "--ctx") && has_next)     ctx = atoll(argv[++i]);
        else if (!strcmp(a, "-n") && has_next)        n_new = atoll(argv[++i]);
        else if (!strcmp(a, "--temp") && has_next)    temp = (float)atof(argv[++i]);
        else if (!strcmp(a, "--top-p") && has_next)   top_p = (float)atof(argv[++i]);
        else if (!strcmp(a, "--top-k") && has_next)   top_k = atoi(argv[++i]);
        else if (!strcmp(a, "--seed") && has_next)    seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--threads") && has_next) threads = atoi(argv[++i]);
        else if (!strcmp(a, "--ram-gb") && has_next)  ram_gb = atof(argv[++i]);
        else if (a[0] == '-') return tt_fail("chat: opcao desconhecida '%s'", a);
        else if (!path) path = a;
        else return tt_fail("chat: argumento extra '%s'", a);
    }

    if (!path || !tok_path)
        return tt_fail("uso: tootega chat <modelo.tootega> --tokenizer "
                       "<tokenizer.json> [--system \"...\"] [--ctx 8192]");

    tt_tokenizer tk;
    if (tt_tokenizer_load(&tk, tok_path, 154880) != 0) return -1;

    tt_model m;
    if (tt_model_open(&m, path, ctx, TT_ATTN_ABSORBED, TT_KV_F32) != 0) {
        tt_tokenizer_free(&tk);
        return -1;
    }

    tt_plan plan;
    tt_plan_compute(&plan, &m, ctx, ram_gb);
    if (tt_plan_check(&plan, ctx) != 0) {
        tt_model_close(&m); tt_tokenizer_free(&tk);
        return -1;
    }

    tt_pool *pool = tt_pool_create(threads, TT_AFFINITY_ALL);
    tt_state s;
    if (!pool || tt_state_init(&s, &m, ctx) != 0) {
        if (pool) tt_pool_destroy(pool);
        tt_model_close(&m); tt_tokenizer_free(&tk);
        return -1;
    }
    tt_kernels_set_pool(pool);

    tt_sampler sampler;
    tt_sampler_init(&sampler, temp, top_p, top_k, seed);

    int32_t eos[8];
    int n_eos = 0;
    const tt_json *ej = tt_json_get(m.c.config, "eos_token_id");
    if (ej && ej->type == TT_JSON_ARRAY)
        for (const tt_json *x = ej->child; x && n_eos < 8; x = x->next)
            eos[n_eos++] = (int32_t)tt_json_int(x, -1);

    int32_t *conv = (int32_t *)malloc((size_t)ctx * sizeof(int32_t));
    char    *turn = (char *)malloc(65536);
    char    *line = (char *)malloc(16384);
    if (!conv || !turn || !line) {
        free(conv); free(turn); free(line);
        tt_kernels_set_pool(NULL); tt_pool_destroy(pool); tt_state_free(&s);
        tt_model_close(&m); tt_tokenizer_free(&tk);
        return tt_fail("sem memoria");
    }

    int64_t n_conv = 0;
    snprintf(turn, 65536, "%s%s%s", TT_CHAT_PREFIX,
             system ? TT_CHAT_SYSTEM : "", system ? system : "");
    n_conv = chat_append(&tk, turn, conv, 0, ctx);

    fprintf(stderr, "chat -- ctx %lld, %d threads, temp %.2f. Linha vazia "
            "encerra.\n\n", (long long)ctx, tt_pool_threads(pool), (double)temp);

    uint64_t total_prompt = 0, total_cached = 0;
    int rc = 0;

    for (;;) {
        fputs("> ", stderr);
        fflush(stderr);
        if (!fgets(line, 16384, stdin)) break;

        size_t len = strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (!len) break;

        /* Turno do usuario mais a abertura do turno do assistente.  Os ids sao
         * ACRESCENTADOS, nunca re-renderizados: re-tokenizar a conversa
         * inteira nao garante os mesmos ids que foram gerados, e ai o cache de
         * KV deixaria de casar por um motivo invisivel. */
        snprintf(turn, 65536, "%s%s%s%s", TT_CHAT_USER, line,
                 TT_CHAT_ASSIST, TT_CHAT_THINK);
        int64_t after = chat_append(&tk, turn, conv, n_conv, ctx);
        if (after < 0) { rc = tt_fail("contexto cheio"); break; }
        n_conv = after;

        /* Casamento de prefixo: o que ja esta no KV nao volta a ser
         * computado.  Em conversa continuada isto e a diferenca entre prefill
         * zero e recomputar tudo (docs/05, criterio do M4). */
        const int64_t reused = tt_state_match_prefix(&s, conv, n_conv);
        total_prompt += (uint64_t)n_conv;
        total_cached += (uint64_t)reused;

        const double t0 = tt_now();
        for (int64_t i = reused; i < n_conv; i++)
            if (tt_forward(&s, conv[i], NULL) != 0) { rc = -1; break; }
        if (rc) break;
        const double t_prefill = tt_now() - t0;

        fprintf(stderr, "[prefill %lld de %lld tokens, %lld do cache, %.2f s]\n",
                (long long)(n_conv - reused), (long long)n_conv,
                (long long)reused, t_prefill);

        /* O ultimo token do prompt ja foi processado; os logits dele estao em
         * s.logits e sao a base do primeiro token gerado. */
        int64_t made = 0;
        const double t1 = tt_now();
        while (made < n_new && n_conv < ctx) {
            int32_t next = tt_sample(&sampler, s.logits, tk.n_tokens);

            int done = 0;
            for (int k = 0; k < n_eos; k++) if (next == eos[k]) done = 1;
            if (done) break;

            int64_t w = tt_decode(&tk, &next, 1, turn, 65535);
            if (w > 0) { turn[w] = '\0'; fputs(turn, stdout); fflush(stdout); }

            conv[n_conv++] = next;
            made++;
            if (tt_forward(&s, next, NULL) != 0) { rc = -1; break; }
        }
        const double t_dec = tt_now() - t1;

        fprintf(stderr, "\n[%lld tokens, %.2f tok/s]\n\n",
                (long long)made, made ? (double)made / t_dec : 0.0);
        if (rc) break;
    }

    if (total_prompt)
        fprintf(stderr, "\ncache de prefixo: %.1f%% (%llu de %llu tokens de "
                "prompt vieram do KV)\n",
                100.0 * (double)total_cached / (double)total_prompt,
                (unsigned long long)total_cached,
                (unsigned long long)total_prompt);

    free(conv); free(turn); free(line);
    tt_state_free(&s);
    tt_kernels_set_pool(NULL);
    tt_pool_destroy(pool);
    tt_model_close(&m);
    tt_tokenizer_free(&tk);
    return rc;
}

/* ---------------------------------------------------------- tokenize */

/* Compara os ids do engine contra os do `tokenizers` da HF, caso a caso.  E o
 * teste de docs/16 que tem de passar ANTES de qualquer teste de forward. */
static int tokenize_check(const tt_tokenizer *tk, const char *fixture)
{
    tt_map map;
    if (tt_map_open(&map, fixture) != 0)
        return tt_fail("nao consegui abrir '%s'", fixture);

    tt_arena dom;
    if (tt_arena_init(&dom, tt_json_bound((const char *)map.base, map.size),
                      "fixture") != 0) {
        tt_map_close(&map);
        return -1;
    }

    tt_json *root = tt_json_parse((const char *)map.base, map.size, &dom);
    const tt_json *cases = tt_json_get(root, "cases");
    if (!cases || cases->type != TT_JSON_ARRAY) {
        tt_arena_free(&dom); tt_map_close(&map);
        return tt_fail("'%s' nao tem o array 'cases'", fixture);
    }

    int32_t ids[8192];
    char    text[65536];
    int passed = 0, failed = 0, rt_ok = 0, n = 0;

    for (const tt_json *c = cases->child; c; c = c->next, n++) {
        size_t tlen = 0;
        const char *t = tt_json_str(tt_json_get(c, "text"), &tlen);
        const tt_json *want = tt_json_get(c, "ids");
        if (!t || !want) continue;

        int64_t got = tt_encode(tk, t, tlen, ids, (int64_t)(sizeof(ids) / sizeof(ids[0])));
        if (got < 0) { failed++; continue; }

        int ok = ((size_t)got == want->count);
        if (ok) {
            int64_t i = 0;
            for (const tt_json *w = want->child; w; w = w->next, i++)
                if (ids[i] != (int32_t)tt_json_int(w, -1)) { ok = 0; break; }
        }

        if (ok) {
            passed++;
        } else {
            failed++;
            printf("  caso %d DIFERE (%.*s%s)\n", n,
                   (int)(tlen > 40 ? 40 : tlen), t, tlen > 40 ? "..." : "");
            printf("    engine  %lld ids:", (long long)got);
            for (int64_t i = 0; i < got && i < 16; i++) printf(" %d", ids[i]);
            printf("\n    esperado %zu ids:", want->count);
            int64_t i = 0;
            for (const tt_json *w = want->child; w && i < 16; w = w->next, i++)
                printf(" %lld", (long long)tt_json_int(w, -1));
            printf("\n");
        }

        /* Round-trip: decode(encode(s)) == s.  Falha aqui e ByteLevel errado,
         * nao BPE errado -- vale separar os dois sintomas. */
        int64_t back = tt_decode(tk, ids, got, text, sizeof(text));
        if (back == (int64_t)tlen && memcmp(text, t, tlen) == 0) rt_ok++;
    }

    printf("\n  %d/%d casos com ids identicos aos do tokenizers da HF\n",
           passed, passed + failed);
    printf("  %d/%d com round-trip exato\n\n", rt_ok, passed + failed);

    tt_arena_free(&dom);
    tt_map_close(&map);
    return failed ? tt_fail("%d caso(s) divergiram", failed) : 0;
}

static int cmd_tokenize(int argc, char **argv)
{
    const char *path = NULL, *check = NULL, *text = NULL;

    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        int has_next = (i + 1 < argc);
        if (!strcmp(a, "--check") && has_next)     check = argv[++i];
        else if (!strcmp(a, "--text") && has_next) text = argv[++i];
        else if (a[0] == '-') return tt_fail("tokenize: opcao desconhecida '%s'", a);
        else if (!path) path = a;
        else return tt_fail("tokenize: argumento extra '%s'", a);
    }

    if (!path) return tt_fail("uso: tootega tokenize <tokenizer.json> "
                              "[--text \"...\"] [--check tokfixture.json]");

    tt_tokenizer tk;
    double t0 = tt_now();
    if (tt_tokenizer_load(&tk, path, 154880) != 0) return -1;
    printf("\n%s\n  %d tokens, %d merges, %d especiais, carregado em %.2f s\n",
           path, tk.n_tokens, tk.n_merges, tk.n_special, tt_now() - t0);

    int rc = 0;
    if (text) {
        int32_t ids[4096];
        int64_t n = tt_encode(&tk, text, strlen(text), ids, 4096);
        if (n < 0) { tt_tokenizer_free(&tk); return tt_fail("texto longo demais"); }
        printf("\n  %lld ids:", (long long)n);
        for (int64_t i = 0; i < n; i++) printf(" %d", ids[i]);
        printf("\n");
    }
    if (check) rc = tokenize_check(&tk, check);

    tt_tokenizer_free(&tk);
    return rc;
}

/* ----------------------------------------------------------- selftest */

/* Cada modulo com invariante verificavel exporta um selftest.  Eles rodam no
 * binario de release, nao num executavel a parte: sao a definicao executavel do
 * modulo e tem de valer no artefato que roda (docs/15, ordem de implementacao). */
int tt_json_selftest(void);
int tt_kernels_selftest(void);
int tt_tokenizer_selftest(void);
int tt_sample_selftest(void);

static int cmd_selftest(int argc, char **argv)
{
    (void)argc; (void)argv;

    struct { const char *name; int (*fn)(void); } suites[] = {
        { "json",      tt_json_selftest      },
        { "kernels",   tt_kernels_selftest   },
        { "pool",      tt_pool_selftest      },
        { "tokenizer", tt_tokenizer_selftest },
        { "sample",    tt_sample_selftest    },
    };

    int failed = 0;
    for (size_t s = 0; s < sizeof(suites) / sizeof(suites[0]); s++) {
        tt_error_clear();
        int rc = suites[s].fn();
        printf("%-10s %s\n", suites[s].name, rc == 0 ? "ok" : "FALHOU");
        if (rc != 0) {
            printf("           %s\n", tt_error());
            failed++;
        }
    }

    if (failed) {
        printf("\n%d suite(s) falharam.\n", failed);
        return -1;
    }
    printf("\ntudo ok.\n");
    return 0;
}
