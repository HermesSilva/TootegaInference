/* gentest -- qualidade de GERACAO, que a perplexidade nao mede.
 *
 * Por que esta ferramenta existe: a perplexidade disse que o container v2 era
 * indistinguivel do BF16, e a geracao livre mostrou o contrario a olho nu --
 * "A capital do Brasil e **Bras**?**Bras** e **Brasilia**.**" contra "A
 * capital do Brasil e Brasilia." do v1.
 *
 * A razao e estrutural, nao azar: perplexidade e TEACHER-FORCED. A cada
 * posicao o prefixo verdadeiro e dado, e o erro nunca realimenta. Em geracao
 * o modelo consome a propria saida, e o KV passa a vir de tokens gerados --
 * um vies pequeno se compoe. Sao regimes diferentes e precisam de medidas
 * diferentes.
 *
 * O que se mede aqui, com amostragem gulosa para ser reproduzivel:
 *
 *   match@ref  quantos tokens iniciais batem com a geracao do BF16 antes de
 *              divergir. E a medida mais direta de "este container gera o que
 *              o original geraria".
 *   rep4       fracao de 4-gramas repetidos. Degeneracao aparece aqui antes
 *              de aparecer para um humano.
 *   parou      quantos prompts terminaram em EOS dentro do orcamento. Modelo
 *              que nao para e modelo que perdeu o fio.
 *
 * O BF16 e obtido pelo mesmo truque do `ablate`: reescrever as entradas do
 * indice apontando para o safetensors mapeado. Mesmo forward, mesma
 * amostragem -- so muda de onde vem o peso.
 *
 *   tools\gentest.exe models\a.tootega models\b.tootega ^
 *       --bf16 models\glm47flash-bf16 ^
 *       --tokenizer models\glm47flash-bf16\tokenizer.json -n 96
 */
#include "safetensors.h"

#include "common.h"
#include "compat.h"
#include "container.h"
#include "forward.h"
#include "json.h"
#include "kernels.h"
#include "model.h"
#include "pool.h"
#include "sample.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Precisao do KV, escolhida por `--kv-prec`.  Global de arquivo: e o ambiente
 * da medicao, nao parametro de um prompt. */
static tt_kv_prec g_kv_prec = TT_KV_F32;

/* Raciocinio ligado/desligado, escolhido por `--think` / `--no-think`. */
static int g_think = 1;

#define MAX_MODELS   6
#define MAX_PROMPTS  24
#define MAX_GEN      512

/* Marcadores do template, conferidos contra `apply_chat_template`. */
#define P_PRE  "[gMASK]<sop><|user|>"

/* `<think>` abre o rascunho; `</think>` o fecha antes de comecar e o modelo vai
 * direto a resposta.  Nao e truque: e a mesma construcao que o template usa
 * para turnos de assistente ja concluidos.
 *
 * O default segue LIGADO para os numeros antigos continuarem comparaveis --
 * mas o descompasso importa: ate aqui esta ferramenta so mediu com raciocinio,
 * enquanto o chat e o agente rodam sem.  Medir um regime e entregar outro e
 * como nao medir. */
#define P_THINK   "<|assistant|><think>"
#define P_NOTHINK "<|assistant|></think>"

/* Conjunto fixo.  Cobre o que o uso real pede: resposta curta e factual,
 * instrucao com formato, raciocinio, codigo, e os dois idiomas.  Prompts
 * curtos de proposito -- o que se mede e a geracao, nao o prefill. */
static const char *PROMPTS[] = {
    "Qual e a capital do Brasil? Responda em uma frase.",
    "Quanto e 17 vezes 23? Mostre a conta.",
    "Escreva uma frase curta sobre o mar.",
    "Liste tres cores primarias, uma por linha.",
    "Explique em duas frases o que e memoria cache.",
    "Traduza para o ingles: o gato dorme no sofa.",
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function that reverses a string.",
    "Name three prime numbers greater than 10.",
    "Explique por que o ceu e azul, em uma frase.",
    "Complete: era uma vez uma casa",
    "Qual a diferenca entre RAM e disco? Uma frase.",
};
#define N_PROMPTS ((int)(sizeof(PROMPTS) / sizeof(PROMPTS[0])))

typedef struct {
    char     label[128];
    int32_t  gen[MAX_PROMPTS][MAX_GEN];
    int      len[MAX_PROMPTS];
    int      stopped[MAX_PROMPTS];
    double   seconds;
    double   tokens;
} run;

/* --------------------------------------------------------- overrides */

/* Igual ao `ablate`: so matrizes.  Normas sao f32 por contrato no container e
 * bf16 no safetensors, e o forward as le com cast direto para `float*`. */
static int override_all(tt_container *c, const st_file *st, int *count)
{
    *count = 0;
    for (size_t i = 0; i < c->n_tensors; i++) {
        tt_tensor *t = &c->tensors[i];
        if (t->ndim < 2 || t->kind == TT_KIND_F32) continue;

        const st_entry *e = st_find(st, t->name, t->name_len);
        if (!e) continue;
        if (e->rows != t->rows || e->cols != t->cols)
            return tt_fail("%.*s: forma diferente entre container e safetensors",
                           (int)t->name_len, t->name);

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

/* ---------------------------------------------------------- metricas */

/* Fracao de 4-gramas que ja apareceram antes.  Degeneracao -- o
 * "**Bras**?**Bras**" -- aparece aqui muito antes de incomodar um humano. */
static double rep4(const int32_t *ids, int n)
{
    if (n < 8) return 0.0;
    int repeated = 0, total = 0;
    for (int i = 0; i + 4 <= n; i++) {
        total++;
        for (int j = 0; j < i; j++)
            if (ids[j] == ids[i] && ids[j+1] == ids[i+1] &&
                ids[j+2] == ids[i+2] && ids[j+3] == ids[i+3]) { repeated++; break; }
    }
    return total ? (double)repeated / total : 0.0;
}

/* Quantos tokens iniciais coincidem. */
static int prefix_match(const int32_t *a, int na, const int32_t *b, int nb)
{
    int i = 0;
    while (i < na && i < nb && a[i] == b[i]) i++;
    return i;
}

/* ----------------------------------------------------------- geracao */

static int generate_all(run *r, const char *model_path, const st_file *bf16,
                        tt_tokenizer *tk, int64_t ctx, int n_new, int threads,
                        float temp, uint64_t seed)
{
    (void)threads;

    tt_model m;
    if (tt_model_open(&m, model_path, ctx, TT_ATTN_ABSORBED, g_kv_prec) != 0)
        return -1;

    if (bf16) {
        int n = 0;
        if (override_all(&m.c, bf16, &n) != 0) { tt_model_close(&m); return -1; }
        snprintf(r->label + strlen(r->label), sizeof(r->label) - strlen(r->label),
                 " (%d tensores em BF16)", n);
    }

    tt_state s;
    if (tt_state_init(&s, &m, ctx) != 0) { tt_model_close(&m); return -1; }

    int32_t eos[8];
    int n_eos = 0;
    const tt_json *ej = tt_json_get(m.c.config, "eos_token_id");
    if (ej && ej->type == TT_JSON_ARRAY)
        for (const tt_json *x = ej->child; x && n_eos < 8; x = x->next)
            eos[n_eos++] = (int32_t)tt_json_int(x, -1);

    int32_t *ids = (int32_t *)malloc((size_t)ctx * sizeof(int32_t));
    char *prompt = (char *)malloc(4096);
    if (!ids || !prompt) { tt_state_free(&s); tt_model_close(&m); return -1; }

    const double t0 = tt_now();
    printf("  %-38s ", r->label);
    fflush(stdout);

    for (int p = 0; p < N_PROMPTS; p++) {
        snprintf(prompt, 4096, "%s%s%s", P_PRE, PROMPTS[p],
                 g_think ? P_THINK : P_NOTHINK);
        const int64_t n_prompt = tt_encode(tk, prompt, strlen(prompt), ids, ctx);
        if (n_prompt < 0) { r->len[p] = 0; continue; }

        /* Guloso por default: a comparacao entre containers so significa
         * alguma coisa se a amostragem for deterministica.
         *
         * `--temp` existe para responder outra pergunta: se a degeneracao
         * some com temperatura, ela e patologia de decode guloso e nao defeito
         * dos pesos.  Com temp > 0 o `match@ref` perde sentido, mas `rep4` e
         * a taxa de parada continuam validos -- e a semente e a mesma para
         * todos os containers, entao a comparacao segue justa. */
        tt_sampler sampler;
        tt_sampler_init(&sampler, temp, temp > 0.0f ? 0.95f : 0.0f,
                        temp > 0.0f ? 40 : 0, seed + (uint64_t)p);

        tt_state_reset(&s);
        for (int64_t i = 0; i < n_prompt; i++)
            if (tt_forward(&s, ids[i], NULL) != 0) break;

        int made = 0, stopped = 0;
        while (made < n_new && n_prompt + made < ctx - 1) {
            const int32_t next = tt_sample(&sampler, s.logits, tk->n_tokens);

            int done = 0;
            for (int k = 0; k < n_eos; k++) if (next == eos[k]) done = 1;
            if (done) { stopped = 1; break; }

            r->gen[p][made++] = next;
            r->tokens += 1.0;
            if (tt_forward(&s, next, NULL) != 0) break;
        }
        r->len[p] = made;
        r->stopped[p] = stopped;
        printf(".");
        fflush(stdout);
    }

    r->seconds = tt_now() - t0;
    printf("  %.0f s\n", r->seconds);
    fflush(stdout);

    free(ids); free(prompt);
    tt_state_free(&s);
    tt_model_close(&m);
    return 0;
}

/* --------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    const char *models[MAX_MODELS];
    int n_models = 0;
    const char *bf16_dir = NULL, *tok_path = NULL;
    int n_new = 96, threads = 20, show = 0;
    float temp = 0.0f;
    uint64_t seed = 1;
    int64_t ctx = 1024;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const int has = (i + 1 < argc);
        if      (!strcmp(a, "--bf16") && has)      bf16_dir = argv[++i];
        else if (!strcmp(a, "--tokenizer") && has) tok_path = argv[++i];
        else if (!strcmp(a, "-n") && has)          n_new = atoi(argv[++i]);
        else if (!strcmp(a, "--threads") && has)   threads = atoi(argv[++i]);
        else if ((!strcmp(a, "--kv-prec") || !strcmp(a, "--kv")) && has) {
            if (tt_kv_prec_parse(argv[++i], &g_kv_prec) != 0) return 1;
        }
        else if (!strcmp(a, "--temp") && has)     temp = (float)atof(argv[++i]);
        else if (!strcmp(a, "--seed") && has)     seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--show"))             show = 1;
        else if (!strcmp(a, "--think"))            g_think = 1;
        else if (!strcmp(a, "--no-think"))         g_think = 0;
        else if (!strcmp(a, "-v"))                 tt_verbose(1);
        else if (a[0] == '-') { fprintf(stderr, "opcao desconhecida: %s\n", a); return 2; }
        else if (n_models < MAX_MODELS) models[n_models++] = a;
    }

    if (!n_models || !tok_path) {
        fprintf(stderr,
            "uso: gentest <a.tootega> [b.tootega ...] --tokenizer <tokenizer.json>\n"
            "     [--bf16 <pasta>] [-n 96] [--show] [--no-think]\n"
            "     [--kv-prec f32|bf16|int8] [--temp 0] [--seed 1]\n\n"
            "  --bf16      gera tambem a referencia com os pesos originais; sem\n"
            "              ela nao ha `match@ref` e so restam rep4 e parada\n"
            "  --show      imprime o texto gerado de cada prompt\n"
            "  --no-think  fecha o </think> antes de comecar: o modelo vai\n"
            "              direto a resposta, como o chat e o agente fazem\n");
        return 2;
    }
    if (n_new > MAX_GEN) n_new = MAX_GEN;

    tt_tokenizer tk;
    if (tt_tokenizer_load(&tk, tok_path, 154880) != 0) {
        fprintf(stderr, "erro: %s\n", tt_error());
        return 1;
    }

    st_file st;
    int have_ref = 0;
    if (bf16_dir) {
        if (st_open(&st, bf16_dir) != 0) { fprintf(stderr, "erro: %s\n", tt_error()); return 1; }
        have_ref = 1;
    }

    tt_pool *pool = tt_pool_create(threads, TT_AFFINITY_ALL);
    tt_kernels_set_pool(pool);

    printf("\n%d prompts, ate %d tokens cada, %s\n", N_PROMPTS, n_new,
           temp > 0.0f ? "com temperatura" : "amostragem gulosa");
    printf("engine %s, %d threads, kv %s, build %s\n\n", tt_kernels_impl_name(),
           tt_pool_threads(pool), tt_kv_prec_name(g_kv_prec), tt_build_id());

    run *runs = (run *)calloc(MAX_MODELS + 1, sizeof(run));
    if (!runs) return 1;
    int n_runs = 0;

    /* A referencia primeiro, para o `match@ref` das demais existir. */
    if (have_ref) {
        snprintf(runs[0].label, sizeof(runs[0].label), "BF16 (referencia)");
        if (generate_all(&runs[0], models[0], &st, &tk, ctx, n_new, threads, temp, seed) != 0) {
            fprintf(stderr, "erro: %s\n", tt_error());
            return 1;
        }
        n_runs = 1;
    }

    for (int i = 0; i < n_models; i++) {
        run *r = &runs[n_runs];
        const char *base = strrchr(models[i], '\\');
        const char *base2 = strrchr(models[i], '/');
        if (base2 > base) base = base2;
        snprintf(r->label, sizeof(r->label), "%s", base ? base + 1 : models[i]);
        /* Nome do container e curto demais para distinguir; usa a pasta. */
        snprintf(r->label, sizeof(r->label), "%s", models[i]);
        if (generate_all(r, models[i], NULL, &tk, ctx, n_new, threads, temp, seed) != 0) {
            fprintf(stderr, "erro: %s\n", tt_error());
            return 1;
        }
        n_runs++;
    }

    printf("\n  %-38s %9s %8s %8s %8s\n", "container", "match@ref", "rep4",
           "parou", "tok/s");

    for (int i = 0; i < n_runs; i++) {
        const run *r = &runs[i];
        double match = 0.0, rep = 0.0;
        int stopped = 0, counted = 0;

        for (int p = 0; p < N_PROMPTS; p++) {
            if (!r->len[p]) continue;
            counted++;
            rep += rep4(r->gen[p], r->len[p]);
            stopped += r->stopped[p];
            if (have_ref)
                match += (double)prefix_match(runs[0].gen[p], runs[0].len[p],
                                              r->gen[p], r->len[p]);
        }
        if (!counted) continue;

        if (have_ref)
            printf("  %-38s %9.1f %7.1f%% %6d/%-2d %8.2f\n", r->label,
                   match / counted, 100.0 * rep / counted, stopped, counted,
                   r->tokens / r->seconds);
        else
            printf("  %-38s %9s %7.1f%% %6d/%-2d %8.2f\n", r->label, "-",
                   100.0 * rep / counted, stopped, counted,
                   r->tokens / r->seconds);
    }

    printf("\n  match@ref: tokens iniciais iguais aos do BF16 antes de divergir\n");
    printf("  rep4:      4-gramas repetidos; acima de ~15%% e degeneracao\n");
    printf("  parou:     prompts que terminaram em EOS dentro do orcamento\n\n");

    if (show) {
        char buf[8192];
        for (int p = 0; p < N_PROMPTS; p++) {
            printf("  --- %s\n", PROMPTS[p]);
            for (int i = 0; i < n_runs; i++) {
                const int64_t w = tt_decode(&tk, runs[i].gen[p], runs[i].len[p],
                                            buf, sizeof(buf) - 1);
                if (w > 0) { buf[w] = '\0'; printf("  [%s] %s\n", runs[i].label, buf); }
            }
            printf("\n");
        }
    }

    free(runs);
    tt_kernels_set_pool(NULL);
    tt_pool_destroy(pool);
    if (have_ref) st_close(&st);
    tt_tokenizer_free(&tk);
    return 0;
}
