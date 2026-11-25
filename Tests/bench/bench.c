/* bench -- prompts, respostas e o que um desenvolvedor precisa ver.
 *
 * Nao e o `gentest`, que julga QUALIDADE contra o BF16. Aqui o assunto e
 * comportamento: onde o tempo vai, quanta memoria e tocada, e quao perto do
 * teto da maquina o engine chega.
 *
 * A estatistica que justifica a ferramenta e a ultima do relatorio. O decode
 * de um MoE residente e memory-bound (docs/08): por token le-se um conjunto
 * fixo de pesos -- atencao inteira, roteador, expert compartilhado e os
 * top-4 -- e a velocidade e esse volume dividido pela banda. Somando os bytes
 * dos tensores REALMENTE tocados e multiplicando por tok/s sai a banda
 * efetiva, e dela a fracao do teto.
 *
 * Sem esse numero, "12 tok/s" nao diz se falta otimizar ou se a maquina
 * acabou.
 *
 *   tools\bench.exe models\glm47flash-v2\model.tootega ^
 *       --tokenizer models\glm47flash-bf16\tokenizer.json
 */
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

#define P_PRE  "[gMASK]<sop><|user|>"
#define P_POST "<|assistant|><think>"

/* Bloco do teste de banda: grande o bastante para nao caber em L2, pequeno o
 * bastante para o particionamento ter granularidade. */
#define BW_CHUNK  4096
#define BW_BYTES  (256u << 20)

static const char *PROMPTS[] = {
    "Qual e a capital do Brasil? Responda em uma frase.",
    "Quanto e 144 dividido por 12?",
    "Escreva uma funcao em C que soma dois inteiros.",
    "Explique em duas frases o que e um mixture of experts.",
    /* Pesado de proposito: exige raciocinio, estrutura, numero e comparacao.
     * Prompt curto mede prefill; prompt que pede resposta elaborada mede o
     * que o usuario sente -- decode longo, e se o modelo mantem o fio. */
    "Explique como funciona o cache de chave-valor em um transformer: por que "
    "ele evita recomputar, quanta memoria custa por token, e o que muda quando "
    "o contexto cresce. De um exemplo numerico com 47 camadas e 20 cabecas.",
};
#define N_PROMPTS ((int)(sizeof(PROMPTS) / sizeof(PROMPTS[0])))

/* ------------------------------------------------------ banda medida */

typedef struct { const float *buf; float *out; } bw_arg;

static void bw_chunk(void *arg, int64_t r0, int64_t r1)
{
    bw_arg *a = (bw_arg *)arg;
    for (int64_t r = r0; r < r1; r++) {
        const float *p = a->buf + r * BW_CHUNK;
        /* `tt_dot` do proprio engine: e SIMD, entao o laco fica limitado por
         * memoria e nao por soma escalar.  Com `/fp:precise` uma soma escrita
         * a mao NAO vetoriza -- mediria compute, nao banda. */
        a->out[r] = tt_dot(p, p, BW_CHUNK);
    }
}

/* Banda de leitura desta maquina, medida agora.
 *
 * Cravar o numero seria mentir para quem roda em outro hardware: o roofline
 * inteiro depende dele. */
static double measure_bandwidth(tt_pool *pool)
{
    const int64_t rows = BW_BYTES / (BW_CHUNK * sizeof(float));
    float *buf = (float *)tt_alloc_aligned(BW_BYTES, TT_ALIGN);
    float *out = (float *)tt_alloc_aligned((size_t)rows * sizeof(float), TT_ALIGN);
    if (!buf || !out) { tt_free_aligned(buf); tt_free_aligned(out); return 0.0; }

    for (int64_t i = 0; i < rows * BW_CHUNK; i++) buf[i] = 1.0f;   /* faulta */

    bw_arg a = { buf, out };
    tt_pool_run(pool, bw_chunk, &a, rows, 64);                     /* aquece */

    double best = 0.0;
    for (int pass = 0; pass < 3; pass++) {
        const double t0 = tt_now();
        tt_pool_run(pool, bw_chunk, &a, rows, 64);
        const double dt = tt_now() - t0;
        const double gbs = dt > 0 ? (double)BW_BYTES / dt / 1e9 : 0.0;
        if (gbs > best) best = gbs;                                /* o melhor
                                    de tres: ruido do SO so tira, nunca poe */
    }

    tt_free_aligned(buf);
    tt_free_aligned(out);
    return best;
}

/* ------------------------------------------------- trafego por token */

static uint64_t tensor_bytes(const tt_tensor *t)
{
    return t ? t->data_bytes + t->scales_bytes : 0;
}

/* Bytes de peso lidos num passe de decode.
 *
 * Nao e o tamanho do modelo: e o que um token TOCA. Num MoE isso e uma
 * fracao pequena -- 4 experts de 64 por camada -- e e exatamente por isso que
 * o modelo cabe no orcamento de banda. */
static uint64_t decode_traffic(const tt_model *m)
{
    const tt_config *c = &m->cfg;
    uint64_t total = 0;

    for (int64_t li = 0; li < c->num_hidden_layers; li++) {
        const tt_layer *L = &m->layers[li];

        total += tensor_bytes(L->input_layernorm);
        total += tensor_bytes(L->post_attention_layernorm);
        total += tensor_bytes(L->q_a_proj) + tensor_bytes(L->q_a_layernorm);
        total += tensor_bytes(L->q_b_proj);
        total += tensor_bytes(L->kv_a_proj) + tensor_bytes(L->kv_a_layernorm);
        total += tensor_bytes(L->kv_b_proj) + tensor_bytes(L->o_proj);

        if (!L->is_moe) {
            total += tensor_bytes(L->gate_proj) + tensor_bytes(L->up_proj)
                   + tensor_bytes(L->down_proj);
            continue;
        }

        total += tensor_bytes(L->router) + tensor_bytes(L->router_bias);
        total += tensor_bytes(L->sh_gate) + tensor_bytes(L->sh_up)
               + tensor_bytes(L->sh_down);

        /* Os experts roteados: top-k de n_routed, e todos tem o mesmo
         * tamanho, entao o expert 0 serve de amostra. */
        const uint64_t one = tensor_bytes(L->e_gate[0]) + tensor_bytes(L->e_up[0])
                           + tensor_bytes(L->e_down[0]);
        total += one * (uint64_t)c->num_experts_per_tok;
    }

    total += tensor_bytes(m->norm) + tensor_bytes(m->lm_head);
    total += (uint64_t)c->hidden_size * 2;      /* linha do embed, bf16 */
    return total;
}

/* --------------------------------------------------------------- main */

typedef struct {
    double ttft, prefill_s, decode_s;
    int64_t n_prompt, n_gen;
    int stopped;
    char text[4096];
} result;

int main(int argc, char **argv)
{
    const char *path = NULL, *tok_path = NULL;
    int64_t ctx = 2048, n_new = 128;
    int threads = 20, repeat = 0;
    double bandwidth = 0.0;   /* 0 = medir nesta maquina */
    float temp = 0.0f;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const int has = (i + 1 < argc);
        if      (!strcmp(a, "--tokenizer") && has) tok_path = argv[++i];
        else if (!strcmp(a, "--ctx") && has)       ctx = atoll(argv[++i]);
        else if (!strcmp(a, "-n") && has)          n_new = atoll(argv[++i]);
        else if (!strcmp(a, "--threads") && has)   threads = atoi(argv[++i]);
        else if (!strcmp(a, "--temp") && has)      temp = (float)atof(argv[++i]);
        else if (!strcmp(a, "--bandwidth") && has) bandwidth = atof(argv[++i]);
        else if (!strcmp(a, "--repeat"))           repeat = 1;
        else if (!strcmp(a, "-v"))                 tt_verbose(1);
        else if (a[0] == '-') { fprintf(stderr, "opcao desconhecida: %s\n", a); return 2; }
        else if (!path) path = a;
    }

    if (!path || !tok_path) {
        fprintf(stderr,
            "uso: bench <modelo.tootega> --tokenizer <tokenizer.json>\n"
            "     [--ctx 2048] [-n 128] [--threads 20] [--temp 0]\n"
            "     [--bandwidth N] [--repeat]\n\n"
            "  --repeat     roda o primeiro prompt duas vezes: a diferenca e o\n"
            "               custo de subir do disco contra rodar quente\n"
            "  --bandwidth  fixa a banda em GB/s; sem ela, e MEDIDA nesta maquina\n");
        return 2;
    }

    const double rss0 = tt_rss_gb();

    tt_tokenizer tk;
    if (tt_tokenizer_load(&tk, tok_path, 154880) != 0) {
        fprintf(stderr, "erro: %s\n", tt_error()); return 1;
    }

    const double t_open = tt_now();
    tt_model m;
    if (tt_model_open(&m, path, ctx, TT_ATTN_ABSORBED, TT_KV_F32) != 0) {
        fprintf(stderr, "erro: %s\n", tt_error()); return 1;
    }
    const double open_s = tt_now() - t_open;

    tt_pool *pool = tt_pool_create(threads, TT_AFFINITY_ALL);
    tt_kernels_set_pool(pool);

    tt_state s;
    if (tt_state_init(&s, &m, ctx) != 0) {
        fprintf(stderr, "erro: %s\n", tt_error()); return 1;
    }

    int32_t eos[8];
    int n_eos = 0;
    const tt_json *ej = tt_json_get(m.c.config, "eos_token_id");
    if (ej && ej->type == TT_JSON_ARRAY)
        for (const tt_json *x = ej->child; x && n_eos < 8; x = x->next)
            eos[n_eos++] = (int32_t)tt_json_int(x, -1);

    const uint64_t traffic = decode_traffic(&m);

    /* Medida antes dos prompts, com a memoria ainda calma. */
    int measured = 0;
    if (bandwidth <= 0.0) {
        printf("\nmedindo a banda de memoria desta maquina ...");
        fflush(stdout);
        bandwidth = measure_bandwidth(pool);
        measured = 1;
        printf(" %.1f GB/s\n", bandwidth);
    }
    if (bandwidth <= 0.0) bandwidth = 1.0;      /* evita divisao por zero */

    char b1[32], b2[32], b3[32];
    printf("\n%s\n\n", path);
    printf("  maquina   %d threads logicas, %.1f GB de RAM (%.1f livres), "
           "%.1f GB/s de leitura %s\n",
           tt_cpu_threads(), tt_ram_total_gb(), tt_ram_avail_gb(), bandwidth,
           measured ? "(medido agora)" : "(informado por --bandwidth)");
    printf("  engine    %s, %d threads, MLA absorvida, KV f32, build %s\n",
           tt_kernels_impl_name(), tt_pool_threads(pool), tt_build_id());
    printf("  abertura  %.2f s (indice + IR; pesos entram sob demanda)\n", open_s);
    printf("  pesos     %s residentes de %s no container\n",
           tt_bytes(b1, sizeof(b1), tt_model_weight_bytes(&m)),
           tt_bytes(b2, sizeof(b2), m.c.map.size));
    printf("  kv        %s por posicao, %s para %lld\n",
           tt_bytes(b1, sizeof(b1), tt_kv_bytes(&m, 1)),
           tt_bytes(b2, sizeof(b2), tt_kv_bytes(&m, ctx)), (long long)ctx);
    printf("  scratch   %s\n", tt_bytes(b1, sizeof(b1), s.scratch.used));
    printf("  trafego   %s por token de decode -- %lld de %lld experts por camada\n\n",
           tt_bytes(b1, sizeof(b1), traffic),
           (long long)m.cfg.num_experts_per_tok,
           (long long)m.cfg.n_routed_experts);

    const int runs = repeat ? N_PROMPTS + 1 : N_PROMPTS;
    result *res = (result *)calloc((size_t)runs, sizeof(result));
    int32_t *ids = (int32_t *)malloc((size_t)ctx * sizeof(int32_t));
    char *prompt = (char *)malloc(4096);
    if (!res || !ids || !prompt) return 1;

    double rss_peak = rss0;

    for (int r = 0; r < runs; r++) {
        /* Com --repeat, a rodada extra repete o primeiro prompt: a diferenca
         * entre as duas e o custo de trazer os pesos do disco. */
        const int p = (repeat && r == runs - 1) ? 0 : r;
        result *o = &res[r];

        snprintf(prompt, 4096, "%s%s%s", P_PRE, PROMPTS[p], P_POST);
        o->n_prompt = tt_encode(&tk, prompt, strlen(prompt), ids, ctx);
        if (o->n_prompt < 0) continue;

        tt_sampler sampler;
        tt_sampler_init(&sampler, temp, 0.95f, 40, 1);
        tt_state_reset(&s);

        const double t0 = tt_now();
        /* The same entry point the server and the chat use, so that what is
         * timed here is the prefill that actually ships. */
        if (tt_prefill(&s, ids, 0, o->n_prompt) != 0) {
            fprintf(stderr, "erro: %s\n", tt_error()); return 1;
        }
        o->prefill_s = tt_now() - t0;

        size_t w = 0;
        char piece[512];
        const double t1 = tt_now();

        while (o->n_gen < n_new && o->n_prompt + o->n_gen < ctx - 1) {
            float *lg = tt_state_logits(&s);
            if (!lg) { fprintf(stderr, "erro: logits obsoletos\n"); return 1; }
            const int32_t next = tt_sample(&sampler, lg, tk.n_tokens);

            int done = 0;
            for (int k = 0; k < n_eos; k++) if (next == eos[k]) done = 1;
            if (done) { o->stopped = 1; break; }

            if (!o->n_gen) o->ttft = tt_now() - t0;   /* prefill + 1o token */

            const int64_t k = tt_decode(&tk, &next, 1, piece, sizeof(piece) - 1);
            if (k > 0 && w + (size_t)k + 1 < sizeof(o->text)) {
                memcpy(o->text + w, piece, (size_t)k);
                w += (size_t)k;
            }
            o->n_gen++;
            if (tt_forward(&s, next, NULL) != 0) break;
        }
        o->text[w] = '\0';
        o->decode_s = tt_now() - t1;

        const double rss = tt_rss_gb();
        if (rss > rss_peak) rss_peak = rss;

        printf("  [%d] %s\n", r + 1, PROMPTS[p]);
        printf("      %s\n", o->text[0] ? o->text : "(vazio)");
        printf("      ttft %.2f s | prefill %lld tok em %.2f s (%.1f tok/s) | "
               "decode %lld tok em %.2f s (%.2f tok/s) | %s\n\n",
               o->ttft, (long long)o->n_prompt, o->prefill_s,
               o->n_prompt / o->prefill_s, (long long)o->n_gen, o->decode_s,
               o->n_gen ? o->n_gen / o->decode_s : 0.0,
               o->stopped ? "parou no EOS" : "cortado no limite");
        fflush(stdout);
    }

    /* --------------------------------------------------------- resumo */

    double dec_tok = 0, dec_s = 0, pre_tok = 0, pre_s = 0, ttft_sum = 0;
    int counted = 0;
    for (int r = 0; r < runs; r++) {
        if (!res[r].n_gen) continue;
        dec_tok += (double)res[r].n_gen; dec_s += res[r].decode_s;
        pre_tok += (double)res[r].n_prompt; pre_s += res[r].prefill_s;
        ttft_sum += res[r].ttft;
        counted++;
    }

    const double decode_rate = dec_s > 0 ? dec_tok / dec_s : 0.0;
    const double achieved = (double)traffic * decode_rate / 1e9;

    printf("  ---------------------------------------------------------------\n");
    printf("  prefill   %.1f tok/s medio\n", pre_s > 0 ? pre_tok / pre_s : 0.0);
    printf("  decode    %.2f tok/s medio\n", decode_rate);
    printf("  ttft      %.2f s medio (prefill + primeiro token)\n",
           counted ? ttft_sum / counted : 0.0);
    printf("  memoria   RSS %.2f GB -> %.2f GB de pico, %.1f GB livres\n",
           rss0, rss_peak, tt_ram_avail_gb());
    printf("  scratch   %s de pico, arena de KV %s\n",
           tt_bytes(b1, sizeof(b1), s.scratch.peak),
           tt_bytes(b2, sizeof(b2), s.kv.size));

    /* A conta que decide se vale otimizar ou se a maquina acabou. */
    printf("\n  roofline  %s/token x %.2f tok/s = %.1f GB/s efetivos\n",
           tt_bytes(b3, sizeof(b3), traffic), decode_rate, achieved);
    printf("            %.0f%% da banda de %.1f GB/s %s\n",
           100.0 * achieved / bandwidth, bandwidth,
           measured ? "medida nesta maquina" : "informada");
    printf("            teto desta maquina: %.1f tok/s\n",
           bandwidth * 1e9 / (double)traffic);

    if (achieved / bandwidth > 0.7)
        printf("            memory-bound: otimizar compute nao vai ajudar\n");
    else if (achieved / bandwidth < 0.35)
        printf("            longe do teto: ha compute ou latencia a recuperar\n");
    printf("\n");

    if (repeat && runs >= 2) {
        const double frio = res[0].decode_s / (res[0].n_gen ? res[0].n_gen : 1);
        const double quente = res[runs-1].decode_s / (res[runs-1].n_gen ? res[runs-1].n_gen : 1);
        printf("  frio x quente: %.0f ms contra %.0f ms por token (%.1fx) --\n"
               "  a diferenca e o custo de trazer os pesos do disco\n\n",
               frio * 1000, quente * 1000, frio / quente);
    }

    free(res); free(ids); free(prompt);
    tt_state_free(&s);
    tt_kernels_set_pool(NULL);
    tt_pool_destroy(pool);
    tt_model_close(&m);
    tt_tokenizer_free(&tk);
    return 0;
}
