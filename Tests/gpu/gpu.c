/* gpu -- o portao do M2.
 *
 * docs/05: ganho abaixo de 20% e o M2 e CANCELADO, nao adiado. Dois backends
 * sao custo permanente de manutencao e precisam se pagar.
 *
 * Esta ferramenta responde a pergunta que decide isso antes de qualquer
 * planner existir: **um matmul na GPU e quanto mais rapido que o mesmo matmul
 * na CPU, incluindo a travessia do PCIe?**
 *
 * A teoria diz ~6,5x: os dois lados sao memory-bound e a VRAM da 448 GB/s
 * contra 69 GB/s da RAM. Se a medida nao chegar perto disso, o desenho esta
 * errado e nada mais precisa ser escrito.
 *
 * Mede tambem a diferenca numerica. A reducao na GPU e uma arvore por bloco e
 * na CPU e um acumulador escalar por linha -- os bits NAO batem, e o criterio
 * de saida do M2 ("byte-identico entre CPU-only e split") precisa ser lido a
 * luz disso.
 *
 *   Tests\gpu.exe models\glm47flash-v3\model.tootega
 */
#include "backend.h"
#include "common.h"
#include "compat.h"
#include "container.h"
#include "kernels.h"
#include "model.h"
#include "pool.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Tensores representativos: o maior de cada tipo de trabalho por camada. */
static const char *TARGETS[] = {
    "model.layers.1.self_attn.o_proj.weight",       /* 2048 x 5120 */
    "model.layers.1.self_attn.kv_b_proj.weight",    /* 8960 x  512 */
    "model.layers.1.mlp.experts.0.gate_proj.weight",/* 1536 x 2048 */
    "model.layers.1.mlp.experts.0.down_proj.weight",/* 2048 x 1536 */
    "model.layers.0.mlp.gate_proj.weight",          /* 10240 x 2048 */
};
#define N_TARGETS ((int)(sizeof(TARGETS) / sizeof(TARGETS[0])))

int main(int argc, char **argv)
{
    const char *path = NULL, *dll = "tootega_cuda.dll";
    int reps = 200, threads = 20;
    double cpu_stream = 68.7;          /* GB/s de RAM, medidos por Tests/bench */

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const int has = (i + 1 < argc);
        if      (!strcmp(a, "--dll") && has)     dll = argv[++i];
        else if (!strcmp(a, "--reps") && has)    reps = atoi(argv[++i]);
        else if (!strcmp(a, "--cpu-gbs") && has) cpu_stream = atof(argv[++i]);
        else if (!strcmp(a, "--threads") && has) threads = atoi(argv[++i]);
        else if (!strcmp(a, "-v"))               tt_verbose(1);
        else if (a[0] == '-') { fprintf(stderr, "opcao desconhecida: %s\n", a); return 2; }
        else if (!path) path = a;
    }
    if (!path) {
        fprintf(stderr, "uso: gpu <modelo.tootega> [--dll tootega_cuda.dll] "
                        "[--reps 200]\n");
        return 2;
    }

    tt_model m;
    if (tt_model_open(&m, path, 64, TT_ATTN_ABSORBED, TT_KV_F32) != 0) {
        fprintf(stderr, "erro: %s\n", tt_error());
        return 1;
    }

    tt_pool *pool = tt_pool_create(threads, TT_AFFINITY_ALL);
    tt_kernels_set_pool(pool);

    printf("\nportao do M2 -- build %s\n", tt_build_id());
    printf("CPU: %s, %d threads\n", tt_kernels_impl_name(), tt_pool_threads(pool));

    const tt_backend *be = tt_backend_load(dll, 0);
    if (!be) {
        printf("\nsem GPU: %s\n", tt_error());
        printf("O engine roda em CPU normalmente; o M2 e que nao pode ser\n"
               "avaliado sem backend.\n\n");
        tt_model_close(&m);
        return 1;
    }

    /* Tres colunas de tempo, nao duas.  `GPU ms` inclui H2D, lancamento, D2H e
     * sincronizacao -- um piso que existe so porque a interface atual manda um
     * matmul por vez.  `kernel ms` e o kernel sozinho, cronometrado no device,
     * e e o numero que diz se vale escrever o `run_layer` do M2 (e se vale
     * comprar placa maior). */
    printf("\n  %-36s %8s %8s %8s %8s %8s %9s\n", "tensor", "CPU ms",
           "CPU GB/s", "GPU ms", "kernel", "GB/s", "erro rel");

    double sum_cpu = 0, sum_gpu = 0, sum_ker = 0, sum_bytes = 0;
    int measured = 0;

    for (int i = 0; i < N_TARGETS; i++) {
        const tt_tensor *t = tt_container_find(&m.c, TARGETS[i]);
        if (!t) continue;

        float *x  = (float *)tt_alloc_aligned((size_t)t->cols * sizeof(float), TT_ALIGN);
        float *y1 = (float *)tt_alloc_aligned((size_t)t->rows * sizeof(float), TT_ALIGN);
        float *y2 = (float *)tt_alloc_aligned((size_t)t->rows * sizeof(float), TT_ALIGN);
        if (!x || !y1 || !y2) break;

        /* Ativacao com a cara da real: media zero, magnitude ~1. */
        for (int64_t c = 0; c < t->cols; c++)
            x[c] = (float)(((c * 2654435761u) % 2000) - 1000) / 1000.0f;

        const tt_dev_tensor dev = be->upload(t->data, t->data_bytes,
                                             t->scales, t->scales_bytes,
                                             (int)t->kind, t->rows, t->cols,
                                             t->group);
        if (dev == TT_DEV_NONE) {
            printf("  %-46.46s   nao coube na VRAM\n", TARGETS[i]);
            tt_free_aligned(x); tt_free_aligned(y1); tt_free_aligned(y2);
            continue;
        }

        /* Aquece os dois lados antes de cronometrar: a primeira passada da CPU
         * traz o peso do disco e a da GPU compila o kernel. */
        tt_matmul(y1, t, x);
        be->matmul(y2, dev, x);
        be->sync();

        /* MELHOR caso, nao media.
         *
         * A media mede a maquina, nao o kernel: com o editor compilando ao
         * lado, o mesmo matmul de CPU mediu 0,033 ms e 0,612 ms na mesma
         * sessao -- 18x de diferenca vinda de fora.  O kernel e determinista,
         * entao o menor tempo observado E o custo dele; interferencia so
         * adiciona. */
        double cpu_ms = 1e9, gpu_ms = 1e9;
        for (int r = 0; r < reps; r++) {
            const double c0 = tt_now();
            tt_matmul(y1, t, x);
            const double dt = (tt_now() - c0) * 1000.0;
            if (dt < cpu_ms) cpu_ms = dt;
        }
        for (int r = 0; r < reps; r++) {
            const double g0 = tt_now();
            be->matmul(y2, dev, x);
            be->sync();
            const double dt = (tt_now() - g0) * 1000.0;
            if (dt < gpu_ms) gpu_ms = dt;
        }

        /* Kernel puro.  Backend antigo (ABI 1) nao tem isto e devolve -1. */
        const double ker_ms = be->bench ? be->bench(dev, reps * 5) : -1.0;

        /* Trafego do kernel: o tensor inteiro mais escalas.  E o denominador
         * honesto -- este e um matmul memory-bound dos dois lados. */
        const double bytes = (double)t->data_bytes + (double)t->scales_bytes;
        const double gbs = ker_ms > 0 ? bytes / (ker_ms * 1e-3) / 1e9 : 0.0;

        double num = 0, den = 0;
        for (int64_t r = 0; r < t->rows; r++) {
            const double d = (double)y1[r] - (double)y2[r];
            num += d * d;
            den += (double)y1[r] * (double)y1[r];
        }
        const double rel = den > 0 ? sqrt(num / den) : 0.0;

        const double cpu_gbs = cpu_ms > 0 ? bytes / (cpu_ms * 1e-3) / 1e9 : 0.0;
        printf("  %-36.36s %8.3f %8.1f %8.3f %8.3f %8.1f %8.4f%%\n", TARGETS[i],
               cpu_ms, cpu_gbs, gpu_ms, ker_ms, gbs, rel * 100.0);

        sum_cpu   += cpu_ms;
        sum_gpu   += gpu_ms;
        if (ker_ms > 0) { sum_ker += ker_ms; sum_bytes += bytes; }
        measured++;

        be->release(dev);
        tt_free_aligned(x); tt_free_aligned(y1); tt_free_aligned(y2);
    }

    if (measured) {
        const double ida_volta = sum_gpu > 0 ? sum_cpu / sum_gpu : 0.0;
        const double ganho     = sum_ker > 0 ? sum_cpu / sum_ker : ida_volta;
        const double gbs       = sum_ker > 0
                               ? sum_bytes / (sum_ker * 1e-3) / 1e9 : 0.0;

        const double cpu_gbs = sum_cpu > 0 ? sum_bytes / (sum_cpu * 1e-3) / 1e9 : 0;

        /* Banda de STREAMING da CPU, medida por Tests/bench (docs/19).
         *
         * O `CPU GB/s` da tabela acima e otimista de proposito de ninguem:
         * depois de 400 repeticoes o tensor esta no L3 de 36 MB, e o decode
         * real nao tem essa sorte -- ele toca 2,87 GB por token, vindos da RAM.
         * Comparar a GPU (lendo da VRAM de verdade) contra a CPU (lendo do L3)
         * seria comparar coisas diferentes, entao a projecao usa a banda de
         * RAM. */
        const double ganho_real = cpu_stream > 0 ? gbs / cpu_stream : ganho;

        printf("\n  CPU quente (L3)       : %.3f ms -- %.0f GB/s\n",
               sum_cpu, cpu_gbs);
        printf("  CPU streaming (RAM)   : %.1f GB/s  (medido por bench)\n",
               cpu_stream);
        printf("  ida e volta por matmul : %.2fx  (%.3f ms)\n",
               ida_volta, sum_gpu);
        if (sum_ker > 0)
            printf("  kernel sozinho        : %.2fx  (%.3f ms) -- %.0f GB/s\n",
                   ganho, sum_ker, gbs);

        /* O piso por chamada e o que o `run_layer` do M2 elimina; medi-lo
         * separado transforma "a GPU e lenta" em "a INTERFACE e lenta", que sao
         * decisoes de projeto diferentes. */
        if (sum_ker > 0 && measured)
            printf("  piso por chamada      : %.3f ms (H2D + lancamento + D2H)\n",
                   (sum_gpu - sum_ker) / measured);

        /* Se `f` da fracao do trafego vai para a GPU, o tempo cai para
         * (1-f) + f/ganho.  O ganho usado e o do KERNEL: e o que um
         * `run_layer` entregaria. */
        printf("\n  ganho que vale para o decode: %.0f / %.1f = %.2fx\n",
               gbs, cpu_stream, ganho_real);
        printf("\n  o que isso implica para o M2 (%.2fx):\n", ganho_real);
        /* As fracoes sao de TRAFEGO POR TOKEN, nao de VRAM: um expert so ajuda
         * se a camada inteira dele (os 64) estiver residente, porque qualquer
         * um pode ser roteado.  A conta esta em docs/18. */
        static const struct { const char *nome; double f; } P[] = {
            { "8 GB  atencao + 15 das 46 camadas MoE",  0.57 },
            { "12 GB atencao + 27 das 46 camadas MoE",  0.73 },
            { "16 GB atencao + 42 das 46 camadas MoE",  0.92 },
            { "24 GB tudo residente (18,5 + 1,8 de KV)", 1.00 },
        };
        for (int p = 0; p < 4; p++) {
            const double t_rel = (1.0 - P[p].f)
                               + P[p].f / (ganho_real > 0 ? ganho_real : 1.0);
            printf("    %-42s %.2fx -> %.1f tok/s\n",
                   P[p].nome, 1.0 / t_rel, 13.5 / t_rel);
        }
        printf("\n  criterio de docs/05: abaixo de 16,2 tok/s o M2 e cancelado.\n"
               "  as fracoes vem da tabela de trafego de docs/18.\n\n");
    }

    tt_backend_unload();
    tt_kernels_set_pool(NULL);
    tt_pool_destroy(pool);
    tt_model_close(&m);
    return 0;
}
