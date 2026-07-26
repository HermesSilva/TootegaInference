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

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const int has = (i + 1 < argc);
        if      (!strcmp(a, "--dll") && has)     dll = argv[++i];
        else if (!strcmp(a, "--reps") && has)    reps = atoi(argv[++i]);
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

    printf("\n  %-46s %9s %9s %7s %10s\n", "tensor", "CPU ms", "GPU ms",
           "ganho", "erro rel");

    double sum_cpu = 0, sum_gpu = 0;
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

        const double c0 = tt_now();
        for (int r = 0; r < reps; r++) tt_matmul(y1, t, x);
        const double cpu_ms = (tt_now() - c0) * 1000.0 / reps;

        const double g0 = tt_now();
        for (int r = 0; r < reps; r++) be->matmul(y2, dev, x);
        be->sync();
        const double gpu_ms = (tt_now() - g0) * 1000.0 / reps;

        double num = 0, den = 0;
        for (int64_t r = 0; r < t->rows; r++) {
            const double d = (double)y1[r] - (double)y2[r];
            num += d * d;
            den += (double)y1[r] * (double)y1[r];
        }
        const double rel = den > 0 ? sqrt(num / den) : 0.0;

        printf("  %-46.46s %9.3f %9.3f %6.2fx %9.4f%%\n", TARGETS[i],
               cpu_ms, gpu_ms, gpu_ms > 0 ? cpu_ms / gpu_ms : 0.0, rel * 100.0);

        sum_cpu += cpu_ms;
        sum_gpu += gpu_ms;
        measured++;

        be->release(dev);
        tt_free_aligned(x); tt_free_aligned(y1); tt_free_aligned(y2);
    }

    if (measured) {
        const double ganho = sum_gpu > 0 ? sum_cpu / sum_gpu : 0.0;
        printf("\n  agregado: %.2fx  (%.3f ms contra %.3f ms)\n",
               ganho, sum_cpu, sum_gpu);

        /* O veredito do portao.  A partir do ganho por matmul da para estimar
         * o teto do split: se `f` da fracao do trabalho vai para a GPU, o
         * tempo cai para (1-f) + f/ganho. */
        printf("\n  o que isso implica para o M2:\n");
        for (double f = 0.3; f <= 0.71; f += 0.2) {
            const double t_rel = (1.0 - f) + f / (ganho > 0 ? ganho : 1.0);
            printf("    %2.0f%% do trabalho na GPU -> %.2fx no total"
                   " -> %.1f tok/s (base 13,5)\n",
                   f * 100.0, 1.0 / t_rel, 13.5 / t_rel);
        }
        printf("\n  criterio de docs/05: abaixo de 16,2 tok/s o M2 e cancelado.\n\n");
    }

    tt_backend_unload();
    tt_kernels_set_pool(NULL);
    tt_pool_destroy(pool);
    tt_model_close(&m);
    return 0;
}
