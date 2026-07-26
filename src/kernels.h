/* kernels.h -- a aritmetica.  Ingenua de proposito, por enquanto.
 *
 * docs/15-kernels-and-numerics.md fixa tres coisas que este arquivo obedece:
 *
 *   W4A32   Desempacota o peso para float e multiplica com FMA.  O decode e
 *           memory-bound (~33 ms de memoria contra 4-7 ms de compute), entao
 *           gastar a folga de compute comprando precisao e o negocio certo.
 *
 *   fp32    Acumulador float, nada em double.  O objetivo e CASAR com o
 *           oraculo, nao supera-lo.
 *
 *   ordem   A reducao ao longo de I acontece inteira dentro de uma thread, na
 *   fixa    ordem 0..I-1, com acumulador escalar por linha.  Threads nunca
 *           somam parcelas umas das outras -- se somassem, o resultado mudaria
 *           com --threads e o teste de determinismo falharia por construcao.
 *
 * Por isso toda funcao de matmul trabalha sobre uma FAIXA DE LINHAS: e assim
 * que o pool vai particionar o trabalho sem tocar na ordem da reducao.
 *
 * Ordem de implementacao (docs/15): ingenuo -> oraculo passando -> SIMD ->
 * threads.  Estamos no primeiro degrau, e ele permanece como referencia
 * executavel dos proximos.
 */
#ifndef TOOTEGA_KERNELS_H
#define TOOTEGA_KERNELS_H

#include <stdint.h>

#include "container.h"

/* ------------------------------------------------------- implementacao */

/* O kernel vetorizado e o ingenuo NAO dao os mesmos bits: o SIMD acumula em 8
 * lanes e usa FMA (um arredondamento em vez de dois).  Ambos sao validos --
 * nenhum e "o certo" -- mas so um pode ser a referencia, e a referencia e o
 * ingenuo, que casa com o Python.  Poder trocar em runtime e o que transforma
 * "o SIMD divergiu" num diagnostico de minutos (docs/15). */
typedef enum {
    TT_IMPL_AUTO = 0,      /* SIMD quando disponivel */
    TT_IMPL_SCALAR,        /* referencia ingenua, sempre */
} tt_impl;

void        tt_kernels_set_impl(tt_impl impl);
tt_impl     tt_kernels_impl(void);
const char *tt_kernels_impl_name(void);

/* Pool usado pelos matmuls.  NULL = tudo na thread chamadora.  A particao e
 * sempre por linha de saida, entao ligar ou desligar o pool nao muda um bit do
 * resultado -- e isso que o teste de determinismo verifica. */
struct tt_pool;
void           tt_kernels_set_pool(struct tt_pool *p);
struct tt_pool *tt_kernels_pool(void);

/* ------------------------------------------------------------- matmul */

/* y[o] = soma_i W[o][i] * x[i], para o em [row0, row1).
 *
 * W e sempre [O, I] com I contiguo -- a dimensao de reducao.  As variantes
 * quantizadas agrupam ao longo de I, `group` pesos por escala. */

void tt_matmul_f32_rows (float *y, const float *w, const float *x,
                         int64_t cols, int64_t row0, int64_t row1);

void tt_matmul_bf16_rows(float *y, const uint16_t *w, const float *x,
                         int64_t cols, int64_t row0, int64_t row1);

void tt_matmul_int8_rows(float *y, const int8_t *w, const float *scales,
                         const float *x, int64_t cols, int group,
                         int64_t row0, int64_t row1);

/* int4 empacotado metade/metade: dentro de cada grupo, byte[j] tem o peso j no
 * nibble baixo e o peso j + group/2 no alto (docs/14). */
void tt_matmul_int4_rows(float *y, const uint8_t *w, const float *scales,
                         const float *x, int64_t cols, int group,
                         int64_t row0, int64_t row1);

/* Despacho pelo `kind` do tensor.  Retorna -1 se o tensor nao for matriz
 * multiplicavel (1 dimensao, por exemplo). */
int tt_matmul_rows(float *y, const tt_tensor *w, const float *x,
                   int64_t row0, int64_t row1);
int tt_matmul(float *y, const tt_tensor *w, const float *x);

/* Varias matmuls INDEPENDENTES sobre a mesma entrada, num dispatch so.
 *
 * Existe por uma conta de barreira: um expert de 1536x2048 leva 26 us com 20
 * threads, e sincronizar 20 threads custa da mesma ordem. Uma camada MoE faz
 * 15 desses (4 experts + shared, gate/up/down) -- 15 barreiras para 400 us de
 * trabalho. Fundidos, sao 2 barreiras, e as threads veem uma fila de linhas
 * grande o bastante para se equilibrarem.
 *
 * Continua valendo a regra de docs/15: cada thread possui linhas de saida
 * inteiras e a reducao ao longo de I acontece dentro dela. Fundir muda quem
 * pega qual linha, nunca a ordem da soma. */
#define TT_BATCH_MAX 16
int tt_matmul_batch(float **y, const tt_tensor **w, int n, const float *x);

/* --------------------------------------------------------- ativacoes */

/* rmsnorm: y[i] = x[i] * rsqrt(mean(x^2) + eps) * peso[i].
 * A soma dos quadrados e float32 em ordem fixa. */
void tt_rmsnorm(float *y, const float *x, const float *weight,
                int64_t n, float eps);

/* softmax com subtracao do maximo, in-place. */
void tt_softmax(float *x, int64_t n);

/* SwiGLU: dst[i] = silu(gate[i]) * up[i], com silu(v) = v / (1 + exp(-v)). */
void tt_silu_mul(float *dst, const float *gate, const float *up, int64_t n);

/* Produto interno e dst += k*src.  Sao o laco da atencao: com 4k de contexto,
 * 47 camadas e 20 cabecas, sao ~2 GFLOP por token so aqui. */
float tt_dot(const float *a, const float *b, int64_t n);
void  tt_axpy(float *dst, float k, const float *src, int64_t n);

/* Versoes contra um vetor em bf16 -- o cache de KV.  bf16 tem os mesmos 8 bits
 * de expoente do fp32 e 7 de mantissa: nao muda a faixa, so a precisao, e
 * corta o trafego pela metade. */
float tt_dot_bf16(const float *a, const uint16_t *b, int64_t n);
void  tt_axpy_bf16(float *dst, float k, const uint16_t *src, int64_t n);
void  tt_store_bf16(uint16_t *dst, const float *src, int64_t n);

/* Idem contra int8 com escala por grupo -- a mesma quantizacao simetrica do
 * container (docs/14): escala = absmax/127, sem zero-point.  Erra 0,59% contra
 * os ~1,5% do bf16 E ocupa menos: 576 int8 + 9 escalas = 612 B contra 1.152 B
 * do bf16 e 2.304 B do f32. */
float tt_dot_int8g(const float *a, const int8_t *w, const float *scales,
                   int64_t n, int group);
void  tt_axpy_int8g(float *dst, float k, const int8_t *w, const float *scales,
                    int64_t n, int group);
void  tt_store_int8g(int8_t *dst, float *scales, const float *src,
                     int64_t n, int group);

void tt_add(float *dst, const float *src, int64_t n);      /* residuo */
void tt_scale(float *dst, float k, int64_t n);
void tt_copy(float *dst, const float *src, int64_t n);
void tt_zero(float *dst, int64_t n);

int64_t tt_argmax(const float *x, int64_t n);

/* top-k por selecao parcial, ordem decrescente e estavel no empate (indice
 * menor primeiro).  E o roteador do MoE: k = 4 de 64 (docs/04). */
void tt_topk(const float *x, int64_t n, int k, int32_t *idx_out, float *val_out);

/* ------------------------------------------------------------ dequant */

static inline float tt_bf16_to_f32(uint16_t h)
{
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)h << 16;
    return c.f;
}

/* Materializa uma linha do tensor em float32.  E a definicao executavel do
 * layout: se isto e o kernel divergirem, o kernel esta errado. */
int tt_dequant_row(float *dst, const tt_tensor *t, int64_t row);

int tt_kernels_selftest(void);

#endif /* TOOTEGA_KERNELS_H */
