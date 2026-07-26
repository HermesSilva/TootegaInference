/* tootega_cuda.cu -- backend CUDA, carregado em execucao pelo engine.
 *
 * Fica fora do binario por decisao de docs/10: CUDA e C++ e arrasta um runtime
 * pesado. O engine linka nada disto; ele faz `LoadLibrary` e conversa por C
 * ABI (src/backend.h).
 *
 * A regra que molda o codigo inteiro: **peso nao atravessa o PCIe**. O tensor
 * sobe uma vez em `upload` e fica na VRAM; o que cruza a cada chamada sao
 * 8 KB de ativacao. A 12 GB/s do gen3 x16, mandar peso daria 4 tok/s -- pior
 * que a CPU sozinha.
 *
 * O desempacotamento espelha `src/kernels.c` e docs/14: metade/metade dentro
 * do grupo, codigo deslocado de 8. A ordem de reducao NAO e a mesma -- aqui e
 * uma arvore por bloco, la e um acumulador escalar por linha --, entao os bits
 * diferem. Ver docs/18 sobre o que isso significa para o criterio do M2.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
#include "backend.h"
}

/* Um WARP por linha de saida, nao um bloco.
 *
 * A primeira versao dava um bloco de 256 threads para cada linha e varria os
 * grupos em laco escalar de bytes.  Duas patologias somadas:
 *
 *   - com `cols=2048` ha 16 grupos, entao **16 das 256 threads trabalhavam** e
 *     240 ficavam paradas na barreira;
 *   - cada thread lia 32 bytes CONSECUTIVOS, e as threads vizinhas liam 32
 *     bytes muito adiante -- o warp emitia 32 transacoes de 1 byte em vez de
 *     uma de 128.
 *
 * Medido: 5,9 GB/s numa placa que faz 448.  Aqui a lane `l` le o vetor de 16
 * bytes `l`, `l+32`, ... -- lanes vizinhas leem enderecos vizinhos, o warp
 * inteiro vira uma transacao alinhada -- e a soma sai por `__shfl_down_sync`,
 * sem memoria compartilhada e sem barreira. */
#define WARP        32
#define ROWS_BLOCK  4                       /* warps por bloco */
#define ROWS_WARP   4                       /* linhas que um warp resolve junto */
#define THREADS     (WARP * ROWS_BLOCK)
#define MAX_TENSORS 16384

namespace {

__device__ __forceinline__ float warp_sum(float v)
{
    for (int off = WARP / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

struct DevTensor {
    void   *data   = nullptr;
    float  *scales = nullptr;
    int64_t rows = 0, cols = 0;
    int     kind = 0, group = 0;
};

std::vector<DevTensor> g_tensors;     /* indice 0 reservado = TT_DEV_NONE */
tt_backend             g_iface;
float                 *g_x = nullptr; /* buffers persistentes de ativacao: */
float                 *g_y = nullptr; /* alocar por chamada custaria mais que */
size_t                 g_x_cap = 0;   /* o proprio matmul */
size_t                 g_y_cap = 0;

/* --------------------------------------------------------------- kernels */

/* int4, meia/meia dentro do grupo (docs/14).  Um vetor de 16 bytes cabe
 * inteiro dentro de um grupo enquanto `group/2` for multiplo de 16 -- com
 * group=64 sao 32 bytes por grupo, e 16 divide 32.  O caminho generico existe
 * para qualquer outro group. */
__global__ void mm_int4(float *y, const unsigned char *w, const float *scales,
                        const float *x, int cols, int group, int rows)
{
    const int warp = threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    const int row0 = (blockIdx.x * ROWS_BLOCK + warp) * ROWS_WARP;
    if (row0 >= rows) return;

    const int half    = group / 2;
    const int ngroups = cols / group;
    const size_t rb   = (size_t)(cols / 2);          /* bytes por linha */
    const int nrow    = min(ROWS_WARP, rows - row0);

    float acc[ROWS_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_WARP; r++) acc[r] = 0.0f;

    if ((half & 15) == 0 && ((size_t)(w + (size_t)row0 * rb) & 15) == 0) {
        const int nvec = (int)(rb / 16);             /* vetores de 16 bytes */

        for (int v = lane; v < nvec; v += WARP) {
            const int   byte0 = v * 16;
            const int   g     = byte0 / half;
            const int   j0    = byte0 - g * half;
            const float *xlo  = x + (size_t)g * group + j0;

            /* `x` sai em float4 e e lido UMA vez para as ROWS_WARP linhas.
             *
             * Duas correcoes na mesma linha de codigo: antes eram 32 leituras
             * escalares por vetor de peso -- 8 loads de x para cada load de
             * peso, e o kernel ficava presp em emissao, nao em banda -- e cada
             * linha relia o mesmo x.  Agora sao 8 loads vetoriais amortizados
             * por 4 linhas.  Alinhamento garantido: j0 e multiplo de 16 floats
             * e group de 64. */
            const float4 *xl4 = (const float4 *)xlo;
            const float4 *xh4 = (const float4 *)(xlo + half);

            float lo[16], hi[16];
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                const float4 a = __ldg(&xl4[u]);
                const float4 b = __ldg(&xh4[u]);
                lo[u * 4 + 0] = a.x; lo[u * 4 + 1] = a.y;
                lo[u * 4 + 2] = a.z; lo[u * 4 + 3] = a.w;
                hi[u * 4 + 0] = b.x; hi[u * 4 + 1] = b.y;
                hi[u * 4 + 2] = b.z; hi[u * 4 + 3] = b.w;
            }

            #pragma unroll
            for (int r = 0; r < ROWS_WARP; r++) {
                if (r >= nrow) break;
                const uint4 packed =
                    __ldg(&((const uint4 *)(w + (size_t)(row0 + r) * rb))[v]);
                const unsigned int word[4] =
                    { packed.x, packed.y, packed.z, packed.w };
                float a = 0.0f;
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    const unsigned int W = word[u];
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        const unsigned int b = (W >> (8 * k)) & 0xFFu;
                        a += (float)((int)(b & 0x0F) - 8) * lo[u * 4 + k];
                        a += (float)((int)(b >> 4)   - 8) * hi[u * 4 + k];
                    }
                }
                acc[r] += a * __ldg(&scales[(size_t)(row0 + r) * ngroups + g]);
            }
        }
    } else {
        for (int r = 0; r < nrow; r++) {
            const unsigned char *wr = w + (size_t)(row0 + r) * rb;
            const float *sr = scales + (size_t)(row0 + r) * ngroups;
            for (int g = lane; g < ngroups; g += WARP) {
                const unsigned char *p = wr + (size_t)g * half;
                const float *xv = x + (size_t)g * group;
                float a = 0.0f;
                for (int j = 0; j < half; j++) {
                    const unsigned char b = __ldg(&p[j]);
                    a += (float)((int)(b & 0x0F) - 8) * __ldg(&xv[j]);
                    a += (float)((int)(b >> 4)   - 8) * __ldg(&xv[half + j]);
                }
                acc[r] += a * __ldg(&sr[g]);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_WARP; r++) {
        const float v = warp_sum(acc[r]);
        if (lane == 0 && r < nrow) y[row0 + r] = v;
    }
}

__global__ void mm_int8(float *y, const signed char *w, const float *scales,
                        const float *x, int cols, int group, int rows)
{
    const int row = blockIdx.x * ROWS_BLOCK + (threadIdx.x / WARP);
    if (row >= rows) return;
    const int lane = threadIdx.x % WARP;

    const int ngroups = cols / group;
    const signed char *wr = w + (size_t)row * cols;
    const float       *sr = scales + (size_t)row * ngroups;

    float acc = 0.0f;

    if ((group & 15) == 0 && ((size_t)wr & 15) == 0) {
        const int nvec = cols / 16;
        const uint4 *wv = (const uint4 *)wr;

        for (int v = lane; v < nvec; v += WARP) {
            const uint4 packed = __ldg(&wv[v]);
            const int   byte0  = v * 16;
            const int   g      = byte0 / group;
            const float s      = __ldg(&sr[g]);
            const float *xv    = x + byte0;

            const float4 *xv4 = (const float4 *)xv;
            const unsigned int word[4] = { packed.x, packed.y, packed.z, packed.w };

            float a = 0.0f;
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                const unsigned int W = word[u];
                const float4 xf = __ldg(&xv4[u]);
                const float f[4] = { xf.x, xf.y, xf.z, xf.w };
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int b = (int)(signed char)((W >> (8 * k)) & 0xFFu);
                    a += (float)b * f[k];
                }
            }
            acc += a * s;
        }
    } else {
        for (int g = lane; g < ngroups; g += WARP) {
            const signed char *p = wr + (size_t)g * group;
            const float s = __ldg(&sr[g]);
            const float *xv = x + (size_t)g * group;
            float a = 0.0f;
            for (int j = 0; j < group; j++) a += (float)__ldg(&p[j]) * __ldg(&xv[j]);
            acc += a * s;
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

__global__ void mm_f32(float *y, const float *w, const float *x,
                       int cols, int rows)
{
    const int row = blockIdx.x * ROWS_BLOCK + (threadIdx.x / WARP);
    if (row >= rows) return;
    const int lane = threadIdx.x % WARP;

    const float *wr = w + (size_t)row * cols;
    float acc = 0.0f;

    if ((cols & 3) == 0 && ((size_t)wr & 15) == 0) {
        const int nvec = cols / 4;
        const float4 *wv = (const float4 *)wr;
        for (int v = lane; v < nvec; v += WARP) {
            const float4 a = __ldg(&wv[v]);
            const float4 b = __ldg(&((const float4 *)x)[v]);
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int i = lane; i < cols; i += WARP) acc += __ldg(&wr[i]) * __ldg(&x[i]);
    }

    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

/* Despacha o kernel certo.  Sem transferencia: quem chama decide onde `x` e
 * `y` estao. */
int launch(const DevTensor &t, float *dy, const float *dx)
{
    const int rows = (int)t.rows, cols = (int)t.cols;
    const int blocks  = (rows + ROWS_BLOCK - 1) / ROWS_BLOCK;
    const int blocks4 = (rows + ROWS_BLOCK * ROWS_WARP - 1)
                      / (ROWS_BLOCK * ROWS_WARP);

    switch (t.kind) {
    case 0: mm_int4<<<blocks4, THREADS>>>(dy, (const unsigned char *)t.data,
                                          t.scales, dx, cols, t.group, rows);
            return 0;
    case 1: mm_int8<<<blocks, THREADS>>>(dy, (const signed char *)t.data,
                                         t.scales, dx, cols, t.group, rows);
            return 0;
    case 2: mm_f32<<<blocks, THREADS>>>(dy, (const float *)t.data, dx, cols, rows);
            return 0;
    default: return -1;                  /* bf16 fica na CPU por enquanto */
    }
}

/* ------------------------------------------------------------- interface */

int be_init(int device)
{
    if (cudaSetDevice(device) != cudaSuccess) return -1;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return -1;

    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);

    snprintf(g_iface.name, sizeof(g_iface.name), "CUDA %s sm_%d%d",
             prop.name, prop.major, prop.minor);
    g_iface.vram_total = totalB;
    g_iface.vram_free  = freeB;

    g_tensors.clear();
    g_tensors.push_back(DevTensor{});      /* slot 0 = TT_DEV_NONE */
    return 0;
}

void be_shutdown(void)
{
    for (auto &t : g_tensors) {
        if (t.data)   cudaFree(t.data);
        if (t.scales) cudaFree(t.scales);
    }
    g_tensors.clear();
    if (g_x) { cudaFree(g_x); g_x = nullptr; g_x_cap = 0; }
    if (g_y) { cudaFree(g_y); g_y = nullptr; g_y_cap = 0; }
    cudaDeviceReset();
}

tt_dev_tensor be_upload(const void *data, unsigned long long bytes,
                        const float *scales, unsigned long long scales_bytes,
                        int kind, long long rows, long long cols, int group)
{
    if (g_tensors.size() >= MAX_TENSORS) return TT_DEV_NONE;

    DevTensor t;
    t.rows = rows; t.cols = cols; t.kind = kind; t.group = group;

    /* Falta de VRAM nao e erro: e informacao para o planner deixar esta camada
     * na CPU. Por isso devolve TT_DEV_NONE em vez de abortar. */
    if (cudaMalloc(&t.data, (size_t)bytes) != cudaSuccess) return TT_DEV_NONE;
    if (cudaMemcpy(t.data, data, (size_t)bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(t.data);
        return TT_DEV_NONE;
    }

    if (scales && scales_bytes) {
        if (cudaMalloc(&t.scales, (size_t)scales_bytes) != cudaSuccess) {
            cudaFree(t.data);
            return TT_DEV_NONE;
        }
        if (cudaMemcpy(t.scales, scales, (size_t)scales_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(t.data); cudaFree(t.scales);
            return TT_DEV_NONE;
        }
    }

    g_tensors.push_back(t);

    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    g_iface.vram_free = freeB;

    return (tt_dev_tensor)(g_tensors.size() - 1);
}

void be_release(tt_dev_tensor id)
{
    if (id == TT_DEV_NONE || id >= g_tensors.size()) return;
    DevTensor &t = g_tensors[id];
    if (t.data)   { cudaFree(t.data);   t.data = nullptr; }
    if (t.scales) { cudaFree(t.scales); t.scales = nullptr; }
}

int be_matmul(float *y, tt_dev_tensor id, const float *x)
{
    if (id == TT_DEV_NONE || id >= g_tensors.size()) return -1;
    const DevTensor &t = g_tensors[id];
    if (!t.data) return -1;

    const size_t xb = (size_t)t.cols * sizeof(float);
    const size_t yb = (size_t)t.rows * sizeof(float);

    if (g_x_cap < xb) {
        if (g_x) cudaFree(g_x);
        if (cudaMalloc(&g_x, xb) != cudaSuccess) { g_x_cap = 0; return -1; }
        g_x_cap = xb;
    }
    if (g_y_cap < yb) {
        if (g_y) cudaFree(g_y);
        if (cudaMalloc(&g_y, yb) != cudaSuccess) { g_y_cap = 0; return -1; }
        g_y_cap = yb;
    }

    /* A unica travessia do PCIe: 8 KB de ativacao para la, `rows` floats de
     * volta. */
    if (cudaMemcpy(g_x, x, xb, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (launch(t, g_y, g_x) != 0) return -1;
    if (cudaMemcpy(y, g_y, yb, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    return 0;
}

/* Tempo do KERNEL, sem transferencia nem sincronizacao por chamada.
 *
 * Existe porque `be_matmul` mede outra coisa: cada chamada paga H2D, lancamento
 * e D2H com sincronizacao implicita -- um piso de ~0,15 ms que nao tem nada a
 * ver com a velocidade do kernel.  O desenho que o M2 realmente propoe
 * (`run_layer` no device) nao paga esse piso por matmul, entao decidir a compra
 * de uma placa pelo numero do ida-e-volta seria decidir pelo numero errado.
 *
 * Devolve milissegundos por chamada, ou -1. */
double be_bench(tt_dev_tensor id, int reps)
{
    if (id == TT_DEV_NONE || id >= g_tensors.size() || reps <= 0) return -1.0;
    const DevTensor &t = g_tensors[id];
    if (!t.data) return -1.0;

    const size_t xb = (size_t)t.cols * sizeof(float);
    const size_t yb = (size_t)t.rows * sizeof(float);
    if (g_x_cap < xb) {
        if (g_x) cudaFree(g_x);
        if (cudaMalloc(&g_x, xb) != cudaSuccess) { g_x_cap = 0; return -1.0; }
        g_x_cap = xb;
        cudaMemset(g_x, 0, xb);
    }
    if (g_y_cap < yb) {
        if (g_y) cudaFree(g_y);
        if (cudaMalloc(&g_y, yb) != cudaSuccess) { g_y_cap = 0; return -1.0; }
        g_y_cap = yb;
    }

    for (int i = 0; i < 8; i++) launch(t, g_y, g_x);   /* aquece o clock */
    cudaDeviceSynchronize();

    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) launch(t, g_y, g_x);
    cudaEventRecord(b);
    cudaEventSynchronize(b);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return (double)ms / reps;
}

void be_sync(void) { cudaDeviceSynchronize(); }

} /* namespace */

extern "C" {

#if defined(_WIN32)
#  define TT_EXPORT __declspec(dllexport)
#else
#  define TT_EXPORT __attribute__((visibility("default")))
#endif

TT_EXPORT const tt_backend *tootega_backend(void)
{
    memset(&g_iface, 0, sizeof(g_iface));
    g_iface.abi      = TT_BACKEND_ABI;
    g_iface.init     = be_init;
    g_iface.shutdown = be_shutdown;
    g_iface.upload   = (tt_dev_tensor (*)(const void *, uint64_t, const float *,
                                          uint64_t, int, int64_t, int64_t, int))
                       be_upload;
    g_iface.release  = be_release;
    g_iface.matmul   = be_matmul;
    g_iface.sync     = be_sync;
    g_iface.bench    = be_bench;
    snprintf(g_iface.name, sizeof(g_iface.name), "CUDA (nao inicializado)");
    return &g_iface;
}

}
