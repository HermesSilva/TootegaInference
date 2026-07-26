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

#define THREADS 256
#define MAX_TENSORS 16384

namespace {

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

/* Um bloco por linha de saida. Cada thread varre grupos com passo
 * `blockDim.x` e acumula em registrador; a soma do bloco vem depois em
 * arvore, na memoria compartilhada. */
__global__ void mm_int4(float *y, const unsigned char *w, const float *scales,
                        const float *x, int cols, int group, int rows)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int half     = group / 2;
    const int ngroups  = cols / group;
    const unsigned char *wr = w + (size_t)row * (cols / 2);
    const float         *sr = scales + (size_t)row * ngroups;

    float acc = 0.0f;
    for (int g = threadIdx.x; g < ngroups; g += blockDim.x) {
        const unsigned char *p  = wr + (size_t)g * half;
        const float          s  = sr[g];
        const float         *xv = x + (size_t)g * group;

        float a = 0.0f;
        for (int j = 0; j < half; j++) {
            const unsigned char b = p[j];
            /* Metade/metade: nibble baixo e o peso j, alto e j + group/2. */
            a += ((float)((int)(b & 0x0F) - 8) * s) * xv[j];
            a += ((float)((int)(b >> 4)   - 8) * s) * xv[half + j];
        }
        acc += a;
    }

    __shared__ float sm[THREADS];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sm[threadIdx.x] += sm[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sm[0];
}

__global__ void mm_int8(float *y, const signed char *w, const float *scales,
                        const float *x, int cols, int group, int rows)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int ngroups = cols / group;
    const signed char *wr = w + (size_t)row * cols;
    const float       *sr = scales + (size_t)row * ngroups;

    float acc = 0.0f;
    for (int g = threadIdx.x; g < ngroups; g += blockDim.x) {
        const signed char *p  = wr + (size_t)g * group;
        const float        s  = sr[g];
        const float       *xv = x + (size_t)g * group;
        float a = 0.0f;
        for (int j = 0; j < group; j++) a += ((float)p[j] * s) * xv[j];
        acc += a;
    }

    __shared__ float sm[THREADS];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sm[threadIdx.x] += sm[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sm[0];
}

__global__ void mm_f32(float *y, const float *w, const float *x,
                       int cols, int rows)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const float *wr = w + (size_t)row * cols;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) acc += wr[i] * x[i];

    __shared__ float sm[THREADS];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sm[threadIdx.x] += sm[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sm[0];
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

    const int rows = (int)t.rows, cols = (int)t.cols;
    switch (t.kind) {
    case 0: /* TT_KIND_INT4 */
        mm_int4<<<rows, THREADS>>>(g_y, (const unsigned char *)t.data,
                                   t.scales, g_x, cols, t.group, rows);
        break;
    case 1: /* TT_KIND_INT8 */
        mm_int8<<<rows, THREADS>>>(g_y, (const signed char *)t.data,
                                   t.scales, g_x, cols, t.group, rows);
        break;
    case 2: /* TT_KIND_F32 */
        mm_f32<<<rows, THREADS>>>(g_y, (const float *)t.data, g_x, cols, rows);
        break;
    default:
        return -1;                       /* bf16 fica na CPU por enquanto */
    }

    if (cudaMemcpy(y, g_y, yb, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    return 0;
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
    snprintf(g_iface.name, sizeof(g_iface.name), "CUDA (nao inicializado)");
    return &g_iface;
}

}
