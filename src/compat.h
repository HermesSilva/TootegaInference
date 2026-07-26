/* compat.h -- a fina camada entre POSIX e Win32.
 *
 * Tudo que difere entre os dois SOs mora aqui e em lugar nenhum mais. Ver
 * docs/10-implementacao.md: o nucleo e C11 puro, e este arquivo existe para
 * que ele possa continuar sendo.
 */
#ifndef TOOTEGA_COMPAT_H
#define TOOTEGA_COMPAT_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  define TT_WINDOWS 1
#else
#  define TT_POSIX 1
#endif

/* ---------------------------------------------------------------- mmap */

typedef struct {
    void   *base;      /* inicio do mapeamento, NULL se falhou */
    size_t  size;
#if defined(TT_WINDOWS)
    void   *file;      /* HANDLE */
    void   *mapping;   /* HANDLE */
#else
    int     fd;
#endif
} tt_map;

/* Mapeia o arquivo inteiro, somente leitura. base == NULL indica falha. */
int  tt_map_open(tt_map *m, const char *path);
void tt_map_close(tt_map *m);

/* Sinaliza que a faixa sera lida em breve e forca residencia.
 * Duas coisas distintas: o hint e assincrono, o toque e sincrono -- ver
 * docs/11-memoria.md secao 2. */
void tt_map_willneed(const tt_map *m, size_t offset, size_t length);
void tt_map_touch(const tt_map *m, size_t offset, size_t length);

/* ------------------------------------------------------------- memoria */

/* Alocacao alinhada.  `align` e potencia de dois.  Existe aqui porque os tres
 * SOs discordam do nome (`_aligned_malloc`, `aligned_alloc`, `posix_memalign`)
 * e porque o par alocar/liberar tem de casar no Windows. */
void *tt_alloc_aligned(size_t bytes, size_t align);
void  tt_free_aligned(void *p);

double tt_ram_total_gb(void);
double tt_ram_avail_gb(void);
double tt_rss_gb(void);            /* residente deste processo */

/* --------------------------------------------------------------- tempo */

double tt_now(void);               /* segundos, monotonico */

/* ----------------------------------------------------------------- cpu */

int tt_cpu_threads(void);          /* logicos, incluindo E-cores */

/* Topologia.  O 14900K e hibrido e docs/09 manda pinar nos P-cores por
 * default: espalhar um matmul memory-bound pelos E-cores costuma piorar.
 * Ambas devolvem UM logico por core fisico -- irmaos SMT competem pela mesma
 * unidade de execucao e nao ajudam num laco ja limitado por memoria.
 * Retornam quantos escreveram. */
int tt_cpu_pcore_list(int *out, int max);   /* so os P-cores */
int tt_cpu_core_list(int *out, int max);    /* todos os cores fisicos */

/* ------------------------------------------------------------- threads */

typedef struct { void *handle; } tt_thread;

int  tt_thread_start(tt_thread *t, void (*fn)(void *), void *arg);
void tt_thread_join(tt_thread *t);
void tt_thread_pin(int cpu);       /* cpu < 0 nao faz nada */

void tt_cpu_relax(void);           /* dica de espera ativa (`pause`) */
void tt_yield(void);               /* cede o resto do quantum */
void tt_sleep_ms(int ms);          /* bloqueia de verdade */

/* Espera ate que o valor em `addr` deixe de ser `expected`, ou o tempo acabe.
 *
 * Dormir por tempo fixo e polling disfarcado: quem acorda tarde atrasa o
 * trabalho, e quem acorda cedo queima CPU. Aqui quem despacha ACORDA os que
 * esperam, e o timeout existe so como rede contra acordar perdido. */
void tt_futex_wait(const volatile int32_t *addr, int32_t expected, int timeout_ms);
void tt_futex_wake(const volatile int32_t *addr);

/* -------------------------------------------------------------- rede */

/* O minimo para um servidor HTTP: escutar, aceitar, ler, escrever.  Winsock e
 * BSD sockets divergem no bastante (inicializacao global, tipo do descritor,
 * nome do close) para que o nucleo nao possa ver nenhum dos dois. */

typedef struct { intptr_t fd; } tt_socket;

#define TT_INVALID_SOCKET ((intptr_t)-1)

int  tt_net_init(void);            /* uma vez, antes de qualquer socket */
void tt_net_shutdown(void);

/* `host` NULL ou "127.0.0.1" escuta so no loopback; "0.0.0.0" abre para a
 * rede -- e decisao de quem chama, nunca default (docs/05, M3). */
int  tt_listen(tt_socket *s, const char *host, int port, int backlog);
int  tt_accept(const tt_socket *server, tt_socket *client, char *peer, size_t peer_len);

/* Lado cliente: o agente fala com o servidor por HTTP.  So IP literal. */
int  tt_connect(tt_socket *s, const char *host, int port);

/* Retornam bytes movidos, 0 no fim da conexao, -1 em erro. */
long tt_recv(const tt_socket *s, void *buf, size_t n);
long tt_send_all(const tt_socket *s, const void *buf, size_t n);

void tt_socket_close(tt_socket *s);
void tt_socket_nodelay(const tt_socket *s);   /* streaming token a token */
int  tt_socket_timeout(const tt_socket *s, int seconds);

/* ------------------------------------------------------------ atomicos */

/* O minimo que o pool precisa, sem `<stdatomic.h>`: o MSVC so o habilita sob
 * `/experimental:c11atomics`, e depender de flag experimental para o coracao
 * do paralelismo nao vale a economia de trinta linhas.
 *
 * Ordem: fetch_add e barreira total nos dois compiladores; load e acquire e
 * store e release, que em x86-64 sao a leitura e a escrita comuns. */

typedef struct { volatile long      v; } tt_atomic32;
typedef struct { volatile long long v; } tt_atomic64;

#if defined(TT_WINDOWS)
#  include <intrin.h>
static inline int32_t tt_at32_add(tt_atomic32 *a, int32_t d)
{ return (int32_t)_InterlockedExchangeAdd(&a->v, (long)d); }
static inline int64_t tt_at64_add(tt_atomic64 *a, int64_t d)
{ return (int64_t)_InterlockedExchangeAdd64(&a->v, (long long)d); }
static inline int32_t tt_at32_load(const tt_atomic32 *a)
{ int32_t v = (int32_t)a->v; _ReadWriteBarrier(); return v; }
static inline void tt_at32_store(tt_atomic32 *a, int32_t v)
{ _ReadWriteBarrier(); a->v = (long)v; }
static inline void tt_at64_store(tt_atomic64 *a, int64_t v)
{ _ReadWriteBarrier(); a->v = (long long)v; }
#else
static inline int32_t tt_at32_add(tt_atomic32 *a, int32_t d)
{ return (int32_t)__atomic_fetch_add(&a->v, (long)d, __ATOMIC_SEQ_CST); }
static inline int64_t tt_at64_add(tt_atomic64 *a, int64_t d)
{ return (int64_t)__atomic_fetch_add(&a->v, (long long)d, __ATOMIC_SEQ_CST); }
static inline int32_t tt_at32_load(const tt_atomic32 *a)
{ return (int32_t)__atomic_load_n(&a->v, __ATOMIC_ACQUIRE); }
static inline void tt_at32_store(tt_atomic32 *a, int32_t v)
{ __atomic_store_n(&a->v, (long)v, __ATOMIC_RELEASE); }
static inline void tt_at64_store(tt_atomic64 *a, int64_t v)
{ __atomic_store_n(&a->v, (long long)v, __ATOMIC_RELEASE); }
#endif

#endif /* TOOTEGA_COMPAT_H */
