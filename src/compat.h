/* compat.h -- the thin layer between POSIX and Win32.
 *
 * Everything that differs between the two OSes lives here and nowhere else.
 * See docs/10-implementation.md: the core is pure C11, and this file exists so
 * that it can stay that way.
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
    void   *base;      /* start of the mapping, NULL on failure */
    size_t  size;
#if defined(TT_WINDOWS)
    void   *file;      /* HANDLE */
    void   *mapping;   /* HANDLE */
#else
    int     fd;
#endif
} tt_map;

/* Maps the whole file, read-only.  base == NULL means failure. */
int  tt_map_open(tt_map *m, const char *path);
void tt_map_close(tt_map *m);

/* Signals that the range will be read soon, and forces residency.
 * Two distinct things: the hint is asynchronous, the touch is synchronous --
 * see docs/11-memory.md section 2. */
void tt_map_willneed(const tt_map *m, size_t offset, size_t length);
void tt_map_touch(const tt_map *m, size_t offset, size_t length);

/* -------------------------------------------------------------- memory */

/* Aligned allocation.  `align` is a power of two.  It lives here because the
 * three OSes disagree on the name (`_aligned_malloc`, `aligned_alloc`,
 * `posix_memalign`) and because the alloc/free pair has to match on Windows. */
void *tt_alloc_aligned(size_t bytes, size_t align);
void  tt_free_aligned(void *p);

double tt_ram_total_gb(void);
double tt_ram_avail_gb(void);
double tt_rss_gb(void);            /* this process's resident set */

/* --------------------------------------------------------------- files */

/* What a persisted artefact needs and ISO C does not give (docs/12, "Integrity
 * rules"): the bytes on the platter before the name changes, and the name
 * changing in one step.  Without both, a power cut leaves a truncated file under
 * the final name and the next open reads it.
 *
 * `stdio_file` is a `FILE *`, passed as `void *` so that this header does not
 * drag `<stdio.h>` into every translation unit. */
int tt_file_commit(void *stdio_file);                   /* flush + fsync */
int tt_file_replace(const char *from, const char *to);  /* atomic, overwrites */
int tt_mkdir(const char *path);                         /* 0 if it exists now */
int tt_rmdir(const char *path);                         /* empty directories only */
int tt_file_touch(const char *path);                    /* modification time = now */

/* Calls `fn` once per regular file in `dir` (not recursive, no order promised).
 * `mtime` only has to compare: larger is more recent.  Returns -1 if the
 * directory cannot be read. */
typedef void (*tt_dir_fn)(void *arg, const char *name, uint64_t bytes, int64_t mtime);
int tt_dir_list(const char *dir, tt_dir_fn fn, void *arg);

/* ------------------------------------------------------------- console */

/* Puts the console in UTF-8, once, at start-up.  Everything this program
 * prints is UTF-8 -- the model's text, and now the citations the retriever
 * logs -- and a Windows console defaults to the OEM page, which turns
 * "Existência" into "Exist├¬ncia" in the operator's terminal.  A no-op
 * everywhere else. */
void tt_console_utf8(void);

/* ---------------------------------------------------------------- time */

double tt_now(void);               /* seconds, monotonic */

/* ----------------------------------------------------------------- cpu */

int tt_cpu_threads(void);          /* logical, E-cores included */

/* Topology.  The 14900K is hybrid and docs/09 mandates pinning to P-cores by
 * default: spreading a memory-bound matmul across the E-cores usually hurts.
 * Both return ONE logical core per physical core -- SMT siblings compete for
 * the same execution unit and do not help a loop that is already memory-bound.
 * They return how many they wrote. */
int tt_cpu_pcore_list(int *out, int max);   /* P-cores only */
int tt_cpu_core_list(int *out, int max);    /* every physical core */

/* ------------------------------------------------------------- threads */

typedef struct { void *handle; } tt_thread;

int  tt_thread_start(tt_thread *t, void (*fn)(void *), void *arg);
void tt_thread_join(tt_thread *t);
void tt_thread_pin(int cpu);       /* cpu < 0 does nothing */

void tt_cpu_relax(void);           /* spin-wait hint (`pause`) */
void tt_yield(void);               /* gives up the rest of the quantum */
void tt_sleep_ms(int ms);          /* really blocks */

/* Waits until the value at `addr` stops being `expected`, or time runs out.
 *
 * Sleeping for a fixed time is polling in disguise: waking late delays the
 * work, waking early burns CPU.  Here whoever dispatches WAKES the waiters, and
 * the timeout exists only as a net against a lost wake-up. */
void tt_futex_wait(const volatile int32_t *addr, int32_t expected, int timeout_ms);
void tt_futex_wake(const volatile int32_t *addr);

/* ------------------------------------------------------------ network */

/* The minimum for an HTTP server: listen, accept, read, write.  Winsock and BSD
 * sockets diverge enough (global initialisation, descriptor type, the name of
 * close) that the core must not see either of them. */

typedef struct { intptr_t fd; } tt_socket;

#define TT_INVALID_SOCKET ((intptr_t)-1)

int  tt_net_init(void);            /* once, before any socket */
void tt_net_shutdown(void);

/* `host` NULL or "127.0.0.1" listens on loopback only; "0.0.0.0" opens to the
 * network -- that is the caller's decision, never a default (docs/05, M3). */
int  tt_listen(tt_socket *s, const char *host, int port, int backlog);
int  tt_accept(const tt_socket *server, tt_socket *client, char *peer, size_t peer_len);

/* Client side: the agent talks to the server over HTTP.  Literal IP only. */
int  tt_connect(tt_socket *s, const char *host, int port);

/* Return bytes moved, 0 at end of connection, -1 on error. */
long tt_recv(const tt_socket *s, void *buf, size_t n);
long tt_send_all(const tt_socket *s, const void *buf, size_t n);

void tt_socket_close(tt_socket *s);
void tt_socket_nodelay(const tt_socket *s);   /* token-by-token streaming */
int  tt_socket_timeout(const tt_socket *s, int seconds);

/* ------------------------------------------------------------- atomics */

/* The minimum the pool needs, without `<stdatomic.h>`: MSVC only enables it
 * under `/experimental:c11atomics`, and depending on an experimental flag for
 * the heart of the parallelism is not worth saving thirty lines.
 *
 * Ordering: fetch_add is a full barrier on both compilers; load is acquire and
 * store is release, which on x86-64 are the ordinary read and write. */

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
