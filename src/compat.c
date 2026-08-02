#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(TT_WINDOWS)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <psapi.h>
#  include <intrin.h>
#else
#  define _GNU_SOURCE
#  include <pthread.h>
#  include <sched.h>
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <time.h>
#  include <unistd.h>
#endif

/* ---------------------------------------------------------------- mmap */

int tt_map_open(tt_map *m, const char *path)
{
    memset(m, 0, sizeof(*m));

#if defined(TT_WINDOWS)
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) return -1;

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz) || sz.QuadPart <= 0) {
        CloseHandle(fh);
        return -1;
    }

    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) { CloseHandle(fh); return -1; }

    void *base = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!base) { CloseHandle(mh); CloseHandle(fh); return -1; }

    m->file = fh;
    m->mapping = mh;
    m->base = base;
    m->size = (size_t)sz.QuadPart;
    return 0;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return -1; }

    void *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { close(fd); return -1; }

    m->fd = fd;
    m->base = base;
    m->size = (size_t)st.st_size;
    return 0;
#endif
}

void tt_map_close(tt_map *m)
{
    if (!m->base) return;
#if defined(TT_WINDOWS)
    UnmapViewOfFile(m->base);
    CloseHandle((HANDLE)m->mapping);
    CloseHandle((HANDLE)m->file);
#else
    munmap(m->base, m->size);
    close(m->fd);
#endif
    memset(m, 0, sizeof(*m));
}

void tt_map_willneed(const tt_map *m, size_t offset, size_t length)
{
    if (!m->base || offset >= m->size) return;
    if (length > m->size - offset) length = m->size - offset;

#if defined(TT_WINDOWS)
    WIN32_MEMORY_RANGE_ENTRY range;
    range.VirtualAddress = (char *)m->base + offset;
    range.NumberOfBytes = length;
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
#elif defined(MADV_WILLNEED)
    madvise((char *)m->base + offset, length, MADV_WILLNEED);
#else
    (void)offset; (void)length;
#endif
}

void tt_map_touch(const tt_map *m, size_t offset, size_t length)
{
    if (!m->base || offset >= m->size) return;
    if (length > m->size - offset) length = m->size - offset;
    if (!length) return;

    /* Uma leitura por pagina de 4 KB basta para forcar a falta. `volatile`
     * impede o compilador de descartar o laco inteiro por ser sem efeito. */
    const volatile unsigned char *p = (const unsigned char *)m->base + offset;
    unsigned char sink = 0;
    for (size_t i = 0; i < length; i += 4096) sink ^= p[i];
    sink ^= p[length - 1];
    (void)sink;
}

/* ------------------------------------------------------------- memoria */

void *tt_alloc_aligned(size_t bytes, size_t align)
{
    if (!bytes) bytes = align;
#if defined(TT_WINDOWS)
    return _aligned_malloc(bytes, align);
#else
    /* aligned_alloc exige tamanho multiplo do alinhamento (C11 7.22.3.1). */
    size_t rounded = (bytes + align - 1) & ~(align - 1);
    return aligned_alloc(align, rounded);
#endif
}

void tt_free_aligned(void *p)
{
#if defined(TT_WINDOWS)
    _aligned_free(p);
#else
    free(p);
#endif
}

double tt_ram_total_gb(void)
{
#if defined(TT_WINDOWS)
    MEMORYSTATUSEX st = { .dwLength = sizeof(st) };
    if (!GlobalMemoryStatusEx(&st)) return 0.0;
    return (double)st.ullTotalPhys / 1e9;
#else
    long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page <= 0) return 0.0;
    return (double)pages * (double)page / 1e9;
#endif
}

double tt_ram_avail_gb(void)
{
#if defined(TT_WINDOWS)
    MEMORYSTATUSEX st = { .dwLength = sizeof(st) };
    if (!GlobalMemoryStatusEx(&st)) return 0.0;
    return (double)st.ullAvailPhys / 1e9;
#else
    /* MemAvailable e mais honesto que MemFree: conta cache recuperavel. */
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0.0;
    char line[256];
    double kb = 0.0;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "MemAvailable: %lf", &kb) == 1) break;
    fclose(f);
    return kb / 1e6;
#endif
}

double tt_rss_gb(void)
{
#if defined(TT_WINDOWS)
    PROCESS_MEMORY_COUNTERS pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return 0.0;
    return (double)pmc.WorkingSetSize / 1e9;
#else
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return 0.0;
    long total = 0, resident = 0;
    if (fscanf(f, "%ld %ld", &total, &resident) != 2) resident = 0;
    fclose(f);
    return (double)resident * (double)sysconf(_SC_PAGESIZE) / 1e9;
#endif
}

/* ------------------------------------------------------------- console */

void tt_console_utf8(void)
{
#if defined(TT_WINDOWS)
    SetConsoleOutputCP(CP_UTF8);
#endif
}

/* --------------------------------------------------------------- tempo */

double tt_now(void)
{
#if defined(TT_WINDOWS)
    static LARGE_INTEGER freq;
    LARGE_INTEGER now;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* ----------------------------------------------------------------- cpu */

int tt_cpu_threads(void)
{
#if defined(TT_WINDOWS)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

#if defined(TT_WINDOWS)
/* Percorre os cores fisicos.  `only_fast` filtra pela EfficiencyClass mais
 * alta, que e como o Windows expoe "isto e um P-core". */
static int enum_cores(int *out, int max, int only_fast)
{
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (!len) return 0;

    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *buf =
        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)malloc(len);
    if (!buf) return 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, buf, &len)) {
        free(buf);
        return 0;
    }

    BYTE best = 0;
    if (only_fast) {
        for (DWORD off = 0; off < len; ) {
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *e =
                (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)((BYTE *)buf + off);
            if (e->Relationship == RelationProcessorCore &&
                e->Processor.EfficiencyClass > best)
                best = e->Processor.EfficiencyClass;
            off += e->Size;
        }
    }

    int n = 0;
    for (DWORD off = 0; off < len && n < max; ) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *e =
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)((BYTE *)buf + off);
        off += e->Size;
        if (e->Relationship != RelationProcessorCore) continue;
        if (only_fast && e->Processor.EfficiencyClass != best) continue;

        /* Primeiro logico do core: o irmao SMT competiria pela mesma unidade. */
        KAFFINITY mask = e->Processor.GroupMask[0].Mask;
        for (int bit = 0; bit < 64; bit++) {
            if (mask & ((KAFFINITY)1 << bit)) { out[n++] = bit; break; }
        }
    }

    free(buf);
    return n;
}
#endif

int tt_cpu_pcore_list(int *out, int max)
{
#if defined(TT_WINDOWS)
    int n = enum_cores(out, max, 1);
    if (n) return n;
#endif
    return tt_cpu_core_list(out, max);
}

int tt_cpu_core_list(int *out, int max)
{
#if defined(TT_WINDOWS)
    int n = enum_cores(out, max, 0);
    if (n) return n;
#endif
    /* Sem topologia: assume um logico por core, sem SMT.  Vale para a maioria
     * dos servidores e e conservador nas maquinas hibridas. */
    int total = tt_cpu_threads();
    if (total > max) total = max;
    for (int i = 0; i < total; i++) out[i] = i;
    return total;
}

/* ---------------------------------------------------------------- rede */

#if defined(TT_WINDOWS)
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
#  define TT_SOCK(s)   ((SOCKET)(s)->fd)
#  define TT_BAD_SOCK  INVALID_SOCKET
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  define TT_SOCK(s)   ((int)(s)->fd)
#  define TT_BAD_SOCK  (-1)
#endif

int tt_net_init(void)
{
#if defined(TT_WINDOWS)
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0 ? 0 : -1;
#else
    return 0;
#endif
}

void tt_net_shutdown(void)
{
#if defined(TT_WINDOWS)
    WSACleanup();
#endif
}

int tt_listen(tt_socket *s, const char *host, int port, int backlog)
{
    s->fd = TT_INVALID_SOCKET;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);

    /* Sem host, loopback.  O default nunca abre para a rede: o servidor da
     * acesso a 19 GB de modelo e ao disco da maquina (docs/05, M3). */
    if (!host || !*host) host = "127.0.0.1";
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) return -1;

    const intptr_t fd = (intptr_t)socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd == (intptr_t)TT_BAD_SOCK) return -1;
    s->fd = fd;

    /* SO_REUSEADDR e nao SO_EXCLUSIVEADDRUSE: reiniciar o servidor logo apos
     * derruba-lo tem de funcionar, e a porta fica em TIME_WAIT. */
    int yes = 1;
    setsockopt(TT_SOCK(s), SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof(yes));

    if (bind(TT_SOCK(s), (struct sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(TT_SOCK(s), backlog) != 0) {
        tt_socket_close(s);
        return -1;
    }
    return 0;
}

int tt_connect(tt_socket *s, const char *host, int port)
{
    s->fd = TT_INVALID_SOCKET;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);

    /* So endereco literal: quem se conecta aqui e o agente, e o servidor mora
     * em `127.0.0.1`.  Resolver nome traria DNS -- e latencia e falha -- para
     * dentro de um caminho que nunca sai da maquina. */
    if (!host || !*host) host = "127.0.0.1";
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) return -1;

    const intptr_t fd = (intptr_t)socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd == (intptr_t)TT_BAD_SOCK) return -1;
    s->fd = fd;

    if (connect(TT_SOCK(s), (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        tt_socket_close(s);
        return -1;
    }
    return 0;
}

int tt_accept(const tt_socket *server, tt_socket *client, char *peer, size_t peer_len)
{
    struct sockaddr_in addr;
#if defined(TT_WINDOWS)
    int alen = (int)sizeof(addr);
#else
    socklen_t alen = sizeof(addr);
#endif

    const intptr_t fd = (intptr_t)accept(TT_SOCK(server),
                                         (struct sockaddr *)&addr, &alen);
    if (fd == (intptr_t)TT_BAD_SOCK) { client->fd = TT_INVALID_SOCKET; return -1; }

    client->fd = fd;
    if (peer && peer_len)
        inet_ntop(AF_INET, &addr.sin_addr, peer, (socklen_t)peer_len);
    return 0;
}

long tt_recv(const tt_socket *s, void *buf, size_t n)
{
    return (long)recv(TT_SOCK(s), (char *)buf, (int)n, 0);
}

long tt_send_all(const tt_socket *s, const void *buf, size_t n)
{
    const char *p = (const char *)buf;
    size_t sent = 0;
    while (sent < n) {
        const long k = (long)send(TT_SOCK(s), p + sent, (int)(n - sent), 0);
        if (k <= 0) return -1;
        sent += (size_t)k;
    }
    return (long)sent;
}

void tt_socket_close(tt_socket *s)
{
    if (s->fd == TT_INVALID_SOCKET) return;
#if defined(TT_WINDOWS)
    closesocket(TT_SOCK(s));
#else
    close(TT_SOCK(s));
#endif
    s->fd = TT_INVALID_SOCKET;
}

void tt_socket_nodelay(const tt_socket *s)
{
    /* Sem isto, o Nagle segura tokens de poucos bytes ate juntar um pacote --
     * o streaming sairia em blocos, com pausas. */
    int yes = 1;
    setsockopt(TT_SOCK(s), IPPROTO_TCP, TCP_NODELAY, (const char *)&yes, sizeof(yes));
}

int tt_socket_timeout(const tt_socket *s, int seconds)
{
#if defined(TT_WINDOWS)
    DWORD ms = (DWORD)seconds * 1000;
    return setsockopt(TT_SOCK(s), SOL_SOCKET, SO_RCVTIMEO,
                      (const char *)&ms, sizeof(ms));
#else
    struct timeval tv = { seconds, 0 };
    return setsockopt(TT_SOCK(s), SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
}

/* ------------------------------------------------------------- threads */

typedef struct {
    void (*fn)(void *);
    void  *arg;
} thread_start;

#if defined(TT_WINDOWS)
static DWORD WINAPI thread_trampoline(LPVOID p)
{
    thread_start s = *(thread_start *)p;
    free(p);
    s.fn(s.arg);
    return 0;
}
#else
static void *thread_trampoline(void *p)
{
    thread_start s = *(thread_start *)p;
    free(p);
    s.fn(s.arg);
    return NULL;
}
#endif

int tt_thread_start(tt_thread *t, void (*fn)(void *), void *arg)
{
    thread_start *s = (thread_start *)malloc(sizeof(*s));
    if (!s) return -1;
    s->fn = fn;
    s->arg = arg;

#if defined(TT_WINDOWS)
    HANDLE h = CreateThread(NULL, 0, thread_trampoline, s, 0, NULL);
    if (!h) { free(s); return -1; }
    t->handle = h;
    return 0;
#else
    pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t));
    if (!tid) { free(s); return -1; }
    if (pthread_create(tid, NULL, thread_trampoline, s) != 0) {
        free(tid); free(s);
        return -1;
    }
    t->handle = tid;
    return 0;
#endif
}

void tt_thread_join(tt_thread *t)
{
    if (!t->handle) return;
#if defined(TT_WINDOWS)
    WaitForSingleObject((HANDLE)t->handle, INFINITE);
    CloseHandle((HANDLE)t->handle);
#else
    pthread_join(*(pthread_t *)t->handle, NULL);
    free(t->handle);
#endif
    t->handle = NULL;
}

void tt_thread_pin(int cpu)
{
    if (cpu < 0) return;
#if defined(TT_WINDOWS)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu);
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)cpu;
#endif
}

void tt_cpu_relax(void)
{
#if defined(_MSC_VER)
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

void tt_yield(void)
{
#if defined(TT_WINDOWS)
    SwitchToThread();
#else
    sched_yield();
#endif
}

#if !defined(TT_WINDOWS)
/* Sem futex portatil em POSIX generico: um par mutex/condicao global resolve.
 * E grosseiro -- acorda todo mundo -- mas so e alcancado quando o processo
 * esta ocioso, e ai nao ha o que otimizar. */
static pthread_mutex_t g_wait_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_wait_cv = PTHREAD_COND_INITIALIZER;
#endif

void tt_futex_wait(const volatile int32_t *addr, int32_t expected, int timeout_ms)
{
#if defined(TT_WINDOWS)
    /* `WaitOnAddress` re-confere o valor atomicamente antes de dormir, entao
     * nao ha janela entre "vi que nao mudou" e "adormeci". */
    int32_t cmp = expected;
    WaitOnAddress((volatile VOID *)addr, &cmp, sizeof(cmp), (DWORD)timeout_ms);
#else
    pthread_mutex_lock(&g_wait_mx);
    if (*addr == expected) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
        ts.tv_sec  += timeout_ms / 1000 + ts.tv_nsec / 1000000000L;
        ts.tv_nsec %= 1000000000L;
        pthread_cond_timedwait(&g_wait_cv, &g_wait_mx, &ts);
    }
    pthread_mutex_unlock(&g_wait_mx);
#endif
}

void tt_futex_wake(const volatile int32_t *addr)
{
#if defined(TT_WINDOWS)
    WakeByAddressAll((PVOID)addr);
#else
    (void)addr;
    pthread_mutex_lock(&g_wait_mx);
    pthread_cond_broadcast(&g_wait_cv);
    pthread_mutex_unlock(&g_wait_mx);
#endif
}

void tt_sleep_ms(int ms)
{
    /* Bloqueio de verdade.  `SwitchToThread`/`sched_yield` num laco cedem o
     * nucleo e o recebem de volta na hora quando ninguem mais o quer -- o
     * processo fica em 50% de CPU sem fazer nada. */
#if defined(TT_WINDOWS)
    Sleep((DWORD)ms);
#else
    struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
#endif
}
