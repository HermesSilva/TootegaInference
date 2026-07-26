#include "backend.h"
#include "common.h"
#include "compat.h"

#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

static void             *g_handle = NULL;
static const tt_backend *g_backend = NULL;

const tt_backend *tt_backend_get(void) { return g_backend; }

/* Diagnostico do carregamento.
 *
 * A armadilha que custou tempo no colibri (docs/10): a mensagem
 * `coli_cuda.dll not found` significava, na verdade, que uma DEPENDENCIA dela
 * -- `cudart64_12.dll` -- nao estava no PATH. As duas falhas se parecem e
 * pedem acoes opostas: copiar o arquivo contra ajustar o PATH.
 *
 * Aqui os dois casos sao distinguidos antes de reportar. */
static void report_load_failure(const char *path)
{
#if defined(_WIN32)
    const DWORD err = GetLastError();

    FILE *f = fopen(path, "rb");
    const int exists = (f != NULL);
    if (f) fclose(f);

    if (!exists) {
        tt_fail("backend '%s' nao existe. Compile com "
                "tools\\cuda\\build.ps1, ou rode sem GPU.", path);
        return;
    }
    if (err == ERROR_MOD_NOT_FOUND) {
        tt_fail("'%s' existe mas uma DEPENDENCIA dela nao foi achada -- "
                "tipicamente `cudart64_*.dll` fora do PATH. Nao e o arquivo "
                "que falta, e o que ele precisa.", path);
        return;
    }
    if (err == ERROR_BAD_EXE_FORMAT) {
        tt_fail("'%s' existe mas e de outra arquitetura (32 contra 64 bits).",
                path);
        return;
    }
    tt_fail("nao consegui carregar '%s' (erro %lu)", path, (unsigned long)err);
#else
    const char *why = dlerror();
    tt_fail("nao consegui carregar '%s': %s", path, why ? why : "motivo desconhecido");
#endif
}

const tt_backend *tt_backend_load(const char *path, int device)
{
    if (g_backend) return g_backend;
    if (!path || !*path) return NULL;

#if defined(_WIN32)
    g_handle = (void *)LoadLibraryA(path);
#else
    g_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif

    if (!g_handle) { report_load_failure(path); return NULL; }

#if defined(_WIN32)
    tt_backend_entry entry =
        (tt_backend_entry)(void *)GetProcAddress((HMODULE)g_handle, TT_BACKEND_ENTRY);
#else
    tt_backend_entry entry = (tt_backend_entry)dlsym(g_handle, TT_BACKEND_ENTRY);
#endif

    if (!entry) {
        tt_fail("'%s' carregou mas nao exporta `%s` -- nao e um backend do "
                "tootega", path, TT_BACKEND_ENTRY);
        tt_backend_unload();
        return NULL;
    }

    const tt_backend *b = entry();
    if (!b) {
        tt_fail("'%s': `%s` devolveu NULL", path, TT_BACKEND_ENTRY);
        tt_backend_unload();
        return NULL;
    }

    /* ABI conferida antes de qualquer chamada.  Um backend compilado contra
     * outra versao desta struct nao "quase funciona": ele le campos errados. */
    if (b->abi != TT_BACKEND_ABI) {
        tt_fail("'%s' fala ABI %u, este engine fala %u -- recompile o backend",
                path, b->abi, (unsigned)TT_BACKEND_ABI);
        tt_backend_unload();
        return NULL;
    }

    if (!b->init || !b->upload || !b->matmul || !b->shutdown) {
        tt_fail("'%s' nao preencheu a interface inteira", path);
        tt_backend_unload();
        return NULL;
    }

    if (b->init(device) != 0) {
        tt_fail("'%s': init(device=%d) falhou -- placa ausente ou em uso?",
                path, device);
        tt_backend_unload();
        return NULL;
    }

    g_backend = b;
    char t[32], f[32];
    tt_log("  backend  %s -- %s de VRAM, %s livres",
           b->name, tt_bytes(t, sizeof(t), b->vram_total),
           tt_bytes(f, sizeof(f), b->vram_free));
    return b;
}

void tt_backend_unload(void)
{
    if (g_backend && g_backend->shutdown) g_backend->shutdown();
    g_backend = NULL;

    if (g_handle) {
#if defined(_WIN32)
        FreeLibrary((HMODULE)g_handle);
#else
        dlclose(g_handle);
#endif
        g_handle = NULL;
    }
}
