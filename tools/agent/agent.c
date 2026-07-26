/* agent.c -- cliente agentico que fala o contrato de processo do Claude Code
 * CLI, para que uma UI escrita contra aquele CLI funcione sem saber que o
 * motor mudou.
 *
 * A especificacao NAO e a minha leitura do protocolo: e o parser da UI alvo
 * (Cockpit), congelado contra fixtures reais em `shared/events.ts` e
 * `src/cli/CliProcessManager.ts`.  Escrever contra referencia externa e a
 * licao de docs/17 -- tres testes verdes nascidos da mesma leitura foram cegos
 * ao mesmo erro.
 *
 * Forma:
 *
 *     UI  --spawn-->  agent.exe  --HTTP-->  tootega.exe serve
 *         <--NDJSON--            <--SSE---
 *
 * O servidor continua burro e padrao (OpenAI); a orquestracao mora aqui.  E
 * por isso que o Copilot, que so fala OpenAI, tambem funciona contra o
 * servidor -- sem passar por este arquivo.
 *
 * Fase A: conversa, streaming e estatisticas.
 * Fase B: ferramentas locais, sandbox e protocolo de permissao.
 *
 * Os `#ifdef` de SO ficam confinados ao bloco marcado abaixo.  A regra de
 * CLAUDE.md (`so compat.c tem #ifdef`) vale para `src/` -- o engine e auditavel
 * em uma sessao de leitura.  Ferramenta e outra coisa: ela abre diretorio e
 * dispara processo, e nada disso pertence ao engine.
 */
#include "common.h"
#include "compat.h"
#include "json.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dirent.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

#define AGENT_VERSION "0.1.0-m4a"

/* ------------------------------------------------------ buffer que cresce */

typedef struct {
    char  *p;
    size_t len, cap;
} sb;

static void sb_grow(sb *b, size_t need)
{
    if (b->len + need + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 4096;
    while (cap < b->len + need + 1) cap *= 2;
    char *q = (char *)realloc(b->p, cap);
    if (!q) { fprintf(stderr, "sem memoria\n"); exit(1); }
    b->p = q;
    b->cap = cap;
}

static void sb_put(sb *b, const char *s, size_t n)
{
    sb_grow(b, n);
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}

static void sb_str(sb *b, const char *s) { sb_put(b, s, strlen(s)); }

static void sb_free(sb *b) { free(b->p); b->p = NULL; b->len = b->cap = 0; }

/* Escapa para dentro de string JSON.  Bytes >= 0x20 passam crus: a saida e
 * UTF-8 e o parser do outro lado tambem. */
static void sb_esc(sb *b, const char *s, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"':  sb_str(b, "\\\""); break;
        case '\\': sb_str(b, "\\\\"); break;
        case '\n': sb_str(b, "\\n");  break;
        case '\r': sb_str(b, "\\r");  break;
        case '\t': sb_str(b, "\\t");  break;
        default:
            if (c < 0x20) {
                char u[8];
                snprintf(u, sizeof(u), "\\u%04X", c);
                sb_str(b, u);
            } else {
                sb_put(b, (const char *)&c, 1);
            }
        }
    }
}

/* ------------------------------------------------------------- estado */

typedef struct {
    char  host[64];
    int   port;
    char  key[128];
    char  cwd[512];
    char  cwd_abs[1024];    /* normalizado; a raiz do sandbox */
    char  model[64];
    char  session[40];
    char  permission_mode[32];
    int   partial;          /* --include-partial-messages */
    int   think;
    int   no_tools;         /* --no-tools: so conversa */
    int   yes;              /* --yes: nao pede aprovacao (sandbox continua) */
    int   max_steps;        /* teto de voltas do laco por pedido do usuario */
    int   idle_exit;        /* segundos sem stdin ate sair; 0 = nunca */
    sb    msgs;             /* mensagens ja serializadas, separadas por virgula */
    int   n_msgs;
    int   turns;
} agent;

/* Modo terminal: o mesmo laco, outra pele.
 *
 * Global e nao campo do `agent` porque `emit` e chamado de dezenas de lugares
 * e passar o estado por todos eles so para decidir "imprime ou nao" poluiria a
 * assinatura de tudo. */
static int g_tty;

/* `--verbose` -- que a UI sempre passa -- escreve o rastro no stderr, que e
 * onde o Cockpit ja captura.  Sem isto, "nao aconteceu nada" e indistinguivel
 * de "travou no connect", "travou esperando resposta" e "o servidor recusou". */
static int g_verbose;

static void trace(const char *fmt, ...)
{
    if (!g_verbose) return;
    va_list ap;
    va_start(ap, fmt);
    fputs("[agent] ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    fflush(stderr);
}

/* Uma linha NDJSON no stdout.  Sempre com flush: a UI le por linha e um buffer
 * segurando eventos parece travamento do agente.
 *
 * Em `--tty` o canal e uma pessoa, e ela nao le NDJSON: os eventos morrem aqui
 * e a impressao acontece nos pontos onde ha o que mostrar. */
static void emit(const sb *line)
{
    if (g_tty) return;
    fwrite(line->p, 1, line->len, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

/* --------------------------------------------------------------- HTTP */

/* Le a resposta inteira de um POST e entrega cada quadro SSE ao callback.
 * Devolve 0 se falou com o servidor, -1 se nem conectou. */
static int http_post_sse(const agent *a, const char *body,
                         void (*on_frame)(const char *json, size_t n, void *ud),
                         void *ud)
{
    tt_socket s;
    trace("conectando em %s:%d (%zu bytes de corpo)", a->host, a->port,
          strlen(body));
    if (tt_connect(&s, a->host, a->port) != 0) {
        trace("connect FALHOU -- o servidor esta no ar em %s:%d?",
              a->host, a->port);
        return -1;
    }
    tt_socket_nodelay(&s);

    /* Sem timeout, um servidor ocupado (ele atende um pedido por vez) deixa o
     * agente pendurado para sempre, e do lado de fora isso e indistinguivel de
     * "nao fez nada".  Dez minutos e generoso a 13 tok/s e ainda assim um
     * teto. */
    tt_socket_timeout(&s, 600);

    sb req = {0};
    char head[512];
    snprintf(head, sizeof(head),
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n",
        a->host, a->port, strlen(body));
    sb_str(&req, head);
    if (a->key[0]) {
        sb_str(&req, "Authorization: Bearer ");
        sb_str(&req, a->key);
        sb_str(&req, "\r\n");
    }
    sb_str(&req, "\r\n");
    sb_str(&req, body);

    const long sent = tt_send_all(&s, req.p, req.len);
    sb_free(&req);
    if (sent < 0) {
        trace("envio FALHOU");
        tt_socket_close(&s);
        return -1;
    }
    trace("pedido enviado, esperando resposta");

    sb acc = {0};
    char chunk[8192];
    size_t scanned = 0;
    int in_body = 0, frames = 0;

    for (;;) {
        const long k = tt_recv(&s, chunk, sizeof(chunk));
        if (k <= 0) {
            if (!in_body) trace("conexao fechou sem resposta (k=%ld)", k);
            break;
        }
        sb_put(&acc, chunk, (size_t)k);

        if (!in_body) {
            const char *sep = strstr(acc.p, "\r\n\r\n");
            if (!sep) continue;
            scanned = (size_t)(sep - acc.p) + 4;
            in_body = 1;
        }

        /* Quadros SSE terminam em linha em branco. */
        for (;;) {
            char *end = strstr(acc.p + scanned, "\n\n");
            if (!end) break;
            char *line = acc.p + scanned;
            const size_t n = (size_t)(end - line);
            scanned += n + 2;

            if (n > 6 && !strncmp(line, "data: ", 6)) {
                if (!frames++) trace("primeiro quadro recebido");
                on_frame(line + 6, n - 6, ud);
            }
        }
    }
    trace("resposta terminada, %d quadros", frames);

    /* Resposta nao-SSE (erro do servidor, por exemplo) chega sem quadro. */
    if (in_body && scanned < acc.len && acc.len - scanned > 2)
        on_frame(acc.p + scanned, acc.len - scanned, ud);

    sb_free(&acc);
    tt_socket_close(&s);
    return 0;
}

/* Pergunta ao servidor qual e a janela de contexto.
 *
 * A UI mostra um medidor de contexto, e sem este numero ela assume o default do
 * Claude -- 200k sobre uma janela que aqui e de 16k, e a barra nunca sai de 1%.
 * Devolve 0 se o servidor nao responder: melhor a UI usar o default dela do que
 * o agente nao subir por causa de um medidor. */
static long long ask_context_window(const agent *a)
{
    tt_socket s;
    if (tt_connect(&s, a->host, a->port) != 0) return 0;

    char req[256];
    const int n = snprintf(req, sizeof(req),
        "GET /stats HTTP/1.1\r\nHost: %s:%d\r\nConnection: close\r\n\r\n",
        a->host, a->port);
    if (tt_send_all(&s, req, (size_t)n) < 0) { tt_socket_close(&s); return 0; }

    sb acc = {0};
    char chunk[2048];
    for (;;) {
        const long k = tt_recv(&s, chunk, sizeof(chunk));
        if (k <= 0) break;
        sb_put(&acc, chunk, (size_t)k);
        if (acc.len > (64u << 10)) break;
    }
    tt_socket_close(&s);

    long long ctx = 0;
    const char *body = acc.p ? strstr(acc.p, "\r\n\r\n") : NULL;
    if (body) {
        const char *at = strstr(body, "\"ctx\":");
        if (at) ctx = atoll(at + 6);
    }
    sb_free(&acc);
    return ctx;
}

/* ------------------------------------------------ eventos do stream-json */

/* Um turno em andamento.  O modelo emite o raciocinio primeiro e fecha com
 * `</think>`; do lado da UI isso sao DOIS blocos de conteudo -- `thinking` e
 * `text` --, e e aqui que a fronteira e detectada. */
typedef struct {
    agent *a;
    sb     raw;             /* tudo que veio, para o historico */
    sb     answer;          /* so a resposta, depois do </think> */
    int    in_think;        /* ainda antes do </think> */
    int    block_open;      /* indice do bloco aberto, -1 se nenhum */
    int    started;
    int    prompt_tokens, completion_tokens, cached;
    double tps;
    char   finish[32];

    /* Chamadas que o servidor devolveu neste turno, cruas como vieram. */
    sb     calls_json;
    char   msg_id[64];
} turn;

/* Id da mensagem deste passo.
 *
 * A UI indexa a bolha por ele: o `message_start` do streaming e o `assistant`
 * completo TEM de trazer o mesmo id -- se diferirem, o texto e emitido duas
 * vezes -- e passos diferentes TEM de trazer ids diferentes, senao o segundo
 * escreve por cima do primeiro e a tela parece parada. */
static const char *turn_msg_id(turn *t)
{
    static int seq;
    if (!t->msg_id[0])
        snprintf(t->msg_id, sizeof(t->msg_id), "msg_%s_%d", t->a->session, ++seq);
    return t->msg_id;
}

static void ev_stream_start(turn *t)
{
    if (t->started) return;
    t->started = 1;

    char h[200];
    sb e = {0};
    sb_str(&e, "{\"type\":\"stream_event\",\"session_id\":\"");
    sb_str(&e, t->a->session);
    snprintf(h, sizeof(h), "\",\"event\":{\"type\":\"message_start\",\"message\":"
                           "{\"id\":\"%s", turn_msg_id(t));
    sb_str(&e, h);
    sb_str(&e, "\",\"role\":\"assistant\",\"model\":\"");
    sb_str(&e, t->a->model);
    sb_str(&e, "\"}}}");
    emit(&e);
    sb_free(&e);
}

static void ev_block_start(turn *t, int index, const char *kind)
{
    sb e = {0};
    char h[256];
    snprintf(h, sizeof(h),
        "{\"type\":\"stream_event\",\"session_id\":\"%s\",\"event\":"
        "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":"
        "{\"type\":\"%s\",\"%s\":\"\"}}}",
        t->a->session, index, kind, kind);
    sb_str(&e, h);
    emit(&e);
    sb_free(&e);
    t->block_open = index;
}

static void ev_block_stop(turn *t)
{
    if (t->block_open < 0) return;
    sb e = {0};
    char h[192];
    snprintf(h, sizeof(h),
        "{\"type\":\"stream_event\",\"session_id\":\"%s\",\"event\":"
        "{\"type\":\"content_block_stop\",\"index\":%d}}",
        t->a->session, t->block_open);
    sb_str(&e, h);
    emit(&e);
    sb_free(&e);
    t->block_open = -1;
}

static void ev_delta(turn *t, int index, const char *field,
                     const char *text, size_t n)
{
    sb e = {0};
    char h[192];
    snprintf(h, sizeof(h),
        "{\"type\":\"stream_event\",\"session_id\":\"%s\",\"event\":"
        "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":"
        "{\"type\":\"%s_delta\",\"%s\":\"",
        t->a->session, index, field, field);
    sb_str(&e, h);
    sb_esc(&e, text, n);
    sb_str(&e, "\"}}}");
    emit(&e);
    sb_free(&e);
}

/* Texto recem-chegado: decide bloco e emite o delta. */
static void on_text(turn *t, const char *s, size_t n)
{
    sb_put(&t->raw, s, n);

    /* No terminal a resposta sai token a token, e o rascunho nao sai: quem
     * esta lendo quer a resposta, e o raciocinio deste modelo e varias vezes
     * mais longo que ela. */
    if (g_tty) {
        if (t->in_think) {
            const char *close = strstr(t->raw.p, "</think>");
            if (!close) return;
            t->in_think = 0;
            const char *rest = close + 8;
            while (*rest == '\n' || *rest == ' ') rest++;
            fputs(rest, stdout);
            sb_str(&t->answer, rest);
        } else {
            fwrite(s, 1, n, stdout);
            sb_put(&t->answer, s, n);
        }
        fflush(stdout);
        return;
    }

    if (t->in_think) {
        /* O fechamento pode chegar colado com o inicio da resposta. */
        const char *close = strstr(t->raw.p, "</think>");
        if (close) {
            const size_t before = (size_t)(close - t->raw.p);
            const size_t already = t->raw.len - n;
            if (before > already) {
                if (t->block_open < 0) ev_block_start(t, 0, "thinking");
                ev_delta(t, 0, "thinking", t->raw.p + already, before - already);
            }
            ev_block_stop(t);
            t->in_think = 0;

            const char *rest = close + 8;
            while (*rest == '\n' || *rest == ' ') rest++;
            const size_t rn = strlen(rest);
            if (rn) {
                ev_block_start(t, 1, "text");
                ev_delta(t, 1, "text", rest, rn);
                sb_put(&t->answer, rest, rn);
            }
            return;
        }
        if (t->block_open < 0) ev_block_start(t, 0, "thinking");
        ev_delta(t, 0, "thinking", s, n);
        return;
    }

    if (t->block_open < 0) ev_block_start(t, 1, "text");
    ev_delta(t, 1, "text", s, n);
    sb_put(&t->answer, s, n);
}

/* Um quadro `data: {...}` do servidor. */
static void on_frame(const char *json, size_t n, void *ud)
{
    turn *t = (turn *)ud;
    if (n >= 6 && !strncmp(json, "[DONE]", 6)) return;

    tt_arena a;
    if (tt_arena_init(&a, tt_json_bound(json, n) + 1024, "sse") != 0) return;
    const tt_json *root = tt_json_parse(json, n, &a);
    const tt_json *ch = tt_json_at(tt_json_get(root, "choices"), 0);

    size_t cn = 0;
    const char *content = tt_json_str(tt_json_get(tt_json_get(ch, "delta"),
                                                  "content"), &cn);
    if (content && cn) {
        ev_stream_start(t);
        on_text(t, content, cn);
    }

    /* `tool_calls` chega inteiro no quadro final -- ver docs/20 sobre por que
     * nao ha `input_json_delta` ainda. */
    const tt_json *tc = tt_json_get(tt_json_get(ch, "delta"), "tool_calls");
    if (tc && tc->type == TT_JSON_ARRAY && tc->count) {
        /* Copia terminada: o quadro vem de um buffer maior e um `strstr` sem
         * limite acharia texto do quadro seguinte. */
        char *tmp = (char *)malloc(n + 1);
        if (tmp) {
            memcpy(tmp, json, n);
            tmp[n] = '\0';
            const char *at = strstr(tmp, "\"tool_calls\":");
            const char *open = at ? strchr(at, '[') : NULL;
            if (open) {
                int depth = 0;
                const char *q = open;
                for (; *q; q++) {
                    if (*q == '[') depth++;
                    else if (*q == ']' && --depth == 0) { q++; break; }
                }
                sb_put(&t->calls_json, open, (size_t)(q - open));
            }
            free(tmp);
        }
    }

    const tt_json *fin = tt_json_get(ch, "finish_reason");
    size_t fl = 0;
    const char *fs = tt_json_str(fin, &fl);
    if (fs && fl < sizeof(t->finish)) {
        memcpy(t->finish, fs, fl);
        t->finish[fl] = '\0';
    }

    const tt_json *u = tt_json_get(root, "usage");
    if (u) {
        t->prompt_tokens     = (int)tt_json_int(tt_json_get(u, "prompt_tokens"), 0);
        t->completion_tokens = (int)tt_json_int(tt_json_get(u, "completion_tokens"), 0);
        t->cached            = (int)tt_json_int(tt_json_get(u, "prefill_cached"), 0);
        t->tps               = tt_json_real(tt_json_get(u, "tokens_per_second"), 0);
    }

    tt_arena_free(&a);
}

/* ==================================================== FERRAMENTAS ==== */

/* Nomes IGUAIS aos do Claude Code de proposito: a UI renderiza diff, saida de
 * shell e cartao de busca **por nome de ferramenta**.  Um nome nosso cairia no
 * renderizador generico e a UI pareceria quebrada sem estar. */
static const char TOOLS_SCHEMA[] =
"["
"{\"type\":\"function\",\"function\":{\"name\":\"Read\","
"\"description\":\"Le um arquivo do disco. Devolve as linhas numeradas.\","
"\"parameters\":{\"type\":\"object\",\"properties\":{"
"\"file_path\":{\"type\":\"string\",\"description\":\"Caminho do arquivo\"},"
"\"offset\":{\"type\":\"integer\",\"description\":\"Primeira linha (1 = inicio)\"},"
"\"limit\":{\"type\":\"integer\",\"description\":\"Quantas linhas\"}},"
"\"required\":[\"file_path\"]}}},"

"{\"type\":\"function\",\"function\":{\"name\":\"Write\","
"\"description\":\"Escreve um arquivo, criando ou sobrescrevendo.\","
"\"parameters\":{\"type\":\"object\",\"properties\":{"
"\"file_path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},"
"\"required\":[\"file_path\",\"content\"]}}},"

"{\"type\":\"function\",\"function\":{\"name\":\"Edit\","
"\"description\":\"Troca um trecho exato de um arquivo por outro.\","
"\"parameters\":{\"type\":\"object\",\"properties\":{"
"\"file_path\":{\"type\":\"string\"},"
"\"old_string\":{\"type\":\"string\",\"description\":\"Trecho a trocar, exato\"},"
"\"new_string\":{\"type\":\"string\"},"
"\"replace_all\":{\"type\":\"boolean\"}},"
"\"required\":[\"file_path\",\"old_string\",\"new_string\"]}}},"

"{\"type\":\"function\",\"function\":{\"name\":\"Glob\","
"\"description\":\"Lista arquivos por padrao, ex: src/*.c ou **/*.md\","
"\"parameters\":{\"type\":\"object\",\"properties\":{"
"\"pattern\":{\"type\":\"string\"},\"path\":{\"type\":\"string\"}},"
"\"required\":[\"pattern\"]}}},"

"{\"type\":\"function\",\"function\":{\"name\":\"Grep\","
"\"description\":\"Busca um texto literal nos arquivos. Nao e regex.\","
"\"parameters\":{\"type\":\"object\",\"properties\":{"
"\"pattern\":{\"type\":\"string\"},\"path\":{\"type\":\"string\"},"
"\"glob\":{\"type\":\"string\",\"description\":\"Filtro de nome, ex: *.c\"},"
"\"-i\":{\"type\":\"boolean\",\"description\":\"Ignorar maiusculas\"}},"
"\"required\":[\"pattern\"]}}},"

"{\"type\":\"function\",\"function\":{\"name\":\"Bash\","
"\"description\":\"Executa um comando no shell da maquina e devolve a saida.\","
"\"parameters\":{\"type\":\"object\",\"properties\":{"
"\"command\":{\"type\":\"string\"},"
"\"timeout\":{\"type\":\"integer\",\"description\":\"Limite em milissegundos\"}},"
"\"required\":[\"command\"]}}}"
"]";

/* Ferramenta que MUDA estado pede aprovacao; leitura corre solta.  Pedir
 * confirmacao para tudo, a 13 tok/s, transforma uma tarefa em meia hora de
 * Enter -- e treina o usuario a aprovar sem ler, que e o pior resultado. */
static int tool_mutates(const char *name)
{
    return !strcmp(name, "Write") || !strcmp(name, "Edit") ||
           !strcmp(name, "Bash");
}

/* Teto de saida por ferramenta.  Um `Read` sem limite ou um build inteiro
 * despejam 50k tokens e o contexto acaba no segundo turno. */
#define TOOL_OUT_MAX 12000

static void sb_cap(sb *b, const char *what)
{
    if (b->len <= TOOL_OUT_MAX) return;
    b->len = TOOL_OUT_MAX;
    b->p[b->len] = '\0';
    char note[160];
    snprintf(note, sizeof(note),
             "\n\n[cortado em %d bytes -- %s tem mais. Refine a busca ou "
             "use offset/limit.]", TOOL_OUT_MAX, what);
    sb_str(b, note);
}

/* ----------------------------------------------------------- caminhos */

/* Normaliza para barras e resolve `.` e `..` sem tocar no disco. */
static void path_norm(const char *in, char *out, size_t cap)
{
    char tmp[1024];
    size_t w = 0;
    for (size_t i = 0; in[i] && w + 1 < sizeof(tmp); i++)
        tmp[w++] = (in[i] == '\\') ? '/' : in[i];
    tmp[w] = '\0';

    char *parts[128];
    int np = 0;
    char *save = tmp;
    /* Prefixo absoluto (`/` ou `C:`) preservado; o resto vira pilha. */
    size_t start = 0;
    char prefix[8] = "";
    if (tmp[0] && tmp[1] == ':') { prefix[0] = tmp[0]; prefix[1] = ':'; prefix[2] = '\0'; start = 2; }
    const int absolute = (tmp[start] == '/');

    for (char *p = save + start; *p; ) {
        while (*p == '/') p++;
        if (!*p) break;
        char *seg = p;
        while (*p && *p != '/') p++;
        if (*p) *p++ = '\0';

        if (!strcmp(seg, ".")) continue;
        if (!strcmp(seg, "..")) { if (np) np--; continue; }
        if (np < 128) parts[np++] = seg;
    }

    size_t o = 0;
    o += (size_t)snprintf(out + o, cap - o, "%s%s", prefix, absolute ? "/" : "");
    for (int i = 0; i < np && o + 1 < cap; i++)
        o += (size_t)snprintf(out + o, cap - o, "%s%s", parts[i],
                              i + 1 < np ? "/" : "");
    out[o < cap ? o : cap - 1] = '\0';
}

/* Resolve um caminho da ferramenta contra o `--cwd` e RECUSA o que escapar.
 *
 * O sandbox nao e cortesia: e a unica coisa entre um modelo de 31 B e o resto
 * do disco.  Vale mesmo com `--yes`. */
static int path_resolve(const agent *a, const char *rel, char *out, size_t cap)
{
    char joined[1024];
    const int is_abs = (rel[0] == '/' || rel[0] == '\\' ||
                        (rel[0] && rel[1] == ':'));
    if (is_abs) snprintf(joined, sizeof(joined), "%s", rel);
    else        snprintf(joined, sizeof(joined), "%s/%s", a->cwd_abs, rel);

    path_norm(joined, out, cap);

    const size_t root = strlen(a->cwd_abs);
    if (strncmp(out, a->cwd_abs, root) != 0 ||
        (out[root] && out[root] != '/')) return -1;
    return 0;
}

/* ------------------------------------------------------- caminhada */

typedef void (*visit_fn)(const char *rel, void *ud);

/* Pastas que nunca interessam e custam caro: `models/` tem 60 GB. */
static int skip_dir(const char *name)
{
    static const char *no[] = { ".git", "node_modules", "obj", "bin", "dist",
                                "build", ".vs", ".vscode", "models", NULL };
    for (int i = 0; no[i]; i++) if (!strcmp(name, no[i])) return 1;
    return name[0] == '.' && name[1] != '\0' && strcmp(name, ".");
}

static void walk(const char *root, const char *rel, visit_fn visit, void *ud,
                 int depth, int *budget)
{
    if (depth > 12 || *budget <= 0) return;

    char dir[1024];
    snprintf(dir, sizeof(dir), "%s%s%s", root, rel[0] ? "/" : "", rel);

#if defined(_WIN32)
    char pat[1100];
    snprintf(pat, sizeof(pat), "%s/*", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        const char *name = fd.cFileName;
        if (!strcmp(name, ".") || !strcmp(name, "..")) continue;
        char sub[1024];
        snprintf(sub, sizeof(sub), "%s%s%s", rel, rel[0] ? "/" : "", name);
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            if (!skip_dir(name)) walk(root, sub, visit, ud, depth + 1, budget);
        } else {
            visit(sub, ud);
            (*budget)--;
        }
    } while (*budget > 0 && FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *e;
    while (*budget > 0 && (e = readdir(d))) {
        const char *name = e->d_name;
        if (!strcmp(name, ".") || !strcmp(name, "..")) continue;
        char sub[1024], full[2048];
        snprintf(sub, sizeof(sub), "%s%s%s", rel, rel[0] ? "/" : "", name);
        snprintf(full, sizeof(full), "%s/%s", dir, name);
        struct stat st;
        if (stat(full, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            if (!skip_dir(name)) walk(root, sub, visit, ud, depth + 1, budget);
        } else {
            visit(sub, ud);
            (*budget)--;
        }
    }
    closedir(d);
#endif
}

/* Casamento de padrao: `*` nao atravessa `/`, `**` atravessa. */
static int glob_match(const char *pat, const char *s)
{
    while (*pat) {
        if (pat[0] == '*' && pat[1] == '*') {
            pat += 2;
            if (*pat == '/') pat++;
            if (!*pat) return 1;
            for (const char *q = s; ; q++) {
                if (glob_match(pat, q)) return 1;
                if (!*q) return 0;
            }
        }
        if (*pat == '*') {
            pat++;
            for (const char *q = s; ; q++) {
                if (glob_match(pat, q)) return 1;
                if (!*q || *q == '/') return 0;
            }
        }
        if (*pat == '?') {
            if (!*s || *s == '/') return 0;
            pat++; s++;
            continue;
        }
        if (*pat != *s) return 0;
        pat++; s++;
    }
    return *s == '\0';
}

/* --------------------------------------------------------- execucao */

typedef struct {
    const agent *a;
    const char  *pattern;
    const char  *glob;
    int          ci;
    sb          *out;
    int          hits;
} scan;

static int ci_find(const char *hay, const char *needle)
{
    const size_t n = strlen(needle);
    if (!n) return 1;
    for (const char *p = hay; *p; p++) {
        size_t i = 0;
        while (i < n && p[i] && (p[i] | 32) == (needle[i] | 32)) i++;
        if (i == n) return 1;
    }
    return 0;
}

static char *read_all(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0 || n > (64L << 20)) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    const size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    *len = got;
    return buf;
}

static void visit_glob(const char *rel, void *ud)
{
    scan *sc = (scan *)ud;
    if (!glob_match(sc->pattern, rel)) return;
    sb_str(sc->out, rel);
    sb_str(sc->out, "\n");
    sc->hits++;
}

static void visit_grep(const char *rel, void *ud)
{
    scan *sc = (scan *)ud;
    if (sc->glob && *sc->glob) {
        const char *base = strrchr(rel, '/');
        if (!glob_match(sc->glob, rel) &&
            !glob_match(sc->glob, base ? base + 1 : rel)) return;
    }

    char full[1200];
    snprintf(full, sizeof(full), "%s/%s", sc->a->cwd_abs, rel);
    size_t n = 0;
    char *txt = read_all(full, &n);
    if (!txt) return;

    int line = 1;
    char *p = txt;
    while (p && *p && sc->out->len < TOOL_OUT_MAX) {
        char *nl = strchr(p, '\n');
        if (nl) *nl = '\0';
        const int hit = sc->ci ? ci_find(p, sc->pattern)
                               : (strstr(p, sc->pattern) != NULL);
        if (hit) {
            char h[64];
            snprintf(h, sizeof(h), ":%d:", line);
            sb_str(sc->out, rel);
            sb_str(sc->out, h);
            sb_str(sc->out, p);
            sb_str(sc->out, "\n");
            sc->hits++;
        }
        if (!nl) break;
        p = nl + 1;
        line++;
    }
    free(txt);
}

/* Roda um comando e captura a saida.  Com limite de tempo: um comando pendurado
 * congelaria a sessao inteira, e quem esta do outro lado e uma UI. */
static int run_command(const char *cmd, int timeout_ms, sb *out)
{
#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa = { sizeof(sa), NULL, TRUE };
    HANDLE rd = NULL, wr = NULL;
    if (!CreatePipe(&rd, &wr, &sa, 1 << 20)) return -1;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

    char line[4096];
    snprintf(line, sizeof(line), "cmd.exe /d /s /c \"%s\"", cmd);

    STARTUPINFOA si;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError = wr;
    si.hStdInput = NULL;

    PROCESS_INFORMATION pi;
    memset(&pi, 0, sizeof(pi));
    if (!CreateProcessA(NULL, line, NULL, NULL, TRUE, CREATE_NO_WINDOW,
                        NULL, NULL, &si, &pi)) {
        CloseHandle(rd); CloseHandle(wr);
        return -1;
    }
    CloseHandle(wr);

    const double t0 = tt_now();
    char chunk[4096];
    DWORD got = 0;
    for (;;) {
        DWORD avail = 0;
        if (!PeekNamedPipe(rd, NULL, 0, NULL, &avail, NULL)) break;
        if (avail) {
            if (!ReadFile(rd, chunk, sizeof(chunk), &got, NULL) || !got) break;
            sb_put(out, chunk, got);
            if (out->len > TOOL_OUT_MAX * 4) break;
            continue;
        }
        if (WaitForSingleObject(pi.hProcess, 20) == WAIT_OBJECT_0) {
            /* Drena o que sobrou depois que o processo terminou. */
            while (PeekNamedPipe(rd, NULL, 0, NULL, &avail, NULL) && avail &&
                   ReadFile(rd, chunk, sizeof(chunk), &got, NULL) && got)
                sb_put(out, chunk, got);
            break;
        }
        if ((tt_now() - t0) * 1000.0 > timeout_ms) {
            TerminateProcess(pi.hProcess, 1);
            sb_str(out, "\n[interrompido: passou do limite de tempo]");
            break;
        }
    }

    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(rd);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return (int)code;
#else
    /* POSIX: `popen` nao permite matar o filho, entao o limite de tempo aqui e
     * do shell, nao nosso.  Assumido conscientemente -- a plataforma de uso e
     * Windows e o portao de aprovacao ja segura o pior caso. */
    char line[4200];
    snprintf(line, sizeof(line), "timeout %d %s 2>&1", timeout_ms / 1000, cmd);
    FILE *f = popen(line, "r");
    if (!f) return -1;
    char chunk[4096];
    size_t got;
    while ((got = fread(chunk, 1, sizeof(chunk), f)) > 0) {
        sb_put(out, chunk, got);
        if (out->len > TOOL_OUT_MAX * 4) break;
    }
    return pclose(f);
#endif
}

/* Executa uma ferramenta.  `args` e o objeto ja parseado.  Devolve 0 em
 * sucesso; a saida (ou a mensagem de erro) vai em `out`. */
static int tool_run(const agent *a, const char *name, const tt_json *args,
                    sb *out)
{
    char path[1024], abs[1024];

    if (!strcmp(name, "Read")) {
        if (tt_json_strcpy(tt_json_get(args, "file_path"), path, sizeof(path)) != 0)
            { sb_str(out, "faltou file_path"); return -1; }
        if (path_resolve(a, path, abs, sizeof(abs)) != 0)
            { sb_str(out, "caminho fora da pasta de trabalho"); return -1; }

        size_t n = 0;
        char *txt = read_all(abs, &n);
        if (!txt) { sb_str(out, "nao consegui ler o arquivo"); return -1; }

        const int64_t off = tt_json_int(tt_json_get(args, "offset"), 1);
        const int64_t lim = tt_json_int(tt_json_get(args, "limit"), 2000);
        int line = 1;
        char *p = txt;
        while (*p && line < off + lim) {
            char *nl = strchr(p, '\n');
            if (nl) *nl = '\0';
            if (line >= off) {
                char h[32];
                snprintf(h, sizeof(h), "%6d\t", line);
                sb_str(out, h);
                sb_str(out, p);
                sb_str(out, "\n");
            }
            if (!nl) break;
            p = nl + 1;
            line++;
        }
        free(txt);
        if (!out->len) sb_str(out, "(vazio)");
        sb_cap(out, "o arquivo");
        return 0;
    }

    if (!strcmp(name, "Write")) {
        size_t cn = 0;
        const char *content = tt_json_str(tt_json_get(args, "content"), &cn);
        if (tt_json_strcpy(tt_json_get(args, "file_path"), path, sizeof(path)) != 0
            || !content)
            { sb_str(out, "faltou file_path ou content"); return -1; }
        if (path_resolve(a, path, abs, sizeof(abs)) != 0)
            { sb_str(out, "caminho fora da pasta de trabalho"); return -1; }

        FILE *f = fopen(abs, "wb");
        if (!f) { sb_str(out, "nao consegui abrir para escrita"); return -1; }
        fwrite(content, 1, cn, f);
        fclose(f);

        char h[256];
        snprintf(h, sizeof(h), "gravado: %s (%zu bytes)", path, cn);
        sb_str(out, h);
        return 0;
    }

    if (!strcmp(name, "Edit")) {
        size_t on = 0, nn = 0;
        const char *olds = tt_json_str(tt_json_get(args, "old_string"), &on);
        const char *news = tt_json_str(tt_json_get(args, "new_string"), &nn);
        if (tt_json_strcpy(tt_json_get(args, "file_path"), path, sizeof(path)) != 0
            || !olds)
            { sb_str(out, "faltou file_path ou old_string"); return -1; }
        if (path_resolve(a, path, abs, sizeof(abs)) != 0)
            { sb_str(out, "caminho fora da pasta de trabalho"); return -1; }

        size_t n = 0;
        char *txt = read_all(abs, &n);
        if (!txt) { sb_str(out, "nao consegui ler o arquivo"); return -1; }

        /* Trecho ambiguo e recusado: aplicar na primeira ocorrencia e o modo
         * silencioso de editar a linha errada. */
        const int all = tt_json_bool(tt_json_get(args, "replace_all"), 0);
        int count = 0;
        for (const char *p = strstr(txt, olds); p; p = strstr(p + on, olds)) count++;
        if (!count) { free(txt); sb_str(out, "trecho nao encontrado"); return -1; }
        if (count > 1 && !all) {
            free(txt);
            char h[160];
            snprintf(h, sizeof(h), "trecho aparece %d vezes -- torne-o unico "
                     "ou use replace_all", count);
            sb_str(out, h);
            return -1;
        }

        sb nw = {0};
        const char *p = txt;
        for (;;) {
            const char *hit = strstr(p, olds);
            if (!hit) { sb_str(&nw, p); break; }
            sb_put(&nw, p, (size_t)(hit - p));
            sb_put(&nw, news ? news : "", nn);
            p = hit + on;
            if (!all) { sb_str(&nw, p); break; }
        }
        free(txt);

        FILE *f = fopen(abs, "wb");
        if (!f) { sb_free(&nw); sb_str(out, "nao consegui gravar"); return -1; }
        fwrite(nw.p, 1, nw.len, f);
        fclose(f);
        sb_free(&nw);

        char h[256];
        snprintf(h, sizeof(h), "editado: %s (%d ocorrencia%s)", path, count,
                 count > 1 ? "s" : "");
        sb_str(out, h);
        return 0;
    }

    if (!strcmp(name, "Glob") || !strcmp(name, "Grep")) {
        char pattern[512] = "", where[512] = "", filter[128] = "";
        if (tt_json_strcpy(tt_json_get(args, "pattern"), pattern,
                           sizeof(pattern)) != 0)
            { sb_str(out, "faltou pattern"); return -1; }
        tt_json_strcpy(tt_json_get(args, "path"), where, sizeof(where));
        tt_json_strcpy(tt_json_get(args, "glob"), filter, sizeof(filter));

        char root[1024];
        if (path_resolve(a, where[0] ? where : ".", root, sizeof(root)) != 0)
            { sb_str(out, "caminho fora da pasta de trabalho"); return -1; }

        scan sc = {0};
        sc.a = a;
        sc.pattern = pattern;
        sc.glob = filter;
        sc.ci = tt_json_bool(tt_json_get(args, "-i"), 0);
        sc.out = out;

        int budget = 20000;
        walk(root, "", !strcmp(name, "Glob") ? visit_glob : visit_grep,
             &sc, 0, &budget);

        if (!sc.hits) sb_str(out, "(nada encontrado)");
        sb_cap(out, "a busca");
        return 0;
    }

    if (!strcmp(name, "Bash")) {
        char *cmd = NULL;
        size_t cn = 0;
        const char *cs = tt_json_str(tt_json_get(args, "command"), &cn);
        if (!cs) { sb_str(out, "faltou command"); return -1; }
        cmd = (char *)malloc(cn + 1);
        if (!cmd) return -1;
        memcpy(cmd, cs, cn);
        cmd[cn] = '\0';

        const int64_t ms = tt_json_int(tt_json_get(args, "timeout"), 120000);
        const int code = run_command(cmd, (int)(ms > 0 ? ms : 120000), out);
        free(cmd);

        if (!out->len) sb_str(out, "(sem saida)");
        sb_cap(out, "a saida do comando");
        if (code != 0) {
            char h[64];
            snprintf(h, sizeof(h), "\n[codigo de saida %d]", code);
            sb_str(out, h);
        }
        return 0;
    }

    sb_str(out, "ferramenta desconhecida");
    return -1;
}

/* ---------------------------------------------------------- um turno */

static void add_message(agent *a, const char *role, const char *text, size_t n)
{
    if (a->n_msgs++) sb_str(&a->msgs, ",");
    sb_str(&a->msgs, "{\"role\":\"");
    sb_str(&a->msgs, role);
    sb_str(&a->msgs, "\",\"content\":\"");
    sb_esc(&a->msgs, text, n);
    sb_str(&a->msgs, "\"}");
}

/* Prompt de sistema.
 *
 * Entra UMA vez, no inicio do historico, antes de qualquer turno: e a parte que
 * nao muda, e a regra de ordenacao de docs/03 (estavel primeiro) e o que faz o
 * cache de prefixo valer numa sessao de agente.
 *
 * As regras do fim nao sao genericas -- saem do que foi MEDIDO nesta maquina:
 *
 *   - ler o arquivo todo: um Read de 3.425 tokens custou 246 s, quase tudo em
 *     prefill.  Ferramenta que devolve muito e o gargalo real deste agente;
 *   - contar e medir com Bash: nao porque o modelo errou -- ele acertou as 126
 *     linhas --, e sim porque contar pelo Read custa o arquivo inteiro no
 *     contexto para produzir um numero;
 *   - o shell e o `cmd.exe`, e o modelo tentou `wc -l` na primeira chance.
 */
static void add_system(agent *a)
{
    sb s = {0};
    sb_str(&s,
        "Voce e o Tootega, um agente de programacao que roda na maquina do "
        "usuario e responde em portugues do Brasil, salvo se o usuario "
        "escrever em outro idioma.\n\n"
        "Pasta de trabalho: ");
    sb_str(&s, a->cwd_abs);
    sb_str(&s,
        "\nTodo caminho e relativo a ela, e nenhuma ferramenta alcanca nada "
        "fora dela.\n\n"
        "Ferramentas: Read, Write, Edit, Glob, Grep, Bash.\n"
        "Write, Edit e Bash pedem aprovacao do usuario antes de executar.\n"
        "Grep e busca por texto LITERAL, nao expressao regular.\n");
#if defined(_WIN32)
    sb_str(&s,
        "Bash roda no cmd.exe do Windows: nao existem wc, grep, sed, ls nem "
        "cat. Use `powershell -Command \"...\"` quando precisar deles -- por "
        "exemplo `(Get-Content arquivo).Count` para contar linhas.\n");
#endif
    sb_str(&s, "\n"
        "Como trabalhar:\n"
        "- Use ferramenta quando a resposta depender do disco ou da maquina. "
        "Quando voce ja sabe a resposta, responda direto.\n"
        "- Para contar linhas, medir tamanho ou somar, use Bash: ler o arquivo "
        "inteiro so para produzir um numero enche o contexto a toa.\n"
        "- Prefira Grep e Glob a ler arquivos inteiros: cada arquivo lido "
        "entra no contexto e custa caro.\n"
        "- Use Read com offset e limit quando so precisar de um trecho.\n"
        "- Antes de Edit, leia o trecho: o texto de old_string precisa bater "
        "byte a byte e ser unico no arquivo.\n"
        "- Termine dizendo o que foi feito, em uma ou duas frases. Nao repita "
        "a saida das ferramentas.");

    add_message(a, "system", s.p, s.len);
    sb_free(&s);
}

/* ------------------------------------------------------ transcricao */

/* Onde a conversa desta sessao mora entre um processo e o proximo.
 *
 * O agente sai sozinho depois de ocioso, e sem isto a conversa morreria com
 * ele: a proxima mensagem da mesma aba chegaria a um processo novo, de memoria
 * vazia, e o modelo responderia como se nunca tivessem conversado. */
static int transcript_path(const agent *a, char *dst, size_t cap)
{
    const char *base =
#if defined(_WIN32)
        getenv("LOCALAPPDATA");
#else
        getenv("HOME");
#endif
    if (!base || !*base || !a->session[0]) return -1;

#if defined(_WIN32)
    snprintf(dst, cap, "%s\\tootega\\sessions", base);
    CreateDirectoryA(dst, NULL);          /* o pai pode ja existir */
    snprintf(dst, cap, "%s\\tootega", base);
    CreateDirectoryA(dst, NULL);
    snprintf(dst, cap, "%s\\tootega\\sessions", base);
    CreateDirectoryA(dst, NULL);
    snprintf(dst, cap, "%s\\tootega\\sessions\\%s.json", base, a->session);
#else
    snprintf(dst, cap, "%s/.tootega", base);
    mkdir(dst, 0700);
    snprintf(dst, cap, "%s/.tootega/sessions", base);
    mkdir(dst, 0700);
    snprintf(dst, cap, "%s/.tootega/sessions/%s.json", base, a->session);
#endif
    return 0;
}

static void save_history(const agent *a)
{
    char path[1024];
    if (transcript_path(a, path, sizeof(path)) != 0) return;

    FILE *f = fopen(path, "wb");
    if (!f) return;
    fputc('[', f);
    if (a->msgs.p) fwrite(a->msgs.p, 1, a->msgs.len, f);
    fputc(']', f);
    fclose(f);
}

static void load_history(agent *a)
{
    char path[1024];
    if (transcript_path(a, path, sizeof(path)) != 0) return;

    FILE *f = fopen(path, "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 3 || n > (32L << 20)) { fclose(f); return; }

    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return; }
    const size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';

    /* Conta as mensagens parseando; e o `n_msgs` que decide se o prompt de
     * sistema ainda precisa entrar. */
    tt_arena ar;
    int count = 0;
    if (tt_arena_init(&ar, tt_json_bound(buf, got) + 4096, "hist") == 0) {
        const tt_json *arr = tt_json_parse(buf, got, &ar);
        if (arr && arr->type == TT_JSON_ARRAY) count = (int)arr->count;
        tt_arena_free(&ar);
    }
    if (count <= 0) { free(buf); return; }

    /* Guarda sem os colchetes: `msgs` e a lista separada por virgula. */
    size_t start = 0, end = got;
    while (start < end && buf[start] != '[') start++;
    start++;
    while (end > start && buf[end - 1] != ']') end--;
    if (end > start) end--;

    a->msgs.len = 0;
    sb_put(&a->msgs, buf + start, end - start);
    a->n_msgs = count;
    free(buf);
    trace("historico retomado: %d mensagens de %s", count, path);
}

/* Objeto de mensagem ja serializado (assistente com `tool_calls`, resultado de
 * ferramenta): entra cru no historico. */
static void add_message_raw(agent *a, const char *json)
{
    if (a->n_msgs++) sb_str(&a->msgs, ",");
    sb_str(&a->msgs, json);
}

/* --------------------------------------------- entrada da UI, em fila */

/* Ha byte esperando no stdin?  -1 quando nao da para saber (console).
 *
 * ATENCAO: isto olha o PIPE, nao o buffer do stdio.  Se `fgetc` ja tiver
 * puxado um bloco de 4 kB contendo duas linhas, a segunda esta no buffer e o
 * pipe esta vazio -- a sondagem diria "nada chegou" com a mensagem na mao.
 *
 * Foi exatamente o que aconteceu: a UI escreve o `initialize` e a mensagem do
 * usuario quase juntos, o agente lia as duas de uma vez, processava a primeira
 * e depois esperava 15 s por algo que ja estava no buffer -- e saia por ocio
 * sem nunca falar com o servidor.
 *
 * Por isso o stdin roda SEM BUFFER (`setvbuf` no main): o que esta no pipe e
 * tudo o que existe. */
static int stdin_ready(void)
{
#if defined(_WIN32)
    const HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    DWORD avail = 0;
    if (!PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL)) return -1;
    return avail > 0;
#else
    fd_set r;
    FD_ZERO(&r);
    FD_SET(0, &r);
    struct timeval tv = { 0, 0 };
    const int n = select(1, &r, NULL, NULL, &tv);
    return n < 0 ? -1 : n > 0;
#endif
}

/* Le uma linha do stdin.  Devolve 1 com linha, 0 no fim do fluxo, -1 se
 * `idle_sec` passou sem nada chegar.
 *
 * O timeout existe para o agente poder ir embora sozinho: a UI sobe um
 * processo por aba e nao mata o que sobra quando a aba fecha (nem quando o
 * host da extensao recarrega).  Sem isto os agentes se acumulam -- foram
 * sete de uma vez nesta maquina. */
static int read_line_timeout(sb *line, int idle_sec)
{
    line->len = 0;
    if (line->p) line->p[0] = '\0';

    const double deadline = idle_sec > 0 ? tt_now() + idle_sec : 0;
    for (;;) {
        if (idle_sec > 0 && !line->len) {
            const int ready = stdin_ready();
            if (ready == 0) {
                if (tt_now() > deadline) return -1;
                tt_sleep_ms(25);
                continue;
            }
            /* `ready < 0` e console: cai no `fgetc` bloqueante, que e o certo
             * quando quem digita e uma pessoa. */
        }

        const int c = fgetc(stdin);
        if (c == EOF) return line->len ? 1 : 0;
        if (c == '\n') return 1;
        if (c == '\r') continue;
        const char ch = (char)c;
        sb_put(line, &ch, 1);
    }
}

static int read_line(sb *line) { return read_line_timeout(line, 0) > 0; }

/* Linhas que chegaram enquanto esperavamos uma aprovacao.  A UI pode enfileirar
 * mensagem enquanto o agente trabalha; descartar seria perder o que o usuario
 * digitou. */
#define PENDING_MAX 32
static char *g_pending[PENDING_MAX];
static int   g_npending;

static void pending_push(const char *s, size_t n)
{
    if (g_npending >= PENDING_MAX) return;
    char *copy = (char *)malloc(n + 1);
    if (!copy) return;
    memcpy(copy, s, n);
    copy[n] = '\0';
    g_pending[g_npending++] = copy;
}

/* ------------------------------------------------------- aprovacao */

/* Pergunta a UI se pode usar a ferramenta, pelo protocolo de controle, e
 * BLOQUEIA ate a resposta.  Bloquear aqui e seguro porque a geracao ja
 * terminou -- o modelo esta parado esperando o resultado.
 *
 * Devolve 1 (pode), 0 (negado) ou -1 (o canal fechou). */
static int ask_permission(agent *a, const char *tool, const char *args_json,
                          const char *tool_use_id, sb *deny_msg)
{
    if (a->yes || !tool_mutates(tool)) return 1;

    if (g_tty) {
        printf("\n  ! %s quer executar:\n    %s\n    permitir? [s/N] ",
               tool, args_json);
        fflush(stdout);
        sb ans = {0};
        if (!read_line(&ans)) { sb_free(&ans); return -1; }
        const int ok = ans.len && (ans.p[0] == 's' || ans.p[0] == 'S' ||
                                   ans.p[0] == 'y' || ans.p[0] == 'Y');
        sb_free(&ans);
        if (!ok) sb_str(deny_msg, "negado pelo usuario no terminal");
        return ok;
    }

    static int seq;
    char id[64];
    snprintf(id, sizeof(id), "perm_%d", ++seq);

    sb e = {0};
    sb_str(&e, "{\"type\":\"control_request\",\"request_id\":\"");
    sb_str(&e, id);
    sb_str(&e, "\",\"request\":{\"subtype\":\"can_use_tool\",\"tool_name\":\"");
    sb_str(&e, tool);
    sb_str(&e, "\",\"tool_use_id\":\"");
    sb_str(&e, tool_use_id);
    sb_str(&e, "\",\"input\":");
    sb_str(&e, args_json && *args_json ? args_json : "{}");
    sb_str(&e, "}}");
    emit(&e);
    sb_free(&e);

    sb line = {0};
    int verdict = -1;
    while (verdict < 0 && read_line(&line)) {
        if (!line.len) continue;

        tt_arena ar;
        if (tt_arena_init(&ar, tt_json_bound(line.p, line.len) + 4096,
                          "perm") != 0) break;
        const tt_json *root = tt_json_parse(line.p, line.len, &ar);

        if (tt_json_str_eq(tt_json_get(root, "type"), "control_response")) {
            const tt_json *resp = tt_json_get(root, "response");
            char rid[64] = "";
            tt_json_strcpy(tt_json_get(resp, "request_id"), rid, sizeof(rid));
            if (!strcmp(rid, id)) {
                const tt_json *inner = tt_json_get(resp, "response");
                verdict = tt_json_str_eq(tt_json_get(inner, "behavior"),
                                         "allow") ? 1 : 0;
                if (!verdict) {
                    size_t mn = 0;
                    const char *m = tt_json_str(tt_json_get(inner, "message"),
                                                &mn);
                    if (m && mn) sb_put(deny_msg, m, mn);
                }
            }
        } else if (tt_json_str_eq(tt_json_get(root, "request"), "interrupt") ||
                   tt_json_str_eq(tt_json_get(tt_json_get(root, "request"),
                                              "subtype"), "interrupt")) {
            verdict = 0;
            sb_str(deny_msg, "interrompido pelo usuario");
        } else {
            pending_push(line.p, line.len);
        }
        tt_arena_free(&ar);
    }
    sb_free(&line);

    if (verdict < 0) return -1;
    if (!verdict && !deny_msg->len) sb_str(deny_msg, "negado pelo usuario");
    return verdict;
}

/* ---------------------------------------------------- uma ida ao modelo */

/* Devolve 0 se falou com o servidor. */
static int step(agent *a, turn *t)
{
    sb body = {0};
    char h[256];
    snprintf(h, sizeof(h), "{\"model\":\"%s\",\"stream\":true,"
             "\"temperature\":0.6,\"think\":%s,\"messages\":[",
             a->model, a->think ? "true" : "false");
    sb_str(&body, h);
    sb_put(&body, a->msgs.p ? a->msgs.p : "", a->msgs.len);
    sb_str(&body, "]");
    if (!a->no_tools) {
        sb_str(&body, ",\"tools\":");
        sb_str(&body, TOOLS_SCHEMA);
    }
    sb_str(&body, "}");

    const int rc = http_post_sse(a, body.p, on_frame, t);
    sb_free(&body);

    if (rc != 0) {
        sb e = {0};
        sb_str(&e, "{\"type\":\"result\",\"subtype\":\"error_during_execution\","
                   "\"is_error\":true,\"session_id\":\"");
        sb_str(&e, a->session);
        sb_str(&e, "\",\"result\":\"");
        snprintf(h, sizeof(h), "sem resposta do engine em %s:%d -- "
                 "o servidor esta no ar?", a->host, a->port);
        sb_esc(&e, h, strlen(h));
        sb_str(&e, "\",\"total_cost_usd\":0}");
        emit(&e);
        sb_free(&e);
        return -1;
    }

    ev_block_stop(t);
    return 0;
}

static void run_turn(agent *a, const char *user_text, size_t n)
{
    if (!a->n_msgs) add_system(a);
    add_message(a, "user", user_text, n);

    turn t = {0};
    char h[256];
    int steps = 0;
    const double t0 = tt_now();
    int in_tokens = 0, out_tokens = 0, cached = 0;
    sb last_answer = {0};

    for (;;) {
        memset(&t, 0, sizeof(t));
        t.a = a;
        t.in_think = a->think;
        t.block_open = -1;
        snprintf(t.finish, sizeof(t.finish), "stop");

        if (step(a, &t) != 0) {
            sb_free(&t.raw); sb_free(&t.answer); sb_free(&t.calls_json);
            sb_free(&last_answer);
            return;
        }
        steps++;
        in_tokens  += t.prompt_tokens;
        out_tokens += t.completion_tokens;
        cached     += t.cached;

        if (t.started) {
            sb e = {0};
            snprintf(h, sizeof(h),
                "{\"type\":\"stream_event\",\"session_id\":\"%s\",\"event\":"
                "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"%s\"},"
                "\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d}}}",
                a->session, t.finish, t.prompt_tokens, t.completion_tokens);
            sb_str(&e, h);
            emit(&e);
            e.len = 0;
            snprintf(h, sizeof(h),
                "{\"type\":\"stream_event\",\"session_id\":\"%s\",\"event\":"
                "{\"type\":\"message_stop\"}}", a->session);
            sb_str(&e, h);
            emit(&e);
            sb_free(&e);
        }

        /* As chamadas vem como array JSON; parseamos uma vez e usamos para o
         * evento, para a aprovacao e para a execucao. */
        tt_arena car;
        const tt_json *calls = NULL;
        int ncalls = 0;
        if (t.calls_json.len &&
            tt_arena_init(&car, tt_json_bound(t.calls_json.p,
                                              t.calls_json.len) + 4096,
                          "calls") == 0) {
            calls = tt_json_parse(t.calls_json.p, t.calls_json.len, &car);
            if (calls && calls->type == TT_JSON_ARRAY) ncalls = (int)calls->count;
            else calls = NULL;
        }

        /* Mensagem completa: a UI usa os deltas para animar e ESTA para o
         * estado final.  Manda as duas coisas, como o CLI oficial. */
        {
            sb e = {0};
            sb_str(&e, "{\"type\":\"assistant\",\"session_id\":\"");
            sb_str(&e, a->session);
            snprintf(h, sizeof(h), "\",\"message\":{\"id\":\"%s",
                     turn_msg_id(&t));
            sb_str(&e, h);
            sb_str(&e, "\",\"role\":\"assistant\",\"model\":\"");
            sb_str(&e, a->model);
            sb_str(&e, "\",\"content\":[");

            int first = 1;
            const char *close = t.raw.p ? strstr(t.raw.p, "</think>") : NULL;
            if (close && close > t.raw.p) {
                sb_str(&e, "{\"type\":\"thinking\",\"thinking\":\"");
                sb_esc(&e, t.raw.p, (size_t)(close - t.raw.p));
                sb_str(&e, "\"}");
                first = 0;
            }
            if (t.answer.len || !ncalls) {
                if (!first) sb_str(&e, ",");
                sb_str(&e, "{\"type\":\"text\",\"text\":\"");
                if (t.answer.p) sb_esc(&e, t.answer.p, t.answer.len);
                sb_str(&e, "\"}");
                first = 0;
            }
            for (const tt_json *c = calls ? calls->child : NULL; c; c = c->next) {
                char id[64] = "", nm[128] = "";
                tt_json_strcpy(tt_json_get(c, "id"), id, sizeof(id));
                const tt_json *fn = tt_json_get(c, "function");
                tt_json_strcpy(tt_json_get(fn, "name"), nm, sizeof(nm));
                size_t an = 0;
                const char *as = tt_json_str(tt_json_get(fn, "arguments"), &an);

                if (!first) sb_str(&e, ",");
                sb_str(&e, "{\"type\":\"tool_use\",\"id\":\"");
                sb_str(&e, id);
                sb_str(&e, "\",\"name\":\"");
                sb_str(&e, nm);
                sb_str(&e, "\",\"input\":");
                if (as && an) sb_put(&e, as, an); else sb_str(&e, "{}");
                sb_str(&e, "}");
                first = 0;
            }

            sb_str(&e, "],\"stop_reason\":\"");
            sb_str(&e, ncalls ? "tool_use" : t.finish);
            snprintf(h, sizeof(h), "\",\"usage\":{\"input_tokens\":%d,"
                     "\"output_tokens\":%d,\"cache_read_input_tokens\":%d}}}",
                     t.prompt_tokens, t.completion_tokens, t.cached);
            sb_str(&e, h);
            emit(&e);
            sb_free(&e);
        }

        /* --- sem ferramenta: acabou o pedido ------------------------------ */
        if (!ncalls) {
            if (t.raw.len) add_message(a, "assistant", t.raw.p, t.raw.len);
            last_answer.len = 0;
            if (t.answer.p) sb_put(&last_answer, t.answer.p, t.answer.len);
            if (calls) tt_arena_free(&car);
            sb_free(&t.raw); sb_free(&t.answer); sb_free(&t.calls_json);
            break;
        }

        /* --- com ferramenta: historico, aprovacao, execucao --------------- */
        {
            sb m = {0};
            sb_str(&m, "{\"role\":\"assistant\",\"content\":\"");
            if (t.raw.p) sb_esc(&m, t.raw.p, t.raw.len);
            sb_str(&m, "\",\"tool_calls\":");
            sb_put(&m, t.calls_json.p, t.calls_json.len);
            sb_str(&m, "}");
            add_message_raw(a, m.p);
            sb_free(&m);
        }

        int aborted = 0;
        for (const tt_json *c = calls->child; c && !aborted; c = c->next) {
            char id[64] = "", nm[128] = "";
            tt_json_strcpy(tt_json_get(c, "id"), id, sizeof(id));
            const tt_json *fn = tt_json_get(c, "function");
            tt_json_strcpy(tt_json_get(fn, "name"), nm, sizeof(nm));

            size_t an = 0;
            const char *as = tt_json_str(tt_json_get(fn, "arguments"), &an);
            char *argtxt = (char *)malloc(an + 3);
            if (!argtxt) break;
            if (as && an) { memcpy(argtxt, as, an); argtxt[an] = '\0'; }
            else          { strcpy(argtxt, "{}"); }

            sb out = {0}, deny = {0};
            int is_error = 0;

            if (g_tty) {
                /* Cartao da ferramenta: nome e argumentos, cortados para caber
                 * numa linha.  O conteudo de um Write pode ter 40 kB. */
                char brief[140];
                size_t bn = strlen(argtxt);
                if (bn > sizeof(brief) - 4) bn = sizeof(brief) - 4;
                memcpy(brief, argtxt, bn);
                strcpy(brief + bn, bn < strlen(argtxt) ? "..." : "");
                printf("\n  * %s %s\n", nm, brief);
                fflush(stdout);
            }

            const int allowed = ask_permission(a, nm, argtxt, id, &deny);
            if (allowed < 0) { aborted = 1; free(argtxt); sb_free(&out);
                               sb_free(&deny); break; }

            if (!allowed) {
                sb_str(&out, "Ferramenta negada: ");
                sb_put(&out, deny.p ? deny.p : "", deny.len);
                is_error = 1;
            } else {
                tt_arena aar;
                if (tt_arena_init(&aar, tt_json_bound(argtxt, strlen(argtxt))
                                        + 4096, "args") == 0) {
                    const tt_json *ao = tt_json_parse(argtxt, strlen(argtxt),
                                                      &aar);
                    is_error = (tool_run(a, nm, ao, &out) != 0);
                    tt_arena_free(&aar);
                } else {
                    sb_str(&out, "argumentos ilegiveis");
                    is_error = 1;
                }
            }
            sb_free(&deny);
            free(argtxt);

            if (g_tty) {
                /* Uma linha de resumo: quantas linhas voltaram e a primeira
                 * delas.  Despejar a saida inteira no terminal enterraria a
                 * resposta do modelo, que e o que interessa. */
                int lines = 0;
                for (size_t k = 0; k < out.len; k++) if (out.p[k] == '\n') lines++;
                char head[100] = "";
                size_t hn = 0;
                while (hn < out.len && hn < sizeof(head) - 1 &&
                       out.p[hn] != '\n') { head[hn] = out.p[hn]; hn++; }
                head[hn] = '\0';
                printf("    -> %s%d linha%s  %s\n", is_error ? "ERRO, " : "",
                       lines + 1, lines ? "s" : "", head);
                fflush(stdout);
            }

            /* Evento para a UI: o resultado aparece no cartao da ferramenta. */
            {
                sb e = {0};
                sb_str(&e, "{\"type\":\"user\",\"session_id\":\"");
                sb_str(&e, a->session);
                sb_str(&e, "\",\"message\":{\"role\":\"user\",\"content\":"
                           "[{\"type\":\"tool_result\",\"tool_use_id\":\"");
                sb_str(&e, id);
                sb_str(&e, "\",\"is_error\":");
                sb_str(&e, is_error ? "true" : "false");
                sb_str(&e, ",\"content\":\"");
                sb_esc(&e, out.p ? out.p : "", out.len);
                sb_str(&e, "\"}]}}");
                emit(&e);
                sb_free(&e);
            }

            /* Historico para o modelo. */
            {
                sb m = {0};
                sb_str(&m, "{\"role\":\"tool\",\"tool_call_id\":\"");
                sb_str(&m, id);
                sb_str(&m, "\",\"content\":\"");
                sb_esc(&m, out.p ? out.p : "", out.len);
                sb_str(&m, "\"}");
                add_message_raw(a, m.p);
                sb_free(&m);
            }
            sb_free(&out);
        }

        tt_arena_free(&car);
        sb_free(&t.raw); sb_free(&t.answer); sb_free(&t.calls_json);

        if (aborted) { sb_free(&last_answer); return; }

        /* Teto de voltas: um laco que nunca converge queima a maquina inteira
         * a 13 tok/s sem ninguem perceber. */
        if (steps >= a->max_steps) {
            sb e = {0};
            snprintf(h, sizeof(h),
                "{\"type\":\"result\",\"subtype\":\"error_max_turns\","
                "\"is_error\":true,\"session_id\":\"%s\",\"num_turns\":%d,"
                "\"total_cost_usd\":0,\"result\":\"limite de %d passos "
                "atingido\"}", a->session, steps, a->max_steps);
            sb_str(&e, h);
            emit(&e);
            sb_free(&e);
            sb_free(&last_answer);
            return;
        }
    }

    a->turns++;
    save_history(a);          /* o processo pode sair antes do proximo turno */
    if (g_tty) {
        const double secs = tt_now() - t0;
        /* O tempo e de PAREDE: inclui prefill, execucao das ferramentas e a
         * espera pela aprovacao.  Nao dividir por ele e chamar de tok/s -- o
         * numero pareceria a velocidade do decode, e nao e. */
        printf("\n\n  %d passo%s  %d tokens gerados  %d de contexto  %.1f s\n",
               steps, steps > 1 ? "s" : "", out_tokens, in_tokens, secs);
        fflush(stdout);
    }
    {
        const double secs = tt_now() - t0;
        sb e = {0};
        snprintf(h, sizeof(h),
            "{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":false,"
            "\"session_id\":\"%s\",\"num_turns\":%d,\"duration_ms\":%d,"
            "\"total_cost_usd\":0,\"usage\":{\"input_tokens\":%d,"
            "\"output_tokens\":%d,\"cache_read_input_tokens\":%d},"
            "\"steps\":%d,\"result\":\"",
            a->session, a->turns, (int)(secs * 1000.0),
            in_tokens, out_tokens, cached, steps);
        sb_str(&e, h);
        if (last_answer.p) sb_esc(&e, last_answer.p, last_answer.len);
        sb_str(&e, "\"}");
        emit(&e);
        sb_free(&e);
    }
    sb_free(&last_answer);
}

/* ----------------------------------------------------- entrada da UI */

/* Extrai o texto de `{"type":"user","message":{"content":[{"type":"text",…}]}}`
 * -- ou de `content` como string pura, que e o formato curto. */
static int user_text_of(const tt_json *root, sb *out)
{
    const tt_json *content = tt_json_get(tt_json_get(root, "message"), "content");
    if (!content) return 0;

    size_t n = 0;
    const char *s = tt_json_str(content, &n);
    if (s) { sb_put(out, s, n); return (int)n; }

    if (content->type != TT_JSON_ARRAY) return 0;
    for (const tt_json *b = content->child; b; b = b->next) {
        if (!tt_json_str_eq(tt_json_get(b, "type"), "text")) continue;
        size_t bn = 0;
        const char *bs = tt_json_str(tt_json_get(b, "text"), &bn);
        if (bs && bn) {
            if (out->len) sb_str(out, "\n");
            sb_put(out, bs, bn);
        }
    }
    return (int)out->len;
}

static void handle_line(agent *a, const char *line, size_t n)
{
    tt_arena ar;
    if (tt_arena_init(&ar, tt_json_bound(line, n) + 4096, "in") != 0) return;
    const tt_json *root = tt_json_parse(line, n, &ar);

    if (tt_json_str_eq(tt_json_get(root, "type"), "user")) {
        sb text = {0};
        if (user_text_of(root, &text) > 0) {
            trace("turno do usuario: %zu caracteres", text.len);
            run_turn(a, text.p, text.len);
        } else {
            /* Mensagem sem texto extraivel e o modo silencioso de nao
             * acontecer nada: o agente fica parado e ninguem sabe por que. */
            trace("mensagem de usuario SEM texto -- linha: %.200s", line);
        }
        sb_free(&text);
    } else if (tt_json_str_eq(tt_json_get(root, "type"), "control_request")) {
        /* O handshake `initialize` e os pedidos que ainda nao atendemos sao
         * respondidos com sucesso vazio: a UI trata resposta desconhecida com
         * tolerancia, mas silencio absoluto a deixa esperando o timeout. */
        char id[128] = "";
        tt_json_strcpy(tt_json_get(root, "request_id"), id, sizeof(id));
        sb e = {0};
        sb_str(&e, "{\"type\":\"control_response\",\"response\":"
                   "{\"subtype\":\"success\",\"request_id\":\"");
        sb_esc(&e, id, strlen(id));
        sb_str(&e, "\",\"response\":{}}}");
        emit(&e);
        sb_free(&e);
    }

    tt_arena_free(&ar);
}

/* --------------------------------------------------------------- CLI */

static void print_help(void)
{
    printf("tootega-agent " AGENT_VERSION " -- cliente agentico do TootegaEngine\n\n"
           "  agent -p --output-format stream-json --input-format stream-json\n\n"
           "Fala o contrato de processo do Claude Code CLI: uma UI escrita\n"
           "contra aquele CLI funciona apontando o caminho do executavel para\n"
           "este binario.\n\n"
           "  --tty, -i             sessao no terminal, para usar sem UI\n"
           "  --server HOST:PORTA   engine (default 127.0.0.1:8080)\n"
           "  --key CHAVE           quando o servidor subiu com --key\n"
           "  --cwd DIR             raiz do sandbox (default: pasta corrente)\n"
           "  --think / --no-think  raciocinio antes de responder (default: nao)\n"
           "  --no-tools            so conversa, sem ferramentas\n"
           "  --yes                 nao pede aprovacao (o sandbox continua)\n"
           "  --max-steps N         teto de voltas do laco (default 16)\n"
           "  --resume ID           retoma a conversa salva desta sessao\n"
           "  --idle-exit N         sai apos N s sem receber nada (default 15;\n"
           "                        0 desliga, e o default no --tty)\n"
           "  --version             versao e build\n"
           "\nFerramentas: Read, Write, Edit, Glob, Grep, Bash.  As tres que\n"
           "mudam estado (Write, Edit, Bash) pedem aprovacao pela UI; nenhuma\n"
           "delas alcanca caminho fora de --cwd, nem com --yes.\n");
}

static void gen_session(char *dst, size_t cap)
{
    const uint64_t t = (uint64_t)(tt_now() * 1e6);
    snprintf(dst, cap, "tootega-%08llx", (unsigned long long)(t & 0xffffffffu));
}

int main(int argc, char **argv)
{
    /* Stdin sem buffer: a sondagem de `stdin_ready` olha o pipe, e um buffer
     * do stdio escondendo linhas dele e a diferenca entre responder e sair por
     * ocio com a pergunta na mao.  O volume aqui e de dezenas de kB por
     * sessao; ler byte a byte nao custa nada. */
    setvbuf(stdin, NULL, _IONBF, 0);

    agent a = {0};
    snprintf(a.host, sizeof(a.host), "127.0.0.1");
    a.port = 8080;
    snprintf(a.model, sizeof(a.model), "glm-4.7-flash");
    snprintf(a.permission_mode, sizeof(a.permission_mode), "default");
    gen_session(a.session, sizeof(a.session));

    for (int i = 1; i < argc; i++) {
        const char *g = argv[i];
        const int has = (i + 1 < argc);

        if (!strcmp(g, "--version")) {
            printf("tootega-agent %s (build %s)\n", AGENT_VERSION, tt_build_id());
            return 0;
        }
        if (!strcmp(g, "--help") || !strcmp(g, "-h")) { print_help(); return 0; }

        if      (!strcmp(g, "--server") && has) {
            const char *v = argv[++i];
            const char *colon = strrchr(v, ':');
            if (colon) {
                size_t hn = (size_t)(colon - v);
                if (hn >= sizeof(a.host)) hn = sizeof(a.host) - 1;
                memcpy(a.host, v, hn);
                a.host[hn] = '\0';
                a.port = atoi(colon + 1);
            } else {
                snprintf(a.host, sizeof(a.host), "%s", v);
            }
        }
        else if (!strcmp(g, "--key") && has)
            snprintf(a.key, sizeof(a.key), "%s", argv[++i]);
        else if (!strcmp(g, "--cwd") && has)
            snprintf(a.cwd, sizeof(a.cwd), "%s", argv[++i]);
        else if (!strcmp(g, "--model") && has)
            snprintf(a.model, sizeof(a.model), "%s", argv[++i]);
        else if (!strcmp(g, "--permission-mode") && has)
            snprintf(a.permission_mode, sizeof(a.permission_mode), "%s", argv[++i]);
        else if (!strcmp(g, "--include-partial-messages")) a.partial = 1;
        else if (!strcmp(g, "--think"))    a.think = 1;
        else if (!strcmp(g, "--no-think")) a.think = 0;
        else if (!strcmp(g, "--tty") || !strcmp(g, "-i")) g_tty = 1;
        else if (!strcmp(g, "--verbose")) g_verbose = 1;
        else if (!strcmp(g, "--no-tools")) a.no_tools = 1;
        else if (!strcmp(g, "--yes"))      a.yes = 1;
        else if (!strcmp(g, "--max-steps") && has) a.max_steps = atoi(argv[++i]);
        else if (!strcmp(g, "--idle-exit") && has) a.idle_exit = atoi(argv[++i]);
        /* A sessao vem de fora quando a UI esta retomando uma conversa: e o id
         * que aponta para a transcricao em disco. */
        else if ((!strcmp(g, "--resume") || !strcmp(g, "--session-id")) && has)
            snprintf(a.session, sizeof(a.session), "%s", argv[++i]);

        /* Flags do CLI oficial que nao mudam nada aqui.  Sao consumidas em
         * silencio -- inclusive o argumento -- porque a UI as manda sempre, e
         * abortar por flag desconhecida quebraria o encaixe por nada. */
        else if (!strcmp(g, "--effort") || !strcmp(g, "--settings") ||
                 !strcmp(g, "--disallowedTools") || !strcmp(g, "--allowedTools") ||
                 !strcmp(g, "--append-system-prompt") ||
                 !strcmp(g, "--append-system-prompt-file") ||
                 !strcmp(g, "--permission-prompt-tool") ||
                 !strcmp(g, "--mcp-config") || !strcmp(g, "--add-dir") ||
                 !strcmp(g, "--output-format") || !strcmp(g, "--input-format")) {
            if (has) i++;
        }
    }

    if (!a.cwd[0]) {
#if defined(_WIN32)
        GetCurrentDirectoryA((DWORD)sizeof(a.cwd), a.cwd);
#else
        if (!getcwd(a.cwd, sizeof(a.cwd))) snprintf(a.cwd, sizeof(a.cwd), ".");
#endif
    }
    path_norm(a.cwd, a.cwd_abs, sizeof(a.cwd_abs));
    if (!a.max_steps) a.max_steps = 16;
    /* No terminal quem manda e a pessoa, e sair sozinho no meio de uma sessao
     * seria hostil.  Pela UI, sair e o certo: um processo por aba que nunca
     * morre vira uma dezena deles. */
    if (!a.idle_exit && !g_tty) a.idle_exit = 15;
    if (a.session[0]) load_history(&a);

    if (tt_net_init() != 0) {
        fprintf(stderr, "rede indisponivel\n");
        return 1;
    }

    /* --- modo terminal ------------------------------------------------ */
    if (g_tty) {
#if defined(_WIN32)
        /* O console do Windows nasce em code page 850; sem isto toda acentuacao
         * da resposta vira lixo. */
        SetConsoleOutputCP(CP_UTF8);
#endif
        printf("tootega-agent %s  ->  %s:%d\n", AGENT_VERSION, a.host, a.port);
        printf("pasta        %s\n", a.cwd_abs);
        printf("ferramentas  %s\n",
               a.no_tools ? "(nenhuma)" : "Read Write Edit Glob Grep Bash");
        printf("raciocinio   %s%s\n", a.think ? "ligado" : "desligado",
               a.yes ? "    aprovacao AUTOMATICA (--yes)" : "");
        printf("\nEscreva o pedido e Enter. /sair termina.\n");

        sb line = {0};
        for (;;) {
            printf("\n> ");
            fflush(stdout);
            if (!read_line(&line)) break;
            if (!line.len) continue;
            if (!strcmp(line.p, "/sair") || !strcmp(line.p, "/quit")) break;
            if (!strcmp(line.p, "/limpar")) {
                a.msgs.len = 0;
                a.n_msgs = 0;
                if (a.msgs.p) a.msgs.p[0] = '\0';
                printf("  (conversa esquecida)\n");
                continue;
            }
            printf("\n");
            run_turn(&a, line.p, line.len);
        }
        sb_free(&line);
        sb_free(&a.msgs);
        tt_net_shutdown();
        return 0;
    }

    /* `init` primeiro: a UI so mostra a sessao depois dele. */
    {
        sb e = {0};
        sb_str(&e, "{\"type\":\"system\",\"subtype\":\"init\",\"session_id\":\"");
        sb_str(&e, a.session);
        sb_str(&e, "\",\"cwd\":\"");
        sb_esc(&e, a.cwd_abs, strlen(a.cwd_abs));
        sb_str(&e, "\",\"model\":\"");
        sb_str(&e, a.model);
        sb_str(&e, "\",\"permissionMode\":\"");
        sb_str(&e, a.permission_mode);
        sb_str(&e, a.no_tools
            ? "\",\"tools\":[]"
            : "\",\"tools\":[\"Read\",\"Write\",\"Edit\",\"Glob\",\"Grep\","
              "\"Bash\"]");
        sb_str(&e, ",\"mcp_servers\":[],\"apiKeySource\":\"none\"");
        const long long ctx = ask_context_window(&a);
        if (ctx > 0) {
            char cw[64];
            snprintf(cw, sizeof(cw), ",\"context_window\":%lld", ctx);
            sb_str(&e, cw);
        }
        sb_str(&e, "}");
        emit(&e);
        sb_free(&e);
    }

    /* Uma linha NDJSON por evento da UI.  Linha longa (prompt colado com
     * arquivo inteiro) cresce o buffer em vez de ser cortada. */
    trace("pronto, lendo stdin (servidor %s:%d, cwd %s, ocioso %ds)",
          a.host, a.port, a.cwd_abs, a.idle_exit);
    sb line = {0};
    for (;;) {
        const int r = read_line_timeout(&line, a.idle_exit);
        if (r == 0) break;                       /* a UI fechou o stdin */
        if (r < 0) {
            /* Ocioso: sai e deixa a conversa no disco.  A proxima mensagem
             * desta aba sobe um processo novo, que retoma de onde parou. */
            trace("ocioso por %d s -- saindo (a conversa fica salva)",
                  a.idle_exit);
            break;
        }
        if (line.len) {
            trace("stdin: %.160s", line.p);
            handle_line(&a, line.p, line.len);
        }

        /* O que chegou enquanto uma aprovacao estava pendente entra agora, na
         * ordem em que o usuario mandou. */
        for (int i = 0; i < g_npending; i++) {
            handle_line(&a, g_pending[i], strlen(g_pending[i]));
            free(g_pending[i]);
        }
        g_npending = 0;
    }

    sb_free(&line);
    sb_free(&a.msgs);
    tt_net_shutdown();
    return 0;
}
