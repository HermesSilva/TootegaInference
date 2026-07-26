#include "server.h"

#include "budget.h"
#include "common.h"
#include "compat.h"
#include "forward.h"
#include "json.h"
#include "kernels.h"
#include "model.h"
#include "pool.h"
#include "sample.h"
#include "tokenizer.h"
#include "webui.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define REQ_MAX     (1u << 20)     /* 1 MB de requisicao: prompt grande cabe */
#define OUT_CHUNK   4096

/* Marcadores do template do modelo -- conferidos byte a byte contra
 * `apply_chat_template` (see cmd_chat in main.c). */
#define CHAT_PREFIX "[gMASK]<sop>"
#define CHAT_SYSTEM "<|system|>"
#define CHAT_USER   "<|user|>"
#define CHAT_ASSIST "<|assistant|>"
#define CHAT_THINK  "<think>"

/* The model id in the protocol.  One place only: `/v1/models` and the replies
 * have to agree, or a client that validates the field rejects the answer. */
#define TT_MODEL_ID "glm-4.7-flash"

/* ------------------------------------------------------------ verbose */

/* `--verbose` shows the SHAPE of what comes in and out, and the statistics --
 * conteudo.
 *
 * The user's text and the model's were left out on purpose: the server
 * roda em terminal aberto, e a conversa de quem o usa pode ter caminho de
 * cliente, credencial colada num prompt, trecho de codigo fechado.  Nada disso
 * ajuda a depurar o protocolo -- o que ajuda e quantas mensagens, de que
 * papel, com quantos tokens, quanto veio do cache e o quao confiante o modelo
 * estava.  Isso o log tem, e com mais detalhe do que antes. */
static int g_verbose;

static void vlog(const char *fmt, ...)
{
    if (!g_verbose) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    fflush(stderr);
}

/* Data e hora de parede, para casar com o log do cliente.
 *
 * `tt_now()` is monotonic -- good for measuring duration, and it says nothing
 * about WHEN.  Correlating the server console with Cockpit's Output needs
 * relogio de parede nos dois. */
static const char *stamp(char *buf, size_t cap)
{
    const time_t t = time(NULL);
    struct tm tmv;
#if defined(_WIN32)
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    strftime(buf, cap, "%Y-%m-%d %H:%M:%S", &tmv);
    return buf;
}

/* ------------------------------------------------- estatistica do passe */

/* What the model thought of each token it chose itself.
 *
 * `nll` e a log-verossimilhanca negativa media; `exp(nll)` e a perplexidade --
 * how many alternatives the model was effectively weighing per position.
 * A entropia mede o espalhamento da distribuicao INTEIRA, e a diferenca entre
 * as duas separa dois casos que a perplexidade sozinha confunde: modelo
 * confident and right (low nll, low entropy) from confident for lack of
 * options (low nll, high entropy) -- the second is what precedes repetition. */
typedef struct {
    double  nll;        /* soma de -ln p(escolhido) */
    double  ent;        /* soma da entropia, em nats */
    double  top1;       /* soma da probabilidade do mais provavel */
    double  chosen_p;   /* soma da probabilidade do escolhido */
    int64_t n;
} pass_stats;

/* Uma passada estavel sobre os logits: max, soma dos exp, e as tres somas.
 *
 * It costs two scans of 154,880 floats per token -- ~0.1 ms against the pass's
 * 74 ms.  Cheap enough to stay on whenever `--verbose` is. */
static void stat_add(pass_stats *g, const float *logits, int64_t n, int32_t pick)
{
    if (n <= 0 || pick < 0 || pick >= n) return;

    float mx = logits[0];
    for (int64_t i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];

    double sum = 0.0, top = 0.0, wsum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        const double e = exp((double)logits[i] - (double)mx);
        sum += e;
        if (e > top) top = e;
        /* sum of p*ln(p) unnormalised; normalisation comes later */
        wsum += e * ((double)logits[i] - (double)mx);
    }
    if (sum <= 0.0) return;

    const double p = exp((double)logits[pick] - (double)mx) / sum;
    g->nll      += -log(p);
    g->top1     += top / sum;
    g->chosen_p += p;
    /* H = ln(sum) - (1/sum) * soma(e_i * z_i), com z ja deslocado pelo max. */
    g->ent      += log(sum) - wsum / sum;
    g->n++;
}

static void stat_log(const char *rotulo, const pass_stats *g)
{
    if (!g->n) return;
    const double nll = g->nll / (double)g->n;
    vlog("  %-11s perplexidade %.2f  nll %.4f  entropia %.2f bits  "
         "confianca %.1f%% (topo %.1f%%)  em %lld tokens",
         rotulo, exp(nll), nll, (g->ent / (double)g->n) / 0.693147180559945,
         100.0 * g->chosen_p / (double)g->n, 100.0 * g->top1 / (double)g->n,
         (long long)g->n);
}

typedef struct {
    tt_model     *model;
    tt_state     *state;
    tt_tokenizer *tok;
    tt_pool      *pool;
    tt_plan      *plan;

    int32_t  *conv;          /* ids da conversa corrente */
    int64_t   n_conv;
    int64_t   ctx;

    int32_t   eos[8];
    int       n_eos;

    const char *key;         /* NULL when loopback only */
    uint64_t    served;
    double      started;
} server;

/* ------------------------------------------------------------- HTTP */

typedef struct {
    char  method[8];
    char  path[512];
    const char *body;
    size_t body_len;
    char  auth[256];
} request;

static const char *header_value(const char *head, const char *name)
{
    /* HTTP headers are case-insensitive; we compare accordingly. */
    const size_t n = strlen(name);
    for (const char *p = head; *p; p++) {
        if (p != head && p[-1] != '\n') continue;
        size_t i = 0;
        while (i < n && p[i] && (p[i] | 0x20) == (name[i] | 0x20)) i++;
        if (i == n && p[i] == ':') {
            p += n + 1;
            while (*p == ' ') p++;
            return p;
        }
    }
    return NULL;
}

/* Reads the whole request.  Returns 0 on success. */
static int read_request(const tt_socket *c, char *buf, size_t cap, request *r)
{
    size_t have = 0;
    const char *sep = NULL;

    while (have + 1 < cap) {
        const long k = tt_recv(c, buf + have, cap - have - 1);
        if (k <= 0) return -1;
        have += (size_t)k;
        buf[have] = '\0';

        sep = strstr(buf, "\r\n\r\n");
        if (sep) break;
    }
    if (!sep) return -1;

    if (sscanf(buf, "%7s %511s", r->method, r->path) != 2) return -1;

    const size_t head_len = (size_t)(sep - buf) + 4;
    r->auth[0] = '\0';
    const char *a = header_value(buf, "authorization");
    if (a) {
        size_t i = 0;
        while (a[i] && a[i] != '\r' && i + 1 < sizeof(r->auth)) { r->auth[i] = a[i]; i++; }
        r->auth[i] = '\0';
    }

    size_t want = 0;
    const char *cl = header_value(buf, "content-length");
    if (cl) want = (size_t)strtoull(cl, NULL, 10);

    /* The body may or may not have arrived together with the header. */
    size_t body_have = have - head_len;
    while (body_have < want && have + 1 < cap) {
        const long k = tt_recv(c, buf + have, cap - have - 1);
        if (k <= 0) break;
        have += (size_t)k;
        body_have = have - head_len;
    }

    buf[have] = '\0';
    r->body = buf + head_len;
    r->body_len = body_have < want ? body_have : want;
    return 0;
}

static void send_status(const tt_socket *c, int code, const char *reason,
                        const char *ctype, const char *body, size_t len)
{
    char head[512];
    /* CORS is open because the server is local and already has a key for what
     * costs.  Without it, a client running INSIDE a browser -- and VS Code's
     * Copilot is one -- cannot even get through the preflight. */
    const int n = snprintf(head, sizeof(head),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Cache-Control: no-store\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Connection: close\r\n\r\n",
        code, reason, ctype, len);
    tt_send_all(c, head, (size_t)n);
    if (len) tt_send_all(c, body, len);
}

static void send_json_error(const tt_socket *c, int code, const char *msg)
{
    char body[512];
    const int n = snprintf(body, sizeof(body),
        "{\"error\":{\"message\":\"%s\",\"type\":\"tootega\"}}", msg);
    send_status(c, code, "Error", "application/json", body, (size_t)n);
}

/* --------------------------------------------------------- conversa */

static int64_t append_text(server *s, const char *text)
{
    const int64_t n = tt_encode(s->tok, text, strlen(text),
                                s->conv + s->n_conv, s->ctx - s->n_conv);
    if (n < 0) return -1;
    s->n_conv += n;
    return n;
}

static size_t json_escape(const char *src, size_t n, char *dst, size_t cap);

/* ------------------------------------------------------- ferramentas */

/* Appends bytes, clipping at the limit.  Returns the new length. */
static size_t jw_put(char *dst, size_t cap, size_t w, const char *s, size_t n)
{
    for (size_t i = 0; i < n && w + 1 < cap; i++) dst[w++] = s[i];
    return w;
}

/* Serialises a DOM node back into JSON.
 *
 * Needed because the model template dumps each tool as JSON inside the prompt
 * (`{{ tool | tojson }}`), and our DOM is zero-copy: it keeps pointers to the
 * pieces, not the object's original text. */
static size_t json_write(const tt_json *n, char *dst, size_t cap, size_t w)
{
    char num[64];
    if (!n) return jw_put(dst, cap, w, "null", 4);

    switch (n->type) {
    case TT_JSON_NULL:   return jw_put(dst, cap, w, "null", 4);
    case TT_JSON_BOOL:   return n->v.boolean ? jw_put(dst, cap, w, "true", 4)
                                             : jw_put(dst, cap, w, "false", 5);
    case TT_JSON_INT:
        return jw_put(dst, cap, w, num,
                      (size_t)snprintf(num, sizeof(num), "%lld",
                                       (long long)n->v.i));
    case TT_JSON_REAL:
        return jw_put(dst, cap, w, num,
                      (size_t)snprintf(num, sizeof(num), "%.17g", n->v.r));
    case TT_JSON_STRING: {
        w = jw_put(dst, cap, w, "\"", 1);
        if (w + 1 < cap) w += json_escape(n->v.str.ptr, n->v.str.len,
                                          dst + w, cap - w);
        return jw_put(dst, cap, w, "\"", 1);
    }
    case TT_JSON_ARRAY:
        w = jw_put(dst, cap, w, "[", 1);
        for (const tt_json *e = n->child; e; e = e->next) {
            if (e != n->child) w = jw_put(dst, cap, w, ",", 1);
            w = json_write(e, dst, cap, w);
        }
        return jw_put(dst, cap, w, "]", 1);
    case TT_JSON_OBJECT:
        w = jw_put(dst, cap, w, "{", 1);
        for (const tt_json *m = n->child; m; m = m->next) {
            if (m != n->child) w = jw_put(dst, cap, w, ",", 1);
            w = jw_put(dst, cap, w, "\"", 1);
            if (w + 1 < cap) w += json_escape(m->key, m->key_len, dst + w, cap - w);
            w = jw_put(dst, cap, w, "\":", 2);
            w = json_write(m, dst, cap, w);
        }
        return jw_put(dst, cap, w, "}", 1);
    }
    return w;
}

/* Bloco de sistema com as ferramentas, byte a byte como `chat_template.jinja`.
 *
 * It sits at the start of the prefix on purpose: it is the part that does NOT
 * change between turns, and the docs/03 ordering rule (stable first) is what
 * prefixo valer alguma coisa numa sessao de agente. */
static int append_tools(server *s, const tt_json *tools)
{
    if (append_text(s, CHAT_SYSTEM "\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> "
        "XML tags:\n<tools>\n") < 0) return -1;

    char *one = (char *)malloc(64u << 10);
    if (!one) return tt_fail("out of memory");
    for (const tt_json *t = tools->child; t; t = t->next) {
        const size_t n = json_write(t, one, 64u << 10, 0);
        one[n] = '\0';
        if (append_text(s, one) < 0 || append_text(s, "\n") < 0) {
            free(one);
            return -1;
        }
    }
    free(one);

    const int64_t got = append_text(s,
        "</tools>\n\nFor each function call, output the "
        "function name and arguments within the following XML format:\n"
        "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>"
        "<arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key>"
        "<arg_value>{arg-value-2}</arg_value>...</tool_call>");
    return got < 0 ? -1 : 0;
}

/* Calls the assistant has already made, in the format the model emitted.
 * `arguments` chega como string JSON (OpenAI) ou objeto; os dois viram pares
 * <arg_key>/<arg_value>. */
static int append_tool_calls(server *s, const tt_json *calls, tt_arena *dom)
{
    for (const tt_json *tc = calls->child; tc; tc = tc->next) {
        const tt_json *fn = tt_json_get(tc, "function");
        if (!fn) fn = tc;

        char name[128];
        if (tt_json_strcpy(tt_json_get(fn, "name"), name, sizeof(name)) != 0)
            continue;
        if (append_text(s, "<tool_call>") < 0 || append_text(s, name) < 0)
            return -1;

        const tt_json *args = tt_json_get(fn, "arguments");
        size_t alen = 0;
        const char *atext = tt_json_str(args, &alen);
        if (atext) args = tt_json_parse(atext, alen, dom);  /* string JSON */

        if (args && args->type == TT_JSON_OBJECT) {
            char val[8192];
            for (const tt_json *a = args->child; a; a = a->next) {
                char key[128];
                const size_t kn = a->key_len < sizeof(key) - 1
                                ? a->key_len : sizeof(key) - 1;
                memcpy(key, a->key, kn);
                key[kn] = '\0';

                /* A string goes raw; everything else goes as JSON -- which is
                 * faz com `tojson if v is not string else v`. */
                size_t vn = 0;
                const char *vs = tt_json_str(a, &vn);
                if (vs) {
                    if (vn >= sizeof(val)) vn = sizeof(val) - 1;
                    memcpy(val, vs, vn);
                    val[vn] = '\0';
                } else {
                    val[json_write(a, val, sizeof(val), 0)] = '\0';
                }

                if (append_text(s, "<arg_key>") < 0 ||
                    append_text(s, key) < 0 ||
                    append_text(s, "</arg_key><arg_value>") < 0 ||
                    append_text(s, val) < 0 ||
                    append_text(s, "</arg_value>") < 0) return -1;
            }
        }
        if (append_text(s, "</tool_call>") < 0) return -1;
    }
    return 0;
}

/* Monta o prefixo da conversa a partir das mensagens do pedido.
 *
 * The UI sends the whole conversation every turn (that is what an OpenAI
 * Reconstruimos os ids do zero e deixamos o casamento de prefixo decidir o que
 * recomputed -- which is why the second turn only costs the new tokens. */
static int build_conversation(server *s, const tt_json *messages, int think,
                              const tt_json *tools, tt_arena *dom)
{
    s->n_conv = 0;
    if (append_text(s, CHAT_PREFIX) < 0) return -1;
    if (tools && tools->type == TT_JSON_ARRAY && tools->count &&
        append_tools(s, tools) != 0) return -1;

    for (const tt_json *m = messages->child; m; m = m->next) {
        const tt_json *role = tt_json_get(m, "role");
        const tt_json *content = tt_json_get(m, "content");
        size_t clen = 0;
        const char *ctext = tt_json_str(content, &clen);

        /* Resultado de ferramenta: `<|observation|>` uma vez, um
         * `<tool_response>` por resultado.  E o unico papel em que a mensagem
         * is NOT the model's nor the user's -- it is the world answering. */
        if (tt_json_str_eq(role, "tool")) {
            if (!ctext) continue;
            char *tmp = (char *)malloc(clen + 1);
            if (!tmp) return tt_fail("out of memory");
            memcpy(tmp, ctext, clen);
            tmp[clen] = '\0';
            const int bad = append_text(s, "<|observation|><tool_response>") < 0
                         || append_text(s, tmp) < 0
                         || append_text(s, "</tool_response>") < 0;
            free(tmp);
            if (bad) return -1;
            continue;
        }

        const int assistant = tt_json_str_eq(role, "assistant");
        const tt_json *calls = assistant ? tt_json_get(m, "tool_calls") : NULL;
        if (!ctext && !calls) continue;

        const char *marker = CHAT_USER;
        if (tt_json_str_eq(role, "system")) marker = CHAT_SYSTEM;
        else if (assistant)                 marker = CHAT_ASSIST;

        if (append_text(s, marker) < 0) return -1;

        /* An assistant turn that only called a tool has no text. */
        if (!ctext) {
            if (append_text(s, "</think>") < 0) return -1;
            if (append_tool_calls(s, calls, dom) != 0) return -1;
            continue;
        }

        /* `tt_encode` needs a terminator; the DOM has none. */
        char *tmp = (char *)malloc(clen + 1);
        if (!tmp) return tt_fail("out of memory");
        memcpy(tmp, ctext, clen);
        tmp[clen] = '\0';

        /* Turno de assistente JA CONCLUIDO entra como `</think>` mais a
         * answer, without the reasoning -- which is what `chat_template.jinja`
         * so o turno aberto ganha `<think>`.
         *
         * It lives in the server and not the client because the template is
         * the model's: any OpenAI client sends plain `content` and has to
         * work.  And stripping the reasoning from the history is what keeps
         * the context from swelling every round. */
        const char *body = tmp;
        if (assistant) {
            const char *close = strstr(tmp, "</think>");
            if (close) body = close + 8;
            while (*body == '\n' || *body == ' ') body++;
            if (append_text(s, "</think>") < 0) { free(tmp); return -1; }
        }

        const int64_t got = append_text(s, body);
        free(tmp);
        if (got < 0) return -1;

        /* Text AND calls in the same turn: the model explains, then calls. */
        if (calls && append_tool_calls(s, calls, dom) != 0) return -1;
    }

    if (append_text(s, CHAT_ASSIST) < 0) return -1;

    /* `<think>` opens the draft; `</think>` closes it before it starts, and the
     * model goes straight to the answer.
     *
     * It is not a trick: it is the same construction the template uses for
     * assistant turns ALREADY completed, so the model saw it in training.  The
     * cost is elsewhere -- see docs/20. */
    if (append_text(s, think ? CHAT_THINK : "</think>") < 0) return -1;
    return 0;
}

static int is_eos(const server *s, int32_t id)
{
    for (int i = 0; i < s->n_eos; i++) if (s->eos[i] == id) return 1;
    return 0;
}

/* Escapes a string for embedding in JSON. */
static size_t json_escape(const char *src, size_t n, char *dst, size_t cap)
{
    size_t w = 0;
    for (size_t i = 0; i < n && w + 8 < cap; i++) {
        const unsigned char ch = (unsigned char)src[i];
        switch (ch) {
        case '"':  dst[w++] = '\\'; dst[w++] = '"';  break;
        case '\\': dst[w++] = '\\'; dst[w++] = '\\'; break;
        case '\n': dst[w++] = '\\'; dst[w++] = 'n';  break;
        case '\r': dst[w++] = '\\'; dst[w++] = 'r';  break;
        case '\t': dst[w++] = '\\'; dst[w++] = 't';  break;
        default:
            if (ch < 0x20) w += (size_t)snprintf(dst + w, cap - w, "\\u%04X", ch);
            else dst[w++] = (char)ch;
        }
    }
    dst[w] = '\0';
    return w;
}

/* -------------------------------------- model output -> tool_calls */

/* Tipo declarado do argumento no schema que o CLIENTE mandou.
 *
 * The model emits `<arg_value>3</arg_value>` without saying whether it is the
 * string "3"; OpenAI exige `arguments` como JSON tipado.  Sem consultar o
 * the schema, every tool with a numeric or boolean argument arrives wrong on
 * the other side. */
static char arg_kind(const tt_json *tools, const char *fname, const char *key)
{
    for (const tt_json *t = tools ? tools->child : NULL; t; t = t->next) {
        const tt_json *fn = tt_json_get(t, "function");
        if (!fn) fn = t;
        if (!tt_json_str_eq(tt_json_get(fn, "name"), fname)) continue;

        const tt_json *props = tt_json_get(tt_json_get(fn, "parameters"),
                                           "properties");
        const tt_json *p = tt_json_get(props, key);
        if (!p) return '?';
        if (tt_json_str_eq(tt_json_get(p, "type"), "string")) return 's';
        return 'j';                                   /* number, bool, array… */
    }
    return '?';
}

/* A raw `<arg_value>` becomes JSON.  `kind` decides; `?` (a silent schema)
 * accepts the raw value only when it ALREADY is valid JSON. */
static size_t arg_value_json(const char *v, size_t n, char kind,
                             char *dst, size_t cap)
{
    int looks_json = 0;
    if (n) {
        const char c0 = v[0];
        looks_json = (c0 == '{' || c0 == '[' || c0 == '-' ||
                      (c0 >= '0' && c0 <= '9') ||
                      !strncmp(v, "true", 4) || !strncmp(v, "false", 5) ||
                      !strncmp(v, "null", 4));
    }
    if (kind == 'j' || (kind == '?' && looks_json))
        return jw_put(dst, cap, 0, v, n);

    size_t w = jw_put(dst, cap, 0, "\"", 1);
    w += json_escape(v, n, dst + w, cap - w);
    return jw_put(dst, cap, w, "\"", 1);
}

/* Extracts every call from the generated text and assembles OpenAI's
 * `tool_calls` array.  Returns how many it found. */
static int parse_tool_calls(const char *text, const tt_json *tools,
                            char *dst, size_t cap, uint64_t seq)
{
    size_t w = jw_put(dst, cap, 0, "[", 1);
    int found = 0;

    for (const char *p = strstr(text, "<tool_call>"); p;
         p = strstr(p, "<tool_call>")) {
        p += 11;
        const char *end = strstr(p, "</tool_call>");
        if (!end) break;

        const char *name_end = strstr(p, "<arg_key>");
        if (!name_end || name_end > end) name_end = end;

        char fname[128];
        size_t nn = (size_t)(name_end - p);
        while (nn && (p[nn - 1] == '\n' || p[nn - 1] == ' ')) nn--;
        if (nn >= sizeof(fname)) nn = sizeof(fname) - 1;
        memcpy(fname, p, nn);
        fname[nn] = '\0';

        char args[16384];
        size_t aw = jw_put(args, sizeof(args), 0, "{", 1);
        int nargs = 0;

        for (const char *k = strstr(p, "<arg_key>"); k && k < end;
             k = strstr(k, "<arg_key>")) {
            k += 9;
            const char *ke = strstr(k, "</arg_key>");
            if (!ke || ke > end) break;
            const char *v = strstr(ke, "<arg_value>");
            if (!v || v > end) break;
            v += 11;
            const char *ve = strstr(v, "</arg_value>");
            if (!ve || ve > end) break;

            char key[128];
            size_t kn = (size_t)(ke - k);
            if (kn >= sizeof(key)) kn = sizeof(key) - 1;
            memcpy(key, k, kn);
            key[kn] = '\0';

            if (nargs++) aw = jw_put(args, sizeof(args), aw, ",", 1);
            aw = jw_put(args, sizeof(args), aw, "\"", 1);
            aw += json_escape(key, kn, args + aw, sizeof(args) - aw);
            aw = jw_put(args, sizeof(args), aw, "\":", 2);
            aw += arg_value_json(v, (size_t)(ve - v),
                                 arg_kind(tools, fname, key),
                                 args + aw, sizeof(args) - aw);
            k = ve;
        }
        args[jw_put(args, sizeof(args), aw, "}", 1)] = '\0';

        /* The id carries the REQUEST number, not just the index inside it.
         *
         * With `call_0` in every response, a client that dedupes by id -- and
         * Cockpit does dedupe -- shows the first step's card and discards
         * every following one as a repeat.  The loop worked and the
         * tela ficava parada. */
        char one[17408];
        const int n = snprintf(one, sizeof(one),
            "%s{\"id\":\"call_%llu_%d\",\"type\":\"function\","
            "\"function\":{\"name\":\"%s\",\"arguments\":\"",
            found ? "," : "", (unsigned long long)seq, found, fname);
        w = jw_put(dst, cap, w, one, (size_t)n);
        /* `arguments` is a STRING in the protocol -- the JSON goes escaped inside. */
        w += json_escape(args, strlen(args), dst + w, cap - w);
        w = jw_put(dst, cap, w, "\"}}", 3);

        found++;
        p = end + 12;
    }

    dst[jw_put(dst, cap, w, "]", 1)] = '\0';
    return found;
}

/* ------------------------------------------------------ /v1/chat/... */

static int handle_chat(server *s, const tt_socket *c, const request *r)
{
    tt_arena dom;
    if (tt_arena_init(&dom, tt_json_bound(r->body, r->body_len) + 4096,
                      "req") != 0)
        return -1;

    tt_json *root = tt_json_parse(r->body, r->body_len, &dom);
    const tt_json *messages = tt_json_get(root, "messages");
    if (!messages || messages->type != TT_JSON_ARRAY || !messages->count) {
        tt_arena_free(&dom);
        send_json_error(c, 400, "faltou 'messages'");
        return 0;
    }

    const tt_json *tools = tt_json_get(root, "tools");
    if (tools && (tools->type != TT_JSON_ARRAY || !tools->count)) tools = NULL;
    /* `tool_choice: "none"` desliga as ferramentas para ESTE pedido. */
    if (tt_json_str_eq(tt_json_get(root, "tool_choice"), "none")) tools = NULL;

    const int stream = tt_json_bool(tt_json_get(root, "stream"), 1);
    /* `think` is ours; `chat_template_kwargs.enable_thinking` is what vLLM and
     * SGLang popularised and what a third-party client sends. */
    const int think = tt_json_bool(tt_json_get(root, "think"),
        tt_json_bool(tt_json_get(tt_json_get(root, "chat_template_kwargs"),
                                 "enable_thinking"), 1));
    const float temp = (float)tt_json_real(tt_json_get(root, "temperature"), 0.8);
    const float top_p = (float)tt_json_real(tt_json_get(root, "top_p"), 0.95);
    const int   top_k = (int)tt_json_int(tt_json_get(root, "top_k"), 40);
    /* Sem `max_tokens`, gera ate o EOS.  O unico teto e o contexto -- cortar
     * by count interrupts the model mid-reasoning and delivers a
     * rascunho como se fosse resposta. */
    int64_t max_new = tt_json_int(tt_json_get(root, "max_tokens"), 0);

    /* The request arena survives to the end: `tools` is consulted after
     * geracao, para tipar os argumentos da chamada. */
    if (build_conversation(s, messages, think, tools, &dom) != 0 ||
        s->n_conv + 8 >= s->ctx) {
        tt_arena_free(&dom);
        send_json_error(c, 400, "conversation does not fit in the context");
        return 0;
    }
    const int64_t room = s->ctx - s->n_conv - 1;
    if (max_new <= 0 || max_new > room) max_new = room;

    if (g_verbose) {
        /* Who spoke, how much, and in what shape.  The TEXT stays out: it is
         * ajuda a depurar o protocolo e o console e um lugar publico. */
        int n_user = 0, n_asst = 0, n_tool = 0, n_sys = 0;
        size_t b_user = 0, b_asst = 0, b_tool = 0, b_sys = 0;
        for (const tt_json *m = messages->child; m; m = m->next) {
            const tt_json *role = tt_json_get(m, "role");
            size_t cn = 0;
            if (!tt_json_str(tt_json_get(m, "content"), &cn)) cn = 0;
            if      (tt_json_str_eq(role, "user"))      { n_user++; b_user += cn; }
            else if (tt_json_str_eq(role, "assistant")) { n_asst++; b_asst += cn; }
            else if (tt_json_str_eq(role, "tool"))      { n_tool++; b_tool += cn; }
            else if (tt_json_str_eq(role, "system"))    { n_sys++;  b_sys  += cn; }
        }

        vlog("--- request ----------------------------------------------");
        vlog("  conversation %llu messages  (role x count x characters)",
             (unsigned long long)messages->count);
        vlog("               system %d/%zu   user %d/%zu   "
             "assistant %d/%zu   tool %d/%zu",
             n_sys, b_sys, n_user, b_user, n_asst, b_asst, n_tool, b_tool);

        if (tools) {
            char names[240];
            size_t w = 0;
            for (const tt_json *t = tools->child; t; t = t->next) {
                const tt_json *fn = tt_json_get(t, "function");
                char nm[64] = "";
                if (tt_json_strcpy(tt_json_get(fn ? fn : t, "name"), nm,
                                   sizeof(nm)) == 0 && w + strlen(nm) + 3 < sizeof(names))
                    w += (size_t)snprintf(names + w, sizeof(names) - w, "%s%s",
                                          w ? ", " : "", nm);
            }
            names[w] = '\0';
            vlog("  tools       %llu declared: %s",
                 (unsigned long long)tools->count, names);
        }

        vlog("  options     reasoning %s, temperature %.2f, top_p %.2f, "
             "top_k %d, %s", think ? "ON" : "off", temp, top_p, top_k,
             stream ? "streaming" : "single response");

        vlog("  context     %lld tokens assembled, %lld positions left (%.1f%% "
             "of the window)", (long long)s->n_conv, (long long)room,
             100.0 * (double)s->n_conv / (double)s->ctx);
    }

    /* Prefix matching: what is already in the KV is not computed again.
     * In a continued conversation only the new tokens enter the prefill. */
    const int64_t reused = tt_state_match_prefix(s->state, s->conv, s->n_conv);

    /* Prompt perplexity, measured only over the RECOMPUTED part.
     *
     * What came from the cache has no new logits -- which is precisely why it
     * matched.  So the number is "how predictable the new piece of the
     * conversation was", which is what matters: a tool result with high
     * perplexity is what pushes the model off topic. */
    pass_stats st_prompt = {0};
    const double t0 = tt_now();
    for (int64_t i = reused; i < s->n_conv; i++) {
        if (tt_forward(s->state, s->conv[i], NULL) != 0) {
            tt_arena_free(&dom);
            send_json_error(c, 500, tt_error());
            return 0;
        }
        /* Os logits apos o token i predizem o token i+1. */
        if (g_verbose && i + 1 < s->n_conv)
            stat_add(&st_prompt, s->state->logits, s->tok->n_tokens,
                     s->conv[i + 1]);
    }
    const double t_prefill = tt_now() - t0;

    tt_log("  %lld tokens (%lld do cache) em %.2f s", (long long)s->n_conv,
           (long long)reused, t_prefill);
    vlog("  prefill     %lld of %lld from cache (%.0f%%), %lld recomputed in "
         "%.2f s (%.1f tok/s)%s",
         (long long)reused, (long long)s->n_conv,
         s->n_conv ? 100.0 * (double)reused / (double)s->n_conv : 0.0,
         (long long)(s->n_conv - reused), t_prefill,
         t_prefill > 0 ? (double)(s->n_conv - reused) / t_prefill : 0.0,
         reused ? "" : "  <- cache frio: recomputou o prompt inteiro");
    stat_log("prompt", &st_prompt);

    tt_sampler sampler;
    tt_sampler_init(&sampler, temp, top_p, top_k, (uint64_t)(tt_now() * 1e6));

    if (stream) {
        static const char head[] =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n\r\n";
        tt_send_all(c, head, sizeof(head) - 1);
    }

    char piece[512], esc[2048], frame[4096];
    /* The whole text is accumulated even when streaming: tool calls can only
     * be extracted from the whole, never from a chunk. */
    const size_t full_cap = 1u << 20;
    char *full = (char *)malloc(full_cap);
    size_t full_len = 0;
    if (full) full[0] = '\0';

    int64_t made = 0;
    int hit_eos = 0, in_call = 0, done_call = 0, streamed = 0;
    const double t1 = tt_now();

    pass_stats st_gen = {0};
    while (made < max_new && s->n_conv < s->ctx) {
        const int32_t next = tt_sample(&sampler, s->state->logits, s->tok->n_tokens);
        if (g_verbose) stat_add(&st_gen, s->state->logits, s->tok->n_tokens, next);
        if (is_eos(s, next)) { hit_eos = 1; break; }

        const int64_t w = tt_decode(s->tok, &next, 1, piece, sizeof(piece) - 1);
        if (w > 0) {
            piece[w] = '\0';
            if (!strcmp(piece, "<tool_call>")) in_call = 1;

            if (full && full_len + (size_t)w + 1 < full_cap) {
                memcpy(full + full_len, piece, (size_t)w);
                full_len += (size_t)w;
                full[full_len] = '\0';
            }

            /* Inside a call, the text is NOT content: it goes into the
             * `tool_calls` do quadro final, como o protocolo pede. */
            if (stream && !in_call) {
                json_escape(piece, (size_t)w, esc, sizeof(esc));
                const int n = snprintf(frame, sizeof(frame),
                    "data: {\"choices\":[{\"delta\":{\"content\":\"%s\"},"
                    "\"index\":0}]}\n\n", esc);
                if (tt_send_all(c, frame, (size_t)n) < 0) break;  /* aba fechou */
                streamed++;
            }
        }

        s->conv[s->n_conv++] = next;
        made++;

        /* `</tool_call>` closes the turn: waiting for EOS after it is burning
         * tokens at 13 tok/s to receive text the client throws away. */
        if (in_call && w > 0 && !strcmp(piece, "</tool_call>")) {
            done_call = 1;
            break;
        }
        if (tt_forward(s->state, next, NULL) != 0) break;
    }

    const double t_dec = tt_now() - t1;
    tt_log("  %lld tokens gerados, %.2f tok/s", (long long)made,
           made ? (double)made / t_dec : 0.0);

    /* `length` means the context is full, not an answer quota -- there is no
     * quota.  It is actionable information: the user needs a bigger `--ctx`. */
    const char *reason = done_call ? "tool_calls" : (hit_eos ? "stop" : "length");

    /* A copy of the RAW text before any trimming: just below, `full` loses the
     * trecho da chamada de ferramenta, e o log verboso quer justamente ele. */
    char *raw = NULL;
    if (g_verbose && full) {
        raw = (char *)malloc(full_len + 1);
        if (raw) { memcpy(raw, full, full_len); raw[full_len] = '\0'; }
    }

    char *calls = NULL;
    if (done_call && full) {
        calls = (char *)malloc(64u << 10);
        if (calls && parse_tool_calls(full, tools, calls, 64u << 10,
                                      s->served) == 0) {
            /* `</tool_call>` with no extractable call: the model emitted junk.
             * Better to return the text than an empty list the client treats
             * as "nothing to do", with the loop dying in silence. */
            free(calls);
            calls = NULL;
            reason = "stop";
        } else if (calls) {
            /* The call's text is not content: it already went to `tool_calls`.
             * Leaving it in `content` would make the client show the raw
             * markup and, worse, resend it all next turn. */
            char *cut = strstr(full, "<tool_call>");
            if (cut) {
                while (cut > full && (cut[-1] == '\n' || cut[-1] == ' ')) cut--;
                *cut = '\0';
                full_len = (size_t)(cut - full);
            }
        }
    }

    if (g_verbose) {
        const char *body = raw ? raw : "";
        const char *close = strstr(body, "</think>");
        const char *tc    = strstr(body, "<tool_call>");

        /* Sizes, not text: how much was draft and how much was answer is what
         * que se quer saber para julgar o custo de `think`. */
        if (close)
            vlog("  reasoning   %zu characters before </think>",
                 (size_t)(close - body));
        {
            const char *ans = close ? close + 8 : body;
            const size_t alen = tc && tc > ans ? (size_t)(tc - ans) : strlen(ans);
            vlog("  answer      %zu characters", alen);
        }

        /* Tool: the NAME and how many arguments.  The values stay out -- a
         * `Write` carries the whole file in its argument, and a `Bash` carries
         * the command, the most sensitive thing that passes through here. */
        for (const char *p = tc; p; p = strstr(p, "<tool_call>")) {
            p += 11;
            const char *end = strstr(p, "</tool_call>");
            if (!end) break;

            const char *ne = strstr(p, "<arg_key>");
            if (!ne || ne > end) ne = end;
            size_t nn = (size_t)(ne - p);
            while (nn && (p[nn - 1] == '\n' || p[nn - 1] == ' ')) nn--;

            int nargs = 0;
            size_t vbytes = 0;
            for (const char *k = strstr(p, "<arg_key>"); k && k < end;
                 k = strstr(k, "<arg_key>")) {
                const char *v  = strstr(k, "<arg_value>");
                const char *ve = v ? strstr(v, "</arg_value>") : NULL;
                if (!v || !ve || ve > end) break;
                nargs++;
                vbytes += (size_t)(ve - v - 11);
                k = ve;
            }
            vlog("  call        %.*s -- %d argument%s, %zu characters",
                 (int)nn, p, nargs, nargs == 1 ? "" : "s", vbytes);
            p = end + 12;
        }

        stat_log("generation", &st_gen);
        vlog("  end         %s -- %lld tokens in %.1f s (%.1f tok/s)%s",
             reason, (long long)made, t_dec, made ? (double)made / t_dec : 0.0,
             !strcmp(reason, "length") ? "  <- CONTEXTO CHEIO" : "");
    }

    char usage[256];
    snprintf(usage, sizeof(usage),
        "\"usage\":{\"prompt_tokens\":%lld,\"completion_tokens\":%lld,"
        "\"total_tokens\":%lld,\"prefill_cached\":%lld,\"context_left\":%lld,"
        "\"tokens_per_second\":%.2f}",
        (long long)(s->n_conv - made), (long long)made,
        (long long)s->n_conv, (long long)reused,
        (long long)(s->ctx - s->n_conv), made ? (double)made / t_dec : 0.0);

    if (stream) {
        const size_t cap = (calls ? strlen(calls) : 0) + 1024;
        char *tail = (char *)malloc(cap);
        if (tail) {
            const int n = snprintf(tail, cap,
                "data: {\"id\":\"chatcmpl-%llu\",\"object\":"
                "\"chat.completion.chunk\",\"created\":%lld,\"model\":\"%s\","
                "\"choices\":[{\"delta\":{%s%s%s},\"finish_reason\":\"%s\","
                "\"index\":0}],%s}\n\n"
                "data: [DONE]\n\n",
                (unsigned long long)s->served, (long long)tt_now(), TT_MODEL_ID,
                calls ? "\"tool_calls\":" : "", calls ? calls : "",
                calls ? "" : "", reason, usage);
            tt_send_all(c, tail, (size_t)n);
            /* What went out over the wire, as a count.  The lines above count
             * what the model produced; this one counts what the client
             * received -- and when a client renders nothing, the difference
             * between the two is the clue.  The body itself stays out: it is the message. */
            vlog("  returned    %d content frames + 1 final (%d bytes)%s",
                 streamed, n, calls ? " com tool_calls" : "");
            free(tail);
        }
    } else if (full) {
        /* Escaping can sextuple the size (`\u00XX` per control byte), so both
         * buffers are sized for the WORST case.  The previous version
         * dimensionava o corpo por `full_len * 2` e truncava respostas longas
         * em silencio -- que e o modo mais chato de errar. */
        const size_t esc_cap = full_len * 6 + 16;
        char *big = (char *)malloc(esc_cap);
        if (big) {
            const size_t esc_len = json_escape(full, full_len, big, esc_cap);
            const size_t body_cap = esc_len + (calls ? strlen(calls) : 0) + 640;
            char *body = (char *)malloc(body_cap);
            if (body) {
                const int n = snprintf(body, body_cap,
                    "{\"id\":\"chatcmpl-%llu\",\"object\":\"chat.completion\","
                    "\"created\":%lld,\"model\":\"%s\","
                    "\"choices\":[{\"message\":{\"role\":\"assistant\","
                    "\"content\":\"%s\"%s%s},\"finish_reason\":\"%s\","
                    "\"index\":0}],%s}",
                    (unsigned long long)s->served, (long long)tt_now(),
                    TT_MODEL_ID, big,
                    calls ? ",\"tool_calls\":" : "", calls ? calls : "",
                    reason, usage);
                send_status(c, 200, "OK", "application/json", body, (size_t)n);
                vlog("  returned    single response, %d bytes%s", n,
                     calls ? " com tool_calls" : "");
                free(body);
            }
            free(big);
        }
    }

    free(raw);
    free(calls);
    free(full);
    tt_arena_free(&dom);
    s->served++;
    return 0;
}

/* ------------------------------------------------------------- loop */

static void handle(server *s, const tt_socket *c, const char *peer)
{
    char *buf = (char *)malloc(REQ_MAX);
    if (!buf) return;

    request r;
    if (read_request(c, buf, REQ_MAX, &r) != 0) { free(buf); return; }

    /* The key protects what COSTS -- generating tokens and touching the cache
     * pagina.
     *
     * Requiring a key on `GET /` would make the UI unopenable: the browser has
     * como mandar `Authorization` ao digitar um endereco. A pagina e HTML
     * public, with no secret in it; it asks for the key and sends it on the
     * API. Proteger o HTML daria a sensacao de seguranca sem a coisa. */
    const int needs_key = s->key &&
        (!strcmp(r.path, "/v1/chat/completions") || !strcmp(r.path, "/reset"));

    if (needs_key) {
        const char *given = strstr(r.auth, "Bearer ");
        if (!given || strcmp(given + 7, s->key) != 0) {
            tt_log("  %s refused: invalid key at %s", peer, r.path);
            send_json_error(c, 401, "invalid key");
            free(buf);
            return;
        }
    }

    /* Routes everything that arrives -- `/stats` is left out because the web UI
     * polls it every 3 s and would drown the console. */
    if (g_verbose && strcmp(r.path, "/stats") != 0) {
        char ts[32];
        vlog("");
        vlog("[%s] %s  %s %s  (%zu bytes)", stamp(ts, sizeof(ts)), peer,
             r.method, r.path, r.body_len);
    }

    if (!strcmp(r.method, "OPTIONS")) {
        send_status(c, 204, "No Content", "text/plain", "", 0);
    } else if (!strcmp(r.method, "GET") &&
        (!strcmp(r.path, "/") || !strcmp(r.path, "/index.html"))) {
        send_status(c, 200, "OK", "text/html; charset=utf-8",
                    TT_WEBUI_HTML, sizeof(TT_WEBUI_HTML) - 1);
    } else if (!strcmp(r.method, "GET") && !strcmp(r.path, "/stats")) {
        /* What a developer needs to see without opening the server console:
         * where the memory is, how much context is left, and whether the
         * vectorised kernels really are active. */
        char body[768];
        const int n = snprintf(body, sizeof(body),
            "{\"version\":\"" TT_VERSION "\","
            "\"build\":\"%s\",\"kernel\":\"%s\",\"threads\":%d,"
            "\"attn\":\"%s\",\"ctx\":%lld,\"ctx_used\":%lld,"
            "\"weights\":%llu,\"kv_per_token\":%llu,\"kv_total\":%llu,"
            "\"rss_gb\":%.2f,\"budget_gb\":%.1f,\"ram_free_gb\":%.1f,"
            "\"served\":%llu,\"uptime\":%.0f}",
            tt_build_id(), tt_kernels_impl_name(), tt_pool_threads(s->pool),
            s->model->attn == TT_ATTN_ABSORBED ? "absorvida" : "expandida",
            (long long)s->ctx, (long long)s->state->len,
            (unsigned long long)tt_model_weight_bytes(s->model),
            (unsigned long long)tt_kv_bytes(s->model, 1),
            (unsigned long long)tt_kv_bytes(s->model, s->ctx),
            tt_rss_gb(), s->plan->budget_gb, tt_ram_avail_gb(),
            (unsigned long long)s->served, tt_now() - s->started);
        send_status(c, 200, "OK", "application/json", body, (size_t)n);
    } else if (!strcmp(r.method, "GET") && !strcmp(r.path, "/v1/models")) {
        char body[512];
        const int n = snprintf(body, sizeof(body),
            "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\","
            "\"owned_by\":\"tootega\"}]}", TT_MODEL_ID);
        send_status(c, 200, "OK", "application/json", body, (size_t)n);
    } else if (!strcmp(r.method, "POST") &&
               !strcmp(r.path, "/v1/chat/completions")) {
        handle_chat(s, c, &r);
    } else if (!strcmp(r.method, "POST") && !strcmp(r.path, "/reset")) {
        /* "Sem historico" e o comportamento pedido: esquecer e so mover o
         * the KV length back to zero. */
        tt_state_reset(s->state);
        s->n_conv = 0;
        send_status(c, 200, "OK", "application/json", "{\"ok\":true}", 11);
    } else {
        send_json_error(c, 404, "rota desconhecida");
    }

    free(buf);
}

/* --------------------------------------------------------------- CLI */

int tt_cmd_serve(int argc, char **argv)
{
    const char *path = NULL, *tok_path = NULL, *host = NULL, *key = NULL;
    int port = 8080, threads = 20, no_warm = 0;
    int64_t ctx = 8192;
    double ram_gb = 0;
    /* KV precision used to be hard-coded to f32 here: the option existed in
     * the engine and did not reach the server, which is exactly where the
     * cabe 202.752 posicoes -- o teto do modelo -- em 5,7 GB. */
    tt_kv_prec kv_prec = TT_KV_F32;

    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        const int has = (i + 1 < argc);
        if      (!strcmp(a, "--tokenizer") && has) tok_path = argv[++i];
        else if (!strcmp(a, "--port") && has)      port = atoi(argv[++i]);
        else if (!strcmp(a, "--host") && has)      host = argv[++i];
        else if (!strcmp(a, "--key") && has)       key = argv[++i];
        else if (!strcmp(a, "--ctx") && has)       ctx = atoll(argv[++i]);
        else if (!strcmp(a, "--threads") && has)   threads = atoi(argv[++i]);
        else if (!strcmp(a, "--ram-gb") && has)    ram_gb = atof(argv[++i]);
        else if (!strcmp(a, "--no-warm"))          no_warm = 1;
        else if (!strcmp(a, "--verbose") || !strcmp(a, "-v")) g_verbose = 1;
        else if ((!strcmp(a, "--kv-prec") || !strcmp(a, "--kv")) && has) {
            if (tt_kv_prec_parse(argv[++i], &kv_prec) != 0) return -1;
        }
        else if (a[0] == '-') return tt_fail("serve: opcao desconhecida '%s'", a);
        else if (!path) path = a;
        else return tt_fail("serve: extra argument '%s'", a);
    }

    if (!path || !tok_path)
        return tt_fail("uso: tootega serve <modelo.tootega> --tokenizer "
                       "<tokenizer.json> [--port 8080] [--ctx 8192] "
                       "[--kv-prec f32|bf16|int8] [--verbose] [--no-warm]");

    /* Expor na rede exige as duas coisas.  Um servidor de inferencia aberto e
     * remote access to the model and to this machine's disk; refusing here
     * costs one line and prevents the silent accident. */
    const int public_bind = host && strcmp(host, "127.0.0.1") != 0 &&
                            strcmp(host, "localhost") != 0;
    if (public_bind && !key)
        return tt_fail("--host %s exposes the server on the network. Pass "
                       "--key <key> as well; without it I do not start", host);

    tt_tokenizer tok;
    if (tt_tokenizer_load(&tok, tok_path, 154880) != 0) return -1;

    tt_model model;
    if (tt_model_open(&model, path, ctx, TT_ATTN_ABSORBED, kv_prec) != 0) {
        tt_tokenizer_free(&tok);
        return -1;
    }

    tt_plan plan;
    tt_plan_compute(&plan, &model, ctx, ram_gb);
    if (tt_plan_check(&plan, ctx) != 0) {
        tt_model_close(&model); tt_tokenizer_free(&tok);
        return -1;
    }

    tt_pool *pool = tt_pool_create(threads, TT_AFFINITY_ALL);
    tt_kernels_set_pool(pool);

    /* Warm up before accepting connections.  Without this the first user pays
     * for the 18 GB disk read spread through the decode loop -- 5x slower per
     * token -- and nothing on screen explains the wait. */
    if (!no_warm) {
        char wb[64];
        const double tw = tt_now();
        tt_log("  warming up: bringing the weights into RAM ...");
        const uint64_t touched = tt_model_warm(&model);
        tt_log("  %s in %.1f s (%.1f GB/s)",
               tt_bytes(wb, sizeof(wb), touched), tt_now() - tw,
               (double)touched / (tt_now() - tw) / 1e9);
    }

    tt_state state;
    if (!pool || tt_state_init(&state, &model, ctx) != 0) {
        if (pool) { tt_kernels_set_pool(NULL); tt_pool_destroy(pool); }
        tt_model_close(&model); tt_tokenizer_free(&tok);
        return -1;
    }

    server s;
    memset(&s, 0, sizeof(s));
    s.model = &model;
    s.state = &state;
    s.tok = &tok;
    s.pool = pool;
    s.plan = &plan;
    s.ctx = ctx;
    s.key = key;
    s.started = tt_now();
    s.conv = (int32_t *)malloc((size_t)ctx * sizeof(int32_t));

    const tt_json *ej = tt_json_get(model.c.config, "eos_token_id");
    if (ej && ej->type == TT_JSON_ARRAY)
        for (const tt_json *x = ej->child; x && s.n_eos < 8; x = x->next)
            s.eos[s.n_eos++] = (int32_t)tt_json_int(x, -1);

    if (tt_net_init() != 0 || !s.conv) {
        tt_log("could not start the network");
        return -1;
    }

    tt_socket srv;
    if (tt_listen(&srv, host, port, 16) != 0) {
        tt_net_shutdown();
        return tt_fail("could not listen on %s:%d (port in use?)",
                       host ? host : "127.0.0.1", port);
    }

    char b[64];
    tt_log("");
    tt_log("  tootega serve  --  http://%s:%d", host ? host : "127.0.0.1", port);
    tt_log("  model    %s", path);
    tt_log("  memory   %s of %.1f GB of budget",
           tt_bytes(b, sizeof(b), plan.total), plan.budget_gb);
    tt_log("  context  %lld positions, %d threads, %s",
           (long long)ctx, tt_pool_threads(pool), tt_kernels_impl_name());
    tt_log("  network  %s%s", public_bind ? "EXPOSED on the network" : "loopback only",
           key ? ", with a key" : "");
    tt_log("  one request at a time; the conversation does not survive the process");
    tt_log("");

    for (;;) {
        tt_socket cli;
        char peer[64] = "?";
        if (tt_accept(&srv, &cli, peer, sizeof(peer)) != 0) continue;

        tt_socket_nodelay(&cli);
        tt_socket_timeout(&cli, 30);
        handle(&s, &cli, peer);
        tt_socket_close(&cli);
    }
}
