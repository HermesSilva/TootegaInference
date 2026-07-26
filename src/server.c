#include "server.h"

#include "budget.h"
#include "common.h"
#include "compat.h"
#include "forward.h"
#include "offload.h"
#include "json.h"
#include "kernels.h"
#include "model.h"
#include "pool.h"
#include "sample.h"
#include "tokenizer.h"
#include "webui.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQ_MAX     (1u << 20)     /* 1 MB de requisicao: prompt grande cabe */
#define OUT_CHUNK   4096

/* Marcadores do template do modelo -- conferidos byte a byte contra
 * `apply_chat_template` (ver cmd_chat em main.c). */
#define CHAT_PREFIX "[gMASK]<sop>"
#define CHAT_SYSTEM "<|system|>"
#define CHAT_USER   "<|user|>"
#define CHAT_ASSIST "<|assistant|>"
#define CHAT_THINK  "<think>"

/* Id do modelo no protocolo.  Um so lugar: `/v1/models` e as respostas tem de
 * concordar, senao cliente que valida o campo rejeita a resposta. */
#define TT_MODEL_ID "glm-4.7-flash"

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

    const char *key;         /* NULL quando so no loopback */
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
    /* Cabecalhos HTTP sao case-insensitive; comparamos assim. */
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

/* Le a requisicao inteira.  Devolve 0 em sucesso. */
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

    /* O corpo pode ter vindo junto com o cabecalho ou nao. */
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
    /* CORS aberto porque o servidor e local e ja tem a chave para o que custa.
     * Sem isto, cliente que roda DENTRO de um navegador -- e o Copilot do VS
     * Code e um -- nao consegue nem o preflight. */
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

/* Anexa bytes com corte no limite.  Devolve o novo comprimento. */
static size_t jw_put(char *dst, size_t cap, size_t w, const char *s, size_t n)
{
    for (size_t i = 0; i < n && w + 1 < cap; i++) dst[w++] = s[i];
    return w;
}

/* Serializa um no do DOM de volta para JSON.
 *
 * Necessario porque o template do modelo despeja cada ferramenta como JSON
 * dentro do prompt (`{{ tool | tojson }}`), e o nosso DOM e zero-copia: guarda
 * ponteiros para os pedacos, nao o texto original do objeto. */
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
 * Fica no inicio do prefixo de proposito: e a parte que NAO muda entre turnos,
 * e a regra de ordenacao de docs/03 (estavel primeiro) e o que faz o cache de
 * prefixo valer alguma coisa numa sessao de agente. */
static int append_tools(server *s, const tt_json *tools)
{
    if (append_text(s, CHAT_SYSTEM "\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> "
        "XML tags:\n<tools>\n") < 0) return -1;

    char *one = (char *)malloc(64u << 10);
    if (!one) return tt_fail("sem memoria");
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

/* Chamadas que o assistente ja fez, no formato que o modelo emitiu.
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

                /* String vai crua; o resto vai como JSON -- e o que o template
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
 * A UI manda a conversa inteira a cada turno (e o que um cliente OpenAI faz).
 * Reconstruimos os ids do zero e deixamos o casamento de prefixo decidir o que
 * recomputar -- e por isso que o segundo turno custa so os tokens novos. */
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
         * NAO e do modelo nem do usuario -- e o mundo respondendo. */
        if (tt_json_str_eq(role, "tool")) {
            if (!ctext) continue;
            char *tmp = (char *)malloc(clen + 1);
            if (!tmp) return tt_fail("sem memoria");
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

        /* Turno de assistente que so chamou ferramenta nao tem texto. */
        if (!ctext) {
            if (append_text(s, "</think>") < 0) return -1;
            if (append_tool_calls(s, calls, dom) != 0) return -1;
            continue;
        }

        /* `tt_encode` precisa de terminador; o DOM nao tem. */
        char *tmp = (char *)malloc(clen + 1);
        if (!tmp) return tt_fail("sem memoria");
        memcpy(tmp, ctext, clen);
        tmp[clen] = '\0';

        /* Turno de assistente JA CONCLUIDO entra como `</think>` mais a
         * resposta, sem o raciocinio -- e o que `chat_template.jinja` faz, e
         * so o turno aberto ganha `<think>`.
         *
         * Fica no servidor e nao no cliente porque o template e do modelo: um
         * cliente OpenAI qualquer manda `content` puro e tem de funcionar. E
         * cortar o raciocinio do historico e o que impede o contexto de
         * inchar a cada volta. */
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

        /* Texto E chamadas no mesmo turno: o modelo explica e depois chama. */
        if (calls && append_tool_calls(s, calls, dom) != 0) return -1;
    }

    if (append_text(s, CHAT_ASSIST) < 0) return -1;

    /* `<think>` abre o rascunho; `</think>` o fecha antes de comecar, e o
     * modelo vai direto a resposta.
     *
     * Nao e truque: e a mesma construcao que o template usa para turnos de
     * assistente JA concluidos, entao o modelo a viu no treino. O custo esta
     * em outro lugar -- ver docs/20. */
    if (append_text(s, think ? CHAT_THINK : "</think>") < 0) return -1;
    return 0;
}

static int is_eos(const server *s, int32_t id)
{
    for (int i = 0; i < s->n_eos; i++) if (s->eos[i] == id) return 1;
    return 0;
}

/* Escapa uma string para dentro de JSON. */
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

/* --------------------------------------- saida do modelo -> tool_calls */

/* Tipo declarado do argumento no schema que o CLIENTE mandou.
 *
 * O modelo emite `<arg_value>3</arg_value>` sem dizer se e o numero 3 ou a
 * string "3"; OpenAI exige `arguments` como JSON tipado.  Sem consultar o
 * schema, toda ferramenta com argumento numerico ou booleano chega errada do
 * outro lado. */
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

/* Um `<arg_value>` cru vira JSON.  `kind` decide; `?` (schema silencioso)
 * aceita o valor cru so quando ele JA e JSON valido. */
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

/* Extrai todas as chamadas do texto gerado e monta o array `tool_calls` da
 * OpenAI.  Devolve quantas achou. */
static int parse_tool_calls(const char *text, const tt_json *tools,
                            char *dst, size_t cap)
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

        char one[17408];
        const int n = snprintf(one, sizeof(one),
            "%s{\"id\":\"call_%d\",\"type\":\"function\","
            "\"function\":{\"name\":\"%s\",\"arguments\":\"",
            found ? "," : "", found, fname);
        w = jw_put(dst, cap, w, one, (size_t)n);
        /* `arguments` e STRING no protocolo -- o JSON vai escapado dentro. */
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
    /* `think` e nosso; `chat_template_kwargs.enable_thinking` e o que vLLM e
     * SGLang popularizaram e o que um cliente de terceiro manda. */
    const int think = tt_json_bool(tt_json_get(root, "think"),
        tt_json_bool(tt_json_get(tt_json_get(root, "chat_template_kwargs"),
                                 "enable_thinking"), 1));
    const float temp = (float)tt_json_real(tt_json_get(root, "temperature"), 0.8);
    const float top_p = (float)tt_json_real(tt_json_get(root, "top_p"), 0.95);
    const int   top_k = (int)tt_json_int(tt_json_get(root, "top_k"), 40);
    /* Sem `max_tokens`, gera ate o EOS.  O unico teto e o contexto -- cortar
     * por contagem interrompe o modelo no meio do raciocinio e entrega um
     * rascunho como se fosse resposta. */
    int64_t max_new = tt_json_int(tt_json_get(root, "max_tokens"), 0);

    /* A arena do pedido sobrevive ate o fim: `tools` e consultado depois da
     * geracao, para tipar os argumentos da chamada. */
    if (build_conversation(s, messages, think, tools, &dom) != 0 ||
        s->n_conv + 8 >= s->ctx) {
        tt_arena_free(&dom);
        send_json_error(c, 400, "conversa nao cabe no contexto");
        return 0;
    }
    const int64_t room = s->ctx - s->n_conv - 1;
    if (max_new <= 0 || max_new > room) max_new = room;

    /* Casamento de prefixo: o que ja esta no KV nao volta a ser computado.
     * Em conversa continuada, so os tokens novos entram no prefill. */
    const int64_t reused = tt_state_match_prefix(s->state, s->conv, s->n_conv);

    const double t0 = tt_now();
    for (int64_t i = reused; i < s->n_conv; i++)
        if (tt_forward(s->state, s->conv[i], NULL) != 0) {
            tt_arena_free(&dom);
            send_json_error(c, 500, tt_error());
            return 0;
        }
    const double t_prefill = tt_now() - t0;

    tt_log("  %lld tokens (%lld do cache) em %.2f s", (long long)s->n_conv,
           (long long)reused, t_prefill);

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
    /* O texto inteiro e acumulado mesmo em streaming: as chamadas de
     * ferramenta so podem ser extraidas do conjunto, e nao de um pedaco. */
    const size_t full_cap = 1u << 20;
    char *full = (char *)malloc(full_cap);
    size_t full_len = 0;
    if (full) full[0] = '\0';

    int64_t made = 0;
    int hit_eos = 0, in_call = 0, done_call = 0;
    const double t1 = tt_now();

    while (made < max_new && s->n_conv < s->ctx) {
        const int32_t next = tt_sample(&sampler, s->state->logits, s->tok->n_tokens);
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

            /* Dentro de uma chamada, o texto NAO e conteudo: vai no
             * `tool_calls` do quadro final, como o protocolo pede. */
            if (stream && !in_call) {
                json_escape(piece, (size_t)w, esc, sizeof(esc));
                const int n = snprintf(frame, sizeof(frame),
                    "data: {\"choices\":[{\"delta\":{\"content\":\"%s\"},"
                    "\"index\":0}]}\n\n", esc);
                if (tt_send_all(c, frame, (size_t)n) < 0) break;  /* aba fechou */
            }
        }

        s->conv[s->n_conv++] = next;
        made++;

        /* `</tool_call>` fecha o turno: esperar o EOS depois disso e queimar
         * tokens a 13 tok/s para receber texto que o cliente descarta. */
        if (in_call && w > 0 && !strcmp(piece, "</tool_call>")) {
            done_call = 1;
            break;
        }
        if (tt_forward(s->state, next, NULL) != 0) break;
    }

    const double t_dec = tt_now() - t1;
    tt_log("  %lld tokens gerados, %.2f tok/s", (long long)made,
           made ? (double)made / t_dec : 0.0);

    /* `length` significa contexto cheio, nao cota de resposta -- nao ha cota.
     * E informacao acionavel: o usuario precisa de `--ctx` maior. */
    const char *reason = done_call ? "tool_calls" : (hit_eos ? "stop" : "length");

    char *calls = NULL;
    if (done_call && full) {
        calls = (char *)malloc(64u << 10);
        if (calls && parse_tool_calls(full, tools, calls, 64u << 10) == 0) {
            /* `</tool_call>` sem chamada extraivel: o modelo emitiu lixo.
             * Melhor devolver o texto do que uma lista vazia que o cliente
             * trata como "nada a fazer" e o laco morre em silencio. */
            free(calls);
            calls = NULL;
            reason = "stop";
        } else if (calls) {
            /* O texto da chamada nao e conteudo: ja foi para `tool_calls`.
             * Deixa-lo em `content` faria o cliente exibir a marcacao crua e,
             * pior, reenviar tudo no proximo turno. */
            char *cut = strstr(full, "<tool_call>");
            if (cut) {
                while (cut > full && (cut[-1] == '\n' || cut[-1] == ' ')) cut--;
                *cut = '\0';
                full_len = (size_t)(cut - full);
            }
        }
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
            free(tail);
        }
    } else if (full) {
        /* Escape pode sextuplicar (`\u00XX` por byte de controle), entao os
         * dois buffers saem do tamanho do PIOR caso.  A versao anterior
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
                free(body);
            }
            free(big);
        }
    }

    free(calls);
    free(full);
    tt_arena_free(&dom);
    s->served++;
    return 0;
}

/* ------------------------------------------------------------- laco */

static void handle(server *s, const tt_socket *c, const char *peer)
{
    char *buf = (char *)malloc(REQ_MAX);
    if (!buf) return;

    request r;
    if (read_request(c, buf, REQ_MAX, &r) != 0) { free(buf); return; }

    /* A chave protege o que CUSTA -- gerar tokens e mexer no cache --, nao a
     * pagina.
     *
     * Exigir chave no `GET /` tornaria a UI inabrivel: o navegador nao tem
     * como mandar `Authorization` ao digitar um endereco. A pagina e HTML
     * publico, sem segredo nenhum; ela pede a chave e a envia nas chamadas de
     * API. Proteger o HTML daria a sensacao de seguranca sem a coisa. */
    const int needs_key = s->key &&
        (!strcmp(r.path, "/v1/chat/completions") || !strcmp(r.path, "/reset"));

    if (needs_key) {
        const char *given = strstr(r.auth, "Bearer ");
        if (!given || strcmp(given + 7, s->key) != 0) {
            tt_log("  %s recusado: chave invalida em %s", peer, r.path);
            send_json_error(c, 401, "chave invalida");
            free(buf);
            return;
        }
    }

    if (!strcmp(r.method, "OPTIONS")) {
        send_status(c, 204, "No Content", "text/plain", "", 0);
    } else if (!strcmp(r.method, "GET") &&
        (!strcmp(r.path, "/") || !strcmp(r.path, "/index.html"))) {
        send_status(c, 200, "OK", "text/html; charset=utf-8",
                    TT_WEBUI_HTML, sizeof(TT_WEBUI_HTML) - 1);
    } else if (!strcmp(r.method, "GET") && !strcmp(r.path, "/stats")) {
        /* O que um desenvolvedor precisa ver sem abrir o console do servidor:
         * onde a memoria esta, quanto do contexto sobrou, e se os kernels
         * vetorizados estao mesmo ativos. */
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
         * comprimento do KV de volta para zero. */
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
    double ram_gb = 0, vram_gb = 0;
    const char *gpu_dll = NULL;
    tt_offload_mode gpu_mode = TT_OFFLOAD_MOE;

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
        else if (!strcmp(a, "--vram-gb") && has)   vram_gb = atof(argv[++i]);
        else if (!strcmp(a, "--gpu"))
            /* `--gpu` sozinho usa o nome padrao; com argumento, o caminho da
             * DLL.  O proximo token so e consumido se nao for outra opcao. */
            gpu_dll = (has && argv[i + 1][0] != '-') ? argv[++i]
                                                     : "tootega_cuda.dll";
        /* `--gpu-all` inclui atencao e lm_head, que sao chamados um a um e
         * pagam o protocolo inteiro em cada matmul.  Fica como opcao porque em
         * placa com mais VRAM a conta muda; o default e so MoE. */
        else if (!strcmp(a, "--gpu-all")) {
            gpu_mode = TT_OFFLOAD_ALL;
            if (!gpu_dll) gpu_dll = "tootega_cuda.dll";
        }
        else if (a[0] == '-') return tt_fail("serve: opcao desconhecida '%s'", a);
        else if (!path) path = a;
        else return tt_fail("serve: argumento extra '%s'", a);
    }

    if (!path || !tok_path)
        return tt_fail("uso: tootega serve <modelo.tootega> --tokenizer "
                       "<tokenizer.json> [--port 8080] [--ctx 8192]");

    /* Expor na rede exige as duas coisas.  Um servidor de inferencia aberto e
     * acesso remoto ao modelo e ao disco desta maquina; recusar aqui custa uma
     * linha e evita o acidente silencioso. */
    const int public_bind = host && strcmp(host, "127.0.0.1") != 0 &&
                            strcmp(host, "localhost") != 0;
    if (public_bind && !key)
        return tt_fail("--host %s expoe o servidor na rede. Passe tambem "
                       "--key <chave>; sem ela eu nao subo", host);

    tt_tokenizer tok;
    if (tt_tokenizer_load(&tok, tok_path, 154880) != 0) return -1;

    tt_model model;
    if (tt_model_open(&model, path, ctx, TT_ATTN_ABSORBED, TT_KV_F32) != 0) {
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

    /* Aquece antes de aceitar conexao. Sem isto o primeiro usuario paga a
     * leitura de 18 GB do disco espalhada pelo laco de decode -- 5x mais lento
     * por token -- e nao ha nada na tela que explique a espera. */
    /* GPU antes do aquecimento: o que subir para a VRAM nao precisa ser
     * tocado na RAM depois. */
    tt_gpu_plan gplan;
    if (gpu_dll) {
        tt_gpu_place(&model.c, gpu_dll, vram_gb, gpu_mode, &gplan);
        tt_gpu_print(&gplan);
    }

    if (!no_warm) {
        char wb[64];
        const double tw = tt_now();
        tt_log("  aquecendo: trazendo os pesos para a RAM ...");
        const uint64_t touched = tt_model_warm(&model);
        tt_log("  %s em %.1f s (%.1f GB/s)",
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
        tt_log("nao consegui iniciar a rede");
        return -1;
    }

    tt_socket srv;
    if (tt_listen(&srv, host, port, 16) != 0) {
        tt_net_shutdown();
        return tt_fail("nao consegui escutar em %s:%d (porta ocupada?)",
                       host ? host : "127.0.0.1", port);
    }

    char b[64];
    tt_log("");
    tt_log("  tootega serve  --  http://%s:%d", host ? host : "127.0.0.1", port);
    tt_log("  modelo   %s", path);
    tt_log("  memoria  %s de %.1f GB de orcamento",
           tt_bytes(b, sizeof(b), plan.total), plan.budget_gb);
    tt_log("  contexto %lld posicoes, %d threads, %s",
           (long long)ctx, tt_pool_threads(pool), tt_kernels_impl_name());
    tt_log("  rede     %s%s", public_bind ? "EXPOSTO na rede" : "so loopback",
           key ? ", com chave" : "");
    tt_log("  um pedido por vez; a conversa nao sobrevive ao processo");
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
