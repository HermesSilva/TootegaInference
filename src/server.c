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
    const int n = snprintf(head, sizeof(head),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Cache-Control: no-store\r\n"
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

/* Monta o prefixo da conversa a partir das mensagens do pedido.
 *
 * A UI manda a conversa inteira a cada turno (e o que um cliente OpenAI faz).
 * Reconstruimos os ids do zero e deixamos o casamento de prefixo decidir o que
 * recomputar -- e por isso que o segundo turno custa so os tokens novos. */
static int build_conversation(server *s, const tt_json *messages)
{
    s->n_conv = 0;
    if (append_text(s, CHAT_PREFIX) < 0) return -1;

    for (const tt_json *m = messages->child; m; m = m->next) {
        const tt_json *role = tt_json_get(m, "role");
        const tt_json *content = tt_json_get(m, "content");
        size_t clen = 0;
        const char *ctext = tt_json_str(content, &clen);
        if (!ctext) continue;

        const int assistant = tt_json_str_eq(role, "assistant");
        const char *marker = CHAT_USER;
        if (tt_json_str_eq(role, "system")) marker = CHAT_SYSTEM;
        else if (assistant)                 marker = CHAT_ASSIST;

        if (append_text(s, marker) < 0) return -1;

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
    }

    if (append_text(s, CHAT_ASSIST) < 0) return -1;
    if (append_text(s, CHAT_THINK) < 0) return -1;
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

    const int stream = tt_json_bool(tt_json_get(root, "stream"), 1);
    const float temp = (float)tt_json_real(tt_json_get(root, "temperature"), 0.8);
    const float top_p = (float)tt_json_real(tt_json_get(root, "top_p"), 0.95);
    const int   top_k = (int)tt_json_int(tt_json_get(root, "top_k"), 40);
    /* Sem `max_tokens`, gera ate o EOS.  O unico teto e o contexto -- cortar
     * por contagem interrompe o modelo no meio do raciocinio e entrega um
     * rascunho como se fosse resposta. */
    int64_t max_new = tt_json_int(tt_json_get(root, "max_tokens"), 0);

    if (build_conversation(s, messages) != 0) {
        tt_arena_free(&dom);
        send_json_error(c, 400, "conversa nao cabe no contexto");
        return 0;
    }
    tt_arena_free(&dom);

    if (s->n_conv + 8 >= s->ctx) {
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
    char *full = stream ? NULL : (char *)malloc(1u << 20);
    size_t full_len = 0;

    int64_t made = 0;
    int hit_eos = 0;
    const double t1 = tt_now();

    while (made < max_new && s->n_conv < s->ctx) {
        const int32_t next = tt_sample(&sampler, s->state->logits, s->tok->n_tokens);
        if (is_eos(s, next)) { hit_eos = 1; break; }

        const int64_t w = tt_decode(s->tok, &next, 1, piece, sizeof(piece) - 1);
        if (w > 0) {
            piece[w] = '\0';
            json_escape(piece, (size_t)w, esc, sizeof(esc));

            if (stream) {
                const int n = snprintf(frame, sizeof(frame),
                    "data: {\"choices\":[{\"delta\":{\"content\":\"%s\"},"
                    "\"index\":0}]}\n\n", esc);
                if (tt_send_all(c, frame, (size_t)n) < 0) break;  /* aba fechou */
            } else if (full && full_len + (size_t)w + 1 < (1u << 20)) {
                memcpy(full + full_len, piece, (size_t)w);
                full_len += (size_t)w;
            }
        }

        s->conv[s->n_conv++] = next;
        made++;
        if (tt_forward(s->state, next, NULL) != 0) break;
    }

    const double t_dec = tt_now() - t1;
    tt_log("  %lld tokens gerados, %.2f tok/s", (long long)made,
           made ? (double)made / t_dec : 0.0);

    if (stream) {
        char tail[256];
        /* `length` significa contexto cheio, nao cota de resposta -- nao ha
         * cota. E informacao acionavel: o usuario precisa de `--ctx` maior. */
        const int n = snprintf(tail, sizeof(tail),
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"%s\","
            "\"index\":0}],\"usage\":{\"prompt_tokens\":%lld,"
            "\"completion_tokens\":%lld,\"prefill_cached\":%lld,"
            "\"context_left\":%lld,\"tokens_per_second\":%.2f}}\n\n"
            "data: [DONE]\n\n",
            hit_eos ? "stop" : "length",
            (long long)s->n_conv - made, (long long)made, (long long)reused,
            (long long)(s->ctx - s->n_conv),
            made ? (double)made / t_dec : 0.0);
        tt_send_all(c, tail, (size_t)n);
    } else if (full) {
        full[full_len] = '\0';

        /* Escape pode sextuplicar (`\u00XX` por byte de controle), entao os
         * dois buffers saem do tamanho do PIOR caso.  A versao anterior
         * dimensionava o corpo por `full_len * 2` e truncava respostas longas
         * em silencio -- que e o modo mais chato de errar. */
        const size_t esc_cap = full_len * 6 + 16;
        char *big = (char *)malloc(esc_cap);
        if (big) {
            const size_t esc_len = json_escape(full, full_len, big, esc_cap);
            const size_t body_cap = esc_len + 512;
            char *body = (char *)malloc(body_cap);
            if (body) {
                const int n = snprintf(body, body_cap,
                    "{\"choices\":[{\"message\":{\"role\":\"assistant\","
                    "\"content\":\"%s\"},\"finish_reason\":\"stop\",\"index\":0}],"
                    "\"usage\":{\"completion_tokens\":%lld,"
                    "\"tokens_per_second\":%.2f}}",
                    big, (long long)made, made ? (double)made / t_dec : 0.0);
                send_status(c, 200, "OK", "application/json", body, (size_t)n);
                free(body);
            }
            free(big);
        }
        free(full);
    }

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

    if (!strcmp(r.method, "GET") &&
        (!strcmp(r.path, "/") || !strcmp(r.path, "/index.html"))) {
        send_status(c, 200, "OK", "text/html; charset=utf-8",
                    TT_WEBUI_HTML, sizeof(TT_WEBUI_HTML) - 1);
    } else if (!strcmp(r.method, "GET") && !strcmp(r.path, "/v1/models")) {
        char body[512];
        const int n = snprintf(body, sizeof(body),
            "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\","
            "\"owned_by\":\"tootega\"}]}", "glm-4.7-flash");
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
    double ram_gb = 0;

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
