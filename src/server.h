/* server.h -- HTTP and the web UI, inside the engine itself.
 *
 * It is deliberately not a separate executable.  The state a chat needs -- KV
 * cache, tokenizer, sampler, prefix matching -- already lives here; a separate
 * process would have to talk to this one over IPC or load the 19 GB again.  And
 * the page is EMBEDDED in the binary (`webui.h`), because docs/10 promises an
 * executable that runs without installing anything.
 *
 * Concurrency: one request at a time.  The accept loop is sequential and there
 * is a single model; slots and continuous batching are M3 and have their own
 * exit criterion (docs/05).  Until then a second tab waits -- which is honest
 * and visible.
 *
 * Network: loopback by default.  Exposing it on the network requires an explicit
 * `--host` AND a key, because the server grants access to the model and to the
 * machine's disk.
 */
#ifndef TOOTEGA_SERVER_H
#define TOOTEGA_SERVER_H

int tt_cmd_serve(int argc, char **argv);

#endif /* TOOTEGA_SERVER_H */
