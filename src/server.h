/* server.h -- HTTP e a UI web, dentro do proprio engine.
 *
 * Nao e um executavel a parte de proposito.  O estado que um chat precisa --
 * KV cache, tokenizer, sampler, casamento de prefixo -- ja vive aqui; um
 * processo separado teria de falar com este por IPC ou carregar os 19 GB de
 * novo.  E a pagina vai EMBUTIDA no binario (`webui.h`), porque docs/10
 * promete um executavel que roda sem instalar nada.
 *
 * Concorrencia: um pedido por vez.  O laco de accept e sequencial e o modelo
 * e um so; slots e batching continuo sao o M3 e tem criterio de saida proprio
 * (docs/05). Ate la, uma segunda aba espera -- o que e honesto e visivel.
 *
 * Rede: loopback por default.  Expor na rede exige `--host` explicito E uma
 * chave, porque o servidor da acesso ao modelo e ao disco da maquina.
 */
#ifndef TOOTEGA_SERVER_H
#define TOOTEGA_SERVER_H

int tt_cmd_serve(int argc, char **argv);

#endif /* TOOTEGA_SERVER_H */
