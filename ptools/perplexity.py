#!/usr/bin/env python3
"""Perplexidade e ablação de precisão -- os dois itens que faltavam do M0.

O critério de saída do M0 (docs/05) pede "perplexidade do int4 medida contra o
BF16 num conjunto fixo, com o custo documentado", e uma ablação por tensor
porque "a tabela de docs/04 é hipótese".

O ponto delicado é não ter duas implementações do forward: se a referência BF16
usasse um código e o container outro, a diferença medida seria a soma do erro de
quantização com o erro de quem escreveu duas vezes.  Então este script REUSA
`oracle.forward` e só troca a fonte dos pesos -- `Container` (int4) ou
`Safetensors` (BF16), ambas com a mesma interface `.config/.get/.row`.

    python tools/perplexity.py                       # int4 vs BF16
    python tools/perplexity.py --ablate              # + de onde vem o erro
    python tools/perplexity.py --tokens 128          # mais rápido

Memória: nunca há um modelo inteiro carregado.  As duas fontes desquantizam
tensor a tensor sob demanda, então o pico é um expert (~38 MB).
"""

import argparse
import json
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle import Container, forward          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INT4 = ROOT / "models" / "glm47flash-int4" / "model.tootega"
DEFAULT_BF16 = ROOT / "models" / "glm47flash-bf16"

# Conjunto fixo.  Prosa técnica em português e inglês mais um trecho de código:
# é o regime em que o modelo vai ser usado (docs/04, "uso agêntico"), e não um
# corpus genérico que mediria outra coisa.
CORPUS = """A engine de inferência lê o modelo de um container mapeado em
memória e nunca copia peso nenhum: os tensores são lidos no lugar, alinhados a
64 bytes, e os kernels multiplicam direto do mapeamento. O gargalo de um MoE
residente não é compute, é banda de memória — por token são lidos cerca de dois
gigabytes de peso contra sete gigaflops de conta, o que deixa a utilização de
compute perto de vinte por cento.

The attention layer uses a compressed key-value cache. Instead of storing the
full expanded keys and values for every position, it keeps the low-rank
projection and folds the up-projection matrix into the query. This is
algebraically equivalent and cuts the cache traffic by a factor of eighteen,
which is what decides throughput once the context grows past a few thousand
tokens.

def forward(state, token, position):
    hidden = embed_row(token)
    for layer in range(num_layers):
        normed = rmsnorm(hidden, layer.input_norm)
        hidden = hidden + attention(state, normed, layer, position)
        normed = rmsnorm(hidden, layer.post_norm)
        hidden = hidden + mixture_of_experts(normed, layer)
    return project(rmsnorm(hidden, final_norm))

Cada decisão de precisão custa alguma coisa e compra alguma coisa. Quantizar os
pesos em quatro bits corta o modelo de sessenta e dois para dezenove gigabytes,
e o preço é um erro relativo de onze por cento na saída de cada matmul. Se esse
preço é aceitável, nenhuma tabela de erro por tensor responde: só perplexidade
ou um teste token a token contra o oráculo respondem."""


# ------------------------------------------------------------ safetensors

class Safetensors:
    """Fonte BF16, mesma interface do `Container` de oracle.py."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.config = json.loads((folder / "config.json").read_text(encoding="utf-8"))

        index = json.loads((folder / "model.safetensors.index.json")
                           .read_text(encoding="utf-8"))
        self.weight_map = index["weight_map"]

        self._shards = {}
        self._headers = {}

    def _shard(self, name):
        shard = self.weight_map.get(name)
        if shard is None:
            sys.exit(f"tensor ausente no checkpoint BF16: {name}")
        if shard not in self._shards:
            path = self.folder / shard
            with open(path, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                self._headers[shard] = (json.loads(f.read(hlen)), 8 + hlen)
            self._shards[shard] = np.memmap(path, dtype=np.uint8, mode="r")
        return shard

    def get(self, name) -> np.ndarray:
        shard = self._shard(name)
        header, base = self._headers[shard]
        info = header[name]
        start, end = info["data_offsets"]
        raw = self._shards[shard][base + start: base + end]

        if info["dtype"] == "BF16":
            u = raw.view(np.uint16).astype(np.uint32)
            return (u << 16).view(np.float32).reshape(info["shape"])
        if info["dtype"] == "F32":
            return raw.view(np.float32).reshape(info["shape"]).copy()
        sys.exit(f"dtype {info['dtype']} nao suportado em {name}")

    def row(self, name, index) -> np.ndarray:
        shard = self._shard(name)
        header, base = self._headers[shard]
        info = header[name]
        cols = info["shape"][-1]
        start = base + info["data_offsets"][0] + index * cols * 2
        u = self._shards[shard][start: start + cols * 2].view(np.uint16).astype(np.uint32)
        return (u << 16).view(np.float32)


class Hybrid:
    """Pega os tensores que casam com `pattern` da fonte B, o resto da A.

    É a ablação: trocar UM grupo de tensores pela versão exata e ver quanto da
    degradação vem dele.  Sem isto, "a quantização custou X" não diz onde."""

    def __init__(self, base, exact, pattern):
        self.base = base
        self.exact = exact
        self.rx = re.compile(pattern)
        self.config = base.config
        self.hits = set()

    def _pick(self, name):
        if self.rx.search(name):
            self.hits.add(name)
            return self.exact
        return self.base

    def get(self, name):
        return self._pick(name).get(name)

    def row(self, name, index):
        return self._pick(name).row(name, index)


# --------------------------------------------------------------- medida

def nll(logits, ids):
    """Log-verossimilhança negativa média, em float64.

    Feita em float64 de propósito: aqui o objetivo é MEDIR a diferença entre
    duas configurações, não reproduzir o que o engine faz -- somar 154.880
    exponenciais em float32 introduziria ruído da mesma ordem do efeito."""
    x = logits.astype(np.float64)
    x -= x.max(axis=-1, keepdims=True)
    logZ = np.log(np.exp(x).sum(axis=-1))

    total, n = 0.0, 0
    for i in range(len(ids) - 1):
        total += logZ[i] - x[i, ids[i + 1]]
        n += 1
    return total / n


def run(name, src, ids, verbose=False):
    t0 = time.time()
    logits, _, _ = forward(src, ids, verbose)
    loss = nll(logits, ids)
    dt = time.time() - t0

    top1 = int((logits[:-1].argmax(-1) == np.array(ids[1:])).sum())
    print(f"  {name:<28} nll {loss:7.4f}   ppl {np.exp(loss):9.3f}   "
          f"top-1 {top1:3d}/{len(ids) - 1}   {dt:6.1f} s", flush=True)
    return loss, logits


def run_engine(exe, model, ids, workdir):
    """Logits do engine em C, via `forward --dump-logits`.

    Duas razões para preferir isto ao caminho numpy no lado int4: é ~7x mais
    rápido (30 s contra 190 s por janela), e mede o ARTEFATO QUE RODA em vez de
    uma reimplementação dele.  O oráculo já provou que os dois concordam em
    32/32 tokens; aqui a pergunta é sobre qualidade, não sobre equivalência."""
    import subprocess

    ids_file = workdir / "_ppl_ids.txt"
    bin_file = workdir / "_ppl_logits.bin"
    ids_file.write_text(",".join(str(i) for i in ids), encoding="ascii")

    t0 = time.time()
    r = subprocess.run(
        [str(exe), "forward", str(model), "--ids-file", str(ids_file),
         "--ctx", str(len(ids)), "--threads", "20", "--affinity", "all",
         "--top", "1", "--dump-logits", str(bin_file)],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"engine falhou:\n{r.stdout}\n{r.stderr}")

    raw = np.fromfile(bin_file, dtype=np.float32)
    logits = raw.reshape(len(ids), -1)
    return logits, time.time() - t0


def corpus_from_docs() -> str:
    """Os próprios documentos do projeto como conjunto fixo.

    Reprodutível, offline, versionado junto com o engine, e no domínio em que o
    modelo vai ser usado -- prosa técnica em português com trechos de código.
    Um corpus genérico mediria outra coisa."""
    parts = []
    for path in sorted((ROOT / "docs").glob("*.md")):
        parts.append(path.read_text(encoding="utf-8"))
    parts.append((ROOT / "README.md").read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def paired(name_a, loss_a, name_b, loss_b):
    """Diferença PAREADA por janela.

    Comparar duas médias globais joga fora a informação de que as duas
    configurações viram exatamente o mesmo texto: a variância entre janelas
    (uma é mais previsível que a outra) domina o efeito que se quer medir.  A
    diferença por janela cancela isso, e é ela que tem erro padrão pequeno o
    bastante para decidir alguma coisa."""
    d = np.array(loss_b) - np.array(loss_a)
    n = len(d)
    mean = d.mean()

    print(f"\n  {n} janela(s) pareada(s)")
    print(f"  delta de nll ({name_b} - {name_a}): {mean:+.4f}")
    print(f"  perplexidade: {np.exp(mean):.4f}x "
          f"({(np.exp(mean) - 1) * 100:+.2f}%)")

    if n < 2:
        print("\n  SEM INTERVALO: uma janela so nao tem erro padrao.  Este\n"
              "  numero e uma observacao, nao uma medida.")
        return mean, float("nan")

    se = d.std(ddof=1) / np.sqrt(n)

    print(f"  erro padrao: {se:.4f}   intervalo 95%: "
          f"[{mean - 1.96 * se:+.4f}, {mean + 1.96 * se:+.4f}]")
    print(f"  por janela: " + " ".join(f"{v:+.3f}" for v in d))

    # A comparacao so acontece com `se` valido: `abs(mean) < 1.96*nan` e FALSO,
    # e mandaria o relatorio direto para "CONCLUSIVO" sem intervalo nenhum --
    # que foi exatamente o que a primeira versao fez com uma janela.
    if abs(mean) < 1.96 * se:
        print("\n  INCONCLUSIVO: o intervalo cruza zero.  Com esta amostra nao\n"
              "  da para afirmar que a quantizacao degrada -- nem que nao.")
    else:
        print(f"\n  CONCLUSIVO: {name_b} e mensuravelmente "
              f"{'pior' if mean > 0 else 'melhor'} que {name_a}.")
    return mean, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int4", default=str(DEFAULT_INT4))
    ap.add_argument("--bf16", default=str(DEFAULT_BF16))
    ap.add_argument("--tokens", type=int, default=384, help="tokens por janela")
    ap.add_argument("--chunks", type=int, default=1, help="janelas independentes")
    ap.add_argument("--docs", action="store_true",
                    help="usa docs/*.md como corpus em vez do texto embutido")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--numpy", action="store_true",
                    help="lado int4 pelo numpy em vez do engine (mais lento)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.bf16)

    text = corpus_from_docs() if args.docs else CORPUS
    all_ids = tok(text)["input_ids"]

    need = args.tokens * args.chunks
    if len(all_ids) < need:
        sys.exit(f"corpus tem {len(all_ids)} tokens, precisa de {need} "
                 f"({args.chunks} x {args.tokens}). Use --docs.")

    windows = [all_ids[i * args.tokens:(i + 1) * args.tokens]
               for i in range(args.chunks)]

    print(f"\nconjunto fixo: {args.chunks} janela(s) de {args.tokens} tokens, "
          f"{args.chunks * (args.tokens - 1)} previsoes\n")

    int4 = Container(Path(args.int4))
    bf16 = Safetensors(Path(args.bf16))

    exe = ROOT / "tootega.exe"
    use_engine = not args.numpy and exe.exists()
    if use_engine:
        print(f"  lado int4: {exe.name} (o artefato que roda)\n")

    base_losses, int4_losses = [], []
    for w, ids in enumerate(windows):
        print(f"  janela {w}", flush=True)
        base_losses.append(run("BF16 (referencia)", bf16, ids, args.verbose)[0])

        if use_engine:
            logits, dt = run_engine(exe, args.int4, ids, ROOT)
            loss = nll(logits, ids)
            top1 = int((logits[:-1].argmax(-1) == np.array(ids[1:])).sum())
            print(f"  {'int4 g64 (engine)':<28} nll {loss:7.4f}   "
                  f"ppl {np.exp(loss):9.3f}   top-1 {top1:3d}/{len(ids) - 1}   "
                  f"{dt:6.1f} s", flush=True)
            int4_losses.append(loss)
        else:
            int4_losses.append(run("int4 g64 (container)", int4, ids,
                                   args.verbose)[0])

        # Parcial a cada janela: um `Ctrl-C` no meio de uma hora de medicao nao
        # pode custar a medicao inteira.
        (ROOT / "ppl_partial.json").write_text(json.dumps(
            {"windows": w + 1, "bf16": base_losses, "int4": int4_losses},
            indent=1), encoding="utf-8")

    paired("BF16", base_losses, "int4", int4_losses)

    ids = windows[0]
    base_loss, int4_loss = base_losses[0], int4_losses[0]
    delta = int4_loss - base_loss

    if not args.ablate:
        return

    # Ablacao: cada grupo volta a ser exato, um de cada vez.  O que a tabela de
    # docs/04 chama de hipotese e exatamente isto.
    print("  ablacao -- cada grupo restaurado para BF16, um de cada vez:\n")
    groups = [
        ("atencao exata",   r"self_attn\."),
        ("experts exatos",  r"mlp\.(experts|shared_experts)\."),
        ("lm_head exato",   r"^lm_head\."),
        ("densa exata",     r"layers\.0\.mlp\."),
    ]

    for label, pattern in groups:
        src = Hybrid(int4, bf16, pattern)
        loss, _ = run(label, src, ids)
        recovered = (int4_loss - loss) / delta * 100 if delta else 0.0
        print(f"  {'':28} recupera {recovered:5.1f}% da degradacao "
              f"({len(src.hits)} tensores)\n")


if __name__ == "__main__":
    main()
