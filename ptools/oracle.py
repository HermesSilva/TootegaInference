#!/usr/bin/env python3
"""Oraculo do M1: referencia independente do forward, sobre o MESMO container.

Por que nao rodar `transformers` direto: o modelo em fp32 sao ~125 GB e em
bf16 ~62 GB numa maquina de 64 GB.  E, mais importante, comparar o engine
contra o checkpoint BF16 mediria QUANTIZACAO, nao o engine -- os 11,7% de erro
de saida de docs/04 apareceriam como "o engine esta errado".

Entao o oraculo le os mesmos pesos int4 que o engine le, desquantiza sob
demanda (um expert por vez, nunca o modelo inteiro em memoria) e refaz a conta
em numpy seguindo `modeling_glm4_moe_lite.py` operacao por operacao.  O que ele
testa e exatamente o que o M1 promete: que o forward em C reproduz a
matematica, dados os mesmos pesos.

    python tools/oracle.py --ids 1,2,3 --out ref.json
    python tools/oracle.py --prompt "Explique MoE" --n 32 --out ref.json
    python tools/oracle.py --compare logits.bin --out ref.json   (verifica o C)
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "glm47flash-int4" / "model.tootega"
DEFAULT_SRC = ROOT / "models" / "glm47flash-bf16"

MAGIC = b"TOOTEGA\x01"
HEADER = 128


# ------------------------------------------------------------- container

class Container:
    """Leitor do `.tootega` em Python, espelho de src/container.c."""

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as f:
            head = f.read(HEADER)
            if head[:8] != MAGIC:
                sys.exit(f"{path}: magic invalido")
            version, _flags, n, index_off, index_len = struct.unpack("<IIQQQ", head[8:40])
            if version != 1:
                sys.exit(f"{path}: formato v{version} desconhecido")
            f.seek(index_off)
            index = json.loads(f.read(index_len))

        self.meta = index["meta"]
        self.config = self.meta["config"]
        self.tensors = index["tensors"]
        self.n = n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def raw(self, name):
        e = self.tensors.get(name)
        if e is None:
            sys.exit(f"tensor ausente: {name}")
        return e

    def get(self, name) -> np.ndarray:
        """Desquantiza para float32.  Sem cache: o chamador decide o que segurar."""
        e = self.raw(name)
        off, nbytes = e["data"]
        shape = e["shape"]
        kind = e["kind"]
        blob = self.mm[off:off + nbytes]

        if kind == "f32":
            return blob.view(np.float32).reshape(shape).copy()
        if kind == "bf16":
            u = blob.view(np.uint16).reshape(shape).astype(np.uint32)
            return (u << 16).view(np.float32)

        rows, cols = int(np.prod(shape[:-1])), shape[-1]
        group = e["group"]
        soff, sbytes = e["scales"]
        scales = self.mm[soff:soff + sbytes].view(np.float32).reshape(rows, cols // group)

        if kind == "int8":
            q = blob.view(np.int8).reshape(rows, cols // group, group).astype(np.float32)
        elif kind == "int4":
            half = group // 2
            packed = blob.reshape(rows, cols // group, half)
            q = np.empty((rows, cols // group, group), dtype=np.float32)
            q[:, :, :half] = (packed & 0x0F).astype(np.float32) - 8.0
            q[:, :, half:] = (packed >> 4).astype(np.float32) - 8.0
        else:
            sys.exit(f"kind {kind} nao suportado")

        return (q * scales[:, :, None].astype(np.float32)).reshape(rows, cols)

    def row(self, name, index) -> np.ndarray:
        """Uma linha de um tensor grande, sem materializar o resto."""
        e = self.raw(name)
        off, _ = e["data"]
        cols = e["shape"][-1]
        if e["kind"] != "bf16":
            return self.get(name)[index]
        start = off + index * cols * 2
        u = self.mm[start:start + cols * 2].view(np.uint16).astype(np.uint32)
        return (u << 16).view(np.float32)


# ---------------------------------------------------------------- pecas

def rmsnorm(x, w, eps):
    """Glm4MoeLiteRMSNorm: variancia em float32, depois multiplica pelo peso."""
    x = x.astype(np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return (w * (x / np.sqrt(var + eps))).astype(np.float32)


def silu(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def deinterleave(x):
    """Leva o elemento 2i para a posicao i e o 2i+1 para i + d/2.

    `Glm4MoeLiteConfig.rope_interleave` e True por default e o `config.json`
    deste modelo nao sobrescreve, entao a atencao usa
    `apply_rotary_pos_emb_interleave` -- que faz exatamente isto antes do
    rotate_half:

        q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    Omitir este passo nao estoura nada: na posicao 0 as duas variantes
    coincidem (cos=1, sin=0) e o erro cresce com a posicao."""
    d = x.shape[-1]
    return np.concatenate((x[..., 0::2], x[..., 1::2]), axis=-1).reshape(x.shape[:-1] + (d,))


def rope_tables(positions, dim, theta):
    inv = (1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.int64).astype(np.float64) / dim))
           ).astype(np.float32)
    freqs = positions.astype(np.float32)[:, None] * inv[None, :]
    emb = np.concatenate((freqs, freqs), axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def linear(x, w):
    """y = x @ w.T -- a convencao [O, I] do container e a de nn.Linear."""
    return (x.astype(np.float32) @ w.T.astype(np.float32)).astype(np.float32)


# -------------------------------------------------------------- forward

def forward(c: Container, ids, verbose=False):
    cfg = c.config
    T = len(ids)
    H = cfg["num_attention_heads"]
    nope, rot = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"]
    qk, vd = nope + rot, cfg["v_head_dim"]
    kv_rank = cfg["kv_lora_rank"]
    eps = cfg["rms_norm_eps"]
    n_layers = cfg["num_hidden_layers"]
    first_dense = cfg["first_k_dense_replace"]
    top_k = cfg["num_experts_per_tok"]
    n_exp = cfg["n_routed_experts"]
    scaling = float(cfg["routed_scaling_factor"])
    norm_topk = bool(cfg["norm_topk_prob"])
    scale = qk ** -0.5

    h = np.stack([c.row("model.embed_tokens.weight", int(i)) for i in ids]).astype(np.float32)

    positions = np.arange(T, dtype=np.int64)
    cos, sin = rope_tables(positions, rot, float(cfg["rope_theta"]))

    causal = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)
    trace = []

    for li in range(n_layers):
        p = f"model.layers.{li}."
        hn = rmsnorm(h, c.get(p + "input_layernorm.weight"), eps)

        # --- atencao MLA, formulacao desabsorvida (docs/15) ---
        q = linear(rmsnorm(linear(hn, c.get(p + "self_attn.q_a_proj.weight")),
                           c.get(p + "self_attn.q_a_layernorm.weight"), eps),
                   c.get(p + "self_attn.q_b_proj.weight")).reshape(T, H, qk)

        kv_a = linear(hn, c.get(p + "self_attn.kv_a_proj_with_mqa.weight"))
        c_kv, k_rot = kv_a[:, :kv_rank], kv_a[:, kv_rank:]
        kvb = linear(rmsnorm(c_kv, c.get(p + "self_attn.kv_a_layernorm.weight"), eps),
                     c.get(p + "self_attn.kv_b_proj.weight")).reshape(T, H, nope + vd)
        k_nope, v = kvb[:, :, :nope], kvb[:, :, nope:]

        qr = deinterleave(q[:, :, nope:])
        kr = deinterleave(k_rot)
        q_rot = qr * cos[:, None, :] + rotate_half(qr) * sin[:, None, :]
        k_rot = kr * cos + rotate_half(kr) * sin

        q = np.concatenate((q[:, :, :nope], q_rot), axis=-1)
        k = np.concatenate((k_nope, np.repeat(k_rot[:, None, :], H, axis=1)), axis=-1)

        scores = np.einsum("thd,shd->hts", q, k, optimize=True) * scale + causal[None]
        scores = scores - scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=-1, keepdims=True)
        attn = np.einsum("hts,shd->thd", w.astype(np.float32), v, optimize=True)
        h = h + linear(attn.reshape(T, H * vd), c.get(p + "self_attn.o_proj.weight"))

        # --- MLP ---
        hn = rmsnorm(h, c.get(p + "post_attention_layernorm.weight"), eps)

        if li < first_dense:
            act = silu(linear(hn, c.get(p + "mlp.gate_proj.weight"))) * \
                  linear(hn, c.get(p + "mlp.up_proj.weight"))
            h = h + linear(act, c.get(p + "mlp.down_proj.weight"))
        else:
            logits = linear(hn, c.get(p + "mlp.gate.weight"))
            probs = 1.0 / (1.0 + np.exp(-logits))
            choice = probs + c.get(p + "mlp.gate.e_score_correction_bias")
            idx = np.argsort(-choice, axis=-1, kind="stable")[:, :top_k]
            wts = np.take_along_axis(probs, idx, axis=-1)
            if norm_topk:
                wts = wts / (wts.sum(axis=-1, keepdims=True) + 1e-20)
            wts = wts * scaling

            out = np.zeros_like(hn)
            # Por expert crescente, como o `index_add_` da referencia.
            for e in range(n_exp):
                hit = np.nonzero(idx == e)
                if hit[0].size == 0:
                    continue
                toks = hit[0]
                xw = hn[toks]
                ep = f"{p}mlp.experts.{e}."
                act = silu(linear(xw, c.get(ep + "gate_proj.weight"))) * \
                      linear(xw, c.get(ep + "up_proj.weight"))
                out[toks] += linear(act, c.get(ep + "down_proj.weight")) * \
                             wts[hit][:, None]

            act = silu(linear(hn, c.get(p + "mlp.shared_experts.gate_proj.weight"))) * \
                  linear(hn, c.get(p + "mlp.shared_experts.up_proj.weight"))
            h = h + out + linear(act, c.get(p + "mlp.shared_experts.down_proj.weight"))

        trace.append(float(np.linalg.norm(h[-1])))
        if verbose:
            print(f"  camada {li:2d}  |h| = {trace[-1]:.4f}", flush=True)

    hn = rmsnorm(h, c.get("model.norm.weight"), eps)
    logits = linear(hn, c.get("lm_head.weight"))
    return logits, hn, trace


# ----------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--src", default=str(DEFAULT_SRC),
                    help="pasta HF, so para o tokenizer de --prompt")
    ap.add_argument("--ids", help="ids separados por virgula")
    ap.add_argument("--ids-file", help="idem, de um arquivo (mesma flag do engine)")
    ap.add_argument("--prompt", help="texto, tokenizado com o tokenizer HF")
    ap.add_argument("--n", type=int, default=32, help="quantas posicoes")
    ap.add_argument("--out", default=str(ROOT / "ref.json"))
    ap.add_argument("--save-logits", help="salva os logits do oraculo (.npy)")
    ap.add_argument("--compare", help="logits crus do engine em C (f32, [T][vocab])")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    c = Container(Path(args.model))

    if args.ids_file:
        raw = Path(args.ids_file).read_text(encoding="utf-8")
        ids = [int(v) for v in raw.replace(",", " ").split()]
    elif args.ids:
        ids = [int(v) for v in args.ids.replace(",", " ").split()]
    elif args.prompt:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.src)
        ids = tok(args.prompt)["input_ids"]
    else:
        sys.exit("passe --ids ou --prompt")

    ids = ids[:args.n]
    print(f"oraculo: {len(ids)} posicoes, teacher-forcing")

    logits, hidden, trace = forward(c, ids, args.verbose)
    pred = logits.argmax(axis=-1).astype(int).tolist()

    top = np.argsort(-logits, axis=-1)[:, :3]
    ref = {
        "model": str(args.model),
        "input_ids": [int(v) for v in ids],
        "argmax": pred,
        "top3": [[int(t) for t in row] for row in top],
        "top3_logits": [[float(logits[i, t]) for t in row] for i, row in enumerate(top)],
        "hidden_norm": [float(np.linalg.norm(hidden[i])) for i in range(len(ids))],
        "layer_norm_last": trace,
    }
    Path(args.out).write_text(json.dumps(ref, indent=1), encoding="utf-8")
    print(f"  {args.out} escrito")

    if args.save_logits:
        np.save(args.save_logits, logits)

    for i, (tok_in, tok_out) in enumerate(zip(ids, pred)):
        margin = logits[i, tok_out] - np.partition(logits[i], -2)[-2]
        print(f"  {i:3d} {tok_in:>8} -> {tok_out:>8}  logit {logits[i, tok_out]:8.3f}"
              f"  margem {margin:6.3f}")

    if args.compare:
        raw = np.fromfile(args.compare, dtype=np.float32)
        vocab = logits.shape[1]
        if raw.size != logits.size:
            sys.exit(f"--compare tem {raw.size} floats, esperava {logits.size}")
        got = raw.reshape(-1, vocab)

        got_pred = got.argmax(axis=-1)
        same = int((got_pred == np.array(pred)).sum())
        diff = np.abs(got - logits)
        rel = np.linalg.norm(got - logits) / np.linalg.norm(logits)

        print(f"\ncomparacao com o engine:")
        print(f"  tokens iguais      {same}/{len(pred)}")
        print(f"  erro relativo      {rel * 100:.4f}%")
        print(f"  maior diferenca    {diff.max():.6f}")
        for i in range(len(pred)):
            if got_pred[i] != pred[i]:
                print(f"  pos {i}: engine {got_pred[i]} (logit "
                      f"{got[i, got_pred[i]]:.4f}), oraculo {pred[i]} (logit "
                      f"{logits[i, pred[i]]:.4f})")
        sys.exit(0 if same == len(pred) else 1)


if __name__ == "__main__":
    main()
