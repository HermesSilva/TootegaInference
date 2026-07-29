#!/usr/bin/env python3
"""Run the evaluation questions through the C engine on the int4 container.

This closes the loop the whole project is about: the same questions that were
answered in bf16 on the GPU, now answered by TootegaEngine reading the
quantised container on CPU.  Anything that differs between the two is either
quantisation error or a bug in the engine -- and the first probe showed the
LoRA delta survives, so what is left to check is whether it survives on every
kind of question, not just one.

The prompt is built exactly as the trained model expects: retrieved passages,
the question, then the chat template with `</think>` closed immediately --
which is what server.c emits.

Usage:
    python scripts/engine_eval.py --threads 16 --out out/engine_eval.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieve import BM25, format_context, load_sections

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
ENGINE = os.path.join(ROOT, "engine", "tootega")
MODEL = os.path.join(ROOT, "models", "glm47flash-v5-sft.tootega")
TOKENIZER = os.path.join(ROOT, "base", "glm47flash-sft", "tokenizer.json")

QUESTIONS = [
    ("definicao", "O que e a Mente Universal e por que ela nao e Deus?"),
    ("definicao", "O que significa o termo 'germe de origem'?"),
    ("estrutura", "Na morte, a Mente individual dissolve-se na Mente Universal?"),
    ("estrutura", "Quem projeta a vida: a Mente Universal ou a Mente individual?"),
    ("objecao", "Se a Mente Universal nao tem vontade nem escuta ninguem, em que "
                "difere de um universo puramente material? Parece superflua."),
    ("recusa", "O que a Cosmologia Composicional diz sobre a interpretacao de "
               "Copenhague da mecanica quantica?"),
]

STATS = re.compile(r"(prefill|decode)\s+(\d+) tokens em ([\d.]+) s \(([\d.]+) tok/s\)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--passage-chars", type=int, default=1400)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "engine_eval.json"))
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.dirname(TOKENIZER))
    bm = BM25(load_sections())

    rows = []
    for kind, q in QUESTIONS:
        hits = [c for c, _ in bm.search(q, k=a.k)]
        user = ("Trechos da obra:\n\n"
                + format_context([{**c, "text": c["text"][:a.passage_chars]}
                                  for c in hits])
                + f"\n\n---\nPergunta: {q}")
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": user}], add_generation_prompt=True,
            tokenize=False, enable_thinking=False)
        n_in = len(tok(prompt).input_ids)

        t0 = time.time()
        p = subprocess.run(
            [ENGINE, "generate", MODEL, "--tokenizer", TOKENIZER,
             "--prompt", prompt, "-n", str(a.max_new),
             "--threads", str(a.threads)],
            capture_output=True, text=True, timeout=1800)
        wall = time.time() - t0
        out = p.stdout

        # The engine echoes the prompt; the answer starts after </think>
        answer = out.split("</think>", 1)[-1]
        answer = STATS.split(answer)[0]
        answer = re.sub(r"\n(prefill|decode|rss)\b.*", "", answer, flags=re.S).strip()

        stats = {m[0]: {"tokens": int(m[1]), "seconds": float(m[2]),
                        "tok_per_s": float(m[3])}
                 for m in STATS.findall(out)}
        rows.append({"kind": kind, "question": q, "answer": answer,
                     "cites": [c["cite"] for c in hits], "prompt_tokens": n_in,
                     "wall_seconds": wall, "stats": stats})
        print("=" * 76)
        print(f"[{kind}] {q}")
        print(f"  contexto: {n_in} tokens | {', '.join(c['cite'] for c in hits)}")
        pf = stats.get("prefill", {}); dc = stats.get("decode", {})
        print(f"  prefill {pf.get('seconds', 0):.0f}s ({pf.get('tok_per_s', 0):.1f} tok/s) "
              f"| decode {dc.get('tok_per_s', 0):.1f} tok/s | total {wall:.0f}s")
        print("-" * 76)
        print(answer[:1100])
        print()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "threads": a.threads, "rows": rows},
                  f, ensure_ascii=False, indent=1)
    pf = [r["stats"].get("prefill", {}).get("tok_per_s", 0) for r in rows]
    dc = [r["stats"].get("decode", {}).get("tok_per_s", 0) for r in rows]
    print("=" * 76)
    print(f"{len(rows)} perguntas | prefill medio {sum(pf)/len(pf):.1f} tok/s "
          f"| decode medio {sum(dc)/len(dc):.1f} tok/s -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
