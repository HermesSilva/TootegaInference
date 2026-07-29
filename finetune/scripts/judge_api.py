#!/usr/bin/env python3
"""Judge the generated pairs with Claude, not with the model being trained.

Why an external judge at all.  The self-instruct judge was the GLM base model --
the same model that does not know this work.  It rejected 12 of 12 and then 30
of 33, and reading the verdicts showed it was wrong: it called "Dicionário" an
invented term (it is a headword in the author's own index), and claimed the work
contradicts "a Mente não sente" when the text says exactly that.  Using the
ignorant model to validate the data that exists to cure its ignorance is
circular, and no prompt fixes it.

Two API features do the heavy lifting:

  prompt caching       The system prompt carries ALL 222 index entries -- the
                       author's canonical glosses.  That is what gives the judge
                       the doctrine the GLM lacked.  It is ~20k tokens, constant
                       across every call, and cached: written once at 1.25x, read
                       at 0.1x thereafter.  Without caching this would be the
                       dominant cost; with it, it is the cheapest part.

  structured outputs   The verdict comes back as schema-validated JSON.  The
                       local judge produced "FIEL: a afirmacao nao sustentada"
                       and "SIM, a resposta afirma apenas..." -- both broke a
                       hand-rolled parser, the second silently rejecting every
                       well-formed verdict that bothered to explain itself.
                       There is nothing left to parse.

Runs off the GPU entirely, so generation and judging are separable: generate
once with --no-judge, then judge (and re-judge, after a rule change) as often as
needed without touching the card.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/judge_api.py data/sample.audit.jsonl --out data/judged.jsonl
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
MANIFEST = os.path.join(ROOT, "data", "manifest.json")
MODEL = "claude-opus-5"

BOOKS = ("A Estrutura da Existência", "O Instrumento e o Afeto",
         "O Ciclo e o Horizonte")

RULES = """Voce avalia respostas geradas sobre a Cosmologia Composicional, uma \
obra de filosofia em tres livros. As respostas vao treinar um modelo, entao uma \
afirmacao falsa que passe daqui sera aprendida como verdadeira.

A obra tem exatamente tres livros:
  Livro I  — A Estrutura da Existencia
  Livro II — O Instrumento e o Afeto
  Livro III — O Ciclo e o Horizonte

Abaixo esta o INDICE DE TERMOS COMPLETO da obra, escrito pelo autor. E a \
definicao canonica de cada termo: se a resposta contradiz uma glosa do indice, \
esta errada, por mais plausivel que soe.

Julgue a resposta como ERRO se, e somente se, ela:
  (a) contradiz os trechos fornecidos ou uma glosa do indice;
  (b) inverte uma relacao do sistema (quem constitui quem, quem projeta o que, \
o que precede o que, o que contem o que);
  (c) inventa um nome proprio, autor, titulo de livro ou de ensaio que nao \
exista na obra;
  (d) atribui a obra uma tese que ela nao sustenta;
  (e) usa afirmativamente um termo que o indice marca como *termo recusado*, \
como se fosse vocabulario do sistema.

NAO e ERRO: explicar por outras palavras, desenvolver, exemplificar, ser \
incompleto, escolher uma enfase, ou usar vocabulario filosofico comum para \
explicar uma ideia que os trechos de fato sustentam. Incompletude nao e erro; \
falsidade e.

Na duvida entre "impreciso" e "falso", classifique como OK — o objetivo e \
barrar falsidade, nao exigir transcricao."""


def load_index_entries():
    """All 222 index entries -- the author's canonical glosses."""
    man = json.load(open(MANIFEST, encoding="utf-8"))
    out = []
    for d in man["docs"]:
        if d["kind"] != "indice_de_termos":
            continue
        out.append(f"\n### {d['book_title']}\n")
        for line in open(os.path.join(ROOT, d["path"]), encoding="utf-8"):
            if re.match(r"^\*\*.+?\*\*\s*[—–-]+\s*.+$", line.strip()):
                out.append(line.rstrip())
    return "\n".join(out)


SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["ok", "erro"]},
        "problem": {"type": "string",
                    "description": "Se erro: a afirmacao falsa, numa frase. "
                                   "Se ok: string vazia."},
    },
    "required": ["verdict", "problem"],
    "additionalProperties": False,
}


def build_user(row):
    parts = []
    ctx = row.get("context") or []
    if ctx:
        parts.append("TRECHOS DA OBRA:\n" + "\n\n".join(
            f"[{i+1}] {c.get('cite','')}"
            + (f" — {c['section']}" if c.get("section") else "")
            + f"\n{c.get('head') or c.get('text','')}"
            for i, c in enumerate(ctx)))
    else:
        parts.append("(Nenhum trecho foi dado ao gerador: a resposta foi "
                     "escrita de memoria. Julgue contra o indice de termos.)")
    parts.append(f"PERGUNTA:\n{row['question']}")
    parts.append(f"RESPOSTA A AVALIAR:\n{row['answer']}")
    if row.get("mode") == "distractors":
        parts.append("NOTA: esta pergunta e sobre assunto que a obra NAO trata. "
                     "A resposta correta reconhece isso. Indicar assuntos "
                     "vizinhos que a obra realmente trata NAO e erro; inventar "
                     "uma posicao da obra sobre o assunto perguntado E erro.")
    return "\n\n---\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "judged.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default="medium",
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")

    rows = [json.loads(l) for l in open(a.infile, encoding="utf-8")]
    rows = [r for r in rows if r.get("answer")]
    if a.limit:
        rows = rows[:a.limit]
    index_text = load_index_entries()
    print(f"{len(rows)} pairs to judge | index {len(index_text)} chars "
          f"| model {a.model} effort {a.effort}")

    client = anthropic.Anthropic()
    system = [
        {"type": "text", "text": RULES},
        # The index is the expensive, constant part -- cache it.  Everything
        # before the breakpoint is reused across every call in the run.
        {"type": "text", "text": "INDICE DE TERMOS:\n" + index_text,
         "cache_control": {"type": "ephemeral"}},
    ]

    lock = threading.Lock()
    stats = {"ok": 0, "erro": 0, "fail": 0, "cache_read": 0, "in": 0, "out": 0}
    t0 = time.time()

    def judge(i_row):
        i, row = i_row
        try:
            resp = client.messages.create(
                model=a.model,
                max_tokens=8000,
                system=system,
                output_config={"effort": a.effort,
                               "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": build_user(row)}],
            )
            if resp.stop_reason == "refusal":
                raise RuntimeError("refusal")
            text = next(b.text for b in resp.content if b.type == "text")
            v = json.loads(text)
            with lock:
                stats[v["verdict"]] += 1
                stats["cache_read"] += resp.usage.cache_read_input_tokens or 0
                stats["in"] += resp.usage.input_tokens
                stats["out"] += resp.usage.output_tokens
                done = stats["ok"] + stats["erro"] + stats["fail"]
                el = time.time() - t0
                print(f"  {done}/{len(rows)}  ok={stats['ok']} erro={stats['erro']}"
                      f"  {el:.0f}s", end="\r", flush=True)
            return i, {**row, "judge": v["verdict"], "judge_problem": v["problem"]}
        except Exception as e:
            with lock:
                stats["fail"] += 1
            return i, {**row, "judge": "fail", "judge_problem": f"{type(e).__name__}: {e}"}

    out = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, rec in ex.map(judge, enumerate(rows)):
            out[i] = rec
    print(" " * 70, end="\r")

    with open(a.out, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(rows)
    print(f"\njudged {n} in {time.time()-t0:.0f}s -> {a.out}")
    print(f"  ok {stats['ok']} ({100*stats['ok']/max(1,n):.0f} %) | "
          f"erro {stats['erro']} | falhas {stats['fail']}")
    print(f"  tokens: {stats['in']} in, {stats['cache_read']} cache-read, "
          f"{stats['out']} out")
    # Opus 5: $5/M in, $25/M out, cache read 0.1x
    cost = (stats["in"] * 5 + stats["cache_read"] * 0.5 + stats["out"] * 25) / 1e6
    print(f"  custo estimado: ${cost:.2f}"
          f"  (sem cache seria ~${(stats['in']+stats['cache_read'])*5/1e6 + stats['out']*25/1e6:.2f})")

    bad = [r for r in out if r["judge"] == "erro"]
    for r in bad[:3]:
        print(f"\n--- ERRO [{r.get('family')}/{r.get('mode')}] {r['question'][:80]}")
        print(f"    {r['judge_problem']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
