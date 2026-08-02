#!/usr/bin/env python3
"""Ask the domain container a question, from Python, with retrieval.

THIS IS NO LONGER HOW YOU USE THE MODEL.  D15 landed: BM25 lives in
`src/retrieve.c`, the index is built by `tools/index.exe`, and the engine
retrieves on its own --

    tools\\index.exe build "models\\Versão Completa" --out models\\v5-sft\\cosmologia.ttindex
    .\\serve-book.cmd

-- which is faster by two orders of magnitude, because the container stays
resident instead of being read off disk for every question.

What is left here is the CROSS-CHECK.  This script and `src/retrieve.c` are two
implementations of the same algorithm -- k1=1.5, b=0.75, the section as the unit,
accents folded -- and the C is only trustworthy while the two rank identically.
They do, on the numbers in docs/19.  When someone changes the tokeniser, this is
what says whether the change was harmless.

Self-contained on purpose: it reads the corpus directly instead of the manifest
built on the GPU box, whose paths point at /workspace.

    python finetune\\ask.py "O que e a Arena?"
    python finetune\\ask.py "..." --k 2 --threads 8 --show-prompt
"""
import argparse
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter

# The answer comes back as UTF-8 and the Windows console is cp850/cp1252, so
# printing it unchanged turns every "é" into "Ã©".  Reconfiguring stdout costs
# nothing and fixes it at the one place it can go wrong.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS = os.path.join(ROOT, "models", "Versão Completa")
ENGINE = os.path.join(ROOT, "tootega.exe")
MODEL = os.path.join(ROOT, "models", "v5-sft", "glm47flash-v5-sft.tootega")
TOKENIZER = os.path.join(ROOT, "models", "v5-sft", "tokenizer.json")

BOOKS = {"I - A Estrutura da Existência": "A Estrutura da Existencia",
         "II - O Instrumento e o Afeto": "O Instrumento e o Afeto",
         "III - O Ciclo e o Horizonte": "O Ciclo e o Horizonte"}

STOP = set("""a o as os um uma de do da dos das em no na nos nas por para com sem
sob sobre entre ate apos e ou mas que se nao nem como quando onde qual quais ao
aos pelo pela pelos pelas este esta estes estas esse essa esses essas aquele
aquela isto isso aquilo seu sua seus suas meu minha nosso nossa e sao era eram
ser sendo sido tem tinha havia ha foi foram sera serao mais menos muito pouco
todo toda todos todas outro outra outros outras mesmo mesma ja tambem so apenas
ainda entao assim porque pois lhe lhes ele ela eles elas eu nos si""".split())


def fold(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def words(s):
    return [w for w in re.findall(r"[a-z0-9]+", fold(s))
            if len(w) > 2 and w not in STOP]


def load_sections():
    """Every ## section of every document, with the citation the books carry."""
    out = []
    if not os.path.isdir(CORPUS):
        sys.exit(f"corpus not found: {CORPUS}")
    for book in sorted(os.listdir(CORPUS)):
        bdir = os.path.join(CORPUS, book)
        if not os.path.isdir(bdir):
            continue
        btitle = BOOKS.get(book, book)
        for name in sorted(os.listdir(bdir)):
            if not name.endswith(".md"):
                continue
            text = open(os.path.join(bdir, name), encoding="utf-8").read()
            m = re.search(r"^#\s+(.+?)\s*$", text, re.M)
            title = m.group(1).strip() if m else name
            m = re.search(r"^\*\(?Livro\s+\w+[^\n]*?Ensaio\s+([IVXLC]+)", text, re.M)
            essay = m.group(1) if m else None
            for part in re.split(r"^##\s+", text, flags=re.M)[1:]:
                head, _, body = part.partition("\n")
                mm = re.match(r"([IVXLC]+)\s*[.)-]\s*(.+)", head.strip())
                roman, sec = (mm.group(1), mm.group(2)) if mm else (None, head.strip())
                body = body.strip()
                if len(body) < 400:
                    continue
                cite = f"{btitle}, Ensaio {essay}" if essay else f"{btitle}, {title}"
                if roman:
                    cite += f", §{roman}"
                out.append({"title": title, "section": sec, "cite": cite, "text": body})
    return out


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        docs = [words(c["text"] + " " + c["section"]) for c in chunks]
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, query, k=3):
        scores = [0.0] * len(self.tf)
        for w in words(query):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w)
                if f:
                    d = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                    scores[i] += idf * f * (self.k1 + 1) / d
        ranked = sorted(((s, i) for i, s in enumerate(scores) if s > 0), reverse=True)
        return [(self.chunks[i], s) for s, i in ranked[:k]]


def decode(raw):
    """Decode engine output without assuming a code page.

    UTF-8 first because that is what the model emits; cp1252 as the fallback
    because that is what a Windows console hands back; and errors="replace" so a
    mangled byte costs a character, not the whole answer.
    """
    if raw is None:
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    # The stream is MIXED: the model's text is UTF-8, the engine's own log lines
    # come out in the console code page.  A single 0xE7 from a log line used to
    # send the whole buffer down the cp1252 path, which decodes the UTF-8 answer
    # into "Ã©" for every accent.  Decoding as UTF-8 with replacement damages
    # only the log bytes, which are not what we print.
    return raw.decode("utf-8", errors="replace")


def build_prompt(question, hits, passage_chars):
    """The exact shape the model was trained on.  Getting it wrong is
    equivalent to not having trained: server.c closes </think> immediately so
    the model answers directly, and that is how every training pair looked."""
    ctx = "\n\n".join(
        f"[{i+1}] {c['title']}" + (f" — {c['section']}" if c["section"] else "")
        + f"\n{c['text'][:passage_chars]}" for i, c in enumerate(hits))
    user = f"Trechos da obra:\n\n{ctx}\n\n---\nPergunta: {question}"
    return f"[gMASK]<sop><|user|>{user}<|assistant|></think>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=3, help="passages to retrieve")
    ap.add_argument("--threads", type=int, default=8,
                    help="the pool does not scale past ~12; 8 P-cores here")
    ap.add_argument("--max-new", "-n", type=int, default=250)
    ap.add_argument("--passage-chars", type=int, default=1400)
    ap.add_argument("--show-prompt", action="store_true")
    a = ap.parse_args()

    for path, what in ((MODEL, "container"), (TOKENIZER, "tokenizer"),
                       (ENGINE, "engine")):
        if not os.path.exists(path):
            sys.exit(f"{what} not found: {path}")

    t0 = time.time()
    hits = BM25(load_sections()).search(a.question, k=a.k)
    if not hits:
        sys.exit("retrieved nothing -- the question shares no terms with the corpus")
    print(f"recuperado em {time.time()-t0:.1f}s:")
    for c, s in hits:
        print(f"  {s:6.2f}  {c['cite']} — {c['section']}")

    prompt = build_prompt(a.question, [c for c, _ in hits], a.passage_chars)
    if a.show_prompt:
        print(f"\n--- prompt ({len(prompt)} chars) ---\n{prompt}\n---")

    print(f"\ngerando (threads={a.threads}) ...", flush=True)
    t0 = time.time()
    # Capture bytes, not text: the engine writes in the console code page on
    # Windows (cp1252 here), so decoding as UTF-8 dies on the first "ç" -- and
    # subprocess raises inside a reader thread, leaving stdout as None and the
    # failure looking like an AttributeError three frames away from the cause.
    p = subprocess.run([ENGINE, "generate", MODEL, "--tokenizer", TOKENIZER,
                        "--prompt", prompt.encode("utf-8").decode("utf-8"),
                        "-n", str(a.max_new), "--threads", str(a.threads)],
                       capture_output=True)
    if p.returncode != 0:
        err = decode(p.stderr)
        sys.exit(f"engine failed ({p.returncode}):\n{err[-800:]}")

    out = decode(p.stdout)
    answer = out.split("</think>", 1)[-1]
    answer = re.split(r"\n(?:prefill|decode|rss)\s", answer)[0].strip()
    print("\n" + "=" * 72)
    print(answer)
    print("=" * 72)
    for line in out.splitlines():
        if line.strip().startswith(("prefill", "decode", "rss")):
            print("  " + line.strip())
    print(f"  total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
