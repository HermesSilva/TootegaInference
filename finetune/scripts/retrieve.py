#!/usr/bin/env python3
"""BM25 over the corpus sections -- the retriever for RAFT.

Why BM25 and not embeddings.  RAFT trains the model to answer with retrieved
passages in the context, so whatever retrieves during data generation has to
retrieve at inference too -- eventually inside TootegaEngine, which is C with no
dependencies.  BM25 is an inverted index and two divisions; it ports.  An
embedding retriever would drag a second neural model into src/ and end the
independence the project is built on.

The unit is the SECTION, not the document.  An essay is 5-14k tokens and would
swamp the context; a section is one argument with its own roman numeral, which
is also the unit the citation points at.

Distractors are drawn from the top of the ranking EXCLUDING the oracle.  That is
the point of RAFT: a distractor that shares no vocabulary teaches nothing, while
a neighbouring section of the same book is genuinely confusable and forces the
model to read rather than pattern-match.
"""
import json
import math
import os
import re
import unicodedata
from collections import Counter

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
MANIFEST = os.path.join(ROOT, "data", "manifest.json")

HOLDOUT = ["livro_1_ensaio_9_", "livro_1_ensaio_16_", "livro_2_ensaio_1_",
           "livro_2_ensaio_15_", "livro_3_ensaio_6_", "livro_3_ensaio_23_"]

# Portuguese function words carry no discriminative signal and dominate raw tf.
STOP = set("""a o as os um uma uns umas de do da dos das em no na nos nas por
para com sem sob sobre entre ate apos e ou mas que se nao nem como quando onde
qual quais quanto ao aos à às pelo pela pelos pelas este esta estes estas esse
essa esses essas aquele aquela aqueles aquelas isto isso aquilo seu sua seus
suas meu minha nosso nossa é sao era eram ser sendo sido tem têm tinha havia ha
foi foram sera serao mais menos muito pouco todo toda todos todas outro outra
outros outras mesmo mesma ja tambem so apenas ainda entao assim porque pois
lhe lhes ele ela eles elas eu tu nos vos me te nos vos si""".split())


def norm(s):
    """Fold accents for matching only -- the stored text keeps them."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def tokens(s):
    return [w for w in re.findall(r"[a-z0-9]+", norm(s))
            if len(w) > 2 and w not in STOP]


def is_holdout(fname):
    return any(fname.startswith(h) for h in HOLDOUT)


def load_sections(include_holdout=False):
    """Every ## section of every document, with the citation already built."""
    man = json.load(open(MANIFEST, encoding="utf-8"))
    out = []
    for d in man["docs"]:
        if not include_holdout and is_holdout(d["file"]):
            continue
        text = open(os.path.join(ROOT, d["path"]), encoding="utf-8").read()
        parts = re.split(r"^##\s+", text, flags=re.M)[1:]
        for p in parts:
            head, _, body = p.partition("\n")
            m = re.match(r"([IVXLC]+)\s*[.)-]\s*(.+)", head.strip())
            roman, title = (m.group(1), m.group(2)) if m else (None, head.strip())
            body = body.strip()
            if len(body) < 400:
                continue
            cite = d.get("cite") or f"{d['book_title']}, {d['title']}"
            if roman:
                cite += f", §{roman}"
            out.append({
                "id": f"{d['file']}#{roman or len(out)}",
                "file": d["file"], "kind": d["kind"], "book": d["book_title"],
                "doc_title": d["title"], "section": title, "roman": roman,
                "cite": cite, "text": body, "holdout": is_holdout(d["file"]),
            })
    return out


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokens(c["text"] + " " + c["section"]) for c in chunks]
        self.len = [len(d) for d in self.docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.by_id = {c["id"]: i for i, c in enumerate(chunks)}

    def scores(self, query):
        q = tokens(query)
        out = [0.0] * len(self.docs)
        for w in q:
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out

    def search(self, query, k=5, exclude=()):
        ex = {self.by_id[e] for e in exclude if e in self.by_id}
        ranked = sorted(((s, i) for i, s in enumerate(self.scores(query))
                         if i not in ex and s > 0), reverse=True)
        return [(self.chunks[i], s) for s, i in ranked[:k]]

    def distractors(self, query, oracle_id, k=3, same_book_bias=True):
        """Top-ranked sections that are NOT the oracle -- the confusable ones."""
        hits = self.search(query, k=k * 3, exclude=(oracle_id,))
        if not same_book_bias:
            return [c for c, _ in hits[:k]]
        oi = self.by_id.get(oracle_id)
        book = self.chunks[oi]["book"] if oi is not None else None
        same = [c for c, _ in hits if c["book"] == book]
        other = [c for c, _ in hits if c["book"] != book]
        picked = (same + other)[:k]
        return picked


def format_context(chunks):
    """Numbered passages, the shape the model will see at inference too."""
    return "\n\n".join(
        f"[{i+1}] {c['doc_title']}" + (f" — {c['section']}" if c["section"] else "")
        + f"\n{c['text']}" for i, c in enumerate(chunks))


if __name__ == "__main__":
    import sys
    secs = load_sections()
    bm = BM25(secs)
    print(f"{len(secs)} sections indexed "
          f"({sum(1 for s in secs if s['kind']=='ensaio')} from essays), "
          f"vocabulary {len(bm.idf)}")
    q = sys.argv[1] if len(sys.argv) > 1 else "o que e a Mente Universal e por que nao e Deus"
    print(f"\nquery: {q}\n")
    for c, s in bm.search(q, k=5):
        print(f"  {s:6.2f}  {c['cite']}")
        print(f"          {c['section']}")
    top = bm.search(q, k=1)
    if top:
        print(f"\ndistractors for oracle {top[0][0]['cite']}:")
        for c in bm.distractors(q, top[0][0]["id"], k=3):
            print(f"  {c['cite']} -- {c['section']}")
