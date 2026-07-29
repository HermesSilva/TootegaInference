#!/usr/bin/env python3
"""Build the corpus manifest: one record per document, with its citation.

Every answer this model is trained to give must be able to say where it came
from -- "Ensaio VI, SVI".  That is only honest if the citation is *extracted*
from the text rather than invented at generation time, so this script parses the
structure the books already carry:

    # <title>                      -> document title
    *Livro Primeiro -- ... Ensaio I*  -> book and essay, in roman as printed
    ## IV. <heading>               -> section IV, the unit a citation points at

Documents that are not essays (prefacio, posfacio, indice de termos,
referencias, conspecto) are classified as such: the term index is the source of
definition pairs, the essays the source of exposition, and neither should be
treated like the other.

The manifest is also the encoding check.  Every file is decoded as strict UTF-8
and re-encoded; the corpus travelled from a Windows path with accented folder
names, and a mojibake title would be silently trained on.
"""
import json
import os
import re
import sys
import unicodedata

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
CORPUS = os.path.join(ROOT, "corpus")
OUT = os.path.join(ROOT, "data", "manifest.json")

BOOK_TITLES = {
    "livro1": "A Estrutura da Existencia",
    "livro2": "O Instrumento e o Afeto",
    "livro3": "O Ciclo e o Horizonte",
}

# "*Livro Primeiro -- A Estrutura da Existencia . Ensaio I*"
RE_SUB = re.compile(r"^\*(Livro\s+\w+)\s*[^\n]*?Ensaio\s+([IVXLC]+)\s*\*", re.M)
# "## IV. O problema do todo"  /  "## I. ..."
RE_SEC = re.compile(r"^##\s+([IVXLC]+)\s*[.)-]\s*(.+?)\s*$", re.M)
RE_SEC_ANY = re.compile(r"^##\s+(.+?)\s*$", re.M)
RE_H1 = re.compile(r"^#\s+(.+?)\s*$", re.M)


def classify(stem):
    for kind in ("indice_de_termos", "referencias", "prefacio", "posfacio", "conspecto"):
        if kind in stem:
            return kind
    if "ensaio" in stem:
        return "ensaio"
    return "outro"


def essay_number(stem):
    m = re.search(r"ensaio_(\d+)", stem)
    return int(m.group(1)) if m else None


def main():
    if not os.path.isdir(CORPUS):
        sys.exit(f"corpus not found: {CORPUS}")

    docs, problems = [], []
    for book in sorted(os.listdir(CORPUS)):
        bdir = os.path.join(CORPUS, book)
        if not os.path.isdir(bdir):
            continue
        for name in sorted(os.listdir(bdir)):
            if not name.endswith(".md"):
                continue
            path = os.path.join(bdir, name)
            raw = open(path, "rb").read()
            try:
                text = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as e:
                problems.append(f"{book}/{name}: not valid UTF-8 ({e})")
                continue
            if text.encode("utf-8") != raw:
                problems.append(f"{book}/{name}: utf-8 round-trip differs")

            stem = name[:-3]
            h1 = RE_H1.search(text)
            title = h1.group(1).strip() if h1 else stem

            sub = RE_SUB.search(text)
            essay_roman = sub.group(2) if sub else None

            secs = [{"roman": r, "heading": h} for r, h in RE_SEC.findall(text)]
            n_h2 = len(RE_SEC_ANY.findall(text))

            kind = classify(stem)
            cite = None
            if kind == "ensaio" and essay_roman:
                cite = f"{BOOK_TITLES.get(book, book)}, Ensaio {essay_roman}"

            docs.append({
                "book": book,
                "book_title": BOOK_TITLES.get(book, book),
                "file": name,
                "path": os.path.relpath(path, ROOT),
                "kind": kind,
                "essay_arabic": essay_number(stem),
                "essay_roman": essay_roman,
                "title": title,
                "cite": cite,
                "sections": secs,
                "n_h2": n_h2,
                "n_sections_numbered": len(secs),
                "bytes": len(raw),
                "chars": len(text),
                "nfc": unicodedata.is_normalized("NFC", text),
            })

    docs.sort(key=lambda d: (d["book"], d["kind"] != "ensaio", d["essay_arabic"] or 0, d["file"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"root": ROOT, "n_docs": len(docs), "docs": docs}, f,
                  ensure_ascii=False, indent=1)

    by_kind, by_book = {}, {}
    for d in docs:
        by_kind[d["kind"]] = by_kind.get(d["kind"], 0) + 1
        by_book.setdefault(d["book"], [0, 0])
        by_book[d["book"]][0] += 1
        by_book[d["book"]][1] += d["chars"]

    print(f"manifest -> {OUT}")
    print(f"  {len(docs)} docs, {sum(d['bytes'] for d in docs)/1e6:.2f} MB, "
          f"{sum(d['chars'] for d in docs)/1e6:.2f} M chars")
    for b, (n, c) in sorted(by_book.items()):
        print(f"  {b:8s} {n:3d} docs  {c/1e6:.2f} M chars  {BOOK_TITLES.get(b,'')}")
    print("  kinds:", ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))

    missing_cite = [d["file"] for d in docs if d["kind"] == "ensaio" and not d["cite"]]
    unnumbered = [d["file"] for d in docs if d["kind"] == "ensaio" and d["n_sections_numbered"] == 0]
    nonfc = [d["file"] for d in docs if not d["nfc"]]
    print(f"  essays without a citation: {len(missing_cite)}")
    if missing_cite:
        print("   ", ", ".join(missing_cite[:6]))
    print(f"  essays without numbered sections: {len(unnumbered)}")
    if unnumbered:
        print("   ", ", ".join(unnumbered[:6]))
    print(f"  not NFC-normalised: {len(nonfc)}")
    if problems:
        print("  ENCODING PROBLEMS:")
        for p in problems:
            print("   ", p)
        return 1
    print("  encoding: all files strict UTF-8, round-trip clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
