#!/usr/bin/env python3
"""RAFT data generation: questions, grounded answers, and a faithfulness judge.

Rewritten after measuring what raw continued pre-training produced.  The failure
was never fluency -- it was that the model learned how the books SOUND and
invented the relations between their terms: a creator Fonte above the Mente
Universal (which Ensaio I spends sections denying), the Mente individual
dissolving back into the whole (which the index lists as a *termo recusado*),
and "germe de origem" attributed to Vygotsky on one run and to an invented
author on the next.

Four stages, per the multi-stage grounded pipelines the literature converged on:

  A  questions   written FROM a section, so they are answerable and specific
  B  answers     written WITH passages in context, RAFT-style
  C  judge       a second pass asks whether the answer is supported by the
                 oracle passage; unsupported answers are dropped, not filtered
                 by regex -- regex catches tics, not a plausible falsehood
  D  assembly    into the exact chat format the engine emits

Three context modes, which is the RAFT part:

  oracle          the right passage plus confusable distractors  (70%)
  distractors     only wrong passages -- correct answer is to decline  (15%)
  closed_book     no context at all -- the engine may run without a retriever,
                  and these are the pairs that put the fact in the weights (15%)

Six families.  Five were in the plan; `estrutura` is new and exists because of
the measured failures: who constitutes whom, who projects what, and above all
that NOTHING stands above the Mente Universal.

Usage:
    python scripts/gen_raft.py --limit 12 --out data/sample.jsonl   # inspect
    python scripts/gen_raft.py --target 1000 --out data/raft.jsonl  # real run
"""
import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieve import BM25, format_context, load_sections, is_holdout

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
BASE = os.path.join(ROOT, "base", "glm47flash-bf16")
MANIFEST = os.path.join(ROOT, "data", "manifest.json")

MAX_PASSAGE = 2000   # chars per passage: 4 passages ~ 3k tokens of context

# The three book titles are stated because the model invented one: it called a
# section heading ("A Arena") the name of the first volume.  Titles are cheap to
# give and the error class disappears entirely.
SYSTEM = (
    "Voce conhece a Cosmologia Composicional e responde sobre ela com precisao "
    "e elegancia, defendendo as suas teses pelos argumentos da propria obra.\n"
    "A obra tem exatamente tres livros, com estes titulos e nenhum outro:\n"
    "  Livro I  — A Estrutura da Existencia\n"
    "  Livro II — O Instrumento e o Afeto\n"
    "  Livro III — O Ciclo e o Horizonte\n"
    "Cada livro reune ensaios, e cada ensaio tem secoes numeradas em romano. "
    "NUNCA invente titulos de livros, de ensaios ou nomes de autores: o titulo "
    "de uma secao nao e o titulo de um livro."
)

ASK = {
    "exposicao":
        "Escreva {n} perguntas que um leitor atento faria sobre a tese desta "
        "secao. Abertas, pedindo explicacao, nao um dado solto.",
    "objecao":
        "Escreva {n} OBJECOES que um critico culto levantaria contra esta secao. "
        "Na voz do critico, como desafio direto, sem responder a elas.",
    "aplicacao":
        "Escreva {n} perguntas sobre casos concretos da vida que esta secao "
        "ajudaria a compreender ou a decidir.",
    "estrutura":
        "Escreva {n} perguntas sobre as RELACOES entre os elementos do sistema "
        "que esta secao estabelece: o que constitui o que, o que projeta o que, "
        "o que precede o que, o que esta contido em que. Perguntas cuja resposta "
        "errada inverteria a arquitetura do sistema.",
}

ANSWER = {
    "exposicao": "Exponha a tese de modo completo e ordenado.",
    "objecao":   "DEFENDA a obra: reconheca o que a critica tem de legitimo e "
                 "mostre por que a tese resiste, com os argumentos da obra.",
    "aplicacao": "Aplique o sistema ao caso, sem moralizar e sem prometer mais "
                 "do que a obra sustenta.",
    "estrutura": "Descreva a relacao com exatidao, nomeando cada elemento e o "
                 "seu lugar. Se a pergunta pressupoe uma relacao invertida, "
                 "corrija-a explicitamente.",
    "definicao": "Defina o termo com precisao, diga o que ele NAO e, e situe-o "
                 "no sistema.",
}

DECLINE = (
    "A resposta correta e declinar. Diga, com cortesia e sem rodeios, que a obra "
    "nao trata desse ponto, e -- se houver -- indique o que de mais proximo ela "
    "trata, sem inventar posicao nenhuma. NAO invente uma tese da obra sobre o "
    "assunto perguntado."
)

STANDALONE = (
    "A resposta deve ser auto-suficiente: nunca escreva 'o trecho', 'o texto "
    "acima', 'o excerto', 'o material fornecido', nem cite os trechos pelo "
    "numero ('[2]', 'o ensaio [3]'). Escreva como quem conhece a obra e a expoe."
)

# The judge caught the generator ornamenting: it attributed to the Absoluto a
# "raiz ontologica" and a "condicao de unidade", terms from a philosophy manual
# and not from these books.  The cause was the instruction to be "complete and
# ordered" with 700 tokens of room -- length the model had to fill with
# something, and what it reached for was generic vocabulary.  Restrict the
# lexicon and shorten the room.
DISCIPLINE = (
    "Use APENAS o vocabulario e as afirmacoes que os trechos sustentam. Nao "
    "introduza terminologia filosofica externa a obra (por exemplo 'raiz "
    "ontologica', 'condicao de unidade', 'substrato transcendental') nem "
    "caracterizacoes que os trechos nao facam. Prefira ser curto e exato a ser "
    "longo e ornamentado: e melhor dizer menos do que acrescentar o que a obra "
    "nao diz."
)

# The first version asked for "FIEL, ou INFIEL: <afirmacao>" and got back
# "FIEL: a afirmacao nao sustentada" -- verdict and template glued together,
# which a startswith("INFIEL") test waves through.  The verdict is now a bare
# word on its own first line, and anything else counts as a failure to judge.
# Two things were wrong with the first judge, and together they rejected 12 of
# 12.  It saw only the oracle passage while the answer had been written with
# four in context, so every legitimate use of the other three read as invention.
# And it asked whether the answer says ONLY what the passage says, which demands
# paraphrase and fails exposition -- the thing we are trying to produce.
#
# What we actually need to catch is what the raw run did: a Fonte creating the
# Mente Universal, "germe de origem" attributed to Vygotsky, a section heading
# promoted to a book title.  Those are contradictions and inventions, not
# elaborations.  So that is what is asked, and OK/ERRO replaces SIM/NAO because
# the polarity of "no, it is not faithful" kept reading backwards.
JUDGE = (
    "TRECHOS DA OBRA:\n{passage}\n\nRESPOSTA DADA:\n{answer}\n\n"
    "A resposta CONTRADIZ os trechos, ou inventa algo que eles nao sustentam "
    "— um nome proprio, um titulo de livro, um autor, uma tese, ou uma relacao "
    "invertida entre conceitos do sistema?\n"
    "Explicar por outras palavras, desenvolver e exemplificar NAO e invencao. "
    "So conta como problema o que contradiz os trechos ou acrescenta um fato "
    "novo e verificavelmente ausente.\n"
    "Escreva na PRIMEIRA linha uma unica palavra: OK ou ERRO.\n"
    "Se ERRO, escreva na segunda linha o que esta errado."
)

# Declines are judged too, and for a different failure: the model declined
# correctly about Copenhagen and, in the same breath, invented a book title.
# The first version rejected a correct decline.  The answer said the work does
# not treat Copenhagen and then pointed at what it treats nearby (the status of
# mathematics) -- which is precisely what the generator was told to do -- and
# the judge called that "attributing a position".  The distinction has to be
# spelt out: naming an adjacent subject is not taking a position on the one
# that was asked.
JUDGE_DECLINE = (
    "PERGUNTA:\n{question}\n\nRESPOSTA DADA:\n{answer}\n\n"
    "A obra (Cosmologia Composicional) NAO trata do assunto perguntado.\n"
    "A resposta falha SOMENTE se: (a) nao reconhecer que a obra nao trata do "
    "assunto perguntado; ou (b) inventar uma tese, posicao, titulo de livro ou "
    "autor QUE A OBRA NAO TENHA.\n"
    "NAO e falha indicar assuntos vizinhos que a obra realmente trata, nem "
    "expor teses da obra sobre esses assuntos vizinhos: isso foi pedido.\n"
    "Escreva na PRIMEIRA linha uma unica palavra: OK ou ERRO.\n"
    "Se ERRO, escreva na segunda linha o problema."
)

# The judge does not just reject, it diagnoses -- and the diagnoses are precise:
# "the answer says the Absoluto precedes the Mente individual, which contradicts
# the thesis that it is a constituent".  That answer was otherwise almost a
# transcription of the author's own gloss.  Discarding it whole, over one
# opening clause, threw away 90% of a good pair; the literature's pipeline has
# a refinement stage for exactly this and I had left it out.
REFINE = (
    "TRECHOS DA OBRA:\n{passage}\n\nPERGUNTA:\n{question}\n\n"
    "RESPOSTA ANTERIOR:\n{answer}\n\n"
    "PROBLEMA APONTADO POR UM REVISOR:\n{verdict}\n\n"
    "Reescreva a resposta corrigindo APENAS o problema apontado. Mantenha o "
    "resto como esta: nao reorganize, nao alongue, nao acrescente ideias novas. "
    "Se o problema for uma afirmacao que a obra nao sustenta, remova-a ou "
    "substitua-a pelo que os trechos de fato dizem.\n"
    "Escreva somente a resposta corrigida."
)


def verdict_word(v):
    """First alphabetic WORD of the verdict: 'ok', 'erro', or None.

    The previous version stripped non-letters from the whole first line, so
    "SIM, a resposta afirma apenas o que o trecho sustenta" collapsed into
    SIMARESPOSTAAFIRMA... and failed the equality test.  Every well-formed
    verdict that bothered to explain itself was counted as unfaithful -- 11 of
    11 on the sample, which is the signature of a broken filter, not of bad
    data.  Only the first word is the verdict; whatever follows is commentary.
    """
    m = re.match(r"[^0-9A-Za-zÀ-ÿ]*([A-Za-zÀ-ÿ]+)", (v or "").strip())
    if not m:
        return None
    w = unicodedata.normalize("NFD", m.group(1).lower())
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    return w if w in ("ok", "erro") else None

# Adjacent subjects the work does not treat -- the source of `recusa` questions.
OUT_OF_SCOPE = [
    "a interpretacao de Copenhague da mecanica quantica",
    "a datacao do universo pela radiacao cosmica de fundo",
    "o problema mente-corpo em Descartes, ponto a ponto",
    "as provas da existencia de Deus em Tomas de Aquino",
    "o que acontece nos buracos negros",
    "a teoria da evolucao por selecao natural e as suas evidencias fosseis",
    "o funcionamento dos neurotransmissores na depressao",
    "a reencarnacao segundo o espiritismo de Allan Kardec",
    "a estrutura do DNA e o codigo genetico",
    "a politica economica mais justa para um pais",
]

NUMBERED = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*(.+?)\s*$")
BAD = re.compile(
    r"\b(o|no|neste|deste|nesse|desse|do)\s+(trecho|excerto|texto\s+acima|"
    r"material\s+fornecido|passagem\s+acima)\b|\bconforme\s+(o\s+)?(trecho|"
    r"excerto|acima)\b|\bsegundo\s+o\s+trecho\b|"
    # "no ensaio [2]" -- pointing at the numbered passage, which the reader of
    # the trained model will never see.  The markdown emphasis the model puts
    # around the citation ("o ensaio **[1]** — O Sonho") slipped past the first
    # version, which assumed the bracket followed the word directly.
    r"\b(trecho|ensaio|passagem|item|documento|se[çc][ãa]o)s?\s*\**\s*\[\d+\]|"
    r"\[\d+\]\s*\**\s*(—|-|:|diz|afirma|estabelece|trata)", re.I)
PT = re.compile(r"\b(que|nao|para|uma|dos|das|com|por|como|mas|ele|ela|isso)\b", re.I)


def clip(t, n=MAX_PASSAGE):
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " [...]"


# KV cache cost per token, this architecture, measured from the config rather
# than assumed: MLA is stored DECOMPRESSED by transformers -- 20 heads x (256 K
# + 256 V) x 47 layers x 2 bytes ~ 1 MB per token.  docs/04 quotes 576 values
# per token per layer for compressed MLA; this is 18x that, because the
# implementation does not absorb kv_b_proj.  A batch of 16 RAFT prompts is
# ~48k tokens = 46 GB of cache on top of 55.8 GB of weights, which is the OOM.
KV_BYTES_PER_TOKEN = 47 * 20 * (256 + 256) * 2


class Runner:
    """Batches by TOKEN budget, not by count.

    A fixed batch size is wrong here because prompt length varies by an order of
    magnitude between stages -- a judge prompt is short, a RAFT answer prompt
    carries four passages.  The budget is what the cache can hold.
    """

    def __init__(self, model, tok, bs, reserve_gb=6.0):
        self.m, self.t, self.bs = model, tok, bs
        free = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated())
        self.budget = max(2048, int((free - reserve_gb * 2**30) / KV_BYTES_PER_TOKEN))
        print(f"  kv budget: {self.budget} tokens per batch "
              f"({KV_BYTES_PER_TOKEN/1024:.0f} KB/token, "
              f"{free/2**30:.1f} GB free)")

    def _groups(self, prompts, max_new):
        """Longest-first, filling each batch until the token budget is spent."""
        lens = [(len(self.t(p, add_special_tokens=False).input_ids) + max_new, i)
                for i, p in enumerate(prompts)]
        lens.sort(reverse=True)
        out, cur, cur_max = [], [], 0
        for n, i in lens:
            m = max(cur_max, n)
            if cur and (len(cur) + 1 > self.bs or m * (len(cur) + 1) > self.budget):
                out.append(cur)
                cur, cur_max = [i], n
            else:
                cur.append(i)
                cur_max = m
        if cur:
            out.append(cur)
        return out

    def __call__(self, prompts, max_new, temp, label=""):
        order = self._groups(prompts, max_new)
        outs = [None] * len(prompts)
        done, t0 = 0, time.time()
        for idxs in order:
            chunk = [prompts[i] for i in idxs]
            texts = [self.t.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=False,
                enable_thinking=False) for p in chunk]
            enc = self.t(texts, return_tensors="pt", padding=True,
                         padding_side="left", add_special_tokens=False).to("cuda")
            kw = dict(do_sample=True, temperature=temp, top_p=0.9) if temp > 0 \
                else dict(do_sample=False)
            try:
                with torch.inference_mode():
                    g = self.m.generate(**enc, max_new_tokens=max_new,
                                        pad_token_id=self.t.pad_token_id, **kw)
                n_in = enc["input_ids"].shape[1]
                for j, o in zip(idxs, g):
                    outs[j] = self.t.decode(o[n_in:], skip_special_tokens=True).strip()
            except torch.cuda.OutOfMemoryError:
                # Fall back to one at a time rather than losing the batch.
                torch.cuda.empty_cache()
                print(f"    {label}: OOM on {len(idxs)}, retrying one by one")
                for j in idxs:
                    e1 = self.t([texts[idxs.index(j)]], return_tensors="pt",
                                add_special_tokens=False).to("cuda")
                    with torch.inference_mode():
                        g1 = self.m.generate(**e1, max_new_tokens=max_new,
                                             pad_token_id=self.t.pad_token_id, **kw)
                    outs[j] = self.t.decode(g1[0][e1["input_ids"].shape[1]:],
                                            skip_special_tokens=True).strip()
                    torch.cuda.empty_cache()
            done += len(idxs)
            el = time.time() - t0
            print(f"    {label} {done}/{len(prompts)}  {el:.0f}s "
                  f"(eta {el/done*(len(prompts)-done):.0f}s)", end="\r", flush=True)
        print(" " * 70, end="\r")
        return ["" if o is None else o for o in outs]


def parse_list(text, limit):
    out = []
    for line in text.splitlines():
        m = NUMBERED.match(line)
        if m:
            q = m.group(1).strip().strip('"').strip()
            if 15 < len(q) < 300:
                out.append(q)
    return out[:limit]


def terms_of(manifest):
    """**Termo** — glosa.  The em dash is the separator the author typed."""
    out = []
    for d in manifest["docs"]:
        if d["kind"] != "indice_de_termos" or is_holdout(d["file"]):
            continue
        for line in open(os.path.join(ROOT, d["path"]), encoding="utf-8"):
            m = re.match(r"^\*\*(.+?)\*\*\s*[—–-]+\s*(.+)$", line.strip())
            if m and len(m.group(2)) > 60:
                out.append({"term": m.group(1).strip(), "gloss": m.group(2).strip(),
                            "refused": "termo recusado" in m.group(2).lower(),
                            "book": d["book_title"], "file": d["file"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "raft.jsonl"))
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--moe", default="eager", choices=["eager", "batched_mm"],
                    help="eager is required: batched_mm copies expert weights "
                         "per token and OOMs on long prompts")
    ap.add_argument("--temp", type=float, default=0.85)
    # Two, not three.  With three, the oracle was one voice in four and the
    # answer synthesised across essays from different books -- one rejection
    # blended Ensaio XIX of book I with three sections of Ensaio XXVII of
    # book III and invented a substance to join them.
    ap.add_argument("--n-distractors", type=int, default=2)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--p-oracle", type=float, default=0.70)
    ap.add_argument("--p-decline", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed)

    sections = load_sections()
    bm = BM25(sections)
    man = json.load(open(MANIFEST, encoding="utf-8"))
    terms = terms_of(man)
    essays = [s for s in sections if s["kind"] == "ensaio"]
    print(f"corpus: {len(essays)} essay sections, {len(terms)} terms "
          f"({sum(t['refused'] for t in terms)} refused), holdout excluded")

    tok = AutoTokenizer.from_pretrained(BASE)
    # eager, NOT batched_mm.  batched_mm does `self.gate_up_proj[expert_ids]`,
    # which materialises a copy of the expert weight matrix per token: gate_up
    # is [64, 3072, 2048] = 12.6 MB per expert, so N tokens at top-4 allocate
    # N * 4 * 12.6 MB.  It survives short prompts (a decode step is one token)
    # and dies on a RAFT prompt: 4 prompts of ~1300 tokens asked for 300 GB.
    # The ceiling is roughly 700 tokens across the whole batch; RAFT contexts
    # are 3k each by design.  eager loops over experts and touches only the
    # tokens routed to each, so its memory does not depend on the weight size.
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", experts_implementation=a.moe)
    model.eval()
    run = Runner(model, tok, a.bs)

    # ---- plan how many items of each family -------------------------------
    n_sec = max(1, (a.target - len(terms) - len(OUT_OF_SCOPE) * 2) // 2)
    picks = random.sample(essays, min(n_sec, len(essays)))
    if a.limit:
        picks = picks[:max(2, a.limit // 3)]
        terms = terms[:max(2, a.limit // 3)]
    fams = ["exposicao", "objecao", "aplicacao", "estrutura"]

    # ---- stage A: questions ------------------------------------------------
    aprompts, ameta = [], []
    for s in picks:
        fam = random.choice(fams)
        aprompts.append(f"Secao \"{s['section']}\" de \"{s['doc_title']}\".\n\n"
                        f"{clip(s['text'], 5000)}\n\n---\n"
                        + ASK[fam].format(n=2) +
                        "\nUma por linha, numeradas. Apenas as perguntas.")
        ameta.append((s, fam))
    print(f"\nstage A: {len(aprompts)} prompts")
    questions = []
    for (s, fam), text in zip(ameta, run(aprompts, 320, a.temp, "A")):
        for q in parse_list(text, 2):
            questions.append({"q": q, "fam": fam, "oracle": s})
    for t in terms:
        q = (f"Por que a Cosmologia Composicional recusa o termo \"{t['term']}\"?"
             if t["refused"] else
             f"O que e \"{t['term']}\" na Cosmologia Composicional?")
        questions.append({"q": q, "fam": "definicao", "oracle": None, "term": t})
    for topic in (OUT_OF_SCOPE if not a.limit else OUT_OF_SCOPE[:2]):
        questions.append({"q": f"O que a Cosmologia Composicional diz sobre "
                               f"{topic}?", "fam": "recusa", "oracle": None})
        questions.append({"q": f"Qual e a posicao da obra sobre {topic}?",
                          "fam": "recusa", "oracle": None})
    random.shuffle(questions)
    if a.limit:
        questions = questions[:a.limit]
    print(f"  {len(questions)} questions "
          f"({sum(1 for q in questions if q['fam']=='definicao')} definicao, "
          f"{sum(1 for q in questions if q['fam']=='recusa')} recusa)")

    # ---- stage B: grounded answers ----------------------------------------
    bprompts = []
    for item in questions:
        fam = item["fam"]
        # distractors-only means "the correct answer is to decline", so it may
        # ONLY be used for questions the work does not answer.  Sending an
        # answerable objection down this path produced a pair that teaches the
        # model to refuse a question about its own subject -- three of the
        # first three survivors were exactly that, which is worse than useless.
        if fam == "recusa":
            mode = "distractors"
        elif fam == "definicao":
            mode = "oracle" if random.random() < 0.8 else "closed_book"
        else:
            mode = "oracle" if random.random() < a.p_oracle else "closed_book"
        # assemble the passages
        if fam == "definicao" and item.get("term"):
            t = item["term"]
            oracle = {"cite": f"{t['book']}, Índice de Termos",
                      "doc_title": "Índice de Termos", "section": t["term"],
                      "text": t["gloss"], "id": f"{t['file']}#{t['term']}"}
        else:
            oracle = item["oracle"]
        hits = bm.search(item["q"], k=a.n_distractors + 2,
                         exclude=(oracle["id"],) if oracle else ())
        distr = [c for c, _ in hits[:a.n_distractors]]
        if mode == "oracle" and oracle:
            ctx = distr + [dict(oracle)]
            random.shuffle(ctx)
        elif mode == "distractors":
            ctx = distr
        else:
            ctx = []
        item["mode"], item["context"] = mode, ctx
        item["oracle_res"] = oracle
        head = (f"Trechos da obra:\n\n"
                f"{format_context([{**c, 'text': clip(c['text'])} for c in ctx])}\n\n---\n"
                if ctx else "")
        if mode == "distractors":
            tail = DECLINE
        else:
            tail = ANSWER.get(fam, ANSWER["exposicao"]) + "\n" + DISCIPLINE
        bprompts.append(f"{head}Pergunta: {item['q']}\n\n{tail}\n{STANDALONE}")
    print(f"\nstage B: {len(bprompts)} answers "
          f"(oracle {sum(1 for q in questions if q['mode']=='oracle')}, "
          f"decline {sum(1 for q in questions if q['mode']=='distractors')}, "
          f"closed {sum(1 for q in questions if q['mode']=='closed_book')})")
    answers = run(bprompts, 500, 0.6, "B")
    for item, ans in zip(questions, answers):
        ans = ans.strip()
        # The citation is OUR datum, from the manifest -- not something to trust
        # the model to remember to write.  Two of twelve answers were discarded
        # for omitting a line we already knew the contents of.
        if item["mode"] == "oracle" and item.get("oracle_res") and "(Fonte:" not in ans:
            ans = ans.rstrip() + f"\n\n(Fonte: {item['oracle_res']['cite']})"
        item["answer"] = ans

    # ---- stage C: the judge, on BOTH kinds of answer ------------------------
    grounded = [q for q in questions
                if q["mode"] != "distractors" and q.get("oracle_res") and q["answer"]]
    declines = [q for q in questions if q["mode"] == "distractors" and q["answer"]]
    if not a.no_judge:
        if grounded:
            print(f"\nstage C: judging {len(grounded)} grounded")
            # The judge must see everything the answer saw.  With only the
            # oracle in front of it, any use of the other three passages looks
            # like invention -- which is how it rejected 12 of 12.  For
            # closed_book there was no context, so the oracle is the reference.
            jp = []
            for q in grounded:
                ctx = q["context"] or [q["oracle_res"]]
                jp.append(JUDGE.format(
                    passage="\n\n".join(f"[{i+1}] {c.get('doc_title', c.get('title',''))}"
                                        f"\n{clip(c['text'], 1800)}"
                                        for i, c in enumerate(ctx)),
                    answer=clip(q["answer"], 3000)))
            for q, v in zip(grounded, run(jp, 100, 0.0, "C")):
                q["verdict"] = v.strip()
        if declines:
            print(f"stage C: judging {len(declines)} declines")
            jp = [JUDGE_DECLINE.format(question=q["q"], answer=clip(q["answer"], 3000))
                  for q in declines]
            for q, v in zip(declines, run(jp, 100, 0.0, "C")):
                q["verdict"] = v.strip()
    for q in questions:
        q.setdefault("verdict", "SKIPPED")

    # ---- stage D: refine what the judge diagnosed ---------------------------
    if not a.no_judge and not a.no_refine:
        bad = [q for q in questions
               if verdict_word(q["verdict"]) == "erro" and q["answer"]]
        if bad:
            print(f"\nstage D: refining {len(bad)}")
            rp = []
            for q in bad:
                ctx = q["context"] or ([q["oracle_res"]] if q.get("oracle_res") else [])
                rp.append(REFINE.format(
                    passage="\n\n".join(
                        f"[{i+1}] {c.get('doc_title', c.get('title',''))}\n"
                        f"{clip(c['text'], 1800)}" for i, c in enumerate(ctx))
                    or "(sem trechos)",
                    question=q["q"], answer=clip(q["answer"], 3000),
                    verdict=q["verdict"]))
            for q, new in zip(bad, run(rp, 500, 0.3, "D")):
                if len(new.strip()) > 150:
                    q["answer_before"] = q["answer"]
                    q["answer"] = new.strip()
                    if (q["mode"] == "oracle" and q.get("oracle_res")
                            and "(Fonte:" not in q["answer"]):
                        q["answer"] += f"\n\n(Fonte: {q['oracle_res']['cite']})"
            # judge the refined answers again -- a refinement that introduces a
            # new error is worse than the original.
            print(f"stage D: re-judging {len(bad)}")
            jp = []
            for q in bad:
                ctx = q["context"] or ([q["oracle_res"]] if q.get("oracle_res") else [])
                if q["mode"] == "distractors":
                    jp.append(JUDGE_DECLINE.format(question=q["q"],
                                                   answer=clip(q["answer"], 3000)))
                else:
                    jp.append(JUDGE.format(
                        passage="\n\n".join(
                            f"[{i+1}] {c.get('doc_title', c.get('title',''))}\n"
                            f"{clip(c['text'], 1800)}" for i, c in enumerate(ctx)),
                        answer=clip(q["answer"], 3000)))
            n_fixed = 0
            for q, v in zip(bad, run(jp, 100, 0.0, "D")):
                q["verdict"] = v.strip()
                q["refined"] = True
                n_fixed += verdict_word(v) == "ok"
            print(f"  recovered {n_fixed} of {len(bad)}")

    # A judge that never says "sim" is broken, not strict.  Print the raw
    # verdicts before they are turned into a drop count.
    seen = [q["verdict"] for q in questions if q["verdict"] != "SKIPPED"]
    words = {}
    for v in seen:
        words[verdict_word(v) or "UNPARSED"] = words.get(verdict_word(v) or "UNPARSED", 0) + 1
    print(f"\nverdicts: {words}")
    for v in seen[:3]:
        print(f"  raw: {v[:110]!r}")

    # ---- stage D: assemble -------------------------------------------------
    kept, drop = [], {"short": 0, "refers": 0, "lang": 0, "nocite": 0,
                      "unfaithful": 0, "unparsed": 0}
    with open(a.out, "w", encoding="utf-8") as f:
        for q in questions:
            ans = q["answer"]
            if len(ans) < 180:
                drop["short"] += 1; continue
            if BAD.search(ans):
                drop["refers"] += 1; continue
            if len(PT.findall(ans)) < 5:
                drop["lang"] += 1; continue
            if q["mode"] == "oracle" and "(Fonte:" not in ans:
                drop["nocite"] += 1; continue
            if q["verdict"] != "SKIPPED":
                w = verdict_word(q["verdict"])
                if w == "erro":
                    drop["unfaithful"] += 1; continue
                if w is None:
                    # The judge failed to answer in the format.  Keep it and
                    # flag it: dropping every unparsed verdict is how a broken
                    # judge silently empties the dataset.
                    drop["unparsed"] += 1
                    q["verdict"] = "UNPARSED: " + q["verdict"][:120]
            ctx = [{"cite": c["cite"], "title": c["doc_title"],
                    "section": c.get("section"), "text": clip(c["text"])}
                   for c in q["context"]]
            rec = {"family": q["fam"], "mode": q["mode"], "question": q["q"],
                   "answer": ans, "context": ctx,
                   "cite": q["oracle_res"]["cite"] if q.get("oracle_res") else None,
                   "verdict": q["verdict"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept.append(rec)

    # Everything, kept or not, with the full verdict.  Tuning a pipeline from a
    # drop counter and 110 characters of verdict is guesswork; three rounds of
    # it moved retention from 0% to 17% and taught nothing about WHY.
    audit = a.out.rsplit(".", 1)[0] + ".audit.jsonl"
    kept_q = {id(q) for q in questions if any(k["question"] == q["q"] for k in kept)}
    with open(audit, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps({
                "kept": id(q) in kept_q, "family": q["fam"], "mode": q["mode"],
                "question": q["q"], "answer": q["answer"],
                "verdict": q["verdict"],
                "cite": q["oracle_res"]["cite"] if q.get("oracle_res") else None,
                "context": [{"cite": c["cite"], "section": c.get("section"),
                             "head": c["text"][:400]} for c in q["context"]],
            }, ensure_ascii=False) + "\n")

    print(f"\nkept {len(kept)} of {len(questions)} -> {a.out}")
    print(f"  audit (todos, com veredicto completo) -> {audit}")
    print(f"  dropped: {drop}")
    for key in ("family", "mode"):
        c = {}
        for r in kept:
            c[r[key]] = c.get(r[key], 0) + 1
        print(f"  by {key}: {c}")
    if kept:
        for r in random.sample(kept, min(2, len(kept))):
            print(f"\n--- [{r['family']}/{r['mode']}] {r['question']}")
            print(f"    ctx: {len(r['context'])} passages | {r['verdict'][:60]}")
            print(f"    {r['answer'][:600]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
