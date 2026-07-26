# -*- coding: utf-8 -*-
"""Gera o HTML do ForDummies a partir do .md mais o glossary.json.

O casco (CSS, JS, painel) vem do HTML que ja existe; o que este script refaz e
o sumario lateral, o corpo, os cartoes do glossario e o blob de dados. As
figuras sao copiadas do HTML antigo -- o SVG ja esta inline la, e reinlina-lo
nao mudaria nada.

Roda em modo verificacao (--check) antes de qualquer edicao: se o corpo gerado
nao bate byte a byte com o corpo atual, o conversor esta errado e nao serve
para regerar nada.
"""
import io
import json
import re
import os
import sys

# roda de qualquer lugar: os caminhos sao relativos a raiz do repo
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CHIP = {'armadilha': 'ca', 'projeto': 'pr'}


def chip_of(catid):
    return CHIP.get(catid, 'cc')


def slug(s):
    s = re.sub(r'<[^>]+>', '', s)
    s = s.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    s = s.lower()
    s = re.sub(r'[^\w\s-]', '', s, flags=re.U)
    s = s.replace(' ', '-')
    return s


def esc(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


# ---------------------------------------------------------------- inline
def href(u):
    """Link do markdown para link do HTML gerado."""
    if u.startswith('#') or u.startswith('http'):
        return u
    u = u.replace('../docs/', 'docs-html/').replace('docs/', 'docs-html/')
    if u.endswith('.md'):
        u = u[:-3] + '.html'
    return u


def inline(s, terms=None, seen=None, link=True, holes=None):
    """Converte marcacao inline e, se pedido, marca os termos do glossario."""
    top = holes is None
    if top:
        holes = []

    def hole(html):
        holes.append(html)
        return '\x00%d\x00' % (len(holes) - 1)

    s = re.sub(r'`([^`]+)`', lambda m: hole('<code>%s</code>' % esc(m.group(1))), s)

    def a(m):
        txt, u = m.group(1), m.group(2)
        return hole('<a href="%s">%s</a>'
                    % (href(u), inline(txt, terms, seen, holes=holes)))
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', a, s)

    s = esc(s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'(?<![\w*])\*([^*\n]+)\*(?!\w)', r'<em>\1</em>', s)

    if terms and link:
        s = mark_terms(s, terms, seen)

    if top:
        for _ in range(4):
            s = re.sub(r'\x00(\d+)\x00', lambda m: holes[int(m.group(1))], s)
            if '\x00' not in s:
                break
    return s


def mark_terms(s, terms, seen):
    """Marca a primeira ocorrencia de cada termo no bloco, fora de tags."""
    out = []
    for piece in re.split(r'(<[^>]+>|\x00\d+\x00)', s):
        if not piece or piece.startswith('<') or piece.startswith('\x00'):
            out.append(piece)
            continue
        out.append(scan(piece, terms, seen))
    return ''.join(out)


def scan(piece, terms, seen):
    res = []
    i = 0
    low = piece.lower()
    while i < len(piece):
        if not (piece[i].isalnum() or piece[i] in '_'):
            res.append(piece[i])
            i += 1
            continue
        hit = None
        for n in range(terms['maxw'], 0, -1):
            cand = low[i:i + n]
            tid = terms['map'].get(cand)
            if tid and tid not in seen:
                after = piece[i + n:i + n + 1]
                before = piece[i - 1:i] if i else ''
                if after and (after.isalnum() or after in '_-'):
                    continue
                if before and (before.isalnum() or before == '_'):
                    continue
                hit = (tid, n)
                break
        if hit:
            tid, n = hit
            seen.add(tid)
            res.append('<span class="term" tabindex="0" data-t="%s">%s</span>'
                       % (tid, piece[i:i + n]))
            i += n
        else:
            j = i
            while j < len(piece) and (piece[j].isalnum() or piece[j] == '_'):
                j += 1
            res.append(piece[i:j])
            i = j
    return ''.join(res)


def build_terms(g, lang):
    m, maxw = {}, 1
    for t in g['terms']:
        e = t[lang]
        names = [e['term']] + list(e.get('aliases') or [])
        for nm in names:
            k = nm.lower()
            if k and k not in m:
                m[k] = t['id']
                maxw = max(maxw, len(k))
    return {'map': m, 'maxw': maxw}


# ---------------------------------------------------------------- blocos
def is_gloss_table(rows):
    """Tabela de glossario: as celulas batem com definicoes do JSON."""
    return False  # decidido pelo chamador, por marcador de secao


def render(md, g, lang, figs, gloss_at):
    terms = build_terms(g, lang)
    lines = md.split('\n')
    out, toc = [], []
    seen = set()
    i, nfig, gloss_done = 0, 0, False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('```'):
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].startswith('```'):
                buf.append(lines[j])
                j += 1
            out.append('<div class="codewrap"><pre><code>%s</code></pre>'
                       '<button class="copy" type="button"></button></div>'
                       % esc('\n'.join(buf)))
            i = j + 1
            continue
        if re.match(r'^#{1,4} ', ln):
            lvl = len(ln) - len(ln.lstrip('#'))
            txt = ln[lvl + 1:].strip()
            body = inline(txt, terms, set(), link=False)
            sid = slug(body)
            if lvl == 2:
                seen = set()
            if lvl in (2, 3):
                toc.append('<a class="l%d" href="#%s">%s</a>' % (lvl, sid, body))
            out.append('<h%d id="%s">%s<a class="anchor" href="#%s" '
                       'aria-label="link">#</a></h%d>' % (lvl, sid, body, sid, lvl))
            i += 1
            continue
        if ln.strip() == '---':
            out.append('<hr>')
            i += 1
            continue
        if ln.startswith('!['):
            m = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', ln)
            out.append(figs[nfig])
            nfig += 1
            i += 1
            continue
        if ln.startswith('> '):
            buf = []
            while i < len(lines) and lines[i].startswith('>'):
                buf.append(lines[i][2:] if lines[i].startswith('> ') else '')
                i += 1
            seen = set()
            paras = '\n'.join(buf).split('\n\n')
            out.append('<blockquote>%s</blockquote>' % ''.join(
                '<p>%s</p>' % inline(' '.join(p.split('\n')).strip(), terms, seen)
                for p in paras if p.strip()))
            continue
        if ln.startswith('|'):
            j = i
            rows = []
            while j < len(lines) and lines[j].startswith('|'):
                rows.append(lines[j])
                j += 1
            if gloss_at and i in gloss_at:
                if not gloss_done:
                    out.append(gcards(g, lang))
                    gloss_done = True
                i = j
                continue
            out.append(table(rows, terms, seen))
            i = j
            continue
        if re.match(r'^\s*[-*] |^\s*\d+\. ', ln):
            j = i
            items, cur = [], None
            ordered = bool(re.match(r'^\s*\d+\. ', ln))
            while j < len(lines) and lines[j].strip():
                m = re.match(r'^\s*(?:[-*]|\d+\.) (.*)$', lines[j])
                if m:
                    if cur is not None:
                        items.append(cur)
                    cur = m.group(1)
                else:
                    cur += ' ' + lines[j].strip()
                j += 1
            if cur is not None:
                items.append(cur)
            tag = 'ol' if ordered else 'ul'
            out.append('<%s>%s</%s>' % (tag, ''.join(
                '<li>%s</li>' % inline(x, terms, seen) for x in items), tag))
            i = j
            continue
        if not ln.strip():
            i += 1
            continue
        j = i
        buf = []
        while j < len(lines) and lines[j].strip() and not lines[j].startswith(
                ('|', '#', '```', '![', '> ')) and not re.match(
                r'^\s*[-*] |^\s*\d+\. ', lines[j]) and lines[j].strip() != '---':
            buf.append(lines[j].strip())
            j += 1
        if not buf:
            out.append(ln)
            i += 1
            continue
        out.append('<p>%s</p>' % inline(' '.join(buf), terms, seen))
        i = j
    return '\n'.join(out), '\n'.join(toc)


def table(rows, terms, seen):
    cells = [[c.strip() for c in r.strip().strip('|').split('|')] for r in rows]
    head, body = cells[0], cells[2:]
    h = ''.join('<th>%s</th>' % inline(c, terms, seen) for c in head)
    b = ''
    for r in body:
        b += '<tr>%s</tr>' % ''.join(
            '<td>%s</td>' % inline(c, terms, seen) for c in r)
    return ('<div class="tablewrap"><table><thead><tr>%s</tr></thead>'
            '<tbody>%s</tbody></table></div>' % (h, b))


def gcards(g, lang):
    order = list(g['categories'].keys())
    by = {}
    for t in g['terms']:
        by.setdefault(t['category'], []).append(t)
    parts = ['<div class="glossary">']
    for cid in order:
        ts = by.get(cid) or []
        if not ts:
            continue
        parts.append('<h3 class="gcat" id="gcat-%s">%s <span class="gcount">%d'
                     '</span></h3>' % (cid, g['categories'][cid][lang], len(ts)))
        parts.append('<div class="gcards">')
        for t in ts:
            e = t[lang]
            u = ('<p class="usual"><span>%s:</span> %s</p>'
                 % (USUAL[lang], e['usual'])) if e.get('usual') else ''
            parts.append('<article class="gcard" id="g-%s"><h4>%s</h4>'
                         '<span class="chip %s">%s</span>%s<p>%s</p></article>'
                         % (t['id'], e['term'], chip_of(t['category']),
                            g['categories'][t['category']][lang], u,
                            e['definition']))
        parts.append('</div>')
    parts.append('</div>')
    return '\n'.join(parts)


USUAL = {'pt': 'costuma significar', 'en': 'usually means'}


def gdata(g, lang):
    arr = []
    for t in g['terms']:
        e = t[lang]
        arr.append({'id': t['id'], 'term': e['term'], 'def': e['definition'],
                    'usual': e.get('usual', ''),
                    'cat': g['categories'][t['category']][lang],
                    'catid': t['category'], 'k': chip_of(t['category'])})
    return json.dumps(arr, ensure_ascii=False)


# ---------------------------------------------------------------- main
def figures(html):
    return re.findall(r'<figure>.*?</figure>', html, re.S)


SHELL_NOTE = ('**Reading copy.** This page is generated from `docs/%s`, which '
              'remains the normative source. Where the two disagree, the '
              'Markdown wins. [← back to the guide](../index.html)')

# O casco do espelho nasceu em portugues; `docs/` e ingles, entao a interface
# dele tambem e. So as strings visiveis mudam -- o CSS e o JS sao os mesmos.
SHELL_EN = [
    ('placeholder="Buscar…"', 'placeholder="Search…"'),
    ('aria-label="Buscar…"', 'aria-label="Search…"'),
    ('</svg> Glossário</button>', '</svg> Glossary</button>'),
    ('aria-label="Tema"', 'aria-label="Theme"'),
    ('aria-label="Topo"', 'aria-label="Top"'),
    ('<nav class="toc"><h2>Nesta página</h2>', '<nav class="toc"><h2>On this page</h2>'),
    ('<aside id="panel" aria-label="Glossário">', '<aside id="panel" aria-label="Glossary">'),
    ('<strong>Glossário</strong>', '<strong>Glossary</strong>'),
    ('id="pclose" type="button">Fechar<', 'id="pclose" type="button">Close<'),
    ('data-f="">Todos<', 'data-f="">All<'),
    ('data-f="armadilha">Outro sentido<', 'data-f="armadilha">Other meaning<'),
    ('data-f="projeto">Do projeto<', 'data-f="projeto">Project<'),
    ('data-f="cc">Vocabulário<', 'data-f="cc">Vocabulary<'),
]
LDATA_EN = ('{"usual": "usually means", "openg": "open in glossary", '
            '"copy": "copy", "copied": "copied", "nores": "nothing found", '
            '"gterms": "Glossary", "gsecs": "Sections", "gtext": "In the text", '
            '"knav": "navigate", "kopen": "open", "kclose": "close", '
            '"nhits": "results", "gbase": "../TootegaEngine-ForDummies.en.html"}')


def docs_mirror(g, check):
    """Reconstroi ForDummies/docs-html a partir de docs/*.md."""
    import os
    shell = io.open('ForDummies/docs-html/02-hardware.html', encoding='utf-8').read()
    for a, b in SHELL_EN:
        shell = shell.replace(a, b)
    pre = shell[:shell.index('<nav class="toc"><h2>') + len('<nav class="toc"><h2>')]
    tail = shell[shell.index('</main>'):]
    tail = re.sub(r'(<script type="application/json" id="gdata">).*?(</script>)',
                  lambda m: m.group(1) + gdata(g, 'en') + m.group(2), tail, flags=re.S)
    tail = re.sub(r'(<script type="application/json" id="ldata">).*?(</script>)',
                  lambda m: m.group(1) + LDATA_EN + m.group(2), tail, flags=re.S)
    ttl = re.search(r'<h2>([^<]*)</h2>', shell[shell.index('<nav class="toc"'):]).group(1)
    bad = 0
    for f in sorted(os.listdir('docs')):
        if not f.endswith('.md'):
            continue
        md = io.open('docs/' + f, encoding='utf-8').read()
        title = md.split('\n')[0].lstrip('# ').strip()
        note = SHELL_NOTE % f
        body, toc = render('> ' + note + '\n\n' + md, g, 'en', [], set())
        out = ('ForDummies/docs-html/' + f[:-3] + '.html')
        head = re.sub(r'(<title>|<meta name="description" content=")'
                      r'TootegaEngine — [^<"]*', lambda m: m.group(1)
                      + 'TootegaEngine — ' + title, pre)
        html = head + ttl + '</h2>' + toc + '</nav>\n  <main>' + body + tail
        if check:
            if os.path.exists(out):
                cur = io.open(out, encoding='utf-8').read()
                a = cur[cur.index('<main>') + 6:cur.index('</main>')]
                if a != body:
                    bad += 1
                    d = next((i for i, (x, y) in enumerate(
                        zip(a.split('\n'), body.split('\n'))) if x != y), None)
                    print(' ', f, 'difere na linha', d,
                          len(a.split('\n')), len(body.split('\n')))
                    if d is not None:
                        print('    ATUAL:', a.split('\n')[d][:180])
                        print('    GERAD:', body.split('\n')[d][:180])
        else:
            io.open(out, 'w', encoding='utf-8', newline='').write(html)
    print('docs-html', 'divergentes:' if check else 'gerados:', bad if check else 'ok')


HDR_MD = {
    'armadilha': {
        'pt': '| palavra | o que costuma significar | o que significa **aqui** |',
        'en': '| word | what it usually means | what it means **here** |'},
    'projeto': {'pt': '| termo | significado |', 'en': '| term | meaning |'},
    'voc': {'pt': '| termo | uma linha |', 'en': '| term | one line |'},
}
VOC = ['modelo', 'execucao', 'maquina', 'numerica', 'medicao', 'software',
       'matematica']
SUB = {'pt': ['3.1 ', '4 — ', '5.1 ', '5.2 ', '5.3 ', '5.4 ', '5.5 ', '5.6 ',
              '5.7 '],
       'en': ['3.1 ', '4 — ', '5.1 ', '5.2 ', '5.3 ', '5.4 ', '5.5 ', '5.6 ',
              '5.7 ']}


def to_md(s):
    """A definicao mora em HTML no JSON; a tabela do .md quer markdown."""
    s = re.sub(r'<code>(.*?)</code>', r'`\1`', s)
    s = re.sub(r'<strong>(.*?)</strong>', r'**\1**', s)
    s = re.sub(r'<em>(.*?)</em>', r'*\1*', s)
    s = s.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    return s.replace('|', '\\|')


def md_table(g, cat, lang):
    key = 'armadilha' if cat == 'armadilha' else (
        'projeto' if cat == 'projeto' else 'voc')
    out = [HDR_MD[key][lang], '|---|---|---|' if cat == 'armadilha' else '|---|---|']
    for t in g['terms']:
        if t['category'] != cat:
            continue
        e = t[lang]
        if cat == 'armadilha':
            out.append('| %s | %s | %s |' % (e['term'], to_md(e.get('usual', '')),
                                             to_md(e['definition'])))
        else:
            out.append('| %s | %s |' % (e['term'], to_md(e['definition'])))
    return '\n'.join(out)


def sync_tables(g):
    """As tabelas dos capitulos 3, 4 e 5 sao espelho do JSON, nao fonte."""
    for lang in ('pt', 'en'):
        p = 'ForDummies/TootegaEngine-ForDummies.%s.md' % lang
        md = io.open(p, encoding='utf-8').read()
        for anchor, cat in zip(SUB[lang], ['armadilha', 'projeto'] + VOC):
            head = ('### ' if anchor[0] != '4' else '## ') + anchor
            i = md.index(head)
            j = md.index('\n|', i)
            k = j + 1
            while k < len(md):
                e = md.index('\n', k)
                if not md[k:e].startswith('|'):
                    break
                k = e + 1
            md = md[:j + 1] + md_table(g, cat, lang) + '\n' + md[k:]
        io.open(p, 'w', encoding='utf-8', newline='').write(md)


def main():
    check = '--check' in sys.argv
    g = json.load(io.open('ForDummies/glossary.json', encoding='utf-8'))
    if not check:
        sync_tables(g)
    if '--docs' in sys.argv:
        docs_mirror(g, check)
        return
    for lang in ('pt', 'en'):
        p = 'ForDummies/TootegaEngine-ForDummies.%s' % lang
        md = io.open(p + '.md', encoding='utf-8').read()
        html = io.open(p + '.html', encoding='utf-8').read()
        figs = figures(html)
        gloss_at = set()
        lines = md.split('\n')
        for i, ln in enumerate(lines):
            if ln.startswith('|') and i and lines[i - 1].strip() == '':
                hdr = [c.strip().lower() for c in ln.strip('|').split('|')]
                if hdr in (['palavra', 'o que costuma significar',
                            'o que significa **aqui**'],
                           ['word', 'what it usually means',
                            'what it means **here**'],
                           ['termo', 'significado'], ['term', 'meaning'],
                           ['termo', 'uma linha'], ['term', 'one line']):
                    gloss_at.add(i)
        body, toc = render(md, g, lang, figs, gloss_at)
        cur = html[html.index('<main>') + 6:html.index('</main>')]
        if check:
            same = cur == body
            print(lang, 'corpo igual:', same, len(cur), len(body))
            if not same:
                a, b = cur.split('\n'), body.split('\n')
                n = 0
                for x, y in zip(a, b):
                    if x != y:
                        print('  ATUAL:', x[:200])
                        print('  GERAD:', y[:200])
                        n += 1
                        if n > 6:
                            break
                print('  linhas', len(a), len(b))
            continue
        html = (html[:html.index('<main>') + 6] + body
                + html[html.index('</main>'):])
        html = re.sub(r'(<nav class="toc"><h2>[^<]*</h2>).*?(</nav>)',
                      lambda m: m.group(1) + toc + m.group(2), html, flags=re.S)
        html = re.sub(r'(<script type="application/json" id="gdata">).*?(</script>)',
                      lambda m: m.group(1) + gdata(g, lang) + m.group(2),
                      html, flags=re.S)
        io.open(p + '.html', 'w', encoding='utf-8', newline='').write(html)
        print(lang, 'gerado', len(html))


main()
