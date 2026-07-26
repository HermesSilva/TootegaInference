# -*- coding: utf-8 -*-
"""Confere o que da para conferir sozinho: ancoras internas, termos do
glossario citados que nao existem, e tabelas com numero de colunas irregular."""
import io
import json
import re
import os
import sys

# roda de qualquer lugar: os caminhos sao relativos a raiz do repo
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def slug(s):
    s = re.sub(r'`([^`]*)`', r'\1', s)
    s = s.replace('**', '').replace('*', '')
    s = s.lower()
    s = re.sub(r'[^\w\s-]', '', s, flags=re.U)
    return s.replace(' ', '-')


G = json.load(io.open('ForDummies/glossary.json', encoding='utf-8'))
ids = {t['id'] for t in G['terms']}
print('termos:', len(ids), '| categorias:', len(G['categories']))

for lang in ('pt', 'en'):
    p = 'ForDummies/TootegaEngine-ForDummies.%s.md' % lang
    md = io.open(p, encoding='utf-8').read()
    heads = {slug(h) for h in re.findall(r'^#{1,4} (.*)$', md, re.M)}
    bad = sorted({a for a in re.findall(r'\]\(#([^)]+)\)', md)
                  if a not in heads})
    cols = []
    for i, ln in enumerate(md.split('\n')):
        c = ln.replace('\|', '')
        if c.startswith('|') and not set(c) <= set('|- '):
            cols.append((i + 1, c.count('|')))
    runs, cur = [], []
    for n, c in cols:
        if cur and n != cur[-1][0] + 1:
            runs.append(cur)
            cur = []
        cur.append((n, c))
    if cur:
        runs.append(cur)
    irr = [r[0][0] for r in runs if len({c for _, c in r}) > 1]
    print('%s: %d linhas, ancoras quebradas: %d %s | tabelas irregulares: %s'
          % (lang, len(md.split('\n')), len(bad), bad[:6] or '', irr or 'nenhuma'))

    h = io.open(p[:-3] + '.html', encoding='utf-8').read()
    if '<main>' in h:
        body = h[h.index('<main>'):h.index('</main>')]
        used = set(re.findall(r'data-t="([^"]+)"', body))
        print('   html: %d termos citados, %d sem verbete %s'
              % (len(used), len(used - ids), sorted(used - ids)[:5]))
        for tag in ('p', 'table', 'ul', 'ol', 'blockquote', 'article'):
            o = len(re.findall(r'<%s[ >]' % tag, body))
            c = len(re.findall(r'</%s>' % tag, body))
            if o != c:
                print('   html: <%s> %d abre, %d fecha' % (tag, o, c))
        print('   html: %d figuras, %d blocos de codigo, %d cartoes'
              % (body.count('<figure>'), body.count('<pre>'),
                 body.count('class="gcard"')))
