/* Teste de runtime do ForDummies dirigindo o Chrome headless por CDP.
 *
 * Sem npm install: o Node 22 ja tem WebSocket global e o Chrome ja esta na
 * maquina.  Existe porque a corrupcao de e5a949c passou por um commit inteiro
 * sem ninguem notar -- o HTML abre, so o JS morre calado.
 *
 *   node .fd_browser.mjs [arquivo.html ...]
 */
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const CHROME = [
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
].find(existsSync);

const PORT = 9333;
const profile = mkdtempSync(join(tmpdir(), 'fdchrome-'));
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`, '--disable-gpu',
  '--no-first-run', '--no-default-browser-check', '--allow-file-access-from-files',
  `--user-data-dir=${profile}`, 'about:blank',
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function endpoint() {
  for (let i = 0; i < 60; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/version`);
      return (await r.json()).webSocketDebuggerUrl;
    } catch { await sleep(250); }
  }
  throw new Error('Chrome nao subiu');
}

class CDP {
  constructor(ws) { this.ws = ws; this.id = 0; this.waits = new Map(); this.ev = []; }
  static async open(url) {
    const ws = new WebSocket(url);
    await new Promise((ok, no) => { ws.onopen = ok; ws.onerror = no; });
    const c = new CDP(ws);
    ws.onmessage = (m) => {
      const d = JSON.parse(m.data);
      if (d.id && c.waits.has(d.id)) { c.waits.get(d.id)(d); c.waits.delete(d.id); }
      else if (d.method) c.ev.push(d);
    };
    return c;
  }
  send(method, params = {}, sessionId) {
    const id = ++this.id;
    return new Promise((ok, no) => {
      this.waits.set(id, (d) => d.error ? no(new Error(d.error.message)) : ok(d.result));
      this.ws.send(JSON.stringify({ id, method, params, sessionId }));
    });
  }
}

/* Tudo que roda dentro da pagina.  Devolve um objeto plano de verificacoes. */
const PROBE = `(() => {
  const out = {}, q = (s) => document.querySelector(s);
  const fire = (el, type, init) => el.dispatchEvent(new (type.startsWith('key')
      ? KeyboardEvent : type.startsWith('mouse') || type === 'click'
      ? MouseEvent : Event)(type, Object.assign({bubbles:true, cancelable:true}, init)));

  /* -- tema -- */
  const root = document.documentElement;
  const t0 = root.getAttribute('data-theme');
  const btn = q('#theme');
  out.temaBotao = !!btn;
  if (btn) {
    btn.click();
    out.temaAntes = t0;
    out.temaDepois = root.getAttribute('data-theme');
    out.temaAlternou = t0 !== out.temaDepois;
    out.temaSalvo = localStorage.getItem('tt-theme');
    btn.click();
    out.temaVoltou = root.getAttribute('data-theme') === t0;
  }

  /* -- CSS reage ao tema? -- */
  const bg = (th) => { root.setAttribute('data-theme', th);
                       return getComputedStyle(document.body).backgroundColor; };
  const claro = bg('light'), escuro = bg('dark');
  root.setAttribute('data-theme', t0);
  out.cssClaro = claro; out.cssEscuro = escuro;
  out.cssMuda = claro !== escuro;

  /* -- busca -- */
  const inp = q('#q'), res = q('#res');
  out.buscaCampo = !!inp && !!res;
  if (inp) {
    inp.value = 'expert'; fire(inp, 'input');
    out.buscaAberta = res.classList.contains('on');
    out.buscaItens = res.querySelectorAll('.rit').length;
    out.buscaGrupos = [...res.querySelectorAll('.rgrp')].map(x => x.textContent);
    out.buscaMarcas = res.querySelectorAll('mark').length;
    inp.value = 'zzqqxx'; fire(inp, 'input');
    out.buscaVazia = !!res.querySelector('.rempty');
    inp.value = 'banda'; fire(inp, 'input');
    fire(inp, 'keydown', { key: 'ArrowDown' });
    out.buscaSeleciona = !!res.querySelector('.rit.sel');
    fire(inp, 'keydown', { key: 'Escape' });
    out.buscaFecha = !res.classList.contains('on');
    inp.value = '';
  }

  /* -- painel do glossario -- */
  const gb = q('#gbtn'), pan = q('#panel'), pl = q('#plist'), pq = q('#pq');
  out.painelBotao = !!gb;
  if (gb) {
    gb.click();
    out.painelAbre = pan.classList.contains('on');
    out.painelItens = pl.querySelectorAll('.pit').length;
    if (pq) { pq.value = 'kernel'; fire(pq, 'input');
              out.painelFiltraTexto = pl.querySelectorAll('.pit').length;
              pq.value = ''; fire(pq, 'input'); }
    const chip = document.querySelector('.fchip[data-f="projeto"]');
    if (chip) { chip.click(); out.painelFiltraChip = pl.querySelectorAll('.pit').length; }
    const todos = document.querySelector('.fchip[data-f=""]');
    if (todos) todos.click();
    q('#pclose').click();
    out.painelFecha = !pan.classList.contains('on');
  }

  /* -- expansao de sigla (abbr) -- */
  out.cartoesComAbbr = document.querySelectorAll('.gcard .abbr').length;
  const lo = [...document.querySelectorAll('.term')].find(e => e.dataset.t === 'lora');
  if (lo) { fire(lo, 'mouseover');
            out.tooltipAbbr = q('#tip .abbr')?.textContent || '(ausente)'; }
  if (inp) {  /* procurar pela expansao tem de achar a sigla */
    inp.value = 'rotary'; fire(inp, 'input');
    out.buscaPorExpansao = res.querySelectorAll('.rit').length;
    out.buscaAchouRoPE = res.textContent.includes('RoPE');
    inp.value = ''; fire(inp, 'input');
  }
  if (gb) { gb.click();
            out.painelComAbbr = pl.querySelectorAll('.abbr').length;
            if (pq) { pq.value = 'low-rank'; fire(pq, 'input');
                      out.painelBuscaExpansao = pl.querySelectorAll('.pit').length;
                      pq.value = ''; fire(pq, 'input'); }
            q('#pclose').click(); }

  /* -- apelido: quem procura pelo apelido tem de achar o verbete -- */
  out.cartoesComApelido = document.querySelectorAll('.gcard .alias').length;
  if (inp) {
    inp.value = 'matmul'; fire(inp, 'input');
    const rt = res.querySelector('.rit .rt')?.textContent || '';
    out.apelidoItens = res.querySelectorAll('.rit').length;
    out.apelidoPrimeiro = rt.trim().slice(0, 44);
    out.apelidoAchouVerbete = /multiplica/i.test(rt);
    out.apelidoMostraQual = /matmul/i.test(rt);
    inp.value = ''; fire(inp, 'input');
  }
  if (gb) { gb.click();
            if (pq) { pq.value = 'matmul'; fire(pq, 'input');
                      out.painelApelido = pl.querySelectorAll('.pit').length;
                      pq.value = ''; fire(pq, 'input'); }
            q('#pclose').click(); }

  /* -- sigla curta nao pode casar dentro de palavra -- */
  if (inp) {
    inp.value = 'RAG'; fire(inp, 'input');
    out.ragItens = res.querySelectorAll('.rit').length;
    out.ragPrimeiro = (res.querySelector('.rit .rt')?.textContent || '').trim().slice(0, 44);
    out.ragSemRuido = !/amostragem|par.grafo/i.test(res.textContent);
    inp.value = 'kv'; fire(inp, 'input');
    out.kvPrimeiro = (res.querySelector('.rit .rt')?.textContent || '').trim().slice(0, 44);
    inp.value = ''; fire(inp, 'input');
  }

  /* -- tooltip -- */
  const term = q('.term');
  out.termos = document.querySelectorAll('.term').length;
  if (term) {
    fire(term, 'mouseover');
    const tip = q('#tip');
    out.tooltipAbre = !!tip && tip.classList.contains('on');
    out.tooltipTexto = tip ? tip.textContent.slice(0, 40) : '';
    out.tooltipLink = tip ? (tip.querySelector('a.go')?.getAttribute('href') || '') : '';
  }

  /* -- lightbox -- */
  const z = q('.zoom');
  out.figuras = document.querySelectorAll('figure').length;
  if (z) { z.click(); const lb = q('#lb');
           out.lightboxAbre = !!lb && lb.classList.contains('on');
           out.lightboxTemArte = !!(lb && lb.querySelector('svg,img'));
           if (lb) lb.click(); }

  /* -- copiar codigo -- */
  const cps = [...document.querySelectorAll('.copy')];
  out.botoesCopiar = cps.length;
  out.copiarRotulado = cps.length ? cps[0].textContent.trim().length > 0 : null;

  /* -- sumario / scroll spy -- */
  out.tocLinks = document.querySelectorAll('nav.toc a').length;
  out.tocAlvosFaltando = [...document.querySelectorAll('nav.toc a')]
      .filter(a => !document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1)))).length;
  out.crumb = !!q('#crumb');

  /* -- ancoras internas do corpo apontam para ids existentes? -- */
  out.ancorasQuebradas = [...document.querySelectorAll('main a[href^="#"]')]
      .map(a => decodeURIComponent(a.getAttribute('href').slice(1)))
      .filter(h => h && !document.getElementById(h));

  /* -- teclado -- */
  fire(document, 'keydown', { key: '/' });
  out.atalhoBarra = document.activeElement === inp;
  if (inp) inp.blur();

  return out;
})()`;

async function testar(cdp, file) {
  const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
  const erros = [];
  cdp.ev.length = 0;
  await cdp.send('Runtime.enable', {}, sessionId);
  await cdp.send('Page.enable', {}, sessionId);
  await cdp.send('Page.navigate', { url: pathToFileURL(resolve(file)).href }, sessionId);
  await sleep(1200);
  for (const e of cdp.ev) {
    if (e.method === 'Runtime.exceptionThrown')
      erros.push('EXCECAO: ' + (e.params.exceptionDetails.exception?.description
                 || e.params.exceptionDetails.text).split('\n')[0]);
    if (e.method === 'Runtime.consoleAPICalled' && e.params.type === 'error')
      erros.push('CONSOLE: ' + e.params.args.map(a => a.value ?? a.description).join(' '));
  }
  const r = await cdp.send('Runtime.evaluate',
      { expression: PROBE, returnByValue: true, awaitPromise: true }, sessionId);
  await cdp.send('Target.closeTarget', { targetId });
  if (r.exceptionDetails)
    return { erros, falha: r.exceptionDetails.exception?.description?.split('\n')[0] };
  return { erros, res: r.result.value };
}

const alvos = process.argv.slice(2).length ? process.argv.slice(2) : [
  'ForDummies/TootegaEngine-ForDummies.pt.html',
  'ForDummies/TootegaEngine-ForDummies.en.html',
  'ForDummies/docs-html/02-hardware.html',
  'ForDummies/index.html',
];

const cdp = await CDP.open(await endpoint());
for (const f of alvos) {
  console.log('\n###', f);
  const { erros, res, falha } = await testar(cdp, f);
  if (erros.length) erros.forEach((e) => console.log('   !!', e));
  else console.log('   sem erro de console');
  if (falha) { console.log('   !! sonda falhou:', falha); continue; }
  for (const [k, v] of Object.entries(res)) {
    const s = Array.isArray(v) ? (v.length ? JSON.stringify(v) : '[]') : String(v);
    const ruim = v === false || (Array.isArray(v) && v.length) || v === 0;
    console.log('   %s %-20s %s', ruim ? '!!' : '  ', k, s.slice(0, 110));
  }
}
chrome.kill();
try { rmSync(profile, { recursive: true, force: true }); } catch {}
