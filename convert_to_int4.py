#!/usr/bin/env python3
"""Converte GLM-4.7-Flash de BF16 para o container int4 do TootegaEngine.

Marco M0 do roadmap. Tres propriedades que o engine depende:

  1. int4 com escalas f32 por grupo de 64 ao longo da dimensao de entrada.
     E o formato que a conta de teto assume (0,5625 byte por peso).
  2. Todo tensor alinhado a 64 bytes no arquivo. Sem isso nao ha leitura
     zero-copy com SIMD e o carregador precisa de um caminho de copia --
     ver docs/11-memoria.md.
  3. Streaming shard a shard. Nunca os dois lados em memoria: a fonte tem
     62,4 GB e a saida ~19,3 GB (medido; a estimativa antiga de 15,6 GB nao
     contava escalas, MTP em int8 nem os dois `embed_tokens` em bf16).

Precisao por tensor segue docs/04-modelo-alvo.md, incluindo a licao #8 do
colibri: cabeca MTP em int4 derruba a aceitacao de draft para 0-4%, entao a
camada MTP inteira vai em int8.

Uso:
    python convert_to_int4.py --dry-run     # plano e tamanho estimado, nao escreve
    python convert_to_int4.py               # converte
    python convert_to_int4.py --limit 2     # so os 2 primeiros shards (teste)
    python convert_to_int4.py --verify      # confere o container ja escrito
"""

import argparse
import json
import mmap
import os
import re
import struct
import sys
import time
import traceback
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE / "models" / "glm47flash-bf16"
DEFAULT_OUT = HERE / "models" / "glm47flash-int4" / "model.tootega"

MAGIC = b"TOOTEGA\x01"
FORMAT_VERSION = 1
HEADER_SIZE = 128
ALIGN = 64

# Definido por --group. Grupo menor = escala mais ajustada = menos erro, ao
# custo de mais bytes de escala (4 B por grupo por linha). Alavanca FRACA:
# medido, 64 -> 32 corta ~11% do erro e engorda o arquivo 11%. Ver docs/04.
GROUP = 64

# Definido por --clip-grid. Pontos da busca de escala, como fracao do absmax do
# grupo, do maior para o menor. O primeiro e 1,00 -- absmax puro, o
# comportamento anterior -- e desempata os grupos todo-zero para ele.
CLIP_RATIOS = np.linspace(1.0, 0.70, 13, dtype=np.float32)

# ---------------------------------------------------------------- politica

# Ordem importa: o primeiro padrao que casar decide.
#
# As regras de tensor 1-D (normas, vieses) e de embedding vem PRIMEIRO, antes
# da regra da camada MTP. A ordem inversa parecia certa -- "MTP antes de tudo,
# para nao ser capturada pelas regras de expert" -- e engolia os `enorm`,
# `hnorm` e layernorms da propria MTP, mandando um vetor de 2048 posicoes para
# um quantizador que exige matriz.
POLICY = [
    (r"(layernorm|^model\.norm|\.enorm|\.hnorm)\.weight$", "f32"),
    (r"\.shared_head\.norm\.weight$",                  "f32"),
    (r"\.mlp\.gate\.weight$",                          "f32"),    # roteador
    (r"\.e_score_correction_bias$",                    "f32"),
    (r"embed_tokens\.weight$",                         "bf16"),   # 1 linha/token
    (r"^model\.layers\.MTP\..*\.weight$",              "int8"),   # licao #8
    (r"^lm_head\.weight$",                             "int8"),
    (r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$",   "int4"),
    (r"\.mlp\.shared_experts\.(gate|up|down)_proj\.weight$", "int4"),
    (r"\.mlp\.(gate|up|down)_proj\.weight$",           "int4"),   # camada densa
    (r"\.self_attn\..*_proj\.weight$",                 "int4"),
    (r"\.eh_proj\.weight$",                            "int4"),
]

def bytes_per_weight(kind: str) -> float:
    return {"int4": 0.5 + 4 / GROUP, "int8": 1.0 + 4 / GROUP,
            "f32": 4.0, "bf16": 2.0}[kind]


def classify(name: str, mtp_layer: int, lm_head: str, mtp: str) -> str:
    """Precisao alvo. `mtp_layer` e resolvido do config, nao chutado.

    lm_head e mtp sao parametros porque sao as duas escolhas genuinamente
    incertas da tabela em docs/04-modelo-alvo.md -- existem para a ablacao,
    nao para ficarem fixas por palpite.
    """
    probe = re.sub(rf"^model\.layers\.{mtp_layer}\.", "model.layers.MTP.", name)
    for pattern, kind in POLICY:
        if re.search(pattern, probe):
            if kind == "int8" and probe.startswith("model.layers.MTP."):
                return mtp
            if kind == "int8" and probe.startswith("lm_head"):
                return lm_head
            return kind
    return "f32"   # desconhecido: preserva em vez de degradar em silencio


# ------------------------------------------------------------ safetensors

def read_st_header(path: Path):
    """(dict do cabecalho, offset onde os dados comecam). Sem dependencia."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path.name}: arquivo truncado")
        hlen = struct.unpack("<Q", raw)[0]
        if not 0 < hlen <= (512 << 20):
            raise ValueError(f"{path.name}: cabecalho absurdo ({hlen} bytes)")
        header = json.loads(f.read(hlen))
    return header, 8 + hlen


def bf16_to_f32(view: np.ndarray) -> np.ndarray:
    """BF16 sao os 16 bits altos de um float32 -- a conversao e um shift."""
    return (view.astype(np.uint32) << 16).view(np.float32)


# A fonte NAO e homogenea: 9.656 tensores em BF16 e 47 em F32 (os vieses de
# correcao do roteador). Assumir BF16 para tudo produz um reshape invalido no
# primeiro F32 -- e, pior, produziria lixo silencioso se as contas batessem.
ST_DTYPES = {
    "BF16": (np.uint16, 2),
    "F16":  (np.float16, 2),
    "F32":  (np.float32, 4),
}


def as_f32(raw: np.ndarray, dtype: str) -> np.ndarray:
    """View do mmap -> float32, sem copia quando ja e F32."""
    if dtype == "BF16":
        return bf16_to_f32(raw)
    if dtype == "F16":
        return raw.astype(np.float32)
    return raw


# ----------------------------------------------------------- quantizacao

def fit_scales(groups: np.ndarray, qmax: int) -> np.ndarray:
    """Escala por grupo que minimiza o erro quadratico da reconstrucao.

    `absmax/qmax` -- a escolha obvia -- otimiza o pior peso do grupo em vez do
    grupo. Encolher a escala clipa os extremos e, em troca, aperta o passo para
    todos os outros pesos; com poucos niveis a troca compensa quase sempre. O
    formato nao muda: continua uma escala f32 por grupo, e o kernel nao sabe a
    diferenca.

    Efeito colateral que vale por si: com ratio < 1 o codigo -8 passa a ser
    alcancavel. Com escala = absmax/7 exata, `rint` nunca produz -8 e o int4
    usa 15 dos 16 niveis que `dequant_grouped` ja sabe ler.
    """
    absmax = np.abs(groups).max(axis=2)
    # absmax == 0 -> grupo todo zero: escala 1 evita divisao por zero e
    # reconstroi exatamente (todos os codigos ficam 0).
    base = np.where(absmax > 0, absmax / qmax, 1.0).astype(np.float32)

    best_scales, best_err = None, None
    for ratio in CLIP_RATIOS:
        scales = base * ratio
        delta = np.clip(np.rint(groups / scales[:, :, None]),
                        -(qmax + 1), qmax) * scales[:, :, None]
        delta -= groups
        err = np.einsum("rgc,rgc->rg", delta, delta)
        if best_err is None:
            best_scales, best_err = scales, err
        else:
            # Estrito: empate mantem o ratio maior, ja visitado antes.
            better = err < best_err
            best_scales = np.where(better, scales, best_scales)
            best_err = np.where(better, err, best_err)
    return best_scales


def quant_grouped(block: np.ndarray, bits: int):
    """Simetrica, por grupo de GROUP ao longo da ultima dimensao.

    Retorna (codigos, escalas f32). Para int4 os codigos vem como uint8 ja
    empacotados, 2 pesos por byte -- baixo primeiro.
    """
    rows, cols = block.shape
    qmax = 7 if bits == 4 else 127
    groups = block.reshape(rows, cols // GROUP, GROUP)

    scales = fit_scales(groups, qmax)

    q = np.rint(groups / scales[:, :, None])
    q = np.clip(q, -(qmax + 1), qmax).astype(np.int8).reshape(rows, cols)

    if bits == 8:
        return q, scales

    # int4: desloca para 0..15 e empacota dois por byte, METADE/METADE dentro
    # de cada grupo: byte[j] carrega w[j] no nibble baixo e w[j + GROUP/2] no
    # alto.
    #
    # A alternativa obvia -- par no baixo, impar no alto -- entrega ordem
    # INTERCALADA ao desempacotar com SIMD (`v & 0x0F` da 0,2,4,...), obrigando
    # a permutar as ativacoes. Metade/metade entrega dois vetores contiguos,
    # w[0..31] e w[32..63], prontos para o produto. E a convencao do llama.cpp
    # Q4_0 e derivados. Ver docs/13 (D1) e docs/14.
    half = GROUP // 2
    u = ((q.astype(np.uint8) + 8) & 0x0F).reshape(rows, cols // GROUP, GROUP)
    packed = (u[:, :, :half] | (u[:, :, half:] << 4)).astype(np.uint8)
    return packed.reshape(rows, cols // 2), scales


def dequant_grouped(codes: np.ndarray, scales: np.ndarray, bits: int, cols: int):
    """Inversa de quant_grouped. Usada para medir o erro e como referencia
    executavel do layout que o kernel em C tem de reproduzir."""
    rows = codes.shape[0]
    ngroups, half = cols // GROUP, GROUP // 2
    if bits == 4:
        packed = codes.reshape(rows, ngroups, half)
        q = np.empty((rows, ngroups, GROUP), dtype=np.int16)
        q[:, :, :half] = (packed & 0x0F).astype(np.int16) - 8
        q[:, :, half:] = (packed >> 4).astype(np.int16) - 8
    else:
        q = codes.astype(np.int16).reshape(rows, ngroups, GROUP)
    return (q * scales[:, :, None]).reshape(rows, cols).astype(np.float32)


# -------------------------------------------------------------- escritor

class Writer:
    """Escreve blobs alinhados e acumula o indice."""

    def __init__(self, path: Path):
        self.path = path
        self.tmp = path.with_suffix(path.suffix + ".part")
        self.tmp.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.tmp, "wb")
        self.f.write(b"\0" * HEADER_SIZE)     # cabecalho reescrito no fim
        self.index = {}

    def _align(self):
        pad = (-self.f.tell()) % ALIGN
        if pad:
            self.f.write(b"\0" * pad)
        return self.f.tell()

    def blob(self, array: np.ndarray) -> tuple:
        off = self._align()
        data = np.ascontiguousarray(array)
        self.f.write(data.tobytes())
        return off, data.nbytes

    def add(self, name, kind, shape, data_off, data_len,
            scale_off=None, scale_len=None):
        self.index[name] = {
            "kind": kind, "shape": list(shape),
            "data": [data_off, data_len],
            "scales": [scale_off, scale_len] if scale_off is not None else None,
            "group": GROUP if kind in ("int4", "int8") else None,
        }

    def close(self, meta: dict):
        index_off = self._align()
        blob = json.dumps({"meta": meta, "tensors": self.index},
                          separators=(",", ":")).encode("utf-8")
        self.f.write(blob)
        self.f.seek(0)
        self.f.write(MAGIC)
        self.f.write(struct.pack("<IIQQQ", FORMAT_VERSION, 0,
                                 len(self.index), index_off, len(blob)))
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()
        # Troca atomica: uma queda no meio nao deixa um container meio escrito
        # com aparencia de valido.
        os.replace(self.tmp, self.path)


# ------------------------------------------------------------- conversao

def convert_tensor(name, kind, raw, src_dtype, writer, rows_per_chunk, err_stats):
    """Quantiza e escreve um tensor, em blocos de linhas."""
    shape = raw.shape

    if kind == "bf16":
        # Preserva o formato de origem quando ja e BF16; converte se nao for.
        data = raw if src_dtype == "BF16" else as_f32(raw, src_dtype)
        label = "bf16" if src_dtype == "BF16" else "f32"
        off, n = writer.blob(data)
        writer.add(name, label, shape, off, n)
        return n

    if kind == "f32":
        off, n = writer.blob(as_f32(raw, src_dtype))
        writer.add(name, "f32", shape, off, n)
        return n

    if raw.ndim != 2:
        # Um tensor inesperado nao pode custar 40 minutos de conversao. Cai
        # para f32 -- preserva o valor exato, so ocupa mais -- e avisa alto.
        print(f"    AVISO: {name} tem {shape}, nao e matriz; "
              f"gravando f32 em vez de {kind}", flush=True)
        off, n = writer.blob(as_f32(raw, src_dtype))
        writer.add(name, "f32", shape, off, n)
        err_stats.setdefault("_fallback_f32", []).append(name)
        return n

    rows, cols = shape
    if cols % GROUP:
        raise ValueError(f"{name}: dim de entrada {cols} nao divide por {GROUP}")

    bits = 4 if kind == "int4" else 8
    data_off = writer._align()
    total = 0
    all_scales = np.empty((rows, cols // GROUP), dtype=np.float32)

    for start in range(0, rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, rows)
        block = as_f32(raw[start:stop], src_dtype)
        codes, scales = quant_grouped(block, bits)
        writer.f.write(np.ascontiguousarray(codes).tobytes())
        total += codes.nbytes
        all_scales[start:stop] = scales

        # Erro medido no primeiro bloco de cada tensor: barato e suficiente
        # para a ablacao de precisao que o M0 exige.
        #
        # Duas metricas. A da SAIDA e a que decide, e a medicao ja mostrou que
        # ela NAO cai em relacao ao erro por peso -- 11,47% -> 10,81% no
        # modelo inteiro. Nao ha cancelamento; ver o texto impresso no fim.
        #
        # LIMITE DESTA SONDA: x e gaussiano iid, e ativacoes reais de LLM nao
        # sao -- alguns canais tem magnitude 10-100x a mediana, e o erro real
        # e dominado pelas colunas de W alinhadas a eles. O numero daqui e
        # portanto OTIMISTA. Medir com ativacoes de calibracao (ou ponderar x
        # pelo RMS por canal) e o que fecharia o furo; e a premissa do AWQ e o
        # motivo de a rotacao Hadamard estar em docs/04.
        if start == 0:
            recon = dequant_grouped(codes, scales, bits, cols)
            denom = np.abs(block).mean() or 1.0
            per_weight = float(np.abs(recon - block).mean() / denom)

            rng = np.random.default_rng(0)          # fixo: comparavel entre runs
            x = rng.standard_normal(cols, dtype=np.float32)
            y_ref = block @ x
            y_q = recon @ x
            norm = float(np.linalg.norm(y_ref)) or 1.0
            per_output = float(np.linalg.norm(y_q - y_ref) / norm)

            s = err_stats.setdefault(kind, {"peso": [], "saida": []})
            s["peso"].append(per_weight)
            s["saida"].append(per_output)

    scale_off, scale_len = writer.blob(all_scales)
    writer.add(name, kind, shape, data_off, total, scale_off, scale_len)
    return total + scale_len


def plan(src: Path, mtp_layer: int, lm_head: str, mtp: str):
    """(entradas, totais por classe). Nao le dado, so cabecalhos."""
    index = json.loads((src / "model.safetensors.index.json").read_text("utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    entries, totals = [], {}
    for shard in shards:
        header, base = read_st_header(src / shard)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            kind = classify(name, mtp_layer, lm_head, mtp)
            n = int(np.prod(info["shape"])) if info["shape"] else 1
            out_bytes = n * bytes_per_weight(kind)
            entries.append((shard, name, info, base, kind))
            agg = totals.setdefault(kind, [0, 0])
            agg[0] += 1
            agg[1] += out_bytes

    # Recusar antes de escrever, nao no meio. Um --group que nao divide alguma
    # dimensao de entrada so apareceria depois de minutos de conversao, com o
    # arquivo parcial ja no disco.
    bad = sorted({(n, s["shape"][1]) for _sh, n, s, _b, k in entries
                  if k in ("int4", "int8") and len(s["shape"]) == 2
                  and s["shape"][1] % GROUP})
    if bad:
        dims = sorted({d for _n, d in bad})
        ok = [g for g in (32, 64, 128, 256, 512)
              if all(d % g == 0 for d in dims)]
        raise SystemExit(
            f"--group {GROUP} nao divide {len(bad)} tensor(es).\n"
            f"  primeiro: {bad[0][0]} (entrada {bad[0][1]})\n"
            f"  dims de entrada no modelo: {dims}\n"
            f"  grupos validos: {ok or 'nenhum entre 32..512'}")

    return shards, entries, totals


def verify(out: Path):
    with open(out, "rb") as f:
        head = f.read(HEADER_SIZE)
    if head[:8] != MAGIC:
        sys.exit(f"{out}: magic invalido")
    version, _flags, n, index_off, index_len = struct.unpack("<IIQQQ", head[8:40])
    with open(out, "rb") as f:
        f.seek(index_off)
        idx = json.loads(f.read(index_len))

    print(f"  formato v{version} · {n} tensores · indice em {index_off:,}")
    bad = 0
    for name, e in idx["tensors"].items():
        for field in ("data", "scales"):
            span = e.get(field)
            if span and span[0] % ALIGN:
                print(f"  DESALINHADO {name}.{field} @ {span[0]}")
                bad += 1
    size = out.stat().st_size
    print(f"  tamanho {size / 1e9:.2f} GB · desalinhados: {bad}")
    print(f"  {'OK' if bad == 0 else 'FALHOU'}")
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0,
                    help="converter so os N primeiros shards (teste)")
    ap.add_argument("--rows-per-chunk", type=int, default=1024,
                    help="linhas por bloco; menor = menos pico de RAM")
    ap.add_argument("--group", type=int, default=64,
                    help="pesos por escala (32/64/128). Menor = menos erro, "
                         "mais bytes de escala. Alavanca fraca: ver docs/04")
    ap.add_argument("--clip-grid", type=int, default=13,
                    help="pontos da busca de escala MSE entre 0,70 e 1,00 do "
                         "absmax. 1 desliga (absmax puro): mais rapido, mais "
                         "erro. Nao muda o formato nem o kernel")
    ap.add_argument("--lm-head", choices=("int4", "int8"), default="int8",
                    help="lm_head e 8,9%% do trafego por token; int4 economiza "
                         "0,16 GB/token mas pode custar qualidade (default int8)")
    ap.add_argument("--mtp", choices=("int4", "int8"), default="int8",
                    help="camada MTP; a licao #8 do colibri diz int8 (default)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    global GROUP, CLIP_RATIOS
    if args.group < 8 or args.group % 8:
        sys.exit(f"--group {args.group}: precisa ser multiplo de 8 e >= 8")
    GROUP = args.group
    if args.clip_grid < 1:
        sys.exit(f"--clip-grid {args.clip_grid}: precisa ser >= 1")
    CLIP_RATIOS = np.linspace(1.0, 0.70, args.clip_grid, dtype=np.float32)

    src, out = Path(args.src), Path(args.out)

    if args.verify:
        print(f"container {out}")
        return 0 if verify(out) else 1

    cfg = json.loads((src / "config.json").read_text("utf-8"))
    mtp_layer = cfg["num_hidden_layers"]        # camadas 0..N-1 sao o modelo;
                                                # a de indice N e a cabeca MTP
    print(f"fonte   {src}")
    print(f"destino {out}")
    print(f"camadas {cfg['num_hidden_layers']} + MTP em model.layers.{mtp_layer}")

    print(f"politica lm_head={args.lm_head} mtp={args.mtp} "
          f"group={GROUP} clip-grid={args.clip_grid}")

    shards, entries, totals = plan(src, mtp_layer, args.lm_head, args.mtp)
    if args.limit:
        keep = set(shards[:args.limit])
        entries = [e for e in entries if e[0] in keep]
        # Recontar: senao a tabela mostra o modelo inteiro e o TOTAL mostra o
        # subconjunto -- duas verdades diferentes no mesmo quadro.
        totals = {}
        for _shard, name, info, _base, kind in entries:
            n = int(np.prod(info["shape"])) if info["shape"] else 1
            agg = totals.setdefault(kind, [0, 0])
            agg[0] += 1
            agg[1] += n * bytes_per_weight(kind)

    print(f"\n{'classe':<8} {'tensores':>9} {'saida':>10}")
    print("-" * 30)
    est = 0.0
    for kind in ("int4", "int8", "f32", "bf16"):
        if kind in totals:
            count, size = totals[kind]
            est += size
            print(f"{kind:<8} {count:>9,} {size / 1e9:>9.2f}G")
    print(f"{'TOTAL':<8} {len(entries):>9,} {est / 1e9:>9.2f}G")

    if args.dry_run:
        print("\n(dry-run: nada escrito)")
        return 0

    # shutil.disk_usage funciona nos dois SOs; os.statvfs nao existe no Windows.
    import shutil
    probe = out.parent
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    need_gb = est / 1e9 * 1.1
    free_gb = shutil.disk_usage(probe).free / 1e9
    if free_gb < need_gb:
        sys.exit(f"espaco insuficiente em {probe}: {free_gb:.1f} GB livres, "
                 f"precisa de ~{need_gb:.1f} GB")

    writer = Writer(out)
    err_stats, done, written, failure = {}, 0, 0, None
    started = time.time()
    by_shard = {}
    for shard, name, info, base, kind in entries:
        by_shard.setdefault(shard, []).append((name, info, base, kind))

    try:
        for shard in shards:
            if shard not in by_shard:
                continue
            path = src / shard
            with open(path, "rb") as f, \
                 mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for name, info, base, kind in by_shard[shard]:
                    src_dtype = info["dtype"]
                    if src_dtype not in ST_DTYPES:
                        raise ValueError(f"{name}: dtype {src_dtype} nao suportado")
                    np_dtype, width = ST_DTYPES[src_dtype]
                    lo, hi = info["data_offsets"]
                    raw = np.frombuffer(mm, dtype=np_dtype,
                                        count=(hi - lo) // width,
                                        offset=base + lo).reshape(info["shape"])
                    try:
                        written += convert_tensor(name, kind, raw, src_dtype,
                                                  writer, args.rows_per_chunk,
                                                  err_stats)
                    except Exception as error:
                        # frombuffer nao copia, e o traceback mantem vivos os
                        # frames -- com eles, as views para dentro do mmap. Sem
                        # limpar os frames, o `with` estoura com "cannot close
                        # exported pointers exist" e MASCARA o erro real, que e
                        # o unico que importa ver. Guarda, limpa, relanca fora.
                        traceback.clear_frames(error.__traceback__)
                        failure = (name, error)
                    del raw                     # sempre: sucesso ou falha
                    if failure:
                        break
                    done += 1
            if failure:
                bad, error = failure
                raise RuntimeError(f"falhou em {bad}: {error}") from None
            pct = 100 * done / len(entries)
            rate = written / 1e6 / max(time.time() - started, 1e-9)
            print(f"  [{pct:5.1f}%] {shard}  ·  {written / 1e9:6.2f} GB "
                  f"escritos  ·  {rate:5.0f} MB/s", flush=True)

        writer.close({
            "model": cfg.get("_name_or_path", "GLM-4.7-Flash"),
            "source": str(src), "group": GROUP,
            "clip_grid": len(CLIP_RATIOS),
            "num_hidden_layers": cfg["num_hidden_layers"],
            "mtp_layer": mtp_layer,
            "config": cfg,
        })
    except KeyboardInterrupt:
        writer.f.close()
        writer.tmp.unlink(missing_ok=True)
        print("\ninterrompido; parcial descartado")
        return 130

    elapsed = time.time() - started
    print(f"\n  {done:,} tensores · {written / 1e9:.2f} GB · {elapsed / 60:.1f} min")

    fallbacks = err_stats.pop("_fallback_f32", [])
    if fallbacks:
        print(f"\n  {len(fallbacks)} tensor(es) caíram para f32 por não serem "
              f"matriz — revise a POLICY:")
        for n in fallbacks[:10]:
            print(f"    {n}")

    print("\n  erro de quantizacao (o que decide e a coluna 'saida'):")
    print(f"    {'classe':<7} {'por peso':>10} {'saida matmul':>14} "
          f"{'pior saida':>12} {'n':>7}")
    for kind, s in sorted(err_stats.items()):
        print(f"    {kind:<7} {np.mean(s['peso']):>9.2%} "
              f"{np.mean(s['saida']):>13.2%} {max(s['saida']):>11.2%} "
              f"{len(s['saida']):>7,}")
    print(f"    grupo={GROUP} · clip-grid={len(CLIP_RATIOS)}")
    print("    O erro da saida NAO cai com o tamanho da soma: em y=Wx o ruido e")
    print("    o sinal crescem ambos com sqrt(cols), entao a razao fica")
    print("    ~sigma_e/sigma_w. Sanidade: int4/int8 deve dar ~127/7 = 18x.")
    print("    Alavancas, da mais fraca para a mais forte:")
    print("      --group 32       ~11% menos erro, +11% de arquivo. Trade ruim.")
    print("      int8 no tensor   ~18x menos erro, 2x os bytes daquele tensor.")
    print("      Hadamard/GPTQ    2-4x menos erro, formato e kernel iguais.")
    print("    E lembre que x da sonda e gaussiano iid: o numero e otimista.")

    print()
    verify(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
