#!/usr/bin/env python3
"""Gera src/fixture_kernels.h -- a referencia cruzada dos kernels.

O layout do container e definido em Python (`quant_grouped`/`dequant_grouped`
de convert_to_int4.py) e consumido em C.  Convencao tacita entre dois programas
em linguagens diferentes e onde nascem bugs de alinhamento (docs/13, D3), entao
o acordo vira um arquivo: este script quantiza com o MESMO codigo do conversor,
calcula a saida esperada com acumulacao float32 na MESMA ordem que o kernel usa,
e emite tudo como literais hexadecimais -- exatos, sem ida e volta por decimal.

Se o kernel em C divergir por um ulp, `tootega selftest` acusa aqui, e nao 40
minutos depois num oraculo de 47 camadas onde tudo e suspeito.

    python tools/gen_kernel_fixture.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "src" / "fixture_kernels.h"

O, I, GROUP = 8, 192, 64          # 3 grupos por linha; I nao e potencia de 2


def load_converter():
    """Importa convert_to_int4.py sem copiar uma linha dele: a fixture tem de
    sair do quantizador de verdade, nao de uma reimplementacao parecida."""
    spec = importlib.util.spec_from_file_location(
        "convert_to_int4", ROOT / "convert_to_int4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod.GROUP != GROUP:
        sys.exit(f"convert_to_int4.GROUP = {mod.GROUP}, fixture assume {GROUP}")
    return mod


def dot_f32(w_row, x):
    """Produto interno em float32, acumulador escalar, ordem 0..I-1.

    E a regra de docs/15 (D4) escrita em Python: a mesma ordem que o kernel
    percorre, para que a comparacao seja de bits e nao de tolerancia."""
    acc = np.float32(0.0)
    for i in range(len(x)):
        acc = np.float32(acc + np.float32(w_row[i] * x[i]))
    return acc


def hexf(v):
    """float32 -> literal hexadecimal C, exato."""
    f = float(np.float32(v))
    if f != f:
        return "(0.0f/0.0f)"
    if f in (float("inf"), float("-inf")):
        return "(1e30f*1e30f)" if f > 0 else "(-1e30f*1e30f)"
    return float.hex(f) + "f"


def emit_floats(f, name, arr, per_line=4):
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    f.write(f"static const float {name}[{flat.size}] = {{\n")
    for i in range(0, flat.size, per_line):
        chunk = ", ".join(hexf(v) for v in flat[i:i + per_line])
        f.write(f"    {chunk},\n")
    f.write("};\n\n")


def emit_ints(f, name, arr, ctype, per_line=16):
    flat = np.asarray(arr).reshape(-1)
    f.write(f"static const {ctype} {name}[{flat.size}] = {{\n")
    for i in range(0, flat.size, per_line):
        chunk = ", ".join(str(int(v)) for v in flat[i:i + per_line])
        f.write(f"    {chunk},\n")
    f.write("};\n\n")


def main():
    conv = load_converter()

    rng = np.random.default_rng(20260725)
    # Distribuicao com cauda: grupos com absmax muito diferente entre si sao o
    # caso que expoe erro de indexacao de escala.
    w = (rng.standard_normal((O, I)) * rng.uniform(0.05, 3.0, (O, 1))).astype(np.float32)
    w[0, :GROUP] = 0.0                      # grupo todo zero -> escala 1.0
    w[1, 0] = 6.0                           # absmax isolado
    x = rng.standard_normal(I).astype(np.float32)

    packed4, scales4 = conv.quant_grouped(w, 4)
    deq4 = conv.dequant_grouped(packed4, scales4, 4, I)

    codes8, scales8 = conv.quant_grouped(w, 8)
    deq8 = conv.dequant_grouped(codes8, scales8, 8, I)

    # bf16 = 16 bits altos de um float32 (docs/14).  Truncamento, nao
    # arredondamento -- e o que o conversor faz na volta.
    bits = w.view(np.uint32)
    bf16 = (bits >> 16).astype(np.uint16)
    deq_bf16 = (bf16.astype(np.uint32) << 16).view(np.float32)

    y_f32  = np.array([dot_f32(w[o],        x) for o in range(O)], dtype=np.float32)
    y_int4 = np.array([dot_f32(deq4[o],     x) for o in range(O)], dtype=np.float32)
    y_int8 = np.array([dot_f32(deq8[o],     x) for o in range(O)], dtype=np.float32)
    y_bf16 = np.array([dot_f32(deq_bf16[o], x) for o in range(O)], dtype=np.float32)

    # rmsnorm em float32, soma dos quadrados em ordem fixa (docs/15).
    g = rng.uniform(0.5, 1.5, I).astype(np.float32)
    ss = np.float32(0.0)
    for i in range(I):
        ss = np.float32(ss + np.float32(x[i] * x[i]))
    inv = np.float32(1.0 / np.sqrt(np.float32(ss / np.float32(I)) + np.float32(1e-6)))
    y_norm = np.array([np.float32(np.float32(x[i] * inv) * g[i]) for i in range(I)],
                      dtype=np.float32)

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"""/* fixture_kernels.h -- GERADO por tools/gen_kernel_fixture.py.  Nao edite.
 *
 * Referencia cruzada Python -> C dos kernels.  Os pesos sao quantizados pelo
 * proprio `convert_to_int4.py` e as saidas sao acumuladas em float32 na mesma
 * ordem que o kernel usa, entao a comparacao e de BITS, nao de tolerancia.
 *
 * O = {O}, I = {I}, group = {GROUP}
 */
#ifndef TOOTEGA_FIXTURE_KERNELS_H
#define TOOTEGA_FIXTURE_KERNELS_H

#include <stdint.h>

#define FX_O     {O}
#define FX_I     {I}
#define FX_GROUP {GROUP}

""")
        emit_floats(f, "fx_x", x)
        emit_floats(f, "fx_w_f32", w)
        emit_ints(f, "fx_w_int4", packed4, "uint8_t")
        emit_floats(f, "fx_scales_int4", scales4)
        emit_floats(f, "fx_w_deq4", deq4)
        emit_ints(f, "fx_w_int8", codes8.astype(np.int16), "int8_t")
        emit_floats(f, "fx_scales_int8", scales8)
        emit_floats(f, "fx_w_deq8", deq8)
        emit_ints(f, "fx_w_bf16", bf16, "uint16_t")
        emit_floats(f, "fx_y_f32", y_f32)
        emit_floats(f, "fx_y_int4", y_int4)
        emit_floats(f, "fx_y_int8", y_int8)
        emit_floats(f, "fx_y_bf16", y_bf16)
        emit_floats(f, "fx_norm_w", g)
        emit_floats(f, "fx_norm_y", y_norm)
        f.write("#endif /* TOOTEGA_FIXTURE_KERNELS_H */\n")

    rel = deq4 - w
    print(f"{OUT} escrito")
    print(f"  {O}x{I}, grupo {GROUP}")
    print(f"  erro relativo da saida int4: "
          f"{np.linalg.norm(y_int4 - y_f32) / np.linalg.norm(y_f32) * 100:.2f}%")
    print(f"  erro relativo da saida int8: "
          f"{np.linalg.norm(y_int8 - y_f32) / np.linalg.norm(y_f32) * 100:.2f}%")
    print(f"  erro relativo por peso int4: "
          f"{np.linalg.norm(rel) / np.linalg.norm(w) * 100:.2f}%")


if __name__ == "__main__":
    main()
