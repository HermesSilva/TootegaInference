#!/usr/bin/env python3
"""Gera tokfixture.json -- ids de referencia do `tokenizers` da HuggingFace.

docs/16: "o teste que importa" nao e round-trip, e igualdade contra a
referencia.  Um tokenizer errado faz o oraculo do M1 falhar com o forward
perfeitamente correto, e a depuracao comeca no lugar errado.

O corpus cobre o que o padrao do cl100k trata de forma especial: contracoes,
sequencias de espacos, quebras de linha, digitos em grupos de tres, simbolos
grudados, CJK, emoji (par substituto), acentos e tokens especiais.

    python tools/gen_tokenizer_fixture.py
    tootega tokenize models/glm47flash-bf16/tokenizer.json --check tokfixture.json
"""

import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = ROOT / "models" / "glm47flash-bf16" / "tokenizer.json"
OUT = ROOT / "tokfixture.json"

CORPUS = [
    "",
    "a",
    " ",
    "   ",
    "\n",
    "\r\n\r\n",
    "hello world",
    "Hello, World! How's it going?",
    "it's don't they're we've I'm you'll he'd",
    "IT'S DON'T THEY'RE WE'VE",
    "1 12 123 1234 12345 007",
    "x=1;y=2;  z = x+y  // soma",
    "     espacos     no     meio     ",
    "linha um\nlinha dois\n\n\nlinha tres",
    "tabs\tentre\tpalavras",
    "acentuação: coração, ímã, ëxôtiço, ñandu",
    "中文字符测试",
    "日本語のテキスト",
    "한국어 텍스트",
    "русский текст",
    "emoji 😀 e 👨‍👩‍👧‍👦 familia",
    "misto 中文 with english e 123 números!",
    "```python\ndef f(x):\n    return x ** 2\n```",
    "{\"json\": [1, 2, 3], \"nested\": {\"a\": null}}",
    "https://exemplo.com/path?query=1&other=2#frag",
    "C:\\Users\\Hermes\\Documents\\arquivo.txt",
    "muitos!!!???...,,,;;;pontos",
    "<|endoftext|>",
    "antes<|endoftext|>depois",
    "<|user|>oi<|assistant|>ola",
    "[MASK] [gMASK] [sMASK] <sop> <eop>",
    "a" * 200,
    "palavra " * 50,
    "0x1F 0b1010 1e10 3.14159 -42",
    "e-mail: nome.sobrenome+tag@dominio.com.br",
    "aaaa\u00a0bbbb\u2003cccc\u3000dddd",   # espacos Unicode que nao sao 0x20
]


def main():
    tok = Tokenizer.from_file(str(TOKENIZER))

    cases = []
    for text in CORPUS:
        ids = tok.encode(text, add_special_tokens=False).ids
        back = tok.decode(ids, skip_special_tokens=False)
        cases.append({"text": text, "ids": ids, "decoded": back})

    OUT.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    total = sum(len(c["ids"]) for c in cases)
    exact = sum(1 for c in cases if c["decoded"] == c["text"])
    print(f"{OUT} escrito")
    print(f"  {len(cases)} casos, {total} ids")
    print(f"  round-trip exato na referencia: {exact}/{len(cases)}")


if __name__ == "__main__":
    main()
