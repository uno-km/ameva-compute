#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# g2p_tokens.py TOKENS_TXT "text" [--lang en-us|en-gb]
# Kokoro v0.19 phoneme ids for a text: espeak-ng phonemes (phonemizer, espeakng-loader) with the replacements of
# kokoro v0.19's phonemize(), filtered to the vocabulary of tokens.txt. Prints the phonemes on stderr and the
# comma-separated ids (without the two pad ids, which kokoro.cpp adds) on stdout.
import re
import sys

import espeakng_loader
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.wrapper import EspeakWrapper


def vocab(path):
    v = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        sym, idx = line.rsplit(" ", 1)
        v[sym] = int(idx)
    return v


def phonemize(text, lang):
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    b = EspeakBackend(language=lang, preserve_punctuation=True, with_stress=True)
    ps = b.phonemize([text])
    ps = ps[0] if ps else ""
    ps = ps.replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ").replace("kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ")
    ps = ps.replace("ʲ", "j").replace("r", "ɹ").replace("x", "k").replace("ɬ", "l")
    ps = re.sub(r"(?<=[a-zɹː])(?=hˈʌndɹɪd)", " ", ps)
    ps = re.sub(r' z(?=[;:,.!?¡¿—…"«»“” ]|$)', "z", ps)
    if lang == "en-us":
        ps = re.sub(r"(?<=nˈaɪn)ti(?!ː)", "di", ps)
    return ps.strip()


def main():
    args = sys.argv[1:]
    lang = "en-us"
    if "--lang" in args:
        i = args.index("--lang")
        lang = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2:
        sys.exit("usage: g2p_tokens.py TOKENS_TXT TEXT [--lang en-us|en-gb]")
    v = vocab(args[0])
    ps = "".join(p for p in phonemize(args[1], lang) if p in v)
    sys.stderr.buffer.write((ps + "\n").encode("utf-8"))
    print(",".join(str(v[p]) for p in ps))


if __name__ == "__main__":
    main()
