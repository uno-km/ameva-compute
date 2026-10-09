#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# gguf_dequant_f32.py GGUFPY_DIR SRC.gguf DST.gguf
# Copy a GGUF with every quantized (and f16) tensor converted to F32, metadata and other tensors unchanged: the exact
# weight values of SRC with no quantization left, the reference of tests/ameva_vision_parity.cpp --ref-mmproj.
# GGUFPY_DIR is the gguf-py directory of a llama.cpp checkout (Termux has no PyYAML; gguf.metadata's import of it is
# satisfied by an empty module, the metadata helpers are not used).
import sys, types
sys.path.insert(0, sys.argv[1])
sys.modules.setdefault("yaml", types.ModuleType("yaml"))  # gguf.metadata imports yaml; not used here
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType
from gguf.quants import dequantize

src, dst = sys.argv[2], sys.argv[3]
r = GGUFReader(src)
arch = r.fields["general.architecture"].contents()
w = GGUFWriter(dst, arch)
for f in r.fields.values():
    if f.name == "general.architecture" or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]
    if vt == GGUFValueType.ARRAY:
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1])
    else:
        w.add_key_value(f.name, f.contents(), vt)
n_deq = 0
for t in r.tensors:
    if t.tensor_type in (GGMLQuantizationType.F32,):
        w.add_tensor(t.name, np.asarray(t.data))
    elif t.tensor_type == GGMLQuantizationType.F16:
        w.add_tensor(t.name, np.asarray(t.data).astype(np.float32))
        n_deq += 1
    else:
        d = dequantize(t.data, t.tensor_type).astype(np.float32)
        w.add_tensor(t.name, d)
        n_deq += 1
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("dequantized", n_deq, "of", len(r.tensors), "tensors ->", dst)
