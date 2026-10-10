#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# reference_onnx.py MODEL_DIR OUT_DIR --tokens 1,2,3 [--voice N] [--speed S] [--seed K]
# Float, deterministic onnxruntime reference for adapters/tts/kokoro (runs on a PC: needs onnx and onnxruntime).
#
# The sherpa-onnx Kokoro file is rewritten into a float graph with the same weights the GGUF converter writes:
#   - MatMulInteger / ConvInteger / DynamicQuantizeLSTM -> MatMul / Conv / LSTM on the dequantized weights (the float
#     original where the file holds one), so activations are not quantized;
#   - RandomNormalLike (the SineGen noise) -> a slice of the input `noise`, generated from --seed by kokoro_noise()
#     below, the same function as kokoro.cpp's; RandomUniformLike (the initial phase) -> zeros. The initial phase
#     is added to sample 0 only and the export's 1/300 linear down-sampling never reads sample 0, so it has no effect;
#   - the STFT of the harmonic source -> a 20-tap Conv with the converter's basis: onnxruntime's STFT leaves up to
#     2e-5 in the imaginary part of bins 0 and 10, which flips the phase of a bin with a negative real part between
#     about +pi and -pi; the basis has exact zeros there, as an FFT gives.
# Writes audio.f32 (raw little-endian float32), audio.wav (16-bit PCM, 24 kHz), durations.i32 and the intermediate
# tensors named in DUMP as .f32 files with a .shape text file each.
import os
import struct
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

GAMMA = 0x9E3779B97F4A7C15


def kokoro_noise(seed, n):
    # n standard normal float32 values: splitmix64 outputs k = 0, 1, ... from `seed` as uniform (0, 1) pairs,
    # Box-Muller in double (z0 = r cos t, z1 = r sin t for the pair 2j, 2j + 1)
    m = (n + 1) // 2
    k = np.arange(2 * m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + (k + np.uint64(1)) * np.uint64(GAMMA)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = ((z >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / 9007199254740992.0)
    u1, u2 = u[0::2], u[1::2]
    r = np.sqrt(-2.0 * np.log(u1))
    t = 2.0 * np.pi * u2
    out = np.empty(2 * m, np.float64)
    out[0::2] = r * np.cos(t)
    out[1::2] = r * np.sin(t)
    return out[:n].astype(np.float32)


DUMP = {
    "bert_dur": "/bert/encoder/albert_layer_groups.0/albert_layers.0/full_layer_layer_norm_11/LayerNormalization_output_0",
    "d_en": "/bert_encoder/Add_output_0",
    "dur_raw": "/ReduceSum_output_0",
    "durations": "/Clip_output_0",
    "f0": "/If_output_0",
    "har": "/decoder/generator/Concat_3_output_0",
    "conv_post": "/decoder/generator/conv_post/Conv_output_0",
}


def dequant(init, qn, zpn):
    base = qn[: -len("_quantized")]
    if base in init and init[base].dtype == np.float32:
        return init[base]
    q, zp, sc = init[qn], init[zpn], init[base + "_scale"]
    return (q.astype(np.int32) - zp.astype(np.int32)).astype(np.float32) * sc.astype(np.float32)


def float_model(src):
    m = onnx.load(src)
    g = m.graph
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    remove, add, new_init = set(), [], []

    def fname(x):
        p = prod.get(x)
        if p is not None and p.op_type == "DynamicQuantizeLinear":
            return p.input[0]
        # a statically quantized input (the text encoder's embedding): dequantize it
        return None

    def tail(n):
        cast = [c for c in cons[n.output[0]] if c.op_type == "Cast"]
        mul = [c for c in cons[cast[0].output[0]] if c.op_type == "Mul"]
        assert len(cast) == 1 and len(mul) == 1, n.name
        return cast[0], mul[0]

    for n in g.node:
        if n.op_type in ("MatMulInteger", "ConvInteger"):
            w = dequant(init, n.input[1], n.input[3])
            wn = n.name + "_Wf"
            new_init.append(numpy_helper.from_array(w, wn))
            x = fname(n.input[0])
            if x is None:
                zp = n.input[2]
                x = n.name + "_Xf"
                add.append(helper.make_node("DequantizeLinear", [n.input[0], zp.replace("_zero_point", "_scale"), zp], [x]))
            cast, mul = tail(n)
            remove |= {id(n), id(cast), id(mul)}
            if n.op_type == "MatMulInteger":
                add.append(helper.make_node("MatMul", [x, wn], [mul.output[0]], name=n.name + "_f"))
            else:
                attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
                add.append(helper.make_node("Conv", [x, wn], [mul.output[0]], name=n.name + "_f", **attrs))
        elif n.op_type == "DynamicQuantizeLSTM":
            wq, rq = init[n.input[1]], init[n.input[2]]
            ws, wz, rs, rz = (init[n.input[i]] for i in (8, 9, 10, 11))
            w = (wq.astype(np.int32) - wz.astype(np.int32)[:, None, None]).astype(np.float32) * ws[:, None, None]
            r = (rq.astype(np.int32) - rz.astype(np.int32)[:, None, None]).astype(np.float32) * rs[:, None, None]
            new_init += [numpy_helper.from_array(np.ascontiguousarray(w.transpose(0, 2, 1)), n.name + "_W"),
                         numpy_helper.from_array(np.ascontiguousarray(r.transpose(0, 2, 1)), n.name + "_R")]
            ins = [n.input[0], n.name + "_W", n.name + "_R"] + list(n.input[3:8])
            attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            add.append(helper.make_node("LSTM", ins, list(n.output), name=n.name + "_f", **attrs))
            remove.add(id(n))
        elif n.op_type == "RandomNormalLike":
            shp = n.name + "_shape"
            st = n.name + "_starts"
            new_init.append(numpy_helper.from_array(np.zeros(3, np.int64), st))
            add += [helper.make_node("Shape", [n.input[0]], [shp]),
                    helper.make_node("Slice", ["noise", st, shp], [n.output[0]])]
            remove.add(id(n))
        elif n.op_type == "RandomUniformLike":
            add.append(helper.make_node("Identity", [n.input[0]], [n.output[0]]))  # ConstantOfShape zeros
            remove.add(id(n))
        elif n.op_type == "STFT":
            # [1, L] -> Conv over [1, 1, L] with the exact basis -> [1, 22, frames] -> [1, frames, 11, 2]
            k = np.arange(20, dtype=np.float64)
            win = numpy_helper.to_array([c for c in g.node if c.output[0] == n.input[2]][0].attribute[0].t)
            basis = np.zeros((22, 1, 20), np.float64)
            for b in range(11):
                basis[b, 0] = win * np.cos(2 * np.pi * b * k / 20)
                if b not in (0, 10):
                    basis[11 + b, 0] = -win * np.sin(2 * np.pi * b * k / 20)
            p = n.name
            new_init += [numpy_helper.from_array(basis.astype(np.float32), p + "_basis"),
                         numpy_helper.from_array(np.array([1, 2, 11, -1], np.int64), p + "_rs"),
                         numpy_helper.from_array(np.array([1], np.int64), p + "_ax")]
            add += [helper.make_node("Unsqueeze", [n.input[0], p + "_ax"], [p + "_x3"]),
                    helper.make_node("Conv", [p + "_x3", p + "_basis"], [p + "_c"], strides=[5]),
                    helper.make_node("Reshape", [p + "_c", p + "_rs"], [p + "_r"]),
                    helper.make_node("Transpose", [p + "_r"], [n.output[0]], perm=[0, 3, 2, 1])]
            remove.add(id(n))
    nodes = [n for n in g.node if id(n) not in remove] + add
    # topological order
    avail = {i.name for i in g.input} | {t.name for t in g.initializer} | {t.name for t in new_init} | {"noise", ""}
    ordered, pending = [], nodes
    while pending:
        rest = []
        for n in pending:
            if all(i in avail for i in n.input):
                ordered.append(n)
                avail |= set(n.output)
            else:
                rest.append(n)
        if len(rest) == len(pending):
            raise RuntimeError(f"cannot order {[n.name for n in rest[:5]]}")
        pending = rest
    del g.node[:]
    g.node.extend(ordered)
    g.initializer.extend(new_init)
    g.input.append(helper.make_tensor_value_info("noise", TensorProto.FLOAT, [1, None, 9]))
    for k, v in DUMP.items():
        g.output.append(helper.make_tensor_value_info(v, TensorProto.FLOAT, None))
    return m


def write_wav(path, audio, sr=24000):
    pcm = np.clip(np.round(audio * 32767.0), -32768, 32767).astype("<i2")
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + pcm.nbytes) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data" + struct.pack("<I", pcm.nbytes) + pcm.tobytes())


def main():
    args = sys.argv[1:]
    opts = {"--tokens": None, "--voice": "0", "--speed": "1.0", "--seed": "1234", "--max-samples": "720000"}
    pos = []
    i = 0
    while i < len(args):
        if args[i] in opts:
            opts[args[i]] = args[i + 1]
            i += 2
        else:
            pos.append(args[i])
            i += 1
    if len(pos) != 2 or not opts["--tokens"]:
        sys.exit("usage: reference_onnx.py MODEL_DIR OUT_DIR --tokens 1,2,3 [--voice N] [--speed S] [--seed K]")
    model_dir, out_dir = pos
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "..", "kokoro-float-det.onnx")
    if not os.path.exists(cache):
        onnx.save(float_model(os.path.join(model_dir, "model.int8.onnx")), cache)
    phon = [int(t) for t in opts["--tokens"].split(",")]
    voices = np.fromfile(os.path.join(model_dir, "voices.bin"), "<f4").reshape(-1, 511, 256)
    style = voices[int(opts["--voice"]), min(len(phon), 510)][None, :]
    tokens = np.array([[0] + phon + [0]], np.int64)
    nmax = int(opts["--max-samples"])
    noise = kokoro_noise(int(opts["--seed"]), nmax * 9).reshape(1, nmax, 9)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(cache, so, providers=["CPUExecutionProvider"])
    names = ["audio"] + list(DUMP.values())
    res = s.run(names, {"tokens": tokens, "style": style.astype(np.float32),
                        "speed": np.array([float(opts["--speed"])], np.float32), "noise": noise})
    audio = res[0].reshape(-1)
    audio.astype("<f4").tofile(os.path.join(out_dir, "audio.f32"))
    write_wav(os.path.join(out_dir, "audio.wav"), audio)
    for (k, _), v in zip(DUMP.items(), res[1:]):
        if k == "durations":
            v.astype("<i4").tofile(os.path.join(out_dir, "durations.i32"))
        np.ascontiguousarray(v, np.float32).tofile(os.path.join(out_dir, k + ".f32"))
        open(os.path.join(out_dir, k + ".shape"), "w").write(",".join(map(str, v.shape)))
    dur = res[1 + list(DUMP).index("durations")].reshape(-1)
    print(f"tokens {tokens.shape[1]} frames {int(dur.sum())} samples {audio.size} ({audio.size / 24000:.2f} s) "
          f"rms {float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))):.4f} peak {float(np.abs(audio).max()):.4f}")


if __name__ == "__main__":
    main()
