#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# convert_onnx_to_gguf.py MODEL_DIR OUT.gguf [--wtype f16|f32] [--mm-type f16|f32]
# Kokoro-82M v0.19 (the sherpa-onnx package kokoro-int8-en-v0_19: model.int8.onnx, voices.bin) to the GGUF that
# adapters/tts/kokoro/kokoro.cpp reads. Needs only python3 and numpy (no onnx package), so it runs on a Termux device
# against the model file already there.
#
# The ONNX file is onnxruntime's dynamic int8 quantization of hexgrad's export: MatMulInteger / ConvInteger /
# DynamicQuantizeLSTM weights are uint8 or int8 with a per-tensor (per-direction for LSTM) scale and zero point.
# Every weight is dequantized here exactly as onnxruntime's DequantizeLinear does it, (float)(q - zp) * scale, except
# where the file still holds the float original: the ALBERT layer is shared by 12 unrolled calls, the first call
# reads the quantized copy and the other eleven the float initializer, which is taken for all twelve.
# Weights are located through the node names of the export (they carry the module paths), and every expected node
# must be found: a missing one stops the conversion.
#
# --wtype: storage type of the matrices (conv kernels, linear and LSTM weights). f16 is the product file; f32 keeps the
# dequantized values unrounded and is the reference for parity runs. Vectors (biases, norms, Snake alphas, the
# depthwise transposed-conv taps), embeddings, voices and the STFT bases are always f32. --mm-type (default: --wtype)
# overrides the type of the MUL_MAT weights (linear layers, LSTMs, the transposed convolutions' tap matrices) alone.
import hashlib
import struct
import sys
from collections import defaultdict

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Minimal protobuf reader for the parts of onnx.proto used here
# ----------------------------------------------------------------------------------------------------------------------


def _varint(b, i):
    r = 0
    s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        if c < 0x80:
            return r, i
        s += 7


def _fields(b):
    i, end = 0, len(b)
    while i < end:
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v = b[i:i + 8]
            i += 8
        elif wt == 2:
            n, i = _varint(b, i)
            v = b[i:i + n]
            i += n
        elif wt == 5:
            v = b[i:i + 4]
            i += 4
        else:
            raise ValueError(f"protobuf wire type {wt} not handled")
        yield f, wt, v


def _signed64(v):
    return v - (1 << 64) if v >= (1 << 63) else v


def _packed_varints(b):
    out, i = [], 0
    while i < len(b):
        v, i = _varint(b, i)
        out.append(_signed64(v))
    return out


_DTYPES = {1: np.float32, 2: np.uint8, 3: np.int8, 6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.float64}


def _tensor(b):
    dims, dt, name, raw = [], None, "", None
    f32, i32, i64 = [], [], []
    for f, wt, v in _fields(b):
        if f == 1:
            dims += _packed_varints(v) if wt == 2 else [_signed64(v)]
        elif f == 2:
            dt = v
        elif f == 4:
            f32 += list(np.frombuffer(v, "<f4")) if wt == 2 else [struct.unpack("<f", v)[0]]
        elif f == 5:
            i32 += _packed_varints(v) if wt == 2 else [_signed64(v)]
        elif f == 7:
            i64 += _packed_varints(v) if wt == 2 else [_signed64(v)]
        elif f == 8:
            name = bytes(v).decode()
        elif f == 9:
            raw = v
    if dt not in _DTYPES:
        raise ValueError(f"tensor {name}: data type {dt} not handled")
    npt = np.dtype(_DTYPES[dt]).newbyteorder("<")
    if raw is not None:
        a = np.frombuffer(raw, npt)
    elif f32:
        a = np.array(f32, np.float32)
    elif i32:
        a = np.array(i32, np.int64).astype(npt)
    elif i64:
        a = np.array(i64, np.int64).astype(npt)
    else:
        a = np.zeros(0, npt)
    n = int(np.prod(dims)) if dims else 1
    if a.size != n:
        raise ValueError(f"tensor {name}: {a.size} values for dims {dims}")
    return name, a.reshape(dims) if dims else a.reshape(())


class Node:
    __slots__ = ("op", "name", "inputs", "outputs", "attrs")


def _node(b):
    n = Node()
    n.op, n.name, n.inputs, n.outputs, n.attrs = "", "", [], [], {}
    for f, wt, v in _fields(b):
        if f == 1:
            n.inputs.append(bytes(v).decode())
        elif f == 2:
            n.outputs.append(bytes(v).decode())
        elif f == 3:
            n.name = bytes(v).decode()
        elif f == 4:
            n.op = bytes(v).decode()
        elif f == 5:
            an, val = None, None
            ints = []
            for af, awt, av in _fields(v):
                if af == 1:
                    an = bytes(av).decode()
                elif af == 2:
                    val = struct.unpack("<f", av)[0]
                elif af == 3:
                    val = _signed64(av)
                elif af == 4:
                    val = bytes(av)
                elif af == 5:
                    val = _tensor(av)[1]
                elif af == 8:
                    ints += _packed_varints(av) if awt == 2 else [_signed64(av)]
            n.attrs[an] = ints if ints else val
    return n


def load_onnx(path):
    data = memoryview(open(path, "rb").read())
    meta, graph = {}, None
    for f, wt, v in _fields(data):
        if f == 7:
            graph = v
        elif f == 14:
            kv = {ff: bytes(vv).decode() for ff, _, vv in _fields(v)}
            meta[kv.get(1, "")] = kv.get(2, "")
    if graph is None:
        raise ValueError("no graph in the ONNX file")
    nodes, inits = [], {}
    for f, wt, v in _fields(graph):
        if f == 1:
            nodes.append(_node(v))
        elif f == 5:
            name, a = _tensor(v)
            inits[name] = a
    return meta, nodes, inits


# ----------------------------------------------------------------------------------------------------------------------
# Graph lookups
# ----------------------------------------------------------------------------------------------------------------------


class Graph:
    def __init__(self, nodes, inits):
        self.nodes = nodes
        self.init = inits
        self.by_name = {n.name: n for n in nodes}
        self.prod = {o: n for n in nodes for o in n.outputs}
        self.cons = defaultdict(list)
        for n in nodes:
            for i in n.inputs:
                self.cons[i].append(n)
        self.const = {n.outputs[0]: n.attrs["value"] for n in nodes if n.op == "Constant" and "value" in n.attrs}
        self.used = set()

    def node(self, name, op=None):
        n = self.by_name.get(name)
        if n is None or (op and n.op != op):
            raise KeyError(f"node {name} ({op}) not in the graph")
        return n

    def has(self, name):
        return name in self.by_name

    def value(self, name):
        # an initializer, possibly behind Identity nodes
        while name not in self.init and name in self.prod and self.prod[name].op == "Identity":
            name = self.prod[name].inputs[0]
        if name not in self.init:
            raise KeyError(f"{name} is not an initializer")
        self.used.add(name)
        return self.init[name]

    def dequant(self, q_name, zp_name):
        base = q_name[: -len("_quantized")] if q_name.endswith("_quantized") else None
        if base and base in self.init and self.init[base].dtype == np.float32:
            return self.value(base).astype(np.float32)  # the float original is still in the file
        q = self.value(q_name)
        zp = self.value(zp_name)
        sc = self.value(base + "_scale")
        return (q.astype(np.int32) - zp.astype(np.int32)).astype(np.float32) * sc.astype(np.float32)

    def bias_after(self, n):
        # MatMulInteger / ConvInteger -> Cast -> Mul(scale) -> Add(bias): the bias initializer, or None
        x = n.outputs[0]
        for _ in range(2):
            nxt = [c for c in self.cons[x] if c.op in ("Cast", "Mul")]
            if len(nxt) != 1:
                return None
            x = nxt[0].outputs[0]
        for c in self.cons[x]:
            if c.op != "Add":
                continue
            for i in c.inputs:
                if i == x:
                    continue
                if i in self.init:
                    return self.value(i)
                p = self.prod.get(i)
                if p is not None and p.op == "Reshape" and p.inputs[0] in self.init:
                    return self.value(p.inputs[0])
        return None


# ----------------------------------------------------------------------------------------------------------------------
# Kokoro modules
# ----------------------------------------------------------------------------------------------------------------------


class Out:
    def __init__(self, wtype, mmtype):
        self.wtype = wtype    # conv kernels
        self.mmtype = mmtype  # MUL_MAT weights (linear, LSTM, transposed conv, inverse STFT basis excluded)
        self.t = []
        self.names = set()
        self.conv = {}  # conv name -> "stride,pad,dilation" (group 1, symmetric padding)

    def add(self, name, a, matrix=False):
        if len(name) >= 64:
            raise ValueError(f"tensor name {name} exceeds GGML_MAX_NAME")
        if name in self.names:
            raise ValueError(f"tensor {name} written twice")
        a = np.ascontiguousarray(a, np.float32)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"tensor {name} has non-finite values")
        if (matrix == "conv" and self.wtype == "f16") or (matrix == "mm" and self.mmtype == "f16"):
            a = a.astype(np.float16)
        self.names.add(name)
        self.t.append((name, a))


def linear(g, out, name, node_name, bias=True):
    # MatMul(X, W) with W [in, out] -> ggml [ne0 = in, ne1 = out]
    n = g.node(node_name, "MatMulInteger")
    w = g.dequant(n.inputs[1], n.inputs[3])
    out.add(name + ".w", w.T, matrix="mm")
    b = g.bias_after(n)
    if bias:
        if b is None:
            raise KeyError(f"{node_name}: bias not found")
        out.add(name + ".b", b)


def conv(g, out, name, node_name, bias=True):
    # Conv weight [Cout, Cin, K] -> CONV_2D kernel [KW = K, KH = 1, IC = Cin, OC = Cout]
    n = g.node(node_name, "ConvInteger")
    w = g.dequant(n.inputs[1], n.inputs[3])
    if w.ndim != 3:
        raise ValueError(f"{node_name}: weight dims {w.shape}")
    out.add(name + ".w", w.reshape(w.shape[0], w.shape[1], 1, w.shape[2]), matrix="conv")
    a = n.attrs
    pads = a.get("pads", [0, 0])
    if a.get("group", 1) != 1 or pads[0] != pads[1] or a.get("kernel_shape", [w.shape[2]]) != [w.shape[2]]:
        raise ValueError(f"{node_name}: attributes {a}")
    out.conv[name] = f"{a.get('strides', [1])[0]},{pads[0]},{a.get('dilations', [1])[0]}"
    b = g.bias_after(n)
    if bias:
        if b is None:
            raise KeyError(f"{node_name}: bias not found")
        out.add(name + ".b", b)
    elif b is not None:
        raise ValueError(f"{node_name}: unexpected bias")
    return n


def lstm(g, out, name, node_name):
    # DynamicQuantizeLSTM: W [dir, in, 4H], R [dir, H, 4H] (onnxruntime's transposed layout), B [dir, 8H] = Wb | Rb,
    # gate order i, o, f, c. Stored: wi [dir, 4H, in], wh [dir, 4H, H] (ggml ne0 = in / H), b [dir, 4H] = Wb + Rb.
    n = g.node(node_name, "DynamicQuantizeLSTM")
    if n.attrs.get("direction") != b"bidirectional":
        raise ValueError(f"{node_name}: direction {n.attrs.get('direction')}")
    hidden = n.attrs["hidden_size"]
    wq, rq, b = g.value(n.inputs[1]), g.value(n.inputs[2]), g.value(n.inputs[3])
    ws, wz, rs, rz = (g.value(n.inputs[i]) for i in (8, 9, 10, 11))
    w = (wq.astype(np.int32) - wz.astype(np.int32)[:, None, None]).astype(np.float32) * ws.astype(np.float32)[:, None, None]
    r = (rq.astype(np.int32) - rz.astype(np.int32)[:, None, None]).astype(np.float32) * rs.astype(np.float32)[:, None, None]
    if w.shape[2] != 4 * hidden or r.shape[1] != hidden or b.shape != (2, 8 * hidden):
        raise ValueError(f"{node_name}: shapes {w.shape} {r.shape} {b.shape}")
    for i in (4, 5, 6, 7):
        if n.inputs[i] and n.inputs[i] in g.init:
            raise ValueError(f"{node_name}: input {i} is an initializer; zero initial state expected")
    out.add(name + ".wi", w.transpose(0, 2, 1), matrix="mm")
    out.add(name + ".wh", r.transpose(0, 2, 1), matrix="mm")
    out.add(name + ".b", b[:, : 4 * hidden] + b[:, 4 * hidden:])


def adain_fc(g, out, name, node_prefix):
    linear(g, out, name, node_prefix + "/fc/Gemm_MatMul_quant")


def adain_resblk1d(g, out, name, p):
    # AdainResBlk1d: norm1 (AdaIN), leaky 0.2, [pool], conv1, norm2, leaky 0.2, conv2; shortcut [upsample] [conv1x1]
    adain_fc(g, out, name + ".n1", p + "/norm1")
    conv(g, out, name + ".c1", p + "/conv1/Conv_quant")
    adain_fc(g, out, name + ".n2", p + "/norm2")
    conv(g, out, name + ".c2", p + "/conv2/Conv_quant")
    if g.has(p + "/conv1x1/Conv_quant"):
        conv(g, out, name + ".sc", p + "/conv1x1/Conv_quant", bias=False)
    if g.has(p + "/pool/ConvTranspose"):
        n = g.node(p + "/pool/ConvTranspose", "ConvTranspose")
        a = n.attrs
        c = g.value(n.inputs[1])
        if (a.get("group") != c.shape[0] or a.get("strides") != [2] or a.get("pads") != [1, 1]
                or a.get("output_padding") != [1] or c.shape[1:] != (1, 3)):
            raise ValueError(f"{p}/pool: attributes {a} weight {c.shape}")
        out.add(name + ".pool.w", c[:, 0, :].T)  # [3, C]: row k = tap k of every channel
        out.add(name + ".pool.b", g.value(n.inputs[2]))


def adain_resblock1(g, out, name, p, onnx_name):
    # generator AdaINResBlock1: three (AdaIN, Snake, dilated conv, AdaIN, Snake, conv) residual units
    for j in range(3):
        adain_fc(g, out, f"{name}.n1_{j}", f"{p}/adain1.{j}")
        adain_fc(g, out, f"{name}.n2_{j}", f"{p}/adain2.{j}")
        c1 = conv(g, out, f"{name}.c1_{j}", f"{p}/convs1.{j}/Conv_quant")
        c2 = conv(g, out, f"{name}.c2_{j}", f"{p}/convs2.{j}/Conv_quant")
        for nm, cn in ((f"{name}.c1_{j}", c1), (f"{name}.c2_{j}", c2)):
            if cn.attrs.get("strides", [1]) != [1] or cn.attrs.get("group", 1) != 1:
                raise ValueError(f"{nm}: attributes {cn.attrs}")
        for a in (1, 2):
            al = g.value(f"{onnx_name}.alpha{a}.{j}").reshape(-1)
            out.add(f"{name}.a{a}_{j}", al)
            out.add(f"{name}.ia{a}_{j}", np.float32(1.0) / al)  # onnxruntime's Reciprocal, correctly rounded


def conv_transpose(g, out, name, node_name, stride, k):
    # ConvTranspose weight [Cin, Cout, K] -> [Cout, K, Cin] (ggml ne0 = Cin, ne1 = K, ne2 = Cout): one MUL_MAT gives
    # every tap of every output channel, followed by an overlap-add
    n = g.node(node_name, "ConvTranspose")
    w = g.value(n.inputs[1])
    if n.attrs.get("strides") != [stride] or n.attrs.get("kernel_shape") != [k] or w.shape[2] != k:
        raise ValueError(f"{node_name}: attributes {n.attrs} weight {w.shape}")
    out.add(name + ".w", w.transpose(1, 2, 0), matrix="mm")
    out.add(name + ".b", g.value(n.inputs[2]))
    return n.attrs["pads"]


def window_sum_check(g, window, hop):
    # the export divides the inverse transform by a stored window_sum (computed for 10^6 frames); the engine computes
    # it on the host, so the formula is checked against the stored values here
    stored = g.value("decoder.generator.stft.stft.window_sum")
    n = 4096
    w2 = window.astype(np.float64) ** 2
    ws = np.zeros(n + len(window), np.float64)
    for f in range(0, n // hop + 1):
        ws[f * hop: f * hop + len(window)] += w2
    diff = np.max(np.abs(ws[:n].astype(np.float32) - stored[:n]))
    if diff > 1e-6:
        raise ValueError(f"window_sum formula differs from the stored values by {diff}")


def convert(model_dir, out_path, wtype, mmtype):
    onnx_path = model_dir + "/model.int8.onnx"
    h = hashlib.sha256()
    with open(onnx_path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    meta, nodes, inits = load_onnx(onnx_path)
    if meta.get("model_type") != "kokoro" or meta.get("version") != "1":
        raise ValueError(f"not the Kokoro v0.19 export: {meta.get('model_type')} {meta.get('version')}")
    g = Graph(nodes, inits)
    out = Out(wtype, mmtype)

    # --- PL-BERT (ALBERT, one shared layer) ---
    def gather_deq(name, q):
        dq = [n for n in g.cons[q] if n.op == "Gather"]
        if not dq:
            raise KeyError(f"{q}: no Gather")
        base = q[: -len("_quantized")]
        out.add(name, g.dequant(q, base + "_zero_point"))

    gather_deq("bert.word_emb", "bert.embeddings.word_embeddings.weight_quantized")
    gather_deq("bert.pos_emb", "bert.embeddings.position_embeddings.weight_quantized")
    gather_deq("bert.type_emb", "bert.embeddings.token_type_embeddings.weight_quantized")
    out.add("bert.emb_ln.w", g.value("bert.embeddings.LayerNorm.weight"))
    out.add("bert.emb_ln.b", g.value("bert.embeddings.LayerNorm.bias"))
    linear(g, out, "bert.map_in", "/bert/encoder/embedding_hidden_mapping_in/MatMul_quant")
    al = "/bert/encoder/albert_layer_groups.0/albert_layers.0"
    an = "bert.encoder.albert_layer_groups.0.albert_layers.0"
    for short, mod in (("q", "attention/query"), ("k", "attention/key"), ("v", "attention/value"),
                       ("o", "attention/dense"), ("ffn", "ffn"), ("ffn_out", "ffn_output")):
        linear(g, out, "bert." + short, f"{al}/{mod}/MatMul_quant")
    out.add("bert.attn_ln.w", g.value(an + ".attention.LayerNorm.weight"))
    out.add("bert.attn_ln.b", g.value(an + ".attention.LayerNorm.bias"))
    out.add("bert.ffn_ln.w", g.value(an + ".full_layer_layer_norm.weight"))
    out.add("bert.ffn_ln.b", g.value(an + ".full_layer_layer_norm.bias"))
    linear(g, out, "bert_enc", "/bert_encoder/MatMul_quant")

    # --- prosody predictor ---
    for i in range(3):
        lstm(g, out, f"pred.te.lstm{i}", f"/lstms.{2 * i}/LSTM_quant")
        linear(g, out, f"pred.te.aln{i}", f"/lstms.{2 * i + 1}/fc/Gemm_MatMul_quant")
    lstm(g, out, "pred.lstm", "/lstm/LSTM_quant")
    linear(g, out, "pred.dur", "/duration_proj/linear_layer/MatMul_quant")
    lstm(g, out, "pred.shared", "/shared/LSTM_quant")
    for br in ("F0", "N"):
        for i in range(3):
            adain_resblk1d(g, out, f"pred.{br}{i}", f"/{br}.{i}")
        conv(g, out, f"pred.{br}_proj", f"/{br}_proj/Conv_quant")

    # --- text encoder ---
    q = "text_encoder.embedding.weight_quantized"
    out.add("te.emb", g.dequant(q, "text_encoder.embedding.weight_zero_point"))
    for i in range(3):
        conv(g, out, f"te.cnn{i}", f"/cnn.{i}/cnn.{i}.0/Conv_quant")
        out.add(f"te.cnn{i}.g", g.value(f"text_encoder.cnn.{i}.1.gamma"))
        out.add(f"te.cnn{i}.beta", g.value(f"text_encoder.cnn.{i}.1.beta"))
    lstm(g, out, "te.lstm", "/lstm_1/LSTM_quant")

    # --- decoder ---
    for nm in ("F0_conv", "N_conv"):
        n = conv(g, out, f"dec.{nm}", f"/decoder/{nm}/Conv_quant")
        if n.attrs.get("strides") != [2] or n.attrs.get("pads") != [1, 1]:
            raise ValueError(f"{nm}: attributes {n.attrs}")
    conv(g, out, "dec.asr_res", "/decoder/asr_res/asr_res.0/Conv_quant")
    adain_resblk1d(g, out, "dec.enc", "/decoder/encode")
    for i in range(4):
        adain_resblk1d(g, out, f"dec.dec{i}", f"/decoder/decode.{i}")

    # --- generator ---
    gp = "/decoder/generator"
    linear(g, out, "gen.src", gp + "/m_source/l_linear/MatMul_quant")
    harm = g.const[[n for n in nodes if n.name == gp + "/m_source/l_sin_gen/Mul"][0].inputs[1]].reshape(-1)
    if not np.array_equal(harm, np.arange(1, 10, dtype=np.float32)):
        raise ValueError(f"harmonic multipliers {harm}")
    stft = g.node(gp + "/STFT", "STFT")
    window = g.const[stft.inputs[2]].astype(np.float32)
    if g.const[stft.inputs[1]] != 5 or window.shape != (20,):
        raise ValueError("STFT hop / window")
    for i, (stride, k) in enumerate(((10, 20), (6, 12))):
        pads = conv_transpose(g, out, f"gen.ups{i}", f"{gp}/ups.{i}/ConvTranspose", stride, k)
        if pads != [(k - stride) // 2] * 2:
            raise ValueError(f"ups.{i} pads {pads}")
    for i in range(2):
        conv(g, out, f"gen.nc{i}", f"{gp}/noise_convs.{i}/Conv_quant")
        adain_resblock1(g, out, f"gen.nr{i}", f"{gp}/noise_res.{i}", f"decoder.generator.noise_res.{i}")
    for i in range(6):
        adain_resblock1(g, out, f"gen.rb{i}", f"{gp}/resblocks.{i}", f"decoder.generator.resblocks.{i}")
    conv(g, out, "gen.post", gp + "/conv_post/Conv_quant")
    inv = g.value("decoder.generator.stft.stft.inverse_basis")  # [22, 1, 20]
    out.add("gen.istft", inv[:, 0, :].T)  # [20 taps, 22 channels]: ggml ne0 = 22
    # forward STFT of the harmonic source as a 20-tap conv with 22 output channels: 11 real, 11 imaginary bins.
    # X[k] = sum_n w[n] x[n] exp(-2 pi i k n / 20); the imaginary rows of bins 0 and 10 are exactly zero.
    nn_ = np.arange(20, dtype=np.float64)
    basis = np.zeros((22, 1, 1, 20), np.float64)
    for k in range(11):
        ang = 2.0 * np.pi * k * nn_ / 20.0
        basis[k, 0, 0] = window * np.cos(ang)
        if k not in (0, 10):
            basis[11 + k, 0, 0] = -window * np.sin(ang)
    out.add("gen.stft", basis.astype(np.float32))
    window_sum_check(g, window, 5)

    # --- voices: [speakers, 511, 256] ---
    sd = [int(x) for x in meta.get("style_dim", "511,1,256").split(",")]
    voices = np.fromfile(model_dir + "/voices.bin", "<f4")
    rows = sd[0] * sd[2]
    if voices.size % rows:
        raise ValueError(f"voices.bin: {voices.size} floats is not a multiple of {rows}")
    out.add("voices", voices.reshape(-1, sd[0], sd[2]))

    # every weight-bearing initializer must have been taken (a quantized copy counts when its float original was)
    weights = {k for k, v in inits.items() if v.dtype in (np.uint8, np.int8) and k.endswith("_quantized")}
    weights |= {k for k, v in inits.items() if v.dtype == np.float32 and v.size > 1 and not k.endswith("_scale")}
    skipped = sorted(k for k in weights
                     if k not in g.used and not (k.endswith("_quantized") and k[: -len("_quantized")] in g.used))
    if skipped:
        raise ValueError(f"initializers not converted: {skipped[:20]} ({len(skipped)})")

    kv = [
        ("general.architecture", "kokoro"),
        ("general.name", "Kokoro-82M v0.19 (from sherpa-onnx kokoro-int8-en-v0_19)"),
        ("kokoro.source_sha256", h.hexdigest()),
        ("kokoro.sample_rate", int(meta.get("sample_rate", "24000"))),
        ("kokoro.speaker_names", meta.get("speaker_names", "")),
        ("kokoro.weight_type", wtype if mmtype == wtype else f"{wtype} conv / {mmtype} mul_mat"),
    ] + [("kokoro.conv." + k, v) for k, v in out.conv.items()]
    write_gguf(out_path, kv, out.t)
    n_par = sum(a.size for _, a in out.t)
    print(f"{out_path}: {len(out.t)} tensors, {n_par} values, source sha256 {h.hexdigest()}")


# ----------------------------------------------------------------------------------------------------------------------
# Minimal GGUF v3 writer
# ----------------------------------------------------------------------------------------------------------------------


def _gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, kv, tensors, align=32):
    head = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))
    for k, v in kv:
        head += _gstr(k)
        if isinstance(v, str):
            head += struct.pack("<I", 8) + _gstr(v)
        elif isinstance(v, int):
            head += struct.pack("<Ii", 5, v)
        else:
            raise TypeError(k)
    off = 0
    for name, a in tensors:
        head += _gstr(name) + struct.pack("<I", a.ndim)
        for d in reversed(a.shape):
            head += struct.pack("<Q", d)
        head += struct.pack("<IQ", 1 if a.dtype == np.float16 else 0, off)
        off += (a.nbytes + align - 1) // align * align
    pad = (-len(head)) % align
    with open(path, "wb") as f:
        f.write(head)
        f.write(b"\0" * pad)
        for _, a in tensors:
            f.write(a.astype(a.dtype.newbyteorder("<")).tobytes())
            f.write(b"\0" * ((-a.nbytes) % align))


if __name__ == "__main__":
    args = sys.argv[1:]
    wt = "f16"
    if "--wtype" in args:
        i = args.index("--wtype")
        wt = args[i + 1]
        del args[i:i + 2]
    mt = wt
    if "--mm-type" in args:
        i = args.index("--mm-type")
        mt = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2 or wt not in ("f16", "f32") or mt not in ("f16", "f32"):
        sys.exit("usage: convert_onnx_to_gguf.py MODEL_DIR OUT.gguf [--wtype f16|f32] [--mm-type f16|f32]")
    convert(args[0], args[1], wt, mt)
