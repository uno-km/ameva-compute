#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build every AMEVA GPU kernel of the ggml backend into one C header.

    python3 scripts/build_ameva_kernels.py [--glslang glslangValidator] [--spirv-val spirv-val] [--out HEADER]

* QMM family (shaders/ameva_qmm.comp): one variant per weight type and activation-column tile (-DTYPE_<T>
  -DNCOLS=<n>), with the block geometry the host checks against ggml_row_size().
* QMV family (shaders/ameva_qmv.comp): MUL_MAT on weights stored in the AMEVA repacked layout
  (ggml-ameva-repack.h), one variant per weight type and work shape (-DTYPE_<T> -DCOLS=<c> -DROWS=<r>): the
  token-generation shape (COLS=1) with 4, 2 and 1 weight rows per lane group, of which the host chooses per device
  and weight shape, and the prompt shape (COLS=8, ROWS=1). ROWS is a define, not a specialization constant (see
  shaders/ameva_qmv.comp); the build checks that no QMV module declares one.
* MALI_QMV family (shaders/mali_qmv.comp): token-generation MUL_MAT on repacked weights in a 32 x 4 work split, one
  variant per weight type (-DTYPE_<T>); the host chooses between it and the QMV layouts per device and shape.
* QGEMM family (shaders/ameva_qgemm.comp): tiled MUL_MAT for many activation rows on repacked weights, one variant
  per weight type (-DTYPE_<T>).
* MMVQ family (shaders/ameva_mmvq.comp): token-generation MUL_MAT (one activation row) with integer dot products
  on repacked weights and int8 activations (ameva_quant_a8), one variant per weight type and weight rows per lane
  group (-DTYPE_<T> -DROWS=<r>, r = 4, 2, 1).
* MMVQE family (shaders/ameva_mmvq.comp -DEMU_DOT): the same without VK_KHR_shader_integer_dot_product, the packed
  dot product computed with two 32-bit multiplications per four weights; for the weight types whose integer values
  are unsigned and at most 63 (not q8_0, iq4_nl, iq4_xs). Its activations come from the op kernel quant_a8u
  (ameva_quant_a8 -DEMU).
* MMQ family (shaders/ameva_mmq.comp): integer-dot-product MUL_MAT for many activation rows on repacked weights
  and int8 activations (shaders/ameva_quant_a8.comp, an op kernel), one variant per weight type (-DTYPE_<T>).
* FA family (shaders/ameva_flash_attn.comp): tiled FLASH_ATTN_EXT, one variant per head-size class
  (-DDMAX=<largest head size> -DBQ=<rows per tile> -DBK=<keys per block>; shared memory stays below 32 KiB).
* Op kernels (shaders/ameva_<op>.comp): one SPIR-V module each; the operation is selected at dispatch time.

* MMQB family (shaders/ameva_mmqb.comp): the same product with integer accumulation over a 256-weight block, for
  the super-block types whose scales are integers per unit (q4_K, q5_K, q6_K), on int8 activations with one scale
  per 256 elements (shaders/ameva_quant_a8b.comp, an op kernel). No shared memory: an invocation owns DRW weight
  rows x DRA activation rows. One variant per weight type and layout (DRW, DRA, local size).

* SPIR-V-only kernels (VK_ONLY: shaders/ameva_conv.comp, the convolution without a column matrix, per weight form
  and staging, shaders/ameva_conv_w.comp, which brings the weights into its K order, and the two passes of
  GROUP_NORM, shaders/ameva_cnorm_stat.comp and shaders/ameva_cnorm.comp): scripts/glsl_to_opencl.py does not
  translate them yet, so they have no entry in the OpenCL C header and the engine declines their ops on the OpenCL
  backend.

For every module: glslangValidator -V (vulkan1.0), spirv-val, SHA-256 (pinned by the runtime at kernel creation),
binding count (from the source's layout declarations), push-constant size (counted from the source's push-constant
block and cross-checked against glslang's reflection) and local size. Any compiler, validator or consistency
failure aborts the build.
"""
import argparse
import hashlib
import os
import re
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADERS = os.path.join(ROOT, "adapters", "llama", "ggml-ameva", "shaders")
DEFAULT_OUT = os.path.join(ROOT, "adapters", "llama", "ggml-ameva", "ggml_ameva_kernels_spv.h")
DEFAULT_CL_OUT = os.path.join(ROOT, "adapters", "llama", "ggml-ameva", "ggml_ameva_kernels_cl.h")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glsl_to_opencl  # noqa: E402  (scripts/glsl_to_opencl.py: the OpenCL C form of every kernel)

# (ggml_type_name, shader define, weights per block, bytes per block) -- ggml-common.h block structs
QMM_TYPES = [
    ("f32", "TYPE_F32", 32, 128),
    ("f16", "TYPE_F16", 32, 64),
    ("bf16", "TYPE_BF16", 32, 64),
    ("q4_0", "TYPE_Q4_0", 32, 18),
    ("q4_1", "TYPE_Q4_1", 32, 20),
    ("q5_0", "TYPE_Q5_0", 32, 22),
    ("q5_1", "TYPE_Q5_1", 32, 24),
    ("q8_0", "TYPE_Q8_0", 32, 34),
    ("q2_K", "TYPE_Q2_K", 256, 84),
    ("q3_K", "TYPE_Q3_K", 256, 110),
    ("q4_K", "TYPE_Q4_K", 256, 144),
    ("q5_K", "TYPE_Q5_K", 256, 176),
    ("q6_K", "TYPE_Q6_K", 256, 210),
    ("iq4_nl", "TYPE_IQ4_NL", 32, 18),
    ("iq4_xs", "TYPE_IQ4_XS", 256, 136),
]
QMM_NCOLS = (1, 8)

# types with a repacked layout (ggml-ameva-repack.cpp kTypes): (ggml_type_name, shader define)
QMV_TYPES = [
    ("q4_0", "TYPE_Q4_0"),
    ("q4_1", "TYPE_Q4_1"),
    ("q5_0", "TYPE_Q5_0"),
    ("q5_1", "TYPE_Q5_1"),
    ("q8_0", "TYPE_Q8_0"),
    ("q2_K", "TYPE_Q2_K"),
    ("q3_K", "TYPE_Q3_K"),
    ("q4_K", "TYPE_Q4_K"),
    ("q5_K", "TYPE_Q5_K"),
    ("q6_K", "TYPE_Q6_K"),
    ("iq4_nl", "TYPE_IQ4_NL"),
    ("iq4_xs", "TYPE_IQ4_XS"),
]
# (COLS, ROWS): token generation (one activation row; 4, 2 or 1 weight rows per lane group, chosen by the host) and
# prompt processing (8 activation rows per workgroup, 1 weight row per lane group)
QMV_SHAPES = [(1, 4), (1, 2), (1, 1), (8, 1)]
MMVQ_ROWS = [4, 2, 1]
# weight types of the emulated integer dot product (shaders/ameva_mmvq.comp -DEMU_DOT)
MMVQE_TYPES = [t for t in QMV_TYPES if t[0] not in ("q8_0", "iq4_nl", "iq4_xs")]
# block-integer MMQ (shaders/ameva_mmqb.comp): weight types, and layouts (weight rows per invocation, activation rows
# per invocation, local size x, y). Which layout is fastest depends on the device: Mali-G68 (Galaxy A53, OpenCL
# prototype, 352x2048x2048, 2026-10-07) 2x8 19.4 ms, 2x4 22.6, 2x16 21.7, 1x8 31.1, 1x4 35.5, 4x8 63.6; the local
# size moves it by a few per cent.
MMQB_TYPES = [t for t in QMV_TYPES if t[0] in ("q4_K", "q5_K", "q6_K")]
MMQB_LAYOUTS = [(2, 8, 8, 4), (2, 4, 8, 4), (2, 16, 8, 4), (1, 8, 8, 4), (1, 4, 8, 4), (1, 16, 8, 4)]
# units per lane and step of the token-generation variant: the shader default unless AMEVA_BUILD_QMV_UPL is set
# (kernel tuning: builds that differ only in it)
QMV_UPL = os.environ.get("AMEVA_BUILD_QMV_UPL", "")
# kernel tuning: the token-generation variant with a part of its work removed (shaders/ameva_qmv.comp, ABLATE); the
# kernels of such a build give wrong results
QMV_ABLATE = os.environ.get("AMEVA_BUILD_QMV_ABLATE", "")

# op kernel families, in the order of enum AmevaOpKernelId
OPS = ["binary", "unary", "glu", "rms_norm", "rope", "set_rows", "get_rows", "cpy", "fa_combine", "quant_a8",
       "act_fq", "quant_a8b", "upscale", "pad", "soft_max", "mm"]
# op kernels compiled from one source with defines, appended to enum AmevaOpKernelId: (kernel name, source, defines)
OPS_DEFINED = [("set_rows_q8_0", "set_rows_q", ["QTYPE_Q8_0"]), ("set_rows_q4_0", "set_rows_q", ["QTYPE_Q4_0"]),
               ("quant_a8u", "quant_a8", ["EMU"]), ("rope_f16", "rope", ["F16"]),
               ("binary_f16", "binary", ["F16"]), ("im2col", "im2col", []), ("mm_f16", "mm", ["F16"]),
               ("mm_f16w", "mm", ["F16", "F16_WORDS"]), ("mm_f16wu", "mm", ["F16", "F16_WORDS", "F16_UNPACK"]),
               ("get_rows_q4_0", "get_rows_q", ["Q4_0"]), ("get_rows_q5_0", "get_rows_q", ["Q5_0"])]

# Kernels built as SPIR-V only, in the order of enum AmevaVkKernelId: (kernel name, source, defines).
# ameva_conv: weights f16 through the exact decoder, f16 through the built-in conversion (f16u; the engine selects it
# after checking the device's conversion of f16 subnormals) or f32; staging scalar (s, any stride) or aligned vec4 (v).
# ameva_cnorm_stat, ameva_cnorm: GROUP_NORM as group statistics, then the normalisation with the fused chain.
# conv_*_acc (ACCUM): the same convolution kernels continuing from the stored result, for the passes after the first
# over the channel-group windows of a source or of weights above one buffer binding.
# ameva_norm (NORM, the layer norm) and ameva_timestep_embd (TIMESTEP_EMBEDDING): ops of the diffusion models'
# transformer blocks and time embedding.
# New kernels go at the end: the engine addresses these by enum value.
VK_ONLY = ([("conv_%s_%s" % (w, st), "conv", wd + sd)
            for w, wd in (("f16", ["TYPE_F16"]), ("f16u", ["TYPE_F16", "F16_UNPACK"]), ("f32", ["TYPE_F32"]))
            for st, sd in (("s", []), ("v", ["STAGE_VEC4"]))] +
           [("conv_w_f16", "conv_w", []), ("conv_w_f32", "conv_w", ["F32"])] +
           [("cnorm_stat", "cnorm_stat", []), ("cnorm", "cnorm", [])] +
           [("conv_%s_%s_acc" % (w, st), "conv", wd + sd + ["ACCUM"])
            for w, wd in (("f16", ["TYPE_F16"]), ("f16u", ["TYPE_F16", "F16_UNPACK"]), ("f32", ["TYPE_F32"]))
            for st, sd in (("s", []), ("v", ["STAGE_VEC4"]))] +
           [("norm", "norm", []), ("timestep_embd", "timestep_embd", [])])

# the licence header of the generated files (kept identical to the other sources of the repository)
LICENSE_HEADER = ["/**", " * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.",
                  " * SPDX-License-Identifier: Apache-2.0 OR Commercial-License", " */"]

# FLASH_ATTN_EXT tiles: (DMAX, BQ, BK). 128 invocations per workgroup; shared memory per workgroup:
# 64: 14.5 KiB, 128: 27.4 KiB, 256: 25.7 KiB
FA_VARIANTS = [(64, 16, 32), (128, 16, 32), (256, 8, 16)]


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        sys.stderr.write("FAILED: %s\n%s\n" % (" ".join(cmd), p.stdout))
        sys.exit(1)
    return p.stdout


def fail(msg):
    sys.stderr.write("FAILED: %s\n" % msg)
    sys.exit(1)


def read_with_includes(path, seen=None):
    """the source with every #include "file" (relative to the including file) expanded once"""
    seen = set() if seen is None else seen
    path = os.path.abspath(path)
    if path in seen:
        return ""
    seen.add(path)
    out = []
    for line in open(path, encoding="utf-8").read().splitlines():
        m = re.match(r'\s*#include\s+"([^"]+)"', line)
        if m:
            out.append(read_with_includes(os.path.join(os.path.dirname(path), m.group(1)), seen))
        else:
            out.append(line)
    return "\n".join(out)


def source_info(path):
    """binding count, push-constant bytes (all members are 32-bit) and local_size_x declared by a kernel source
    (included files count: a shared include may declare bindings). The local size is None when the source takes it
    from defines."""
    src = read_with_includes(path)
    bindings = [int(b) for b in re.findall(r"layout\(std430,\s*binding\s*=\s*(\d+)\)", src)]
    if not bindings or sorted(bindings) != list(range(len(bindings))):
        fail("%s: bindings must be numbered 0..n-1, found %s" % (path, bindings))
    m = re.search(r"layout\(push_constant\)\s*uniform\s+\w+\s*\{(.*?)\}", src, re.S)
    if not m:
        fail("%s: no push-constant block" % path)
    body = re.sub(r"//[^\n]*", "", m.group(1))
    members = 0
    for decl in body.split(";"):
        decl = decl.strip()
        if not decl:
            continue
        if not decl.startswith("uint "):
            fail("%s: push-constant member is not uint: %r" % (path, decl))
        members += len([n for n in decl[5:].split(",") if n.strip()])
    ls = re.search(r"local_size_x\s*=\s*(\d+)", src)
    return len(bindings), 4 * members, int(ls.group(1)) if ls else None


def compile_module(args, src, defines, out, cl, local_size=None):
    """SPIR-V of one kernel variant, and its OpenCL C translation appended to cl (the same variant, the same order);
    cl None: SPIR-V only. local_size: the (x, y, z) the defines give a source that does not spell its local size
    out."""
    cmd = [args.glslang, "-V", "-q", "--target-env", "vulkan1.0"] + ["-D%s" % d for d in defines] + ["-o", out, src]
    refl = run(cmd)
    if args.spirv_val != "none":
        run([args.spirv_val, "--target-env", "vulkan1.0", out])
    m = re.search(r"PushConstants: offset -1, type \w+, size (\d+)", refl)
    data = open(out, "rb").read()
    if cl is None:
        return data, (int(m.group(1)) if m else None)
    try:
        text, nbind, npush, local, _ = glsl_to_opencl.translate(glsl_to_opencl.preprocess(args.glslang, src, defines))
    except glsl_to_opencl.TranslateError as e:
        fail("%s %s: OpenCL C translation: %s" % (os.path.basename(src), " ".join(defines), e))
    sbind, spush, slocal = source_info(src)
    if local_size is not None:
        if slocal is not None or tuple(local) != tuple(local_size):
            fail("%s %s: local size %s expected from the defines, the OpenCL C translation has %s"
                 % (os.path.basename(src), " ".join(defines), tuple(local_size), tuple(local)))
        slocal = local[0]
    if (nbind, npush, local[0]) != (sbind, spush, slocal):
        fail("%s %s: the OpenCL C translation has %d bindings, %d push-constant bytes, local size %d; the source %d, %d, %d"
             % (os.path.basename(src), " ".join(defines), nbind, npush, local[0], sbind, spush, slocal))
    cl.append(text)
    return data, (int(m.group(1)) if m else None)


def c_string(text):
    """text as C string literals, one per source line"""
    out = []
    for line in text.splitlines():
        out.append('    "%s\\n"' % line.replace("\\", "\\\\").replace('"', '\\"'))
    return "\n".join(out)


# OpenCL C tables of ggml_ameva_kernels_cl.h: (key in cl, table name), in the order of the SPIR-V tables
CL_TABLES = [("qmm", "kAmevaClQmm"), ("qmv", "kAmevaClQmv"), ("mali_qmv", "kAmevaClMaliQmv"),
             ("qgemm", "kAmevaClQgemm"), ("mmq", "kAmevaClMmq"), ("mmqb", "kAmevaClMmqb"), ("mmvq", "kAmevaClMmvq"), ("mmvqe", "kAmevaClMmvqe"),
             ("fa", "kAmevaClFa"), ("ops", "kAmevaClOps")]


def write_cl_header(path, version, cl):
    """Every kernel of the SPIR-V header as OpenCL C: per family an array of sources in the order of its SPIR-V table.
    A kernel's program is kAmevaClPrelude followed by its own text; the entry point is AMEVA_CL_ENTRY_POINT. Functions
    that several kernels of a family share (the decoders) are repeated in each: every variant is one preprocessed
    source."""
    lines = LICENSE_HEADER + [
             "/* Generated by scripts/build_ameva_kernels.py with scripts/glsl_to_opencl.py from",
             "   adapters/llama/ggml-ameva/shaders -- do not edit. OpenCL C translations of the kernels of",
             "   ggml_ameva_kernels_spv.h, table by table in the same order (GLSL preprocessor: %s). */" % version,
             "#ifndef GGML_AMEVA_KERNELS_CL_H", "#define GGML_AMEVA_KERNELS_CL_H", "",
             '#define AMEVA_CL_ENTRY_POINT "%s"' % glsl_to_opencl.ENTRY_POINT, "",
             "static const char kAmevaClPrelude[] =", c_string(glsl_to_opencl.PRELUDE) + ";", ""]
    total = 0
    for key, table in CL_TABLES:
        for i, text in enumerate(cl[key]):
            total += len(text)
            lines += ["static const char %s_%d[] =" % (table, i), c_string(text) + ";"]
        lines += ["static const char* const %s[%d] = {" % (table, len(cl[key])),
                  "    " + ", ".join("%s_%d" % (table, i) for i in range(len(cl[key]))), "};", ""]
    lines += ["#endif /* GGML_AMEVA_KERNELS_CL_H */", ""]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines))
    return total


def spec_defaults(data):
    """{SpecId: default value} of the 32-bit integer specialization constants of a SPIR-V module."""
    words = struct.unpack("<%dI" % (len(data) // 4), data)
    spec_id, value = {}, {}
    i = 5
    while i < len(words):
        op, n = words[i] & 0xFFFF, words[i] >> 16
        if n == 0:
            fail("malformed SPIR-V module")
        if op == 71 and n >= 4 and words[i + 2] == 1:  # OpDecorate <id> SpecId <n>
            spec_id[words[i + 1]] = words[i + 3]
        elif op == 50 and n == 4:  # OpSpecConstant <type> <id> <value>
            value[words[i + 2]] = words[i + 3]
        i += n
    return {sid: value.get(rid) for rid, sid in spec_id.items()}


def emit_array(lines, var, data):
    words = struct.unpack("<%dI" % (len(data) // 4), data)
    lines.append("static const uint32_t %s[] = {" % var)
    for i in range(0, len(words), 8):
        lines.append("    " + ", ".join("0x%08xu" % w for w in words[i:i + 8]) + ",")
    lines.append("};")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glslang", default="glslangValidator")
    # "none" skips validation (a host without SPIRV-Tools); the header then says so in its second line
    ap.add_argument("--spirv-val", default="spirv-val")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--cl-out", default=DEFAULT_CL_OUT)
    args = ap.parse_args()
    cl = {key: [] for key, _ in CL_TABLES}

    version = run([args.glslang, "--version"]).splitlines()[0].strip()
    lines = LICENSE_HEADER + [
             "/* Generated by scripts/build_ameva_kernels.py from adapters/llama/ggml-ameva/shaders -- do not edit. */",
             ("/* compiler: %s, --target-env vulkan1.0, every module checked by spirv-val */" % version
              if args.spirv_val != "none" else
              "/* compiler: %s, --target-env vulkan1.0, NOT checked by spirv-val (--spirv-val none) */" % version),
             "#ifndef GGML_AMEVA_KERNELS_SPV_H", "#define GGML_AMEVA_KERNELS_SPV_H", "",
             "#include <stddef.h>", "#include <stdint.h>", ""]
    qmm, qmv, mali_qmv, qgemm, mmq, mmqb, mmvq, mmvqe, fa, ops = [], [], [], [], [], [], [], [], [], []
    vk = []
    with tempfile.TemporaryDirectory() as tmp:
        qsrc = os.path.join(SHADERS, "ameva_qmm.comp")
        qbind, qpush, qlocal = source_info(qsrc)
        for name, define, belems, bbytes in QMM_TYPES:
            for ncols in QMM_NCOLS:
                spv = os.path.join(tmp, "qmm_%s_n%d.spv" % (name, ncols))
                data, rsize = compile_module(args, qsrc, [define, "NCOLS=%d" % ncols], spv, cl["qmm"])
                if rsize is not None and rsize != qpush:
                    fail("ameva_qmm %s: reflection says %d push-constant bytes, source %d" % (name, rsize, qpush))
                var = "kAmevaQmm_%s_n%d" % (name, ncols)
                emit_array(lines, var, data)
                qmm.append((name, ncols, var, len(data), hashlib.sha256(data).hexdigest(), belems, bbytes))
                print("qmm %-6s ncols=%d %6d bytes sha256=%s" % (name, ncols, len(data), qmm[-1][4]))
        vsrc = os.path.join(SHADERS, "ameva_qmv.comp")
        vbind, vpush, vlocal = source_info(vsrc)
        for name, define in QMV_TYPES:
            for cols, rows in QMV_SHAPES:
                spv = os.path.join(tmp, "qmv_%s_c%d_r%d.spv" % (name, cols, rows))
                defs = [define, "COLS=%d" % cols, "ROWS=%d" % rows]
                if cols == 1 and QMV_UPL:
                    if QMV_UPL not in ("1", "2", "4"):
                        fail("AMEVA_BUILD_QMV_UPL must be 1, 2 or 4 (got %r)" % QMV_UPL)
                    defs.append("UPL=%s" % QMV_UPL)
                if cols == 1 and QMV_ABLATE:
                    if QMV_ABLATE not in ("1", "2", "3"):
                        fail("AMEVA_BUILD_QMV_ABLATE must be 1, 2 or 3 (got %r)" % QMV_ABLATE)
                    defs.append("ABLATE=%s" % QMV_ABLATE)
                data, rsize = compile_module(args, vsrc, defs, spv, cl["qmv"])
                if rsize is not None and rsize != vpush:
                    fail("ameva_qmv %s: reflection says %d push-constant bytes, source %d" % (name, rsize, vpush))
                if spec_defaults(data):
                    fail("ameva_qmv %s cols=%d rows=%d: unexpected specialization constants %r" %
                         (name, cols, rows, spec_defaults(data)))
                var = "kAmevaQmv_%s_c%d_r%d" % (name, cols, rows)
                emit_array(lines, var, data)
                qmv.append((name, cols, rows, var, len(data), hashlib.sha256(data).hexdigest()))
                print("qmv %-6s cols=%d rows=%d %6d bytes sha256=%s" % (name, cols, rows, len(data), qmv[-1][5]))
        msrc_mali = os.path.join(SHADERS, "mali_qmv.comp")
        mbind_mali, mpush_mali, mlocal_mali = source_info(msrc_mali)
        for name, define in QMV_TYPES:
            spv = os.path.join(tmp, "mali_qmv_%s.spv" % name)
            data, rsize = compile_module(args, msrc_mali, [define], spv, cl["mali_qmv"])
            if rsize is not None and rsize != mpush_mali:
                fail("mali_qmv %s: reflection says %d push-constant bytes, source %d" % (name, rsize, mpush_mali))
            var = "kAmevaMaliQmv_%s" % name
            emit_array(lines, var, data)
            mali_qmv.append((name, var, len(data), hashlib.sha256(data).hexdigest()))
            print("mali_qmv %-6s %6d bytes sha256=%s" % (name, len(data), mali_qmv[-1][3]))
        gsrc = os.path.join(SHADERS, "ameva_qgemm.comp")
        gbind, gpush, glocal = source_info(gsrc)
        for name, define in QMV_TYPES:
            spv = os.path.join(tmp, "qgemm_%s.spv" % name)
            data, rsize = compile_module(args, gsrc, [define], spv, cl["qgemm"])
            if rsize is not None and rsize != gpush:
                fail("ameva_qgemm %s: reflection says %d push-constant bytes, source %d" % (name, rsize, gpush))
            var = "kAmevaQgemm_%s" % name
            emit_array(lines, var, data)
            qgemm.append((name, var, len(data), hashlib.sha256(data).hexdigest()))
            print("qgemm %-6s %6d bytes sha256=%s" % (name, len(data), qgemm[-1][3]))
        msrc = os.path.join(SHADERS, "ameva_mmq.comp")
        mbind, mpush, mlocal = source_info(msrc)
        for name, define in QMV_TYPES:
            spv = os.path.join(tmp, "mmq_%s.spv" % name)
            data, rsize = compile_module(args, msrc, [define], spv, cl["mmq"])
            if rsize is not None and rsize != mpush:
                fail("ameva_mmq %s: reflection says %d push-constant bytes, source %d" % (name, rsize, mpush))
            var = "kAmevaMmq_%s" % name
            emit_array(lines, var, data)
            mmq.append((name, var, len(data), hashlib.sha256(data).hexdigest()))
            print("mmq   %-6s %6d bytes sha256=%s" % (name, len(data), mmq[-1][3]))
        bsrc = os.path.join(SHADERS, "ameva_mmqb.comp")
        bbind, bpush, _ = source_info(bsrc)
        for name, define in MMQB_TYPES:
            for drw, dra, lx, ly in MMQB_LAYOUTS:
                tag = "%s_w%da%d_l%dx%d" % (name, drw, dra, lx, ly)
                spv = os.path.join(tmp, "mmqb_%s.spv" % tag)
                data, rsize = compile_module(args, bsrc, [define, "DRW=%du" % drw, "DRA=%du" % dra, "LX=%d" % lx,
                                                          "LY=%d" % ly], spv, cl["mmqb"], (lx, ly, 1))
                if rsize is not None and rsize != bpush:
                    fail("ameva_mmqb %s: reflection says %d push-constant bytes, source %d" % (tag, rsize, bpush))
                var = "kAmevaMmqb_%s" % tag
                emit_array(lines, var, data)
                mmqb.append((name, drw, dra, lx, ly, var, len(data), hashlib.sha256(data).hexdigest()))
                print("mmqb  %-6s w%d a%-2d l%dx%d %6d bytes sha256=%s" % (name, drw, dra, lx, ly, len(data), mmqb[-1][7]))
        vqsrc = os.path.join(SHADERS, "ameva_mmvq.comp")
        vqbind, vqpush, vqlocal = source_info(vqsrc)
        for name, define in QMV_TYPES:
            for rows in MMVQ_ROWS:
                spv = os.path.join(tmp, "mmvq_%s_r%d.spv" % (name, rows))
                data, rsize = compile_module(args, vqsrc, [define, "ROWS=%d" % rows], spv, cl["mmvq"])
                if rsize is not None and rsize != vqpush:
                    fail("ameva_mmvq %s: reflection says %d push-constant bytes, source %d" % (name, rsize, vqpush))
                if spec_defaults(data):
                    fail("ameva_mmvq %s rows=%d: unexpected specialization constants %r" % (name, rows,
                                                                                            spec_defaults(data)))
                var = "kAmevaMmvq_%s_r%d" % (name, rows)
                emit_array(lines, var, data)
                mmvq.append((name, rows, var, len(data), hashlib.sha256(data).hexdigest()))
                print("mmvq  %-6s rows=%d %6d bytes sha256=%s" % (name, rows, len(data), mmvq[-1][4]))
        for name, define in MMVQE_TYPES:
            for rows in MMVQ_ROWS:
                spv = os.path.join(tmp, "mmvqe_%s_r%d.spv" % (name, rows))
                data, rsize = compile_module(args, vqsrc, [define, "ROWS=%d" % rows, "EMU_DOT"], spv, cl["mmvqe"])
                if rsize is not None and rsize != vqpush:
                    fail("ameva_mmvq (emulated) %s: reflection says %d push-constant bytes, source %d" % (name, rsize, vqpush))
                if spec_defaults(data):
                    fail("ameva_mmvq (emulated) %s rows=%d: unexpected specialization constants %r" %
                         (name, rows, spec_defaults(data)))
                var = "kAmevaMmvqe_%s_r%d" % (name, rows)
                emit_array(lines, var, data)
                mmvqe.append((name, rows, var, len(data), hashlib.sha256(data).hexdigest()))
                print("mmvqe %-6s rows=%d %6d bytes sha256=%s" % (name, rows, len(data), mmvqe[-1][4]))
        fsrc = os.path.join(SHADERS, "ameva_flash_attn.comp")
        fbind, fpush, flocal = source_info(fsrc)
        if fpush > 128:
            fail("flash_attn: %d push-constant bytes exceed the runtime limit of 128" % fpush)
        for dmax, bq, bk in FA_VARIANTS:
            spv = os.path.join(tmp, "fa_d%d.spv" % dmax)
            data, rsize = compile_module(args, fsrc, ["DMAX=%du" % dmax, "BQ=%du" % bq, "BK=%du" % bk], spv, cl["fa"])
            if rsize is not None and rsize != fpush:
                fail("ameva_flash_attn d%d: reflection says %d push-constant bytes, source %d" % (dmax, rsize, fpush))
            var = "kAmevaFa_d%d" % dmax
            emit_array(lines, var, data)
            fa.append((dmax, bq, bk, var, len(data), hashlib.sha256(data).hexdigest()))
            print("fa  dmax=%-3d bq=%-2d bk=%-2d %6d bytes sha256=%s" % (dmax, bq, bk, len(data), fa[-1][5]))
        for op, stem, defines in [(o, o, []) for o in OPS] + OPS_DEFINED:
            src = os.path.join(SHADERS, "ameva_%s.comp" % stem)
            nbind, npush, nlocal = source_info(src)
            if npush > 128:
                fail("%s: %d push-constant bytes exceed the runtime limit of 128" % (op, npush))
            spv = os.path.join(tmp, "%s.spv" % op)
            data, rsize = compile_module(args, src, defines, spv, cl["ops"])
            if rsize is not None and rsize != npush:
                fail("%s: reflection says %d push-constant bytes, source %d" % (op, rsize, npush))
            var = "kAmevaOp_%s" % op
            emit_array(lines, var, data)
            ops.append((op, var, len(data), hashlib.sha256(data).hexdigest(), nbind, npush, nlocal))
            print("op  %-10s %6d bytes bindings=%d push=%d local=%d sha256=%s" % (op, len(data), nbind, npush, nlocal,
                                                                                   ops[-1][3]))
        for name, stem, defines in VK_ONLY:
            src = os.path.join(SHADERS, "ameva_%s.comp" % stem)
            nbind, npush, nlocal = source_info(src)
            if npush > 128:
                fail("%s: %d push-constant bytes exceed the runtime limit of 128" % (name, npush))
            spv = os.path.join(tmp, "%s.spv" % name)
            data, rsize = compile_module(args, src, defines, spv, None)
            if rsize is not None and rsize != npush:
                fail("%s: reflection says %d push-constant bytes, source %d" % (name, rsize, npush))
            var = "kAmevaVk_%s" % name
            emit_array(lines, var, data)
            vk.append((name, var, len(data), hashlib.sha256(data).hexdigest(), nbind, npush, nlocal))
            print("vk  %-12s %6d bytes bindings=%d push=%d local=%d sha256=%s" % (name, len(data), nbind, npush, nlocal,
                                                                                     vk[-1][3]))

    lines += ["", "typedef struct {",
              "    const char* type_name;   /* ggml_type_name() of the weight type */",
              "    uint32_t ncols;          /* activation rows per workgroup */",
              "    uint32_t block_elems;    /* weights per ggml block */",
              "    uint32_t block_bytes;    /* bytes per ggml block */",
              "    const uint32_t* spirv;", "    size_t spirv_bytes;", "    const char* spirv_sha256;",
              "} AmevaQmmVariant;", "",
              "static const AmevaQmmVariant kAmevaQmmVariants[] = {"]
    for name, ncols, var, nbytes, sha, belems, bbytes in qmm:
        lines.append('    { "%s", %du, %du, %du, %s, %du, "%s" },' % (name, ncols, belems, bbytes, var, nbytes, sha))
    lines += ["};", "#define AMEVA_QMM_VARIANT_COUNT %d" % len(qmm),
              "#define AMEVA_QMM_BINDINGS %d" % qbind,
              "#define AMEVA_QMM_PUSH_CONSTANT_BYTES %d" % qpush,
              "#define AMEVA_QMM_LOCAL_SIZE_X %d" % qlocal, "",
              "typedef struct {",
              "    const char* type_name;   /* ggml_type_name() of the weight type (stored repacked) */",
              "    uint32_t cols;           /* activation rows per workgroup */",
              "    uint32_t rows;           /* weight rows per lane group */",
              "    const uint32_t* spirv;", "    size_t spirv_bytes;", "    const char* spirv_sha256;",
              "} AmevaQmvVariant;", "",
              "static const AmevaQmvVariant kAmevaQmvVariants[] = {"]
    for name, cols, rows, var, nbytes, sha in qmv:
        lines.append('    { "%s", %du, %du, %s, %du, "%s" },' % (name, cols, rows, var, nbytes, sha))
    lines += ["};", "#define AMEVA_QMV_VARIANT_COUNT %d" % len(qmv),
              "#define AMEVA_QMV_BINDINGS %d" % vbind,
              "#define AMEVA_QMV_PUSH_CONSTANT_BYTES %d" % vpush,
              "#define AMEVA_QMV_LOCAL_SIZE_X %d" % vlocal, "",
              "typedef struct {",
              "    const char* type_name;   /* ggml_type_name() of the weight type (stored repacked) */",
              "    const uint32_t* spirv;", "    size_t spirv_bytes;", "    const char* spirv_sha256;",
              "} AmevaMaliQmvVariant;", "",
              "static const AmevaMaliQmvVariant kAmevaMaliQmvVariants[] = {"]
    for name, var, nbytes, sha in mali_qmv:
        lines.append('    { "%s", %s, %du, "%s" },' % (name, var, nbytes, sha))
    lines += ["};", "#define AMEVA_MALI_QMV_VARIANT_COUNT %d" % len(mali_qmv),
              "#define AMEVA_MALI_QMV_BINDINGS %d" % mbind_mali,
              "#define AMEVA_MALI_QMV_PUSH_CONSTANT_BYTES %d" % mpush_mali,
              "#define AMEVA_MALI_QMV_LOCAL_SIZE_X 32",
              "#define AMEVA_MALI_QMV_LOCAL_SIZE_Y 4", "",
              "typedef struct {",
              "    const char* type_name;   /* ggml_type_name() of the weight type (stored repacked) */",
              "    const uint32_t* spirv;", "    size_t spirv_bytes;", "    const char* spirv_sha256;",
              "} AmevaQgemmVariant;", "",
              "static const AmevaQgemmVariant kAmevaQgemmVariants[] = {"]
    for name, var, nbytes, sha in qgemm:
        lines.append('    { "%s", %s, %du, "%s" },' % (name, var, nbytes, sha))
    lines += ["};", "#define AMEVA_QGEMM_VARIANT_COUNT %d" % len(qgemm),
              "#define AMEVA_QGEMM_BINDINGS %d" % gbind,
              "#define AMEVA_QGEMM_PUSH_CONSTANT_BYTES %d" % gpush,
              "#define AMEVA_QGEMM_LOCAL_SIZE_X %d" % glocal, "",
              "typedef struct {",
              "    const char* type_name;   /* ggml_type_name() of the weight type (stored repacked) */",
              "    const uint32_t* spirv;", "    size_t spirv_bytes;", "    const char* spirv_sha256;",
              "} AmevaMmqVariant;", "",
              "static const AmevaMmqVariant kAmevaMmqVariants[] = {"]
    for name, var, nbytes, sha in mmq:
        lines.append('    { "%s", %s, %du, "%s" },' % (name, var, nbytes, sha))
    lines += ["};", "#define AMEVA_MMQ_VARIANT_COUNT %d" % len(mmq),
              "#define AMEVA_MMQ_BINDINGS %d" % mbind,
              "#define AMEVA_MMQ_PUSH_CONSTANT_BYTES %d" % mpush,
              "#define AMEVA_MMQ_LOCAL_SIZE_X %d" % mlocal, "",
              "typedef struct {",
              "    const char* type_name;   /* ggml_type_name() of the weight type (stored repacked) */",
              "    uint32_t drw, dra;       /* weight rows and activation rows per invocation */",
              "    uint32_t local_x, local_y;",
              "    const uint32_t* spirv;", "    size_t spirv_bytes;", "    const char* spirv_sha256;",
              "} AmevaMmqbVariant;", "",
              "static const AmevaMmqbVariant kAmevaMmqbVariants[] = {"]
    for name, drw, dra, lx, ly, var, nbytes, sha in mmqb:
        lines.append('    { "%s", %du, %du, %du, %du, %s, %du, "%s" },' % (name, drw, dra, lx, ly, var, nbytes, sha))
    lines += ["};", "#define AMEVA_MMQB_VARIANT_COUNT %d" % len(mmqb),
              "#define AMEVA_MMQB_BINDINGS %d" % bbind,
              "#define AMEVA_MMQB_PUSH_CONSTANT_BYTES %d" % bpush, "",
              "typedef struct {",
              "    const char* type_name;   /* ggml_type_name() of the weight type (stored repacked) */",
              "    uint32_t rows;           /* weight rows per lane group */",
              "    const uint32_t* spirv;", "    size_t spirv_bytes;", "    const char* spirv_sha256;",
              "} AmevaMmvqVariant;", "",
              "static const AmevaMmvqVariant kAmevaMmvqVariants[] = {"]
    for name, rows, var, nbytes, sha in mmvq:
        lines.append('    { "%s", %du, %s, %du, "%s" },' % (name, rows, var, nbytes, sha))
    lines += ["};", "#define AMEVA_MMVQ_VARIANT_COUNT %d" % len(mmvq),
              "#define AMEVA_MMVQ_BINDINGS %d" % vqbind,
              "#define AMEVA_MMVQ_PUSH_CONSTANT_BYTES %d" % vqpush,
              "#define AMEVA_MMVQ_LOCAL_SIZE_X %d" % vqlocal, "",
              "/* ameva_mmvq with the integer dot product emulated by 32-bit multiplications */",
              "static const AmevaMmvqVariant kAmevaMmvqeVariants[] = {"]
    for name, rows, var, nbytes, sha in mmvqe:
        lines.append('    { "%s", %du, %s, %du, "%s" },' % (name, rows, var, nbytes, sha))
    lines += ["};", "#define AMEVA_MMVQE_VARIANT_COUNT %d" % len(mmvqe), "",
              "typedef struct {",
              "    uint32_t dmax;           /* largest head size (D and DV) the variant takes */",
              "    uint32_t bq;             /* rows (query position x head) per workgroup */",
              "    uint32_t bk;             /* keys per block */",
              "    const uint32_t* spirv;", "    size_t spirv_bytes;", "    const char* spirv_sha256;",
              "} AmevaFaVariant;", "",
              "static const AmevaFaVariant kAmevaFaVariants[] = {"]
    for dmax, bq, bk, var, nbytes, sha in fa:
        lines.append('    { %du, %du, %du, %s, %du, "%s" },' % (dmax, bq, bk, var, nbytes, sha))
    lines += ["};", "#define AMEVA_FA_VARIANT_COUNT %d" % len(fa),
              "#define AMEVA_FA_BINDINGS %d" % fbind,
              "#define AMEVA_FA_PUSH_CONSTANT_BYTES %d" % fpush,
              "#define AMEVA_FA_LOCAL_SIZE_X %d" % flocal, "",
              "typedef enum {"]
    for op, *_ in ops:
        lines.append("    AMEVA_K_%s," % op.upper())
    lines += ["    AMEVA_K_COUNT", "} AmevaOpKernelId;", "",
              "typedef struct {",
              "    const char* name;", "    uint32_t bindings;", "    uint32_t push_constant_bytes;",
              "    uint32_t local_size_x;", "    const uint32_t* spirv;", "    size_t spirv_bytes;",
              "    const char* spirv_sha256;", "} AmevaOpKernel;", "",
              "static const AmevaOpKernel kAmevaOpKernels[AMEVA_K_COUNT] = {"]
    for op, var, nbytes, sha, nbind, npush, nlocal in ops:
        lines.append('    { "ameva_%s", %du, %du, %du, %s, %du, "%s" },' % (op, nbind, npush, nlocal, var, nbytes, sha))
    lines += ["};", "",
              "/* SPIR-V only: these kernels have no OpenCL C translation (no entry in ggml_ameva_kernels_cl.h). */",
              "typedef enum {"]
    for name, *_ in vk:
        lines.append("    AMEVA_VK_%s," % name.upper())
    lines += ["    AMEVA_VK_COUNT", "} AmevaVkKernelId;", "",
              "static const AmevaOpKernel kAmevaVkKernels[AMEVA_VK_COUNT] = {"]
    for name, var, nbytes, sha, nbind, npush, nlocal in vk:
        lines.append('    { "ameva_%s", %du, %du, %du, %s, %du, "%s" },' % (name, nbind, npush, nlocal, var, nbytes, sha))
    lines += ["};", "", "#endif /* GGML_AMEVA_KERNELS_SPV_H */", ""]
    with open(args.out, "w", newline="\n") as f:
        f.write("\n".join(lines))
    counts = [len(qmm), len(qmv), len(mali_qmv), len(qgemm), len(mmq), len(mmqb), len(mmvq), len(mmvqe), len(fa),
              len(ops)]
    if [len(cl[key]) for key, _ in CL_TABLES] != counts:
        fail("OpenCL C tables %s do not match the SPIR-V tables %s" % ([len(cl[key]) for key, _ in CL_TABLES], counts))
    cl_bytes = write_cl_header(args.cl_out, version, cl)
    print("wrote %s (%d kernels as OpenCL C, %d bytes of kernel text)" % (args.cl_out, sum(counts), cl_bytes))
    # the summary of the SPIR-V header stays the last line (scripts read it)
    print("wrote %s (%d qmm, %d qmv, %d mali_qmv, %d qgemm, %d mmq, %d mmqb, %d mmvq, %d mmvqe, %d fa variants, "
          "%d op kernels, %d SPIR-V-only kernels)" % (args.out, len(qmm), len(qmv), len(mali_qmv), len(qgemm), len(mmq),
                                                      len(mmqb), len(mmvq), len(mmvqe), len(fa), len(ops), len(vk)))


if __name__ == "__main__":
    main()
