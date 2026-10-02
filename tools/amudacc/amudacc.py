#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
amudacc - AMUDA CUDA-like kernel compiler (POC v0.1)

Input  : a .acu file written in a CUDA C subset
         (__global__ / __device__ / __shared__, threadIdx / blockIdx / blockDim / gridDim,
          __syncthreads, atomics)
Output : per __global__ kernel
           Vulkan : GLSL 450 compute shader  -> SPIR-V (glslangValidator or glslc)
           OpenCL : OpenCL C 1.2 source
         plus one C header that embeds both images and typed launch wrappers for amuda_rt.

Policy : anything outside the subset is a compile error with file:line.
         Nothing is silently dropped, and no backend is silently skipped.

Python 3.8+, standard library only (runs inside Termux).
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

VERSION = "0.1.0"
IMAGE_ABI = 1

# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class AmuError(Exception):
    def __init__(self, msg, fname=None, line=None):
        super().__init__(msg)
        self.msg = msg
        self.fname = fname
        self.line = line

    def __str__(self):
        loc = ""
        if self.fname:
            loc = self.fname + (":%d" % self.line if self.line else "") + ": "
        return loc + "error: " + self.msg


# --------------------------------------------------------------------------------------
# Tokens / lexer
# --------------------------------------------------------------------------------------


class Tok:
    __slots__ = ("kind", "text", "line")

    def __init__(self, kind, text, line=0):
        self.kind = kind  # id num op pp str chr raw
        self.text = text
        self.line = line

    def __repr__(self):
        return "Tok(%s,%r,%d)" % (self.kind, self.text, self.line)


def raw(text):
    return Tok("raw", text, 0)


_PUNCTS = ["<<=", ">>=", "...", "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&",
           "||", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "::"]
_SINGLE = set("+-*/%&|^~!=<>?:;,.()[]{}")
_NUM_RE = re.compile(r"0[xX][0-9a-fA-F]+[uUlL]*|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFuUlL]*")
_ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def strip_comments(src, fname):
    """Replace comments with spaces, keep newlines so line numbers stay exact."""
    out = []
    i, n, line = 0, len(src), 1
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            out.append(c)
            i += 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" ")
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            if j < 0:
                raise AmuError("unterminated /* comment", fname, line)
            nl = src.count("\n", i, j)
            line += nl
            out.append(" " + "\n" * nl)
            i = j + 2
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == "\\":
                    j += 1
                if j < n and src[j] == "\n":
                    raise AmuError("unterminated literal", fname, line)
                j += 1
            out.append(src[i:j + 1])
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def lex(src, fname):
    src = strip_comments(src.replace("\r\n", "\n").replace("\r", "\n"), fname)
    toks = []
    i, n, line = 0, len(src), 1
    bol = True
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            bol = True
            continue
        if c in " \t\f\v":
            i += 1
            continue
        if c == "#" and bol:
            start = line
            parts = []
            j = i
            while True:
                k = src.find("\n", j)
                if k < 0:
                    k = n
                seg = src[j:k]
                if seg.rstrip().endswith("\\"):
                    parts.append(seg.rstrip()[:-1])
                    j = k + 1
                    line += 1
                    continue
                parts.append(seg)
                i = k
                break
            toks.append(Tok("pp", " ".join(p.strip() for p in parts).strip(), start))
            continue
        bol = False
        m = _ID_RE.match(src, i)
        if m:
            toks.append(Tok("id", m.group(), line))
            i = m.end()
            continue
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            m = _NUM_RE.match(src, i)
            toks.append(Tok("num", m.group(), line))
            i = m.end()
            continue
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == "\\":
                    j += 1
                j += 1
            toks.append(Tok("str" if c == '"' else "chr", src[i:j + 1], line))
            i = j + 1
            continue
        for p in _PUNCTS:
            if src.startswith(p, i):
                toks.append(Tok("op", p, line))
                i += len(p)
                break
        else:
            if c in _SINGLE:
                toks.append(Tok("op", c, line))
                i += 1
            else:
                raise AmuError("unexpected character %r" % c, fname, line)
    return toks


# --------------------------------------------------------------------------------------
# Token helpers
# --------------------------------------------------------------------------------------

_OPEN = {"(": ")", "[": "]", "{": "}"}


def match(toks, i, fname):
    """Index of the bracket closing toks[i]."""
    o = toks[i].text
    c = _OPEN[o]
    depth = 0
    for j in range(i, len(toks)):
        t = toks[j]
        if t.kind != "op":
            continue
        if t.text == o:
            depth += 1
        elif t.text == c:
            depth -= 1
            if depth == 0:
                return j
    raise AmuError("unbalanced '%s'" % o, fname, toks[i].line)


def split_commas(toks, fname):
    out, cur, depth = [], [], 0
    for t in toks:
        if t.kind == "op" and t.text in "([{":
            depth += 1
        elif t.kind == "op" and t.text in ")]}":
            depth -= 1
        if t.kind == "op" and t.text == "," and depth == 0:
            out.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur or out:
        out.append(cur)
    return out


def first_line(toks):
    for t in toks:
        if t.line:
            return t.line
    return None


# --------------------------------------------------------------------------------------
# Language tables
# --------------------------------------------------------------------------------------

SCALAR_TYPES = {"float", "int", "uint"}
TYPE_WORDS = {"float", "int", "uint", "unsigned", "signed", "uint32_t", "int32_t", "size_t", "bool", "void"}
DECL_TYPES = {"float", "int", "uint", "bool", "void"}

REJECT_IDS = {
    "double": "double(FP64)은 모바일 GPU 서브셋에서 미지원",
    "long": "long/64비트 정수 미지원",
    "short": "short 미지원",
    "char": "char 미지원",
    "half": "half는 POC v0.1에서 미지원",
    "__half": "__half는 POC v0.1에서 미지원",
    "struct": "struct 미지원 (POC v0.1)",
    "union": "union 미지원",
    "class": "class 미지원",
    "typedef": "typedef 미지원",
    "template": "template 미지원",
    "goto": "goto 미지원",
    "new": "동적 할당 미지원",
    "delete": "동적 할당 미지원",
    "malloc": "동적 할당 미지원",
    "free": "동적 할당 미지원",
    "printf": "커널 내 printf 미지원",
    "dim3": "dim3 변수 미지원: threadIdx.x 처럼 성분으로 사용",
    "warpSize": "warp 기능은 POC v0.1에서 미지원",
    "auto": "auto 미지원",
    "enum": "enum 미지원",
    "namespace": "namespace 미지원",
    "texture": "텍스처 미지원",
}

# CUDA math name -> (glsl, opencl). None means unsupported on that backend.
MATH = {
    "sqrtf": ("sqrt", "sqrt"), "rsqrtf": ("inversesqrt", "rsqrt"),
    "expf": ("exp", "exp"), "__expf": ("exp", "exp"), "exp2f": ("exp2", "exp2"),
    "logf": ("log", "log"), "__logf": ("log", "log"), "log2f": ("log2", "log2"),
    "powf": ("pow", "pow"), "__powf": ("pow", "pow"),
    "sinf": ("sin", "sin"), "__sinf": ("sin", "sin"), "cosf": ("cos", "cos"), "__cosf": ("cos", "cos"),
    "tanf": ("tan", "tan"), "tanhf": ("tanh", "tanh"),
    "asinf": ("asin", "asin"), "acosf": ("acos", "acos"), "atanf": ("atan", "atan"), "atan2f": ("atan", "atan2"),
    "fabsf": ("abs", "fabs"), "fabs": ("abs", "fabs"),
    "fminf": ("min", "fmin"), "fmaxf": ("max", "fmax"),
    "floorf": ("floor", "floor"), "ceilf": ("ceil", "ceil"), "truncf": ("trunc", "trunc"),
    "roundf": ("amu_roundf", "round"), "fmodf": ("amu_fmodf", "fmod"),
    "fmaf": ("fma", "fma"), "__fdividef": ("amu_fdividef", "amu_fdividef"),
    "__saturatef": ("amu_saturatef", "amu_saturatef"),
    "isnan": ("isnan", "isnan"), "isinf": ("isinf", "isinf"),
    "erff": (None, "erf"),
}

ATOMICS = {"atomicAdd", "atomicSub", "atomicExch", "atomicMin", "atomicMax", "atomicAnd", "atomicOr",
           "atomicXor", "atomicCAS"}
ATOMIC_GLSL = {"atomicAdd": "atomicAdd", "atomicSub": "atomicAdd", "atomicExch": "atomicExchange",
               "atomicMin": "atomicMin", "atomicMax": "atomicMax", "atomicAnd": "atomicAnd",
               "atomicOr": "atomicOr", "atomicXor": "atomicXor", "atomicCAS": "atomicCompSwap"}
ATOMIC_CL = {"atomicAdd": "atomic_add", "atomicSub": "atomic_sub", "atomicExch": "atomic_xchg",
             "atomicMin": "atomic_min", "atomicMax": "atomic_max", "atomicAnd": "atomic_and",
             "atomicOr": "atomic_or", "atomicXor": "atomic_xor", "atomicCAS": "atomic_cmpxchg"}

BUILTIN_DIM = {
    "threadIdx": ("int(gl_LocalInvocationID.%s)", "((int)get_local_id(%d))"),
    "blockIdx": ("int(gl_WorkGroupID.%s)", "((int)get_group_id(%d))"),
    "blockDim": ("int(gl_WorkGroupSize.%s)", "((int)get_local_size(%d))"),
    "gridDim": ("int(gl_NumWorkGroups.%s)", "((int)get_num_groups(%d))"),
}

SYNC = {
    "__syncthreads": ("{ memoryBarrierShared(); memoryBarrierBuffer(); barrier(); }",
                      "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)"),
    "__threadfence": ("memoryBarrier()", "mem_fence(CLK_GLOBAL_MEM_FENCE)"),
    "__threadfence_block": ("groupMemoryBarrier()", "mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)"),
}

DROP_QUALIFIERS = {"__device__", "__host__", "__forceinline__", "__noinline__", "inline", "static",
                   "register", "__restrict__", "__restrict", "restrict", "volatile"}

# Identifiers that are legal user names in CUDA but reserved / builtin in GLSL or OpenCL C.
GLSL_RESERVED = set("""
attribute uniform varying buffer shared coherent readonly writeonly atomic_uint layout centroid flat smooth
noperspective patch sample subroutine in out inout invariant precise discard lowp mediump highp precision
common partition active asm resource noinline public external interface fixed superp input output filter
cast using vec2 vec3 vec4 ivec2 ivec3 ivec4 uvec2 uvec3 uvec4 bvec2 bvec3 bvec4 dvec2 dvec3 dvec4 mat2 mat3
mat4 dmat2 dmat3 dmat4 hvec2 hvec3 hvec4 fvec2 fvec3 fvec4 sampler1D sampler2D sampler3D image1D image2D
image3D length distance dot cross normalize reflect refract step smoothstep mix clamp fract mod sign radians
degrees any all not equal notEqual lessThan greaterThan lessThanEqual greaterThanEqual sin cos tan asin acos
atan exp log exp2 log2 sqrt inversesqrt abs floor ceil round trunc min max pow fma barrier memoryBarrier
determinant inverse transpose outerProduct faceforward frexp ldexp isnan isinf uintBitsToFloat
floatBitsToUint floatBitsToInt intBitsToFloat atomicCompSwap atomicExchange main
""".split())
OPENCL_RESERVED = set("""
kernel global local constant private read_only write_only read_write image1d_t image2d_t image3d_t sampler_t
event_t uchar ushort ulong float2 float3 float4 float8 float16 int2 int3 int4 int8 int16 uint2 uint3 uint4
char2 char4 get_global_id get_local_id get_group_id get_local_size get_num_groups get_global_size barrier
mem_fence fmin fmax fabs rsqrt atomic_add atomic_sub atomic_xchg atomic_min atomic_max atomic_and atomic_or
atomic_xor atomic_cmpxchg complex imaginary
""".split())
RESERVED = GLSL_RESERVED | OPENCL_RESERVED


# --------------------------------------------------------------------------------------
# AST-ish structures
# --------------------------------------------------------------------------------------


class Param:
    def __init__(self, name, base, is_buffer, readonly, restrict, line):
        self.name = name          # user name (as written)
        self.ident = name         # emitted identifier (renamed if reserved)
        self.base = base          # float | int | uint
        self.is_buffer = is_buffer
        self.readonly = readonly
        self.restrict = restrict
        self.line = line
        self.slot = 0             # binding index (buffer) or push-constant byte offset (scalar)
        self.float_atomic_storage = False  # float buffer accessed only via atomicAdd -> uint storage


class Shared:
    def __init__(self, base, name, dims, line):
        self.base = base
        self.name = name
        self.ident = name
        self.dims = dims          # list of token lists
        self.line = line


class Kernel:
    def __init__(self, name, params, body, line):
        self.name = name
        self.params = params
        self.body = body
        self.line = line
        self.shared = []
        self.shared_bytes = 0


# --------------------------------------------------------------------------------------
# Type normalisation
# --------------------------------------------------------------------------------------


def normalize_type_words(words, fname, line):
    w = " ".join(words)
    table = {
        "float": "float", "int": "int", "signed": "int", "signed int": "int", "int32_t": "int",
        "unsigned": "uint", "unsigned int": "uint", "uint": "uint", "uint32_t": "uint",
        "size_t": "uint", "bool": "bool", "void": "void",
    }
    if w not in table:
        raise AmuError("지원하지 않는 타입 '%s' (float, int, unsigned int만 지원)" % w, fname, line)
    return table[w]


def normalize_types(toks, fname):
    """Merge multi-word integer types into one token: 'unsigned int' -> 'uint' etc."""
    out = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.kind == "id" and t.text in ("unsigned", "signed"):
            words = [t.text]
            j = i + 1
            if j < len(toks) and toks[j].kind == "id" and toks[j].text == "int":
                words.append("int")
                j += 1
            if j < len(toks) and toks[j].kind == "id" and toks[j].text in ("long", "short", "char"):
                raise AmuError("지원하지 않는 타입 '%s %s'" % (" ".join(words), toks[j].text), fname, t.line)
            out.append(Tok("id", normalize_type_words(words, fname, t.line), t.line))
            i = j
            continue
        if t.kind == "id" and t.text in ("uint32_t", "int32_t", "size_t"):
            out.append(Tok("id", normalize_type_words([t.text], fname, t.line), t.line))
            i += 1
            continue
        out.append(t)
        i += 1
    return out


# --------------------------------------------------------------------------------------
# Top level parsing
# --------------------------------------------------------------------------------------


def pp_name(text):
    m = re.match(r"#\s*([A-Za-z_]\w*)", text)
    return m.group(1) if m else ""


def parse_params(toks, fname, kname):
    params = []
    if len(toks) == 1 and toks[0].text == "void":
        return params
    if not toks:
        return params
    for p in split_commas(toks, fname):
        if not p:
            raise AmuError("빈 파라미터", fname, first_line(toks))
        line = first_line(p)
        restrict = any(t.text in ("__restrict__", "__restrict", "restrict") for t in p)
        p = [t for t in p if t.text not in ("__restrict__", "__restrict", "restrict", "volatile")]
        star_idx = [i for i, t in enumerate(p) if t.text == "*"]
        if len(star_idx) > 1:
            raise AmuError("다중 포인터 파라미터 미지원", fname, line)
        if p[-1].kind != "id":
            raise AmuError("파라미터 이름이 필요함", fname, line)
        name = p[-1].text
        readonly = False
        if star_idx:
            readonly = any(t.text == "const" for t in p[:star_idx[0]])
        words = [t.text for t in p[:-1] if t.text not in ("const", "*")]
        if not words:
            raise AmuError("파라미터 '%s'에 타입이 없음" % name, fname, line)
        for w in words:
            if w in REJECT_IDS:
                raise AmuError(REJECT_IDS[w], fname, line)
        base = normalize_type_words(words, fname, line)
        if base not in SCALAR_TYPES:
            raise AmuError("파라미터 '%s' 타입 '%s' 미지원 (float/int/unsigned int만)" % (name, base), fname, line)
        params.append(Param(name, base, bool(star_idx), readonly, restrict, line))
    names = [p.name for p in params]
    if len(set(names)) != len(names):
        raise AmuError("커널 '%s'에 중복 파라미터 이름" % kname, fname)
    return params


def parse_kernel(item, fname):
    i = 0
    line = item[0].line
    if item[i].text != "__global__":
        raise AmuError("__global__ 앞에는 extern \"C\" 외의 수식어를 둘 수 없음", fname, line)
    i += 1
    if i < len(item) and item[i].text == "__launch_bounds__":
        if item[i + 1].text != "(":
            raise AmuError("__launch_bounds__ 형식 오류", fname, line)
        i = match(item, i + 1, fname) + 1
    if i >= len(item) or item[i].text != "void":
        raise AmuError("__global__ 함수는 void를 반환해야 함", fname, line)
    i += 1
    if i >= len(item) or item[i].kind != "id":
        raise AmuError("커널 이름이 필요함", fname, line)
    name = item[i].text
    if name.startswith("amu_") or name.startswith("gl_") or "__" in name:
        raise AmuError("커널 이름 '%s'은 예약된 형태 (amu_, gl_, __ 금지)" % name, fname, line)
    i += 1
    if item[i].text != "(":
        raise AmuError("커널 파라미터 목록이 필요함", fname, line)
    close = match(item, i, fname)
    params = parse_params(item[i + 1:close], fname, name)
    j = close + 1
    if j < len(item) and item[j].text == "__launch_bounds__":
        j = match(item, j + 1, fname) + 1
    if j >= len(item) or item[j].text != "{":
        raise AmuError("커널 '%s'의 본문이 필요함 (선언만 있는 커널 미지원)" % name, fname, line)
    end = match(item, j, fname)
    if end != len(item) - 1:
        raise AmuError("커널 정의 뒤에 예상치 못한 토큰", fname, item[end].line)
    if not any(p.is_buffer for p in params):
        raise AmuError("커널 '%s'에 버퍼(포인터) 파라미터가 하나 이상 필요함" % name, fname, line)
    if sum(1 for p in params if p.is_buffer) > 16:
        raise AmuError("버퍼 파라미터는 최대 16개", fname, line)
    if sum(1 for p in params if not p.is_buffer) * 4 > 128:
        raise AmuError("스칼라 파라미터는 최대 32개(128바이트)", fname, line)
    return Kernel(name, params, item[j + 1:end], line)


def split_top(toks, fname):
    """Returns list of ('pp', tok) | ('kernel', Kernel) | ('code', toks)."""
    items = []
    i = 0
    pp_depth = 0
    while i < len(toks):
        t = toks[i]
        if t.kind == "pp":
            k = pp_name(t.text)
            if k == "include":
                raise AmuError("#include 미지원 (커널은 단일 파일)", fname, t.line)
            if k in ("if", "ifdef", "ifndef"):
                pp_depth += 1
            elif k == "endif":
                pp_depth -= 1
            items.append(("pp", t))
            i += 1
            continue
        if t.text == "extern":
            if i + 1 < len(toks) and toks[i + 1].kind == "str" and toks[i + 1].text == '"C"':
                if i + 2 < len(toks) and toks[i + 2].text == "{":
                    raise AmuError('extern "C" { } 블록 미지원: 커널마다 extern "C"를 붙이거나 생략', fname, t.line)
                i += 2
                continue
            raise AmuError("extern 미지원 (동적 __shared__ 포함)", fname, t.line)
        # collect one top-level item
        j = i
        while j < len(toks):
            tt = toks[j]
            if tt.kind == "op" and tt.text == ";":
                break
            if tt.kind == "op" and tt.text == "{":
                j = match(toks, j, fname)
                break
            if tt.kind == "op" and tt.text in "([":
                j = match(toks, j, fname)
            j += 1
        if j >= len(toks):
            raise AmuError("최상위 선언이 끝나지 않음 (';' 또는 '}' 누락)", fname, t.line)
        item = toks[i:j + 1]
        if any(x.text == "__global__" for x in item):
            if pp_depth != 0:
                raise AmuError("#if 블록 안의 __global__ 커널은 미지원", fname, t.line)
            items.append(("kernel", parse_kernel(item, fname)))
        else:
            check_global_item(item, fname)
            items.append(("code", item))
        i = j + 1
    if pp_depth != 0:
        raise AmuError("#if/#endif 짝이 맞지 않음", fname)
    return items


def check_global_item(item, fname):
    line = item[0].line
    if any(x.text == "__constant__" for x in item):
        raise AmuError("__constant__ 메모리는 POC v0.1에서 미지원 (스칼라 파라미터나 버퍼 사용)", fname, line)
    if any(x.text == "__shared__" for x in item):
        raise AmuError("__shared__는 커널 본문 안에서만 선언 가능", fname, line)
    has_paren = any(x.text == "(" for x in item)
    is_def = item[-1].text == "}"
    if is_def:
        # function definition: '=' before the first '(' means a braced initializer (global var)
        first_paren = next(k for k, x in enumerate(item) if x.text == "(")
        if any(x.text == "=" for x in item[:first_paren]):
            raise AmuError("전역 변수 미지원 (device 메모리는 버퍼 파라미터로 전달)", fname, line)
        close = match(item, first_paren, fname)
        if close + 1 >= len(item) or item[close + 1].text != "{":
            raise AmuError("함수 정의 형식 오류", fname, line)
        if any(x.text == "*" or x.text == "&" for x in item[first_paren + 1:close]):
            raise AmuError("__device__ 함수의 포인터/참조 파라미터 미지원 (값으로 전달)", fname, line)
    elif has_paren:
        pass  # prototype
    else:
        raise AmuError("전역 변수 미지원 (device 메모리는 버퍼 파라미터로 전달)", fname, line)


# --------------------------------------------------------------------------------------
# Body analysis
# --------------------------------------------------------------------------------------


def extract_shared(body, fname):
    out, shared = [], []
    i = 0
    while i < len(body):
        t = body[i]
        if t.text == "__shared__":
            prev = out[-1].text if out else "{"
            if prev == "extern":
                raise AmuError("extern __shared__ (동적 공유 메모리)는 POC v0.1에서 미지원", fname, t.line)
            if prev not in (";", "{", "}") and not (out and out[-1].kind == "pp"):
                raise AmuError("__shared__ 선언은 문장의 시작에 있어야 함", fname, t.line)
            j = i + 1
            if j < len(body) and body[j].text == "volatile":
                j += 1
            words = []
            while j < len(body) and body[j].kind == "id" and body[j].text in TYPE_WORDS:
                words.append(body[j].text)
                j += 1
            if not words:
                raise AmuError("__shared__ 변수에 타입이 필요함", fname, t.line)
            base = normalize_type_words(words, fname, t.line)
            if base not in SCALAR_TYPES:
                raise AmuError("__shared__ 타입은 float/int/unsigned int만", fname, t.line)
            if j >= len(body) or body[j].kind != "id":
                raise AmuError("__shared__ 변수 이름이 필요함", fname, t.line)
            name = body[j].text
            j += 1
            dims = []
            while j < len(body) and body[j].text == "[":
                k = match(body, j, fname)
                if k == j + 1:
                    raise AmuError("__shared__ 배열 크기는 상수로 지정해야 함 (동적 공유 메모리 미지원)", fname, t.line)
                dims.append(body[j + 1:k])
                j = k + 1
            if j >= len(body) or body[j].text != ";":
                raise AmuError("__shared__는 '__shared__ T name[N];' 형태로 하나씩 선언 (초기화 불가)", fname, t.line)
            shared.append(Shared(base, name, dims, t.line))
            i = j + 1
            continue
        out.append(t)
        i += 1
    return out, shared


def is_unary_context(prev):
    if prev is None:
        return True
    if prev.kind in ("id", "num", "raw") and prev.text != "return":
        return False
    if prev.kind == "op" and prev.text in (")", "]", "++", "--"):
        return False
    return True


def check_common(toks, fname, buffers, in_kernel):
    """Subset checks shared by every backend. 'buffers' = set of buffer parameter names."""
    for idx, t in enumerate(toks):
        prev = toks[idx - 1] if idx > 0 else None
        nxt = toks[idx + 1] if idx + 1 < len(toks) else None
        if t.kind in ("str", "chr"):
            raise AmuError("문자열/문자 리터럴 미지원", fname, t.line)
        if t.kind == "num":
            body = t.text.lower()
            if not body.startswith("0x") and ("l" in body):
                raise AmuError("64비트 정수 리터럴(l/ll 접미사) 미지원: %s" % t.text, fname, t.line)
            if body.startswith("0x") and "l" in body:
                raise AmuError("64비트 정수 리터럴 미지원: %s" % t.text, fname, t.line)
        if t.kind == "op" and t.text == "->":
            raise AmuError("'->' 미지원 (포인터/구조체 없음)", fname, t.line)
        if t.kind == "op" and t.text == "::":
            raise AmuError("'::' 미지원", fname, t.line)
        if t.kind != "id":
            continue
        if prev is not None and prev.kind == "op" and prev.text == ".":
            continue  # member / swizzle name (threadIdx.x), not a user identifier
        w = t.text
        if w in REJECT_IDS:
            raise AmuError(REJECT_IDS[w], fname, t.line)
        if w.startswith("cuda"):
            raise AmuError("'%s': CUDA 호스트 API는 커널 안에서 사용할 수 없음" % w, fname, t.line)
        if w.startswith("amu_"):
            raise AmuError("'amu_' 접두사는 amudacc 예약 이름", fname, t.line)
        if w.startswith("__") and w not in SYNC and w not in MATH and w not in DROP_QUALIFIERS \
                and w not in ("__shared__", "__global__"):
            raise AmuError("미지원 내장 함수/수식어 '%s'" % w, fname, t.line)
        if in_kernel and w == "static":
            raise AmuError("커널 안의 static 변수 미지원", fname, t.line)
        if w in buffers:
            if nxt is None or nxt.text != "[":
                raise AmuError("버퍼 '%s'는 %s[인덱스] 형태로만 사용 가능 (포인터 연산/전달 미지원)" % (w, w), fname, t.line)
            if prev is not None and prev.text == "&":
                pp = toks[idx - 2] if idx >= 2 else None
                ppp = toks[idx - 3] if idx >= 3 else None
                if not (pp is not None and pp.text == "(" and ppp is not None and ppp.text in ATOMICS):
                    raise AmuError("&%s[...]는 atomic 함수의 첫 인자에서만 사용 가능" % w, fname, t.line)
        if w in BUILTIN_DIM:
            if not (nxt is not None and nxt.text == "." and idx + 2 < len(toks) and toks[idx + 2].text in ("x", "y", "z")):
                raise AmuError("'%s'는 .x/.y/.z 성분으로만 사용 가능" % w, fname, t.line)
        if w in DECL_TYPES and nxt is not None and nxt.text == "*":
            raise AmuError("포인터 변수/파라미터 미지원 (버퍼는 커널 파라미터로만)", fname, t.line)
    # unary & outside atomics
    for idx, t in enumerate(toks):
        if t.kind == "op" and t.text == "&" and is_unary_context(toks[idx - 1] if idx > 0 else None):
            ok = idx >= 2 and toks[idx - 1].text == "(" and toks[idx - 2].text in ATOMICS
            if not ok:
                raise AmuError("주소 연산자 &는 atomic 함수의 첫 인자에서만 사용 가능", fname, t.line)


VK_MIN_SHARED_BYTES = 16384  # Vulkan-guaranteed minimum of maxComputeSharedMemorySize


def eval_const_expr(toks, macros, fname, line, depth=0):
    """Integer constant expression for __shared__ array sizes (literals, + - * / % ( ), object-like macros)."""
    parts = []
    for t in toks:
        if t.kind == "num":
            txt = t.text.rstrip("uU")
            if not re.fullmatch(r"0[xX][0-9a-fA-F]+|[1-9]\d*|0", txt):
                raise AmuError("__shared__ 배열 크기는 정수 상수여야 함: '%s'" % t.text, fname, line)
            parts.append(str(int(txt, 0)))
        elif t.kind == "id":
            if t.text not in macros or depth > 8:
                raise AmuError("__shared__ 배열 크기를 정적으로 계산할 수 없음: '%s' (정수 상수나 #define 상수만)" % t.text,
                               fname, line)
            inner = lex(macros[t.text], fname)
            parts.append("(%d)" % eval_const_expr(inner, macros, fname, line, depth + 1))
        elif t.kind == "op" and t.text in ("+", "-", "*", "/", "%", "(", ")"):
            parts.append("//" if t.text == "/" else t.text)
        else:
            raise AmuError("__shared__ 배열 크기 식에 허용되지 않는 토큰 '%s'" % t.text, fname, line)
    try:
        v = eval(" ".join(parts), {"__builtins__": {}}, {})  # only digits/operators/parens reach here
    except Exception:
        raise AmuError("__shared__ 배열 크기 식을 계산할 수 없음", fname, line)
    if not isinstance(v, int) or v <= 0:
        raise AmuError("__shared__ 배열 크기는 양의 정수여야 함 (계산값 %r)" % (v,), fname, line)
    return v


def check_float_atomic_buffers(toks, params, fname):
    """A float buffer used as an atomicAdd target must be accessed ONLY through atomicAdd in that kernel.
    The Vulkan path then declares it as uint storage and uses a CAS loop, with no aliased descriptors."""
    for p in params:
        p.float_atomic_storage = False
        if not p.is_buffer or p.base != "float":
            continue
        occ, atom = [], []
        for idx, t in enumerate(toks):
            if t.kind != "id" or t.text != p.name:
                continue
            if idx > 0 and toks[idx - 1].kind == "op" and toks[idx - 1].text == ".":
                continue
            occ.append(idx)
            if idx >= 3 and toks[idx - 1].text == "&" and toks[idx - 2].text == "(" and toks[idx - 3].text == "atomicAdd":
                atom.append(idx)
        if atom and len(atom) != len(occ):
            bad = next(i for i in occ if i not in atom)
            raise AmuError("float atomicAdd 대상 버퍼 '%s'는 이 커널에서 atomicAdd로만 접근해야 함 "
                           "(읽기/쓰기는 별도 버퍼 사용, POC v0.1)" % p.name, fname, toks[bad].line)
        p.float_atomic_storage = bool(atom)


def qualify_buffers(toks, idents):
    """GLSL: buffers are instance-named blocks; NAME[...] -> amu_buf_NAME.v[...] (no global-scope member names)."""
    out = []
    for i, t in enumerate(toks):
        after_dot = i > 0 and toks[i - 1].kind == "op" and toks[i - 1].text == "."
        if t.kind == "id" and t.text in idents and not after_dot and i + 1 < len(toks) and toks[i + 1].text == "[":
            out.append(Tok("raw", "amu_buf_%s.v" % t.text, t.line))
        else:
            out.append(t)
    return out


def collect_renames(toks, params, shared):
    """User identifiers colliding with GLSL/OpenCL reserved words get an amu_u_ prefix."""
    declared = set()
    for idx, t in enumerate(toks):
        if t.kind != "id" or t.text not in RESERVED:
            continue
        prev = toks[idx - 1] if idx > 0 else None
        if prev is not None and prev.kind == "op" and prev.text == ".":
            continue
        nxt = toks[idx + 1] if idx + 1 < len(toks) else None
        is_call = nxt is not None and nxt.text == "("
        user_func_decl = is_call and prev is not None and prev.kind == "id" and prev.text in DECL_TYPES
        if not is_call or user_func_decl:
            declared.add(t.text)
    for p in params:
        if p.name in RESERVED:
            declared.add(p.name)
    for s in shared:
        if s.name in RESERVED:
            declared.add(s.name)
    for idx, t in enumerate(toks):
        if idx > 0 and toks[idx - 1].kind == "op" and toks[idx - 1].text == ".":
            continue
        if t.kind == "id" and (t.text.startswith("gl_") or ("__" in t.text and t.text not in SYNC and t.text not in MATH
                                                              and t.text not in DROP_QUALIFIERS
                                                              and t.text not in ("__shared__", "__global__"))):
            declared.add(t.text)
    return {n: "amu_u_" + n.replace("__", "_") for n in declared}


# --------------------------------------------------------------------------------------
# Expression-level transforms (backend specific)
# --------------------------------------------------------------------------------------


def apply_renames(toks, ren):
    out = []
    for i, t in enumerate(toks):
        after_dot = i > 0 and toks[i - 1].kind == "op" and toks[i - 1].text == "."
        if t.kind == "id" and t.text in ren and not after_dot:
            out.append(Tok("id", ren[t.text], t.line))
        else:
            out.append(t)
    return out


def transform_builtins(toks, backend, fname):
    out = []
    i = 0
    bi = 0 if backend == "glsl" else 1
    while i < len(toks):
        t = toks[i]
        if t.kind == "id" and t.text in BUILTIN_DIM and i + 2 < len(toks) and toks[i + 1].text == "." \
                and toks[i + 2].text in ("x", "y", "z"):
            axis = toks[i + 2].text
            fmt = BUILTIN_DIM[t.text][bi]
            txt = fmt % (axis if backend == "glsl" else "xyz".index(axis))
            out.append(Tok("raw", txt, t.line))
            i += 3
            continue
        if t.kind == "id" and t.text in SYNC:
            if not (i + 2 < len(toks) and toks[i + 1].text == "(" and toks[i + 2].text == ")"):
                raise AmuError("%s()는 인자 없이 호출" % t.text, fname, t.line)
            out.append(Tok("raw", SYNC[t.text][bi], t.line))
            i += 3
            continue
        if t.kind == "id" and t.text in MATH and i + 1 < len(toks) and toks[i + 1].text == "(":
            target = MATH[t.text][bi]
            if target is None:
                raise AmuError("'%s'는 %s 백엔드에서 미지원" % (t.text, "Vulkan" if backend == "glsl" else "OpenCL"),
                               fname, t.line)
            out.append(Tok("id", target, t.line))
            i += 1
            continue
        if t.kind == "id" and t.text == "sizeof":
            if i + 3 < len(toks) and toks[i + 1].text == "(" and toks[i + 2].text in ("float", "int", "uint") \
                    and toks[i + 3].text == ")":
                out.append(Tok("num", "4", t.line))
                i += 4
                continue
            raise AmuError("sizeof는 sizeof(float/int/unsigned int)만 지원", fname, t.line)
        out.append(t)
        i += 1
    return out


def drop_qualifiers(toks, backend):
    out = []
    for t in toks:
        if t.kind == "id" and t.text in DROP_QUALIFIERS:
            if backend == "cl" and t.text in ("__restrict__", "__restrict"):
                out.append(Tok("id", "restrict", t.line))
            continue
        if backend == "glsl" and t.kind == "id" and t.text == "const":
            continue  # GLSL const requires constant initialisers; dropping it never changes semantics
        out.append(t)
    return out


def operand_end(toks, s, fname, line):
    if s >= len(toks):
        raise AmuError("캐스트 뒤에 피연산자가 없음", fname, line)
    t = toks[s]
    if t.kind == "op" and t.text in ("-", "+", "!", "~"):
        return operand_end(toks, s + 1, fname, line)
    if t.kind == "op" and t.text == "(":
        e = match(toks, s, fname) + 1
    elif t.kind in ("id", "num", "raw"):
        e = s + 1
    else:
        raise AmuError("캐스트 피연산자를 해석할 수 없음: '%s' (괄호로 감싸세요)" % t.text, fname, line)
    while e < len(toks):
        x = toks[e]
        if x.kind == "op" and x.text == "[":
            e = match(toks, e, fname) + 1
        elif x.kind == "op" and x.text == "(" and toks[e - 1].kind in ("id", "raw"):
            e = match(toks, e, fname) + 1
        elif x.kind == "op" and x.text == "." and e + 1 < len(toks) and toks[e + 1].kind == "id":
            e += 2
        elif x.kind == "op" and x.text in ("++", "--"):
            e += 1
        else:
            break
    return e


def convert_casts_glsl(toks, fname):
    """(T)expr -> T(expr); innermost/rightmost first so nested casts resolve."""
    while True:
        pos = -1
        for i in range(len(toks) - 2):
            if toks[i].kind == "op" and toks[i].text == "(" and toks[i + 1].kind == "id" \
                    and toks[i + 1].text in ("float", "int", "uint", "bool") and toks[i + 2].text == ")":
                pos = i
        if pos < 0:
            return toks
        ty = toks[pos + 1].text
        line = toks[pos].line
        end = operand_end(toks, pos + 3, fname, line)
        toks = toks[:pos] + [Tok("raw", ty, line), Tok("op", "(", line)] + toks[pos + 3:end] + \
            [Tok("op", ")", line)] + toks[end:]


class AtomicCtx:
    def __init__(self):
        self.counter = 0
        self.cl_float_atomic = False


def rewrite_atomics(toks, backend, symtab, fname, actx):
    out = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if not (t.kind == "id" and t.text in ATOMICS):
            out.append(t)
            i += 1
            continue
        fn = t.text
        if i + 1 >= len(toks) or toks[i + 1].text != "(":
            raise AmuError("%s(...) 호출 형식 오류" % fn, fname, t.line)
        close = match(toks, i + 1, fname)
        args = split_commas(toks[i + 2:close], fname)
        nargs = 3 if fn == "atomicCAS" else 2
        if len(args) != nargs:
            raise AmuError("%s 인자 개수는 %d개" % (fn, nargs), fname, t.line)
        first = args[0]
        if len(first) < 2 or first[0].text != "&" or first[1].kind != "id":
            raise AmuError("%s의 첫 인자는 &이름[인덱스] 또는 &공유변수 형태여야 함" % fn, fname, t.line)
        name = first[1].text
        sym = symtab.get(name)
        if sym is None:
            raise AmuError("원자연산 대상 '%s'는 버퍼 파라미터나 __shared__ 변수여야 함" % name, fname, t.line)
        kind, base, readonly, ident = sym
        if readonly:
            raise AmuError("const 버퍼 '%s'에는 원자연산 불가" % name, fname, t.line)
        target = first[1:]
        target = [Tok("id", ident, target[0].line)] + target[1:]
        vals = [rewrite_atomics(a, backend, symtab, fname, actx) for a in args[1:]]
        if base in ("int", "uint"):
            if backend == "glsl":
                if fn == "atomicSub":
                    vals = [[Tok("raw", "-(", 0)] + vals[0] + [Tok("raw", ")", 0)]]
                seq = [Tok("raw", ATOMIC_GLSL[fn], t.line), Tok("op", "(", t.line)] + target
                for v in vals:
                    seq += [Tok("op", ",", t.line)] + v
                seq.append(Tok("op", ")", t.line))
            else:
                seq = [Tok("raw", ATOMIC_CL[fn], t.line), Tok("op", "(", t.line), Tok("raw", "&", t.line)] + target
                for v in vals:
                    seq += [Tok("op", ",", t.line)] + v
                seq.append(Tok("op", ")", t.line))
            out.extend(seq)
            i = close + 1
            continue
        # float
        if fn != "atomicAdd":
            raise AmuError("float 원자연산은 atomicAdd만 지원 (POC v0.1)", fname, t.line)
        if kind == "shared":
            raise AmuError("__shared__ float에 대한 atomicAdd는 POC v0.1에서 미지원 (int/unsigned int는 가능)", fname, t.line)
        prev = out[-1].text if out else ";"
        if prev not in (";", "{", "}", ")", "else") or close + 1 >= len(toks) or toks[close + 1].text != ";":
            raise AmuError("float atomicAdd는 단독 문장으로만 사용 가능 (반환값 사용 불가)", fname, t.line)
        if len(target) < 3 or target[1].text != "[" or match(target, 1, fname) != len(target) - 1:
            raise AmuError("float atomicAdd 대상은 &버퍼[인덱스] 형태여야 함", fname, t.line)
        idx = target[2:-1]
        if backend == "glsl":
            n = actx.counter
            actx.counter += 1
            sym_param = symtab["__params__"][name]
            if not sym_param.float_atomic_storage:
                raise AmuError("내부 오류: float atomic 저장소 표시 누락 (%s)" % name, fname, t.line)
            u = "amu_buf_%s.v" % ident  # declared as uint storage
            # the index/value are rendered into raw text here, so qualify their buffer accesses now
            bq = set(symtab["__params__"].keys())
            idx = qualify_buffers(idx, bq)
            vals = [qualify_buffers(vals[0], bq)]
            code = ("{ uint amu_i%(n)d = uint(%(idx)s); float amu_v%(n)d = float(%(val)s); "
                    "uint amu_o%(n)d = %(u)s[amu_i%(n)d]; uint amu_a%(n)d; "
                    "do { amu_a%(n)d = amu_o%(n)d; amu_o%(n)d = atomicCompSwap(%(u)s[amu_i%(n)d], amu_a%(n)d, "
                    "floatBitsToUint(uintBitsToFloat(amu_a%(n)d) + amu_v%(n)d)); } "
                    "while (amu_a%(n)d != amu_o%(n)d); }") % {
                "n": n, "u": u, "idx": render_inline(idx), "val": render_inline(vals[0])}
            out.append(Tok("raw", code, t.line))
        else:
            actx.cl_float_atomic = True
            out.extend([Tok("raw", "amu_atomic_add_f32_global", t.line), Tok("op", "(", t.line),
                        Tok("raw", "&", t.line)] + target + [Tok("op", ",", t.line)] + vals[0] +
                       [Tok("op", ")", t.line)])
        i = close + 1
    return out


# --------------------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------------------

_NO_SPACE_BEFORE = {",", ";", ")", "]", "."}
_NO_SPACE_AFTER = {"(", "[", "."}


def render_inline(toks):
    s = []
    prev = None
    for t in toks:
        if t.kind == "pp":
            raise AmuError("식 안에 전처리기 지시문이 올 수 없음")
        if prev is not None:
            sp = True
            if t.text in _NO_SPACE_BEFORE and t.kind == "op":
                sp = False
            if prev.kind == "op" and prev.text in _NO_SPACE_AFTER:
                sp = False
            if t.kind == "op" and t.text in ("(", "[") and prev.kind in ("id", "raw"):
                sp = False
            if t.kind == "op" and t.text == "[" and prev.kind == "op" and prev.text == "]":
                sp = False
            if prev.kind == "raw" and (prev.text.endswith("(") or prev.text == "&"):
                sp = False
            if t.kind == "raw" and t.text == ")":
                sp = False
            if t.kind == "op" and t.text in ("++", "--") and prev.kind in ("id", "raw") or \
                    (t.kind == "op" and t.text in ("++", "--") and prev.text in (")", "]")):
                sp = False
            if sp:
                s.append(" ")
        s.append(t.text)
        prev = t
    return "".join(s)


def render_lines(toks, src_name, indent="    "):
    """Render tokens keeping source line structure and emitting #line directives."""
    out = []
    cur_line = None
    line_buf = []
    synced = False

    def flush():
        if line_buf:
            out.append(indent + render_inline(line_buf))
            line_buf.clear()

    for t in toks:
        if t.kind == "pp":
            flush()
            out.append(t.text)
            synced = False
            continue
        if t.line and t.line != cur_line:
            flush()
            if synced and cur_line is not None and 0 < t.line - cur_line <= 2:
                out.extend([""] * (t.line - cur_line - 1))
            else:
                out.append('#line %d "%s"' % (t.line, src_name))
            cur_line = t.line
            synced = True
        line_buf.append(t)
    flush()
    return out


def pp_transform(tok, backend, fname, ren, bufq=None):
    """Translate the body of #define lines; other directives pass through."""
    k = pp_name(tok.text)
    if k == "pragma":
        if re.match(r"#\s*pragma\s+unroll\b", tok.text):
            return None if backend == "glsl" else tok  # GLSL has no #pragma unroll
        raise AmuError("미지원 #pragma: %s" % tok.text, fname, tok.line)
    if k != "define":
        return tok
    m = re.match(r"#\s*define\s+([A-Za-z_]\w*)(\([^)]*\))?(.*)$", tok.text)
    if not m:
        raise AmuError("#define 형식 오류", fname, tok.line)
    name, args, body = m.group(1), m.group(2) or "", m.group(3)
    if name.startswith("amu_"):
        raise AmuError("'amu_' 접두사는 amudacc 예약 이름", fname, tok.line)
    name = ren.get(name, name)
    btoks = lex(body, fname)
    for bt in btoks:
        bt.line = tok.line
        if bt.kind == "id" and bt.text in ATOMICS:
            raise AmuError("#define 안의 atomic 함수 미지원", fname, tok.line)
    btoks = normalize_types(btoks, fname)
    btoks = apply_renames(btoks, ren)
    btoks = transform_builtins(btoks, backend, fname)
    btoks = drop_qualifiers(btoks, backend)
    if backend == "glsl":
        btoks = convert_casts_glsl(btoks, fname)
        if bufq:
            btoks = qualify_buffers(btoks, bufq)
    return Tok("pp", "#define %s%s %s" % (name, args, render_inline(btoks)), tok.line)


# emitted only when referenced
GLSL_HELPERS = {
    "amu_roundf": "float amu_roundf(float x) { return sign(x) * floor(abs(x) + 0.5); }",
    "amu_fmodf": "float amu_fmodf(float x, float y) { return x - y * trunc(x / y); }",
    "amu_saturatef": "float amu_saturatef(float x) { return clamp(x, 0.0, 1.0); }",
    "amu_fdividef": "float amu_fdividef(float x, float y) { return x / y; }",
}

CL_HELPERS = {
    "amu_saturatef": "inline float amu_saturatef(float x) { return clamp(x, 0.0f, 1.0f); }",
    "amu_fdividef": "inline float amu_fdividef(float x, float y) { return x / y; }",
}


def used_helpers(table, *token_lists):
    text = " ".join(t.text for tl in token_lists for t in tl)
    return [code for name, code in table.items() if re.search(r"\b%s\b" % name, text)]

CL_FLOAT_ATOMIC = """\
inline float amu_atomic_add_f32_global(volatile __global float* p, float v) {
    union { uint u; float f; } o, n;
    do { o.f = *p; n.f = o.f + v; }
    while (atomic_cmpxchg((volatile __global uint*)p, o.u, n.u) != o.u);
    return o.f;
}"""


class Compiler:
    def __init__(self, src, fname, defines):
        self.fname = os.path.basename(fname)
        self.src = src
        self.defines = defines
        toks = lex(src, self.fname)
        self.items = split_top(toks, self.fname)
        self.kernels = [it[1] for it in self.items if it[0] == "kernel"]
        if not self.kernels:
            raise AmuError("__global__ 커널이 없음", self.fname)
        names = [k.name for k in self.kernels]
        if len(set(names)) != len(names):
            raise AmuError("커널 이름 중복", self.fname)
        self.macros = {n: v for n, v in self.defines}
        for kind, it in self.items:
            if kind == "pp":
                m = re.match(r"#\s*define\s+([A-Za-z_]\w*)(\s+(.*))?$", it.text)
                if m:  # object-like only (a '(' right after the name makes it function-like)
                    self.macros[m.group(1)] = (m.group(3) or "").strip()

    # ---- per kernel preparation (backend independent) ----
    def prepare(self, k):
        global_code = []
        for kind, it in self.items:
            if kind == "pp":
                global_code.append(it)
            elif kind == "code":
                global_code.extend(it)
        body, shared = extract_shared(k.body, self.fname)
        k.shared = shared
        total = 0
        for sd in shared:
            n = 1
            for d in sd.dims:
                n *= eval_const_expr(normalize_types(d, self.fname), self.macros, self.fname, sd.line)
            total += 4 * n
        if total > VK_MIN_SHARED_BYTES:
            raise AmuError("정적 __shared__ 합계 %d B > %d B (Vulkan 최소 보장치). 기기별 한도에 기대지 않도록 거부함"
                           % (total, VK_MIN_SHARED_BYTES), self.fname, k.line)
        k.shared_bytes = total
        gcode = normalize_types(global_code, self.fname)
        body = normalize_types(body, self.fname)
        check_common([t for t in gcode if t.kind != "pp"], self.fname, set(), False)
        buffers = {p.name for p in k.params if p.is_buffer}
        check_common([t for t in body if t.kind != "pp"], self.fname, buffers, True)
        snames = [s.name for s in shared]
        if len(set(snames)) != len(snames) or set(snames) & {p.name for p in k.params}:
            raise AmuError("__shared__ 이름이 중복되거나 파라미터와 겹침", self.fname, k.line)
        ren = collect_renames([t for t in gcode + body if t.kind != "pp"], k.params, shared)
        check_float_atomic_buffers([t for t in body if t.kind != "pp"], k.params, self.fname)
        for p in k.params:
            p.ident = ren.get(p.name, p.name)
        for s in shared:
            s.ident = ren.get(s.name, s.name)
        b = 0
        off = 0
        for p in k.params:
            if p.is_buffer:
                p.slot = b
                b += 1
            else:
                p.slot = off
                off += 4
        return gcode, body, ren

    def lower(self, toks, backend, ren, symtab, actx, bufq_pp=None, bufq_tokens=None):
        out = []
        for t in toks:
            if t.kind == "pp":
                nt = pp_transform(t, backend, self.fname, ren, bufq_pp)
                if nt is not None:
                    out.append(nt)
            else:
                out.append(t)
        out = apply_renames(out, ren)
        out = transform_builtins(out, backend, self.fname)
        out = drop_qualifiers(out, backend)
        if backend == "glsl":
            out = convert_casts_glsl(out, self.fname)
        out = rewrite_atomics(out, backend, symtab, self.fname, actx)
        if backend == "glsl" and bufq_tokens:
            out = qualify_buffers(out, bufq_tokens)
        return out

    def symtab(self, k):
        # keyed by the emitted identifier: atomics are rewritten after renaming
        st = {"__params__": {}}
        for p in k.params:
            if p.is_buffer:
                st[p.ident] = ("buffer", p.base, p.readonly, p.ident)
                st["__params__"][p.ident] = p
        for s in k.shared:
            st[s.ident] = ("shared", s.base, False, s.ident)
        return st

    def define_lines(self):
        return ["#define %s %s" % (n, v) for n, v in self.defines]

    # ---- GLSL ----
    def glsl(self, k):
        gcode, body, ren = self.prepare(k)
        actx = AtomicCtx()
        st = self.symtab(k)
        bufq = {p.ident for p in k.params if p.is_buffer}
        g = self.lower(gcode, "glsl", ren, st, actx, bufq_pp=bufq)
        b = self.lower(body, "glsl", ren, st, actx, bufq_pp=bufq, bufq_tokens=bufq)
        shared_lines = []
        for s in k.shared:
            dims = "".join("[" + render_inline(self.lower(d, "glsl", ren, st, actx)) + "]" for d in s.dims)
            shared_lines.append('#line %d "%s"' % (s.line, self.fname))
            shared_lines.append("shared %s %s%s;" % (s.base, s.ident, dims))
        L = ["#version 450",
             "#extension GL_GOOGLE_cpp_style_line_directive : enable",
             "// generated by amudacc %s from %s - kernel %s - DO NOT EDIT" % (VERSION, self.fname, k.name),
             "layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;"]
        for p in k.params:
            if not p.is_buffer:
                continue
            quals = []
            if p.readonly:
                quals.append("readonly")
            if p.restrict:
                quals.append("restrict")
            qual = (" ".join(quals) + " ") if quals else ""
            elem = "uint" if p.float_atomic_storage else p.base
            note = "  // float data, atomicAdd via CAS" if p.float_atomic_storage else ""
            L.append("layout(std430, set = 0, binding = %d) %sbuffer AmuB%d { %s v[]; } amu_buf_%s;%s" % (
                p.slot, qual, p.slot, elem, p.ident, note))
        scalars = [p for p in k.params if not p.is_buffer]
        if scalars:
            L.append("layout(push_constant) uniform AmuPC { %s } amu_pc;" % " ".join(
                "%s %s;" % (p.base, p.ident) for p in scalars))
        L.extend(used_helpers(GLSL_HELPERS, g, b))
        L.extend(self.define_lines())
        L.extend(render_lines(g, self.fname, indent=""))
        L.extend(shared_lines)
        L.append("void main() {")
        for p in scalars:
            L.append("    %s %s = amu_pc.%s;" % (p.base, p.ident, p.ident))
        L.extend(render_lines(b, self.fname))
        L.append("}")
        L.append("")
        return "\n".join(L)

    # ---- OpenCL C ----
    def opencl(self, k):
        gcode, body, ren = self.prepare(k)
        actx = AtomicCtx()
        st = self.symtab(k)
        g = self.lower(gcode, "cl", ren, st, actx)
        b = self.lower(body, "cl", ren, st, actx)
        L = ["// generated by amudacc %s from %s - kernel %s - DO NOT EDIT" % (VERSION, self.fname, k.name)]
        L.extend(used_helpers(CL_HELPERS, g, b))
        if actx.cl_float_atomic:
            L.append(CL_FLOAT_ATOMIC)
        L.extend(self.define_lines())
        L.extend(render_lines(g, self.fname, indent=""))
        plist = []
        for p in k.params:
            if p.is_buffer:
                plist.append("__global %s%s*%s %s" % ("const " if p.readonly else "", p.base,
                                                     " restrict" if p.restrict else "", p.ident))
            else:
                plist.append("%s %s" % (p.base, p.ident))
        L.append('#line %d "%s"' % (k.line, self.fname))
        L.append("__kernel void %s(%s) {" % (k.name, ", ".join(plist)))
        for s in k.shared:
            dims = "".join("[" + render_inline(self.lower(d, "cl", ren, st, actx)) + "]" for d in s.dims)
            L.append('#line %d "%s"' % (s.line, self.fname))
            L.append("    __local %s %s%s;" % (s.base, s.ident, dims))
        L.extend(render_lines(b, self.fname))
        L.append("}")
        L.append("")
        return "\n".join(L)


# --------------------------------------------------------------------------------------
# External tools
# --------------------------------------------------------------------------------------


def find_spirv_compiler(explicit):
    if explicit:
        if not shutil.which(explicit) and not os.path.isfile(explicit):
            raise AmuError("지정한 SPIR-V 컴파일러를 찾을 수 없음: %s" % explicit)
        return explicit
    for cand in ("glslangValidator", "glslc"):
        p = shutil.which(cand)
        if p:
            return p
    raise AmuError("SPIR-V 컴파일러(glslangValidator 또는 glslc)가 없음. Termux: pkg install glslang "
                   "| Vulkan 타깃을 빼려면 --targets opencl")


def compile_spirv(glsl_text, kname, tool, target_env, workdir):
    comp = os.path.join(workdir, kname + ".comp")
    spv = os.path.join(workdir, kname + ".spv")
    with open(comp, "w", encoding="utf-8") as f:
        f.write(glsl_text)
    base = os.path.basename(tool)
    if base.startswith("glslc"):
        cmd = [tool, "-fshader-stage=compute", "--target-env=" + target_env, "-o", spv, comp]
    else:
        cmd = [tool, "-V", "--target-env", target_env, "-S", "comp", "-o", spv, comp]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if r.returncode != 0 or not os.path.isfile(spv):
        raise AmuError("Vulkan(SPIR-V) 컴파일 실패 - kernel %s\n%s\n(생성된 GLSL: %s)" % (kname, r.stdout.strip(), comp))
    with open(spv, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0 or data[:4] != b"\x03\x02\x23\x07":
        raise AmuError("SPIR-V 출력이 손상됨 - kernel %s" % kname)
    return data


def check_opencl_with_clang(cl_text, kname, workdir):
    clang = shutil.which("clang")
    if not clang:
        raise AmuError("--check-opencl 지정됐지만 clang이 없음")
    path = os.path.join(workdir, kname + ".cl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(cl_text)
    r = subprocess.run([clang, "-cl-std=CL1.2", "-fsyntax-only", "-Xclang", "-finclude-default-header", path],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if r.returncode != 0:
        raise AmuError("OpenCL C 검사 실패 - kernel %s\n%s" % (kname, r.stdout.strip()))


# --------------------------------------------------------------------------------------
# Header generation
# --------------------------------------------------------------------------------------

_CTYPE = {"float": "float", "int": "int32_t", "uint": "uint32_t"}
_TENUM = {"float": "AMU_T_F32", "int": "AMU_T_I32", "uint": "AMU_T_U32"}


def c_string(text):
    lines = text.split("\n")
    out = []
    for ln in lines:
        esc = ln.replace("\\", "\\\\").replace('"', '\\"')
        out.append('    "%s\\n"' % esc)
    return "\n".join(out)


def gen_header(src_name, results):
    guard = "AMUDACC_" + re.sub(r"\W", "_", src_name).upper() + "_H"
    L = ["/* Generated by amudacc %s from %s - DO NOT EDIT */" % (VERSION, src_name),
         "#ifndef %s" % guard, "#define %s" % guard, "",
         '#include "amuda_rt.h"', "",
         "#if !defined(AMU_IMAGE_ABI) || AMU_IMAGE_ABI != %du" % IMAGE_ABI,
         '#error "amudacc %s output requires amuda_rt image ABI %d"' % (VERSION, IMAGE_ABI),
         "#endif", ""]
    for r in results:
        k = r["kernel"]
        n = k.name
        if r["spirv"] is not None:
            words = [int.from_bytes(r["spirv"][i:i + 4], "little") for i in range(0, len(r["spirv"]), 4)]
            L.append("static const uint32_t amu_spirv_%s[%d] = {" % (n, len(words)))
            for i in range(0, len(words), 8):
                L.append("    " + ", ".join("0x%08xu" % w for w in words[i:i + 8]) + ",")
            L.append("};")
        if r["opencl"] is not None:
            L.append("static const char amu_opencl_%s[] =" % n)
            L.append(c_string(r["opencl"]) + ";")
        L.append("static const amuParamDesc amu_params_%s[%d] = {" % (n, len(k.params)))
        for p in k.params:
            L.append('    { "%s", %s, %s, %du, %du },' % (
                p.name, "AMU_ARG_BUFFER" if p.is_buffer else "AMU_ARG_SCALAR", _TENUM[p.base],
                1 if p.readonly else 0, p.slot))
        L.append("};")
        nbuf = sum(1 for p in k.params if p.is_buffer)
        pcs = 4 * sum(1 for p in k.params if not p.is_buffer)
        L.append("static const amuKernelImage amu_image_%s = {" % n)
        L.append("    %du, \"%s\", \"%s\"," % (IMAGE_ABI, n, src_name))
        if r["spirv"] is not None:
            L.append("    amu_spirv_%s, sizeof(amu_spirv_%s), \"%s\"," % (n, n, r["sha256"]))
        else:
            L.append("    NULL, 0, NULL,")
        L.append("    %s," % (("amu_opencl_%s" % n) if r["opencl"] is not None else "NULL"))
        L.append("    %du, amu_params_%s, %du, %du, %du" % (len(k.params), n, nbuf, pcs, k.shared_bytes))
        L.append("};")
        sig = ", ".join(("amuBuffer* %s" % p.name) if p.is_buffer else ("%s %s" % (_CTYPE[p.base], p.name))
                        for p in k.params)
        L.append("static inline amuStatus amu_load_%s(amuContext* amu_ctx, amuFunction** amu_out_fn) {" % n)
        L.append("    return amuModuleLoad(amu_ctx, &amu_image_%s, amu_out_fn);" % n)
        L.append("}")
        L.append("static inline amuStatus amu_launch_%s(amuFunction* amu_fn, amuDim3 amu_grid, amuDim3 amu_block, %s) {" % (n, sig))
        L.append("    void* amu_args[%d] = { %s };" % (len(k.params), ", ".join("(void*)&" + p.name for p in k.params)))
        L.append("    return amuLaunchKernel(amu_fn, amu_grid, amu_block, amu_args, %du);" % len(k.params))
        L.append("}")
        L.append("")
    L.append("#endif /* %s */" % guard)
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def compile_file(path, targets, out_header=None, emit_dir=None, spirv_tool=None, target_env="vulkan1.1",
                 defines=(), check_cl=False):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    comp = Compiler(src, path, list(defines))
    tool = find_spirv_compiler(spirv_tool) if "vulkan" in targets else None
    workdir = emit_dir or tempfile.mkdtemp(prefix="amudacc_")
    os.makedirs(workdir, exist_ok=True)
    results = []
    try:
        for k in comp.kernels:
            r = {"kernel": k, "spirv": None, "sha256": None, "opencl": None, "glsl": None}
            if "vulkan" in targets:
                r["glsl"] = comp.glsl(k)
                r["spirv"] = compile_spirv(r["glsl"], k.name, tool, target_env, workdir)
                r["sha256"] = hashlib.sha256(r["spirv"]).hexdigest()
            if "opencl" in targets:
                r["opencl"] = comp.opencl(k)
                if check_cl:
                    check_opencl_with_clang(r["opencl"], k.name, workdir)
                if emit_dir:
                    with open(os.path.join(workdir, k.name + ".cl"), "w", encoding="utf-8") as f:
                        f.write(r["opencl"])
            if emit_dir:
                meta = {"kernel": k.name, "source": comp.fname, "abi": IMAGE_ABI,
                        "spirv_sha256": r["sha256"], "shared_bytes": k.shared_bytes,
                        "params": [{"name": p.name, "kind": "buffer" if p.is_buffer else "scalar",
                                    "type": p.base, "readonly": p.readonly, "slot": p.slot} for p in k.params]}
                with open(os.path.join(workdir, k.name + ".json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
            results.append(r)
    finally:
        if not emit_dir:
            shutil.rmtree(workdir, ignore_errors=True)
    if out_header:
        with open(out_header, "w", encoding="utf-8") as f:
            f.write(gen_header(comp.fname, results))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(prog="amudacc", description="AMUDA CUDA-like kernel compiler (POC)")
    ap.add_argument("input", nargs="?", help=".acu source file")
    ap.add_argument("-o", "--output", help="output C header (default: <input>.h)")
    ap.add_argument("--targets", default="vulkan,opencl", help="vulkan,opencl (default both)")
    ap.add_argument("--emit-dir", help="also write .comp/.spv/.cl/.json per kernel into this directory")
    ap.add_argument("--spirv-tool", help="path to glslangValidator or glslc")
    ap.add_argument("--target-env", default="vulkan1.1")
    ap.add_argument("-D", dest="defines", action="append", default=[], help="NAME=VALUE macro for kernels")
    ap.add_argument("--check-opencl", action="store_true", help="syntax-check OpenCL C output with local clang")
    ap.add_argument("--print-glsl", action="store_true", help="print generated GLSL and exit (no SPIR-V)")
    ap.add_argument("--print-opencl", action="store_true", help="print generated OpenCL C and exit")
    ap.add_argument("--version", action="store_true")
    a = ap.parse_args(argv)
    if a.version:
        print("amudacc %s (image ABI %d)" % (VERSION, IMAGE_ABI))
        return 0
    if not a.input:
        ap.error("input file required")
    defines = []
    for d in a.defines:
        if "=" in d:
            n, v = d.split("=", 1)
        else:
            n, v = d, "1"
        if not re.match(r"^[A-Za-z_]\w*$", n):
            print("amudacc: error: bad -D name %r" % n, file=sys.stderr)
            return 2
        defines.append((n, v))
    try:
        if a.print_glsl or a.print_opencl:
            with open(a.input, "r", encoding="utf-8") as f:
                comp = Compiler(f.read(), a.input, defines)
            for k in comp.kernels:
                print(comp.glsl(k) if a.print_glsl else comp.opencl(k))
            return 0
        targets = [t.strip() for t in a.targets.split(",") if t.strip()]
        for t in targets:
            if t not in ("vulkan", "opencl"):
                raise AmuError("알 수 없는 타깃 '%s' (vulkan, opencl)" % t)
        if not targets:
            raise AmuError("타깃이 비어 있음")
        out = a.output or (a.input + ".h")
        results = compile_file(a.input, targets, out, a.emit_dir, a.spirv_tool, a.target_env, defines,
                               a.check_opencl)
        for r in results:
            k = r["kernel"]
            parts = []
            if r["spirv"] is not None:
                parts.append("spirv %d bytes sha256 %s" % (len(r["spirv"]), r["sha256"][:12]))
            if r["opencl"] is not None:
                parts.append("opencl %d chars" % len(r["opencl"]))
            print("amudacc: %s: kernel %s -> %s" % (os.path.basename(a.input), k.name, ", ".join(parts)))
        print("amudacc: wrote %s" % out)
        return 0
    except AmuError as e:
        print("amudacc: " + str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
