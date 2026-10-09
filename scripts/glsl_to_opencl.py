#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Translate an AMEVA compute shader into an OpenCL C kernel (docs/design/opencl-prompt-path.md, stage B).

    python3 scripts/glsl_to_opencl.py [--glslang glslangValidator] [-DNAME[=VALUE] ...] shader.comp > kernel.cl

The kernels of the ggml backend exist once, as GLSL (adapters/llama/ggml-ameva/shaders). The OpenCL backend of
ac_runtime runs the same arithmetic: this script turns the output of the GLSL preprocessor (glslangValidator -E: includes
and macros expanded, one variant per set of defines) into OpenCL C. It handles the dialect those shaders are written
in and stops with an error on anything else; it is not a general GLSL compiler.

  GLSL                                              OpenCL C
  layout(local_size_x = X, ...) in;                 __attribute__((reqd_work_group_size(X, Y, Z))) of the kernel
  layout(std430, binding = k) buffer B { T f[]; };  kernel argument k: __global [const] T* amv_f
  layout(push_constant) uniform P { uint ...; } pc; the next kernel argument: ameva_pc_t amv_pc (a struct by value)
  f.length() of a buffer                            the byte size of the bound range / element size, from the last
                                                    kernel argument, ameva_ranges_t (every kernel with bindings has
                                                    it: AC_KERNEL_SOURCE_BINDING_RANGES, ac_runtime fills it)
  shared T s[N];                                    __local T amv_s[N]; at the top of the kernel
  T g; at global scope (a value per invocation)     a local of the kernel, passed to the functions using it by pointer
  void main()                                       __kernel void ameva_main(...)
  a function that uses one of the above             takes all of them as leading parameters (and so does every caller)
  out / inout parameter (not an array)              pointer parameter; the caller passes the address
  vecN / uvecN / ivecN, their constructors          floatN / uintN / intN, glsl_vecN(...) overloads of the prelude
  v[i] on a vector (component by index)             glsl_vget(v, i) / glsl_vset(&v, i, x)
  float(x), uint(x), int(x)                         casts
  1.0                                               1.0f (an unsuffixed literal is a double in OpenCL C)
  built-in functions and variables                  the prelude (PRELUDE below)

Names: a local or parameter may carry the name of a buffer, of pc or of a shared array (GLSL scoping). The translated
entities therefore get names of their own (amv_<name>), and inside a function that declares such a name itself every
use of it is the function's own.

ac_runtime's contract for OpenCL kernels (include/ac_runtime.h): argument i < binding count is the pointer of binding
i, followed by the push-constant struct and, when the kernel declares it, the struct of bound range sizes; the
dispatch grid counts workgroups of the kernel's local size.
"""
import argparse
import os
import re
import subprocess
import sys

ENTRY_POINT = "ameva_main"

# GLSL semantics the translated kernels rely on. Overloads need clang's overloadable attribute, which the OpenCL C
# compilers of Mali and Adreno (both built on clang) accept; a compiler without it fails the kernel build.
PRELUDE = r"""/* AMEVA: GLSL compute semantics for kernels translated by scripts/glsl_to_opencl.py -- do not edit. */
#define AMEVA_OVL __attribute__((overloadable))
#define gl_WorkGroupID ((uint3)((uint)get_group_id(0), (uint)get_group_id(1), (uint)get_group_id(2)))
#define gl_LocalInvocationID ((uint3)((uint)get_local_id(0), (uint)get_local_id(1), (uint)get_local_id(2)))
#define gl_GlobalInvocationID ((uint3)((uint)get_global_id(0), (uint)get_global_id(1), (uint)get_global_id(2)))
#define gl_NumWorkGroups ((uint3)((uint)get_num_groups(0), (uint)get_num_groups(1), (uint)get_num_groups(2)))
#define glsl_float(x) ((float)(x))
#define glsl_uint(x) ((uint)(x))
#define glsl_int(x) ((int)(x))
#define glsl_bool(x) ((bool)(x))
#define atomicAnd(mem, v) atomic_and(&(mem), (v))
#define atomicOr(mem, v) atomic_or(&(mem), (v))
#define atomicAdd(mem, v) atomic_add(&(mem), (v))
#define atomicMin(mem, v) atomic_min(&(mem), (v))
#define atomicMax(mem, v) atomic_max(&(mem), (v))

float4 AMEVA_OVL glsl_vec4(float a) { return (float4)(a); }
float4 AMEVA_OVL glsl_vec4(uint a) { return (float4)((float)a); }
float4 AMEVA_OVL glsl_vec4(int a) { return (float4)((float)a); }
float4 AMEVA_OVL glsl_vec4(float a, float b, float c, float d) { return (float4)(a, b, c, d); }
float4 AMEVA_OVL glsl_vec4(float4 v) { return v; }
float4 AMEVA_OVL glsl_vec4(uint4 v) { return convert_float4(v); }
float4 AMEVA_OVL glsl_vec4(int4 v) { return convert_float4(v); }
float4 AMEVA_OVL glsl_vec4(float2 a, float2 b) { return (float4)(a, b); }
float4 AMEVA_OVL glsl_vec4(float2 a, float b, float c) { return (float4)(a, b, c); }
float4 AMEVA_OVL glsl_vec4(float a, float b, float2 c) { return (float4)(a, b, c); }
float4 AMEVA_OVL glsl_vec4(float3 a, float b) { return (float4)(a, b); }
float3 AMEVA_OVL glsl_vec3(float a) { return (float3)(a); }
float3 AMEVA_OVL glsl_vec3(float a, float b, float c) { return (float3)(a, b, c); }
float3 AMEVA_OVL glsl_vec3(float3 v) { return v; }
float2 AMEVA_OVL glsl_vec2(float a) { return (float2)(a); }
float2 AMEVA_OVL glsl_vec2(uint a) { return (float2)((float)a); }
float2 AMEVA_OVL glsl_vec2(int a) { return (float2)((float)a); }
float2 AMEVA_OVL glsl_vec2(float a, float b) { return (float2)(a, b); }
float2 AMEVA_OVL glsl_vec2(float2 v) { return v; }
float2 AMEVA_OVL glsl_vec2(uint2 v) { return convert_float2(v); }
float2 AMEVA_OVL glsl_vec2(int2 v) { return convert_float2(v); }
uint4 AMEVA_OVL glsl_uvec4(uint a) { return (uint4)(a); }
uint4 AMEVA_OVL glsl_uvec4(int a) { return (uint4)((uint)a); }
uint4 AMEVA_OVL glsl_uvec4(uint a, uint b, uint c, uint d) { return (uint4)(a, b, c, d); }
uint4 AMEVA_OVL glsl_uvec4(uint4 v) { return v; }
uint4 AMEVA_OVL glsl_uvec4(int4 v) { return as_uint4(v); }
uint4 AMEVA_OVL glsl_uvec4(float4 v) { return convert_uint4(v); }
uint4 AMEVA_OVL glsl_uvec4(uint2 a, uint2 b) { return (uint4)(a, b); }
uint3 AMEVA_OVL glsl_uvec3(uint a) { return (uint3)(a); }
uint3 AMEVA_OVL glsl_uvec3(uint a, uint b, uint c) { return (uint3)(a, b, c); }
uint3 AMEVA_OVL glsl_uvec3(uint3 v) { return v; }
uint2 AMEVA_OVL glsl_uvec2(uint a) { return (uint2)(a); }
uint2 AMEVA_OVL glsl_uvec2(int a) { return (uint2)((uint)a); }
uint2 AMEVA_OVL glsl_uvec2(uint a, uint b) { return (uint2)(a, b); }
uint2 AMEVA_OVL glsl_uvec2(uint2 v) { return v; }
uint2 AMEVA_OVL glsl_uvec2(int2 v) { return as_uint2(v); }
uint2 AMEVA_OVL glsl_uvec2(float2 v) { return convert_uint2(v); }
int4 AMEVA_OVL glsl_ivec4(int a) { return (int4)(a); }
int4 AMEVA_OVL glsl_ivec4(uint a) { return (int4)((int)a); }
int4 AMEVA_OVL glsl_ivec4(int a, int b, int c, int d) { return (int4)(a, b, c, d); }
int4 AMEVA_OVL glsl_ivec4(int4 v) { return v; }
int4 AMEVA_OVL glsl_ivec4(uint4 v) { return as_int4(v); }
int4 AMEVA_OVL glsl_ivec4(float4 v) { return convert_int4(v); }
int4 AMEVA_OVL glsl_ivec4(int2 a, int2 b) { return (int4)(a, b); }
int3 AMEVA_OVL glsl_ivec3(int a) { return (int3)(a); }
int3 AMEVA_OVL glsl_ivec3(int a, int b, int c) { return (int3)(a, b, c); }
int3 AMEVA_OVL glsl_ivec3(int3 v) { return v; }
int2 AMEVA_OVL glsl_ivec2(int a) { return (int2)(a); }
int2 AMEVA_OVL glsl_ivec2(uint a) { return (int2)((int)a); }
int2 AMEVA_OVL glsl_ivec2(int a, int b) { return (int2)(a, b); }
int2 AMEVA_OVL glsl_ivec2(int2 v) { return v; }
int2 AMEVA_OVL glsl_ivec2(uint2 v) { return as_int2(v); }
int2 AMEVA_OVL glsl_ivec2(float2 v) { return convert_int2(v); }

/* v[i]: component i of a vector (OpenCL C has no subscript on vectors) */
float AMEVA_OVL glsl_vget(float2 v, uint i) { return i == 0u ? v.x : v.y; }
float AMEVA_OVL glsl_vget(float4 v, uint i) { return i == 0u ? v.x : (i == 1u ? v.y : (i == 2u ? v.z : v.w)); }
uint AMEVA_OVL glsl_vget(uint2 v, uint i) { return i == 0u ? v.x : v.y; }
uint AMEVA_OVL glsl_vget(uint4 v, uint i) { return i == 0u ? v.x : (i == 1u ? v.y : (i == 2u ? v.z : v.w)); }
int AMEVA_OVL glsl_vget(int2 v, uint i) { return i == 0u ? v.x : v.y; }
int AMEVA_OVL glsl_vget(int4 v, uint i) { return i == 0u ? v.x : (i == 1u ? v.y : (i == 2u ? v.z : v.w)); }
/* v[i] = x: a store of that one component. In shared memory the invocations of a workgroup fill the components of
   one vector between them (GLSL stores a scalar there); "v->x = x" would be compiled as load the vector, replace the
   component, store the vector, and the stores of two invocations would overwrite each other. */
#define AMEVA_VSET(AS, V, S) void AMEVA_OVL glsl_vset(AS V* v, uint i, S x) { ((AS S*)v)[i] = x; }
AMEVA_VSET(, float2, float) AMEVA_VSET(, uint2, uint) AMEVA_VSET(, int2, int)
AMEVA_VSET(, float4, float) AMEVA_VSET(, uint4, uint) AMEVA_VSET(, int4, int)
AMEVA_VSET(__local, float2, float) AMEVA_VSET(__local, uint2, uint) AMEVA_VSET(__local, int2, int)
AMEVA_VSET(__local, float4, float) AMEVA_VSET(__local, uint4, uint) AMEVA_VSET(__local, int4, int)

float AMEVA_OVL glsl_uintBitsToFloat(uint v) { return as_float(v); }
float2 AMEVA_OVL glsl_uintBitsToFloat(uint2 v) { return as_float2(v); }
float4 AMEVA_OVL glsl_uintBitsToFloat(uint4 v) { return as_float4(v); }
uint AMEVA_OVL glsl_floatBitsToUint(float v) { return as_uint(v); }
uint2 AMEVA_OVL glsl_floatBitsToUint(float2 v) { return as_uint2(v); }
uint4 AMEVA_OVL glsl_floatBitsToUint(float4 v) { return as_uint4(v); }

/* bitfieldExtract: bits [offset, offset + bits) of value, zero-extended (uint) or sign-extended (int); bits >= 1 */
uint AMEVA_OVL glsl_bitfieldExtract(uint v, int offset, int bits) {
    return bits >= 32 ? v >> offset : (v >> offset) & ((1u << bits) - 1u);
}
int AMEVA_OVL glsl_bitfieldExtract(int v, int offset, int bits) {
    return (int)((uint)v << (32 - offset - bits)) >> (32 - bits);
}

float AMEVA_OVL glsl_abs(float x) { return fabs(x); }
float2 AMEVA_OVL glsl_abs(float2 x) { return fabs(x); }
float4 AMEVA_OVL glsl_abs(float4 x) { return fabs(x); }
int AMEVA_OVL glsl_abs(int x) { return (int)abs(x); }
int4 AMEVA_OVL glsl_abs(int4 x) { return as_int4(abs(x)); }
float AMEVA_OVL glsl_sign(float x) { return sign(x); }
float4 AMEVA_OVL glsl_sign(float4 x) { return sign(x); }
int AMEVA_OVL glsl_sign(int x) { return (x > 0) - (x < 0); }

/* unpackHalf2x16: the two binary16 halves of a word (low half first) as floats */
float2 glsl_unpackHalf2x16(uint v) {
    const ushort lo = (ushort)(v & 0xFFFFu), hi = (ushort)(v >> 16);
    return (float2)(vload_half(0, (const half*)&lo), vload_half(0, (const half*)&hi));
}

/* dotPacked4x8EXT(int, int): signed 8-bit x signed 8-bit over the four bytes of each word */
#if defined(cl_arm_integer_dot_product_int8)
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_int8 : enable
#define glsl_dot4x8(a, b) arm_dot(as_char4((int)(a)), as_char4((int)(b)))
#elif defined(__opencl_c_integer_dot_product_input_4x8bit_packed)
#define glsl_dot4x8(a, b) dot_4x8packed_ss_int((uint)(a), (uint)(b))
#elif defined(cl_khr_integer_dot_product)
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#define glsl_dot4x8(a, b) dot_4x8packed_ss_int((uint)(a), (uint)(b))
#else
#define glsl_dot4x8(a, b) AMEVA_ERROR_this_device_has_no_OpenCL_integer_dot_product
#endif
"""

TYPE_MAP = {"vec2": "float2", "vec3": "float3", "vec4": "float4", "uvec2": "uint2", "uvec3": "uint3",
            "uvec4": "uint4", "ivec2": "int2", "ivec3": "int3", "ivec4": "int4"}
CTOR_NAMES = list(TYPE_MAP) + ["float", "uint", "int", "bool"]
SCALAR_TYPES = ["float", "uint", "int", "bool"]
VECTOR_TYPES = sorted(set(TYPE_MAP.values()))
VALUE_TYPES = SCALAR_TYPES + VECTOR_TYPES
RETURN_TYPES = ["void"] + VALUE_TYPES
# bytes of one element of a buffer (std430 arrays; three-component vectors are padded there and not translated)
ELEM_BYTES = {"float": 4, "uint": 4, "int": 4, "float2": 8, "uint2": 8, "int2": 8, "float4": 16, "uint4": 16,
              "int4": 16}
PREFIX = "amv_"
# built-ins with another name or signature in OpenCL C: (regex, replacement)
BUILTIN_RENAMES = [
    (r"\buintBitsToFloat\b", "glsl_uintBitsToFloat"), (r"\bfloatBitsToUint\b", "glsl_floatBitsToUint"),
    (r"\bbitfieldExtract\b", "glsl_bitfieldExtract"), (r"\bunpackHalf2x16\b", "glsl_unpackHalf2x16"),
    (r"\bdotPacked4x8EXT\b", "glsl_dot4x8"), (r"\broundEven\b", "rint"), (r"\binversesqrt\b", "rsqrt"),
    (r"\babs\s*\(", "glsl_abs("), (r"\bsign\s*\(", "glsl_sign("),
    (r"\bbarrier\s*\(\s*\)", "barrier(CLK_LOCAL_MEM_FENCE)"),
]
# GLSL built-ins this translator has no OpenCL C form for: their use is an error, not a silent pass-through
UNSUPPORTED = ["floatBitsToInt", "intBitsToFloat", "packHalf2x16", "bitfieldInsert", "findMSB", "findLSB", "bitCount",
               "mix", "fract", "mod", "trunc", "lessThan", "lessThanEqual", "greaterThan", "greaterThanEqual", "equal",
               "notEqual", "step", "smoothstep", "umulExtended", "imulExtended", "uaddCarry", "usubBorrow", "ldexp",
               "frexp", "atomicExchange", "atomicCompSwap", "atomicXor",
               "memoryBarrier", "memoryBarrierShared", "memoryBarrierBuffer", "gl_LocalInvocationIndex", "discard"]
# identifiers that are free in GLSL and reserved in OpenCL C
RESERVED = ["half", "kernel", "global", "local", "private", "constant", "pipe", "restrict", "register", "auto",
            "static", "signed", "unsigned", "char", "short", "long", "double", "union", "enum", "goto", "typedef",
            "extern", "inline", "sizeof", "uchar", "ushort", "ulong", "size_t", "ptrdiff_t", "event_t"]


class TranslateError(Exception):
    pass


def _fail(msg):
    raise TranslateError(msg)


def _word(name):
    """regex of an identifier use: not part of a longer name and not a member or swizzle (.name)"""
    return r"(?<![\w.])%s\b" % re.escape(name)


def _match_close(text, open_pos, open_ch="{", close_ch="}"):
    """index of the bracket that closes the one at open_pos"""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    _fail("unbalanced %s" % open_ch)


def _split_args(text):
    """the comma-separated parts of an argument list, commas inside brackets left alone"""
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip() or parts:
        parts.append(cur)
    return [p.strip() for p in parts]


def _statement_end(text, pos):
    """index of the ';' (or of the ')' of an enclosing for header) that ends the expression starting at pos"""
    depth = 0
    for i in range(pos, len(text)):
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                return i
            depth -= 1
        elif ch == ";" and depth == 0:
            return i
    _fail("assignment without an end")


_DECL = re.compile(r"\b(%s)\s+(\w+)\s*((?:\[[^\]]*\])*)" % "|".join(VALUE_TYPES))
_NEXT_DECLARATOR = re.compile(r"\s*(\w+)\s*((?:\[[^\]]*\])*)")


def _scan_declarations(text):
    """[(position, name, type, array dimensions, end of scope)] of every variable a function body declares. The scope
    of a declaration ends with the block it is in; one in a for header ends with the loop."""
    block_close, opens = {}, []
    for i, ch in enumerate(text):
        if ch == "{":
            opens.append(i)
        elif ch == "}" and opens:
            block_close[opens.pop()] = i

    def scope_end(pos):
        # a declaration inside the parentheses of for (...): the end of the loop statement
        depth = 0
        for i in range(pos - 1, -1, -1):
            ch = text[i]
            if ch in ")]":
                depth += 1
            elif ch in "([":
                if depth == 0:
                    if ch == "(" and re.search(r"\bfor\s*$", text[:i]):
                        close = _match_close(text, i, "(", ")")
                        j = close + 1
                        while j < len(text) and text[j].isspace():
                            j += 1
                        if j < len(text) and text[j] == "{":
                            return block_close[j]
                        return _statement_end(text, j)
                    break
                depth -= 1
            elif ch in "{};" and depth == 0:
                break
        enclosing = [o for o, c in block_close.items() if o < pos < c]
        return block_close[max(enclosing)] if enclosing else len(text)

    decls = []

    def add(pos, name, vtype, arr):
        decls.append((pos, name, vtype, arr.count("["), scope_end(pos)))

    for m in _DECL.finditer(text):
        add(m.start(), m.group(2), m.group(1), m.group(3))
        i, depth = m.end(), 0
        while i < len(text):  # further declarators of the same statement: T a = x, b[2], c;
            ch = text[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif ch == ";" and depth == 0:
                break
            elif ch == "," and depth == 0:
                m2 = _NEXT_DECLARATOR.match(text, i + 1)
                if not m2 or m2.group(1) in VALUE_TYPES or m2.group(1) in ("const", "in", "out", "inout"):
                    break
                add(m.start(), m2.group(1), m.group(1), m2.group(2))
                i = m2.end()
                continue
            i += 1
    return decls


class _Function:
    def __init__(self, ret, name, params_text, body):
        self.ret, self.name, self.body = ret, name, body
        self.params = []        # (type text, name, array suffix, is_pointer)
        self.param_types = {}   # name: (type, array dimensions)
        for p in _split_args(params_text):
            if not p or p == "void":
                continue
            m = re.match(r"^((?:(?:const|in|out|inout)\s+)*)(\w+)\s+(\w+)\s*((?:\[[^\]]*\])*)$", p)
            if not m:
                _fail("function %s: parameter %r not understood" % (name, p))
            quals, ptype, pname, arr = m.group(1).split(), m.group(2), m.group(3), m.group(4)
            writes = "out" in quals or "inout" in quals
            if arr and not writes and "const" not in quals:
                # a GLSL array parameter is a copy; in C it aliases the caller's array
                _fail("function %s: array parameter %s is passed by value in GLSL (declare it const, out or inout)" %
                      (name, pname))
            self.params.append((("const " if "const" in quals else "") + ptype, pname, arr, writes and not arr))
            self.param_types[pname] = (ptype, arr.count("["))
        self.decls = _scan_declarations(body)
        # every name the function declares anywhere: inside the function it is the function's own (see module text)
        self.syms = set(self.param_types) | set(d[1] for d in self.decls)
        self.needs_ctx = False

    def type_at(self, name, pos, global_types):
        """(type, array dimensions) of name as seen at body position pos, or None"""
        best = None
        for dpos, dname, dtype, dims, dend in self.decls:
            if dname == name and dpos <= pos < dend and (best is None or dpos > best[0]):
                best = (dpos, dtype, dims)
        if best:
            return best[1], best[2]
        return self.param_types.get(name) or global_types.get(name)


def _rewrite_calls(text, name, ctx_args, pointer_positions):
    """every call name(args) in text: the context arguments in front, '&' on the arguments at pointer_positions"""
    out, pos = "", 0
    pattern = re.compile(r"\b%s\s*\(" % re.escape(name))
    while True:
        m = pattern.search(text, pos)
        if not m:
            return out + text[pos:]
        close = _match_close(text, m.end() - 1, "(", ")")
        args = _split_args(_rewrite_calls(text[m.end():close], name, ctx_args, pointer_positions))
        for i in pointer_positions:
            if i >= len(args):
                _fail("call of %s with %d arguments, pointer parameter %d" % (name, len(args), i))
            args[i] = "&(" + args[i] + ")"
        out += text[pos:m.start()] + name + "(" + ", ".join(ctx_args + args) + ")"
        pos = close + 1


_SUBSCRIPTED = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\[")
_ASSIGN = re.compile(r"\s*(=(?!=)|\+=|-=|\*=|/=|%=|\|=|&=|\^=|<<=|>>=)")


def _rewrite_vector_subscripts(text, type_at, base=0):
    """name[i] on a vector and name[..][i] on an element of an array of vectors -> glsl_vget / glsl_vset.
    type_at(name, position in the function body) -> (type, array dimensions) or None; text starts at body position
    base"""
    out, pos = "", 0
    while True:
        m = _SUBSCRIPTED.search(text, pos)
        if not m:
            return out + text[pos:]
        sym = type_at(m.group(1), base + m.start())
        cur, subs = m.end() - 1, []
        if sym and sym[0] in VECTOR_TYPES:
            for _ in range(sym[1]):  # the subscripts that select the array element
                if cur >= len(text) or text[cur] != "[":
                    break
                close = _match_close(text, cur, "[", "]")
                subs.append((cur + 1, text[cur + 1:close]))
                cur = close + 1
                while cur < len(text) and text[cur].isspace():
                    cur += 1
        if not sym or sym[0] not in VECTOR_TYPES or len(subs) != sym[1] or cur >= len(text) or text[cur] != "[":
            out += text[pos:m.end()]  # not a component access: an array, or the vector as a whole
            pos = m.end()
            continue
        close = _match_close(text, cur, "[", "]")
        comp = _rewrite_vector_subscripts(text[cur + 1:close], type_at, base + cur + 1)
        lvalue = m.group(1) + "".join("[%s]" % _rewrite_vector_subscripts(s, type_at, base + at) for at, s in subs)
        ma = _ASSIGN.match(text, close + 1)
        if ma:
            end = _statement_end(text, ma.end())
            rhs = _rewrite_vector_subscripts(text[ma.end():end], type_at, base + ma.end()).strip()
            if ma.group(1) != "=":
                rhs = "glsl_vget(%s, %s) %s (%s)" % (lvalue, comp, ma.group(1)[:-1], rhs)
            out += text[pos:m.start()] + "glsl_vset(&(%s), %s, %s)" % (lvalue, comp, rhs)
            pos = end
        else:
            if re.match(r"\s*(\+\+|--)", text[close + 1:]):
                _fail("++ / -- on a vector component is not translated: %s[...]" % m.group(1))
            out += text[pos:m.start()] + "glsl_vget(%s, %s)" % (lvalue, comp)
            pos = close + 1


def translate(src):
    """(OpenCL C source without the prelude, binding count, push-constant bytes, local size, takes binding ranges) of
    a preprocessed AMEVA compute shader"""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    # GLSL precise (no multiply-add contraction of that value): OpenCL C has only the program-wide FP_CONTRACT pragma,
    # which keeps every operation of the kernel rounded on its own
    precise = re.search(r"\bprecise\b", src) is not None
    src = re.sub(r"\bprecise\s+", "", src)
    kept = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("#"):
            if re.match(r"#\s*(version|extension|line)\b", s):
                continue
            _fail("preprocessor line left in the source: %r" % s)
        kept.append(line)
    src = "\n".join(kept)
    src = re.sub(r"\[\[[^\]]*\]\]", "", src)  # [[unroll]] and other control-flow attributes

    m = re.search(r"layout\s*\(\s*local_size_x\s*=\s*(\d+)\s*,\s*local_size_y\s*=\s*(\d+)\s*,\s*local_size_z\s*=\s*"
                  r"(\d+)\s*\)\s*in\s*;", src)
    if not m:
        _fail("no layout(local_size_x = ..., local_size_y = ..., local_size_z = ...) in;")
    local = tuple(int(g) for g in m.groups())
    src = src[:m.start()] + src[m.end():]

    buffers = {}
    def take_buffer(mm):
        binding, access, elem, field = int(mm.group(1)), mm.group(2), mm.group(4), mm.group(5)
        if binding in buffers:
            _fail("binding %d declared twice" % binding)
        elem = TYPE_MAP.get(elem, elem)
        if elem not in ELEM_BYTES:
            _fail("binding %d: element type %s is not translated" % (binding, elem))
        buffers[binding] = (access, elem, field)
        return ""
    src = re.sub(r"layout\s*\(\s*std430\s*,\s*binding\s*=\s*(\d+)\s*\)\s*(readonly|writeonly)?\s*buffer\s+(\w+)\s*\{"
                 r"\s*(\w+)\s+(\w+)\s*\[\s*\]\s*;\s*\}\s*;", take_buffer, src)
    if sorted(buffers) != list(range(len(buffers))):
        _fail("bindings must be numbered 0..n-1, found %s" % sorted(buffers))

    push_members, push_name = None, None
    m = re.search(r"layout\s*\(\s*push_constant\s*\)\s*uniform\s+\w+\s*\{(.*?)\}\s*(\w+)\s*;", src, re.S)
    if m:
        push_members, push_name = [], m.group(2)
        for decl in m.group(1).split(";"):
            decl = decl.strip()
            if not decl:
                continue
            if not decl.startswith("uint "):
                _fail("push-constant member is not uint: %r" % decl)
            push_members += [n.strip() for n in decl[5:].split(",") if n.strip()]
        src = src[:m.start()] + src[m.end():]
    if re.search(r"\b(layout|buffer|uniform)\b", src):
        _fail("a layout / buffer / uniform declaration this translator does not handle: %r" %
              re.search(r"[^\n]*\b(layout|buffer|uniform)\b[^\n]*", src).group(0).strip())

    shared = []
    def take_shared(mm):
        dims = re.findall(r"\[([^\]]*)\]", mm.group(3))
        if len(dims) > 2:
            _fail("shared array %s has more than two dimensions" % mm.group(2))
        shared.append((TYPE_MAP.get(mm.group(1), mm.group(1)), mm.group(2), dims))
        return ""
    src = re.sub(r"(?m)^[ \t]*shared\s+(\w+)\s+(\w+)\s*((?:\[[^\]]*\])+)\s*;", take_shared, src)
    if re.search(r"\bshared\b", src):
        _fail("a shared declaration this translator does not handle")

    for name in UNSUPPORTED:
        # a call (a variable may carry one of these names); the two that are not functions by their name alone
        if re.search(r"\b%s\b%s" % (name, "" if name in ("gl_LocalInvocationIndex", "discard") else r"\s*\("), src):
            _fail("GLSL built-in %s has no translation" % name)
    for word in RESERVED:
        src = re.sub(r"\b%s\b" % word, "glsl_" + word, src)
    src = re.sub(r"\b(%s)\s*\(" % "|".join(CTOR_NAMES), r"glsl_\1(", src)  # constructors and scalar conversions
    for g, c in TYPE_MAP.items():
        src = re.sub(r"\b%s\b" % g, c, src)
    # 1.0 -> 1.0f (not: hexadecimal, already suffixed, part of an identifier)
    src = re.sub(r"(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![\w.])", r"\1f", src)
    for pat, rep in BUILTIN_RENAMES:
        src = re.sub(pat, rep, src)

    # top level: function definitions; everything between them is a global declaration
    fn_head = re.compile(r"(?m)^[ \t]*(%s)\s+(\w+)\s*\(([^)]*)\)\s*\{" % "|".join(RETURN_TYPES))
    functions, pos = [], 0
    while True:
        m = fn_head.search(src, pos)
        if not m:
            break
        between = src[pos:m.start()]
        close = _match_close(src, m.end() - 1)
        functions.append((between, _Function(m.group(1), m.group(2), m.group(3), src[m.end():close])))
        pos = close + 1
    if src[pos:].strip():
        _fail("text after the last function: %r" % src[pos:].strip()[:80])
    names = [f.name for _, f in functions]
    if len(set(names)) != len(names):
        _fail("overloaded functions are not translated: %s" % sorted(n for n in set(names) if names.count(n) > 1))
    if "main" not in names:
        _fail("no void main()")

    # Global declarations between the functions: constants stay at program scope (the constant address space); a
    # variable is a value of one invocation in GLSL, so it becomes a local of the kernel that the functions using it
    # receive by pointer.
    variables = []  # (type, name, initializer or None)
    def fix_globals(text):
        out = []
        for stmt in text.strip().split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            if stmt.startswith("const "):
                out.append("__constant " + stmt + ";")
                continue
            mv = re.match(r"^(\w+)\s+(\w+)\s*(?:=\s*(.+))?$", stmt, re.S)
            if not mv or mv.group(1) not in VALUE_TYPES:
                _fail("global declaration this translator does not handle: %r" % stmt[:80])
            variables.append((mv.group(1), mv.group(2), mv.group(3)))
        return "\n".join(out)
    global_text = [fix_globals(between) for between, _ in functions]

    # The context: everything a GLSL function reaches without it being a parameter. Each entry: the GLSL name, the
    # type for subscripts, how a function declares it as a parameter, and what main and a function pass for it.
    uses_ranges = bool(re.search(r"\.\s*length\s*\(\s*\)", src))
    ctx = []  # dicts: name, sym (type, dims) or None, decl, arg_main, arg_fn, in_fn (the text of a use in a function)
    kernel_params = []
    for b in sorted(buffers):
        access, elem, field = buffers[b]
        decl = "__global %s%s* %s%s" % ("const " if access == "readonly" else "", elem, PREFIX, field)
        kernel_params.append(decl)
        ctx.append(dict(name=field, sym=(elem, 1), decl=decl, binding=b))
    if push_name:
        kernel_params.append("ameva_pc_t %s%s" % (PREFIX, push_name))
        ctx.append(dict(name=push_name, sym=None, decl="const ameva_pc_t %s%s" % (PREFIX, push_name)))
    if buffers:
        # every translated kernel takes the bound range sizes (AC_KERNEL_SOURCE_BINDING_RANGES), used or not: one
        # contract for all of them
        kernel_params.append("ameva_ranges_t ameva_ranges")
    for stype, sname, dims in shared:
        ctx.append(dict(name=sname, sym=(stype, len(dims)),
                        decl=("__local %s* %s%s" % (stype, PREFIX, sname) if len(dims) == 1 else
                              "__local %s (*%s%s)[%s]" % (stype, PREFIX, sname, dims[1]))))
    for vtype, vname, _ in variables:
        ctx.append(dict(name=vname, sym=(vtype, 0), decl="%s* %s%s" % (vtype, PREFIX, vname), variable=True))
    ctx_decl = [e["decl"] for e in ctx] + (["const ameva_ranges_t ameva_ranges"] if uses_ranges else [])
    ctx_args_fn = [PREFIX + e["name"] for e in ctx] + (["ameva_ranges"] if uses_ranges else [])
    ctx_args_main = ([("&" if e.get("variable") else "") + PREFIX + e["name"] for e in ctx] +
                     (["ameva_ranges"] if uses_ranges else []))

    by_name = {f.name: f for _, f in functions}
    for _, f in functions:
        visible = [e for e in ctx if e["name"] not in f.syms]
        f.needs_ctx = f.name != "main" and any(re.search(_word(e["name"]), f.body) for e in visible)
    changed = True
    while changed:  # a caller of a function that needs the context needs it too
        changed = False
        for _, f in functions:
            if f.needs_ctx or f.name == "main":
                continue
            if any(g.needs_ctx and re.search(r"\b%s\s*\(" % re.escape(g.name), f.body) for _, g in functions):
                f.needs_ctx = changed = True

    out = ["#define gl_WorkGroupSize ((uint3)(%du, %du, %du))" % local]
    if push_name:
        out.append("typedef struct { %s } ameva_pc_t;" % " ".join("uint %s;" % n for n in push_members))
    if buffers:
        out.append("typedef struct { uint bytes[%d]; } ameva_ranges_t;" % len(buffers))
    for (between, f), gtext in zip(functions, global_text):
        out.append(gtext)
        is_main = f.name == "main"
        visible = [e for e in ctx if e["name"] not in f.syms]
        global_types = {e["name"]: e["sym"] for e in visible if e["sym"]}
        # first, on the untouched body: the declarations were located in it
        body = _rewrite_vector_subscripts(f.body, lambda n, p, f=f, g=global_types: f.type_at(n, p, g))
        # buffer.length(): elements in the bound range
        for e in visible:
            if "binding" in e:
                body = re.sub(_word(e["name"]) + r"\s*\.\s*length\s*\(\s*\)",
                              "((int)(ameva_ranges.bytes[%d] / %du))" % (e["binding"], ELEM_BYTES[e["sym"][0]]), body)
        if re.search(r"\.\s*length\s*\(\s*\)", body):
            _fail("function %s: length() of something that is not a buffer" % f.name)
        for e in visible:
            target = PREFIX + e["name"]
            if e.get("variable") and not is_main:
                target = "(*%s)" % target
            body = re.sub(_word(e["name"]), target, body)
        for _, pname, _, is_ptr in f.params:
            if is_ptr:
                body = re.sub(_word(pname), "(*%s)" % pname, body)
        for g in by_name.values():
            ptrs = [i for i, p in enumerate(g.params) if p[3]]
            if g.name != "main" and (g.needs_ctx or ptrs):
                body = _rewrite_calls(body, g.name, (ctx_args_main if is_main else ctx_args_fn) if g.needs_ctx else [],
                                      ptrs)
        if is_main:
            if f.ret != "void" or f.params:
                _fail("main must be void main()")
            decls = "".join("\n    __local %s %s%s%s;" % (st, PREFIX, sn, "".join("[%s]" % d for d in dims))
                            for st, sn, dims in shared)
            decls += "".join("\n    %s %s%s%s;" % (vt, PREFIX, vn, " = " + vi if vi else "") for vt, vn, vi in variables)
            out.append("__kernel __attribute__((reqd_work_group_size(%d, %d, %d)))\nvoid %s(%s) {%s%s}" %
                       (local + (ENTRY_POINT, ", ".join(kernel_params), decls, body)))
        else:
            params = ["%s%s %s%s" % (pt, "*" if is_ptr else "", pn, arr) for pt, pn, arr, is_ptr in f.params]
            out.append("%s %s(%s) {%s}" % (f.ret, f.name, ", ".join((ctx_decl if f.needs_ctx else []) + params), body))
    text = "\n".join(o for o in out if o) + "\n"
    text = re.sub(r"\n[ \t]*\n+", "\n", text)
    if precise:
        text = "#pragma OPENCL FP_CONTRACT OFF\n" + text
    return text, len(buffers), 4 * len(push_members or []), local, uses_ranges


def preprocess(glslang, path, defines):
    """the shader with includes and macros expanded for one set of defines (glslangValidator -E)"""
    cmd = [glslang, "-E", "-S", "comp"] + ["-D%s" % d for d in defines] + [path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=os.path.dirname(path) or ".")
    if p.returncode != 0:
        raise TranslateError("%s failed:\n%s%s" % (" ".join(cmd), p.stdout, p.stderr))
    return p.stdout


def main():
    ap = argparse.ArgumentParser(description="AMEVA compute shader -> OpenCL C kernel")
    ap.add_argument("--glslang", default="glslangValidator")
    ap.add_argument("-D", dest="defines", action="append", default=[])
    ap.add_argument("--no-prelude", action="store_true")
    ap.add_argument("shader")
    args = ap.parse_args()
    try:
        text, nbind, npush, local, ranges = translate(preprocess(args.glslang, os.path.abspath(args.shader), args.defines))
    except TranslateError as e:
        sys.stderr.write("glsl_to_opencl: %s: %s\n" % (args.shader, e))
        sys.exit(1)
    sys.stdout.write(("" if args.no_prelude else PRELUDE) +
                     "/* %s %s: %d bindings, %d push-constant bytes, local size %s%s */\n" %
                     (os.path.basename(args.shader), " ".join("-D" + d for d in args.defines), nbind, npush, local,
                      ", takes binding ranges" if ranges else "") + text)


if __name__ == "__main__":
    main()
