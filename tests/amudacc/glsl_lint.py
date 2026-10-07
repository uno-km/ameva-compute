#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""Static symbol check for amudacc-generated GLSL (no glslang needed).

Every identifier in the generated shader must be one of:
  GLSL keyword / type / qualifier, a GLSL builtin variable or function from the whitelist,
  a name declared in the shader (variable, parameter, function, block, block instance, macro,
  macro parameter), or a member/swizzle after '.'.
Anything else is reported. This catches the main failure class of a source-to-source
translator (an identifier that was renamed in one place but not another).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools", "amudacc"))
import amudacc  # noqa: E402

KEYWORDS = set("""
if else for while do return break continue switch case default layout in out inout uniform buffer shared
readonly writeonly restrict coherent volatile std430 std140 set binding push_constant local_size_x_id local_size_y_id
local_size_z_id true false const void float int uint bool
""".split())
BUILTIN_VARS = {"gl_LocalInvocationID", "gl_WorkGroupID", "gl_WorkGroupSize", "gl_NumWorkGroups",
                "gl_GlobalInvocationID", "gl_LocalInvocationIndex"}
BUILTIN_FUNCS = set("""
barrier memoryBarrier memoryBarrierShared memoryBarrierBuffer groupMemoryBarrier atomicAdd atomicMin atomicMax
atomicAnd atomicOr atomicXor atomicExchange atomicCompSwap floatBitsToUint uintBitsToFloat floatBitsToInt
intBitsToFloat sqrt inversesqrt exp exp2 log log2 pow sin cos tan tanh asin acos atan abs min max floor ceil
trunc fma sign clamp isnan isinf mod
""".split())
TYPES = {"float", "int", "uint", "bool", "void"}


def lint(glsl_text):
    problems = []
    declared = set()
    body_lines = []
    for ln in glsl_text.splitlines():
        s = ln.strip()
        if s.startswith("#"):
            m = re.match(r"#\s*define\s+([A-Za-z_]\w*)(\(([^)]*)\))?(.*)$", s)
            if m:
                declared.add(m.group(1))
                params = [p.strip() for p in (m.group(3) or "").split(",") if p.strip()]
                body = m.group(4)
                for tok in re.findall(r"(?<![\w.])[A-Za-z_]\w*", body):
                    if tok in params:
                        continue
                    body_lines.append(("macro:" + m.group(1), tok))
            continue
        body_lines.append((None, ln))
    code = "\n".join(l for k, l in body_lines if k is None)
    # block members are only reachable through the instance name: drop them from the global symbol scan
    code = re.sub(r"\b(buffer|uniform)\s+(\w+)\s*\{[^}]*\}", r"\1 \2 { }", code)
    toks = amudacc.lex(code, "generated.comp")
    # declarations
    for i, t in enumerate(toks):
        if t.kind != "id":
            continue
        prev = toks[i - 1] if i > 0 else None
        if prev is not None and prev.kind == "id" and prev.text in TYPES:
            declared.add(t.text)
        if prev is not None and prev.kind == "id" and prev.text in ("buffer", "uniform"):
            declared.add(t.text)  # block type name
        if prev is not None and prev.kind == "op" and prev.text == "}" and i + 1 < len(toks) and toks[i + 1].text == ";":
            declared.add(t.text)  # block instance name
        if prev is not None and prev.kind == "op" and prev.text == "," :
            # 'int a, b' style declarators: walk back to statement start
            j = i - 1
            depth = 0
            while j >= 0 and not (toks[j].kind == "op" and toks[j].text in (";", "{", "}") and depth == 0):
                if toks[j].text in (")", "]"):
                    depth += 1
                elif toks[j].text in ("(", "["):
                    depth -= 1
                j -= 1
            if j + 1 < len(toks) and toks[j + 1].kind == "id" and toks[j + 1].text in TYPES and depth == 0:
                declared.add(t.text)
    # uses
    for i, t in enumerate(toks):
        if t.kind != "id":
            continue
        prev = toks[i - 1] if i > 0 else None
        if prev is not None and prev.kind == "op" and prev.text == ".":
            continue
        w = t.text
        if w in KEYWORDS or w in BUILTIN_VARS or w in BUILTIN_FUNCS or w in declared:
            continue
        problems.append("line %d: undeclared identifier '%s'" % (t.line, w))
    for owner, tok in body_lines:
        if owner and tok not in KEYWORDS and tok not in BUILTIN_VARS and tok not in BUILTIN_FUNCS \
                and tok not in declared:
            problems.append("%s: undeclared identifier '%s'" % (owner, tok))
    # structural rules that glslang enforces
    if not glsl_text.startswith("#version 450\n"):
        problems.append("#version 450 must be the first line")
    ext = glsl_text.find("#extension GL_GOOGLE_cpp_style_line_directive")
    first_line = glsl_text.find("#line")
    if first_line >= 0 and (ext < 0 or ext > first_line):
        problems.append("#line with a file name requires GL_GOOGLE_cpp_style_line_directive before it")
    bindings = re.findall(r"binding = (\d+)\)", glsl_text)
    if len(bindings) != len(set(bindings)):
        problems.append("duplicate descriptor binding (aliasing) in one shader")
    if glsl_text.count("layout(push_constant)") > 1:
        problems.append("more than one push_constant block")
    return problems


if __name__ == "__main__":
    for path in sys.argv[1:]:
        c = amudacc.Compiler(open(path, encoding="utf-8").read(), path, [])
        for k in c.kernels:
            pr = lint(c.glsl(k))
            print("%s:%s %s" % (os.path.basename(path), k.name, "OK" if not pr else "PROBLEMS"))
            for p in pr:
                print("   ", p)
