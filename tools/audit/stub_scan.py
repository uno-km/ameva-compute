#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""Stub / hollow-implementation scanner for C, C++, Python and shell sources.

Flags (heuristics, every hit is reviewed by hand afterwards):
  C/C++  : EMPTY body, VOID_ONLY body ((void)x; only), RETURN_CONST-only body,
           marker words (TODO/FIXME/stub/placeholder/not implemented/dummy/fake/mock/hardcod/simulat)
  Python : function body that is pass / ... / raise NotImplementedError / return <constant>,
           class body that is pass / docstring only, `except ...: pass` (silent swallow), markers
  Shell  : `|| true`, `2>/dev/null ||` (failure masking), `touch` of a binary path
"""
import ast
import json
import os
import re
import sys

MARKERS = re.compile(r"\b(TODO|FIXME|XXX|HACK|STUB|stub|placeholder|not implemented|NOT_IMPLEMENTED|"
                     r"NotImplemented|dummy|DUMMY|fake|FAKE|mock|MOCK|hardcod\w*|simulat\w*)\b")
CONST_RE = re.compile(r"^(-?\d+(\.\d+)?[fFuUlL]*|true|false|NULL|nullptr|AC_SUCCESS|AMU_SUCCESS|"
                      r"GGML_STATUS_SUCCESS|VK_SUCCESS|0x[0-9a-fA-F]+|\"[^\"]*\"|[A-Z][A-Z0-9_]+)$")
CTRL = {"if", "for", "while", "switch", "catch", "sizeof", "return", "defined", "alignof", "decltype",
        "__attribute__", "static_assert", "do", "else"}
SKIP_DIRS = {"third_party", "thirdparty", "build", "build-poc", "node_modules", ".git", "__pycache__",
             "evidence", "dist", "vault_git_archives", ".venv", "venv"}


def strip_c(text):
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                if text[j] == "\\":
                    j += 1
                j += 1
            out.append(c + " " * max(0, j - i - 1) + (c if j < n else ""))
            i = j + 1
        else:
            out.append(c)
            i += 1
    s = "".join(out)
    # drop preprocessor lines (macros with braces confuse the scanner)
    s = re.sub(r"(?m)^[ \t]*#.*(?:\\\n.*)*$", lambda m: re.sub(r"[^\n]", " ", m.group(0)), s)
    return s


def match_brace(s, i):
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    return -1


def scan_c(path, text):
    hits = []
    s = strip_c(text)
    line_of = lambda pos: s.count("\n", 0, pos) + 1
    for m in re.finditer(r"\)\s*((const|noexcept|override|final)\s*)*\{", s):
        brace = m.end() - 1
        # find matching '(' for the ')' at m.start()
        depth = 0
        j = m.start()
        while j >= 0:
            if s[j] == ")":
                depth += 1
            elif s[j] == "(":
                depth -= 1
                if depth == 0:
                    break
            j -= 1
        if j < 0:
            continue
        k = j - 1
        while k >= 0 and s[k].isspace():
            k -= 1
        e = k + 1
        while k >= 0 and (s[k].isalnum() or s[k] in "_:~"):
            k -= 1
        name = s[k + 1:e]
        if not name or name in CTRL or not re.match(r"[A-Za-z_~]", name.split("::")[-1] or "x"):
            continue
        end = match_brace(s, brace)
        if end < 0:
            continue
        body = s[brace + 1:end]
        stmts = [x.strip() for x in re.split(r";", body) if x.strip()]
        nb = re.sub(r"\s+", " ", body).strip()
        kind = None
        if nb == "":
            kind = "EMPTY"
        elif all(re.fullmatch(r"\(\s*void\s*\)\s*[\w\.\->\[\]]+", x) for x in stmts):
            kind = "VOID_ONLY"
        else:
            core = [x for x in stmts if not re.fullmatch(r"\(\s*void\s*\)\s*[\w\.\->\[\]]+", x)]
            if len(core) == 1 and "{" not in nb:
                rm = re.fullmatch(r"return\s*(.*)", core[0])
                if rm is not None:
                    val = rm.group(1).strip().strip("()").strip()
                    if val == "" or CONST_RE.match(val):
                        kind = "RETURN_CONST(%s)" % (val or "void")
        if kind:
            hits.append({"line": line_of(brace), "name": name, "kind": kind, "body": nb[:120]})
    for i, ln in enumerate(text.splitlines(), 1):
        mm = MARKERS.search(ln)
        if mm:
            hits.append({"line": i, "name": "", "kind": "MARKER:" + mm.group(0), "body": ln.strip()[:140]})
    return hits


def scan_py(path, text):
    hits = []
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return [{"line": e.lineno or 0, "name": "", "kind": "SYNTAX_ERROR", "body": str(e)}]

    def strip_doc(body):
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            return body[1:]
        return body

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            b = strip_doc(node.body)
            kind = None
            if not b:
                kind = "DOCSTRING_ONLY"
            elif len(b) == 1:
                st = b[0]
                if isinstance(st, ast.Pass):
                    kind = "PASS_ONLY"
                elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and st.value.value is Ellipsis:
                    kind = "ELLIPSIS_ONLY"
                elif isinstance(st, ast.Raise) and st.exc is not None and "NotImplemented" in ast.dump(st.exc):
                    kind = "RAISE_NOT_IMPLEMENTED"
                elif isinstance(st, ast.Return) and (st.value is None or isinstance(st.value, ast.Constant)):
                    kind = "RETURN_CONST(%r)" % (None if st.value is None else st.value.value)
                elif isinstance(st, ast.Return) and isinstance(st.value, (ast.Dict, ast.List, ast.Tuple)) and \
                        not getattr(st.value, "keys", None) and not getattr(st.value, "elts", None):
                    kind = "RETURN_EMPTY_CONTAINER"
            if kind:
                hits.append({"line": node.lineno, "name": node.name, "kind": kind, "body": ""})
        elif isinstance(node, ast.ClassDef):
            b = strip_doc(node.body)
            if not b or all(isinstance(x, ast.Pass) for x in b):
                bases = ",".join(ast.unparse(x) for x in node.bases) if hasattr(ast, "unparse") else ""
                hits.append({"line": node.lineno, "name": node.name, "kind": "EMPTY_CLASS(%s)" % bases, "body": ""})
        elif isinstance(node, ast.ExceptHandler):
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                t = ast.unparse(node.type) if (node.type is not None and hasattr(ast, "unparse")) else "bare"
                hits.append({"line": node.lineno, "name": "", "kind": "EXCEPT_PASS(%s)" % t, "body": ""})
    for i, ln in enumerate(text.splitlines(), 1):
        mm = MARKERS.search(ln)
        if mm:
            hits.append({"line": i, "name": "", "kind": "MARKER:" + mm.group(0), "body": ln.strip()[:140]})
    return hits


def scan_sh(path, text):
    hits = []
    for i, ln in enumerate(text.splitlines(), 1):
        s = ln.strip()
        if s.startswith("#"):
            continue
        if re.search(r"\|\|\s*true\b", s):
            hits.append({"line": i, "name": "", "kind": "OR_TRUE", "body": s[:140]})
        if re.search(r"2>\s*/dev/null\s*\|\|", s):
            hits.append({"line": i, "name": "", "kind": "STDERR_HIDDEN_FALLBACK", "body": s[:140]})
        if re.search(r"\btouch\b.*\b(bin|lib)/", s):
            hits.append({"line": i, "name": "", "kind": "TOUCH_BINARY", "body": s[:140]})
    return hits


def main():
    roots = sys.argv[1:]
    report = {}
    for root in roots:
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in SKIP_DIRS]
            for f in fn:
                p = os.path.join(dp, f)
                if f.endswith("_spv.h"):
                    continue
                ext = os.path.splitext(f)[1]
                try:
                    text = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                if ext in (".c", ".cc", ".cpp", ".h", ".hpp"):
                    h = scan_c(p, text)
                elif ext == ".py":
                    h = scan_py(p, text)
                elif ext == ".sh":
                    h = scan_sh(p, text)
                else:
                    continue
                if h:
                    report[p] = h
    json.dump(report, sys.stdout, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
