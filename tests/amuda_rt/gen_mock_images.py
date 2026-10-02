#!/usr/bin/env python3
"""Builds a test-only header: real amudacc metadata + a placeholder SPIR-V blob (the mock never executes it)."""
import hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools", "amudacc"))
import amudacc
results = []
for name in ("vector_add", "saxpy"):
    path = os.path.join(HERE, "..", "..", "samples", "cuda_like", name + ".acu")
    comp = amudacc.Compiler(open(path).read(), path, [])
    for k in comp.kernels:
        comp.glsl(k)
        blob = b"\x03\x02\x23\x07" + bytes(60)
        results.append({"kernel": k, "spirv": blob, "sha256": hashlib.sha256(blob).hexdigest(),
                        "opencl": comp.opencl(k), "glsl": None})
open(sys.argv[1], "w").write(amudacc.gen_header("mock_images.acu", results))
