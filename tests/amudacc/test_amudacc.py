#!/usr/bin/env python3
"""Unit tests for amudacc (translator only; SPIR-V compile needs glslangValidator)."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools", "amudacc"))
import amudacc  # noqa: E402
import glsl_lint  # noqa: E402

CLANG = shutil.which("clang")


def glsl(src):
    c = amudacc.Compiler(src, "t.acu", [])
    out = [c.glsl(k) for k in c.kernels]
    for g in out:
        problems = glsl_lint.lint(g)
        assert not problems, problems
    return out


def opencl(src):
    c = amudacc.Compiler(src, "t.acu", [])
    return [c.opencl(k) for k in c.kernels]


def clang_ok(cl_text):
    if not CLANG:
        return True
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "k.cl")
        with open(p, "w") as f:
            f.write(cl_text)
        r = subprocess.run([CLANG, "-cl-std=CL1.2", "-fsyntax-only", "-Xclang", "-finclude-default-header", p],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        if r.returncode != 0:
            print(r.stdout)
        return r.returncode == 0
    finally:
        shutil.rmtree(d)


class Rejects(unittest.TestCase):
    def assertRejected(self, src, needle):
        with self.assertRaises(amudacc.AmuError) as cm:
            c = amudacc.Compiler(src, "t.acu", [])
            for k in c.kernels:
                c.glsl(k)
                c.opencl(k)
        self.assertIn(needle, cm.exception.msg)
        self.assertIsNotNone(cm.exception.line)

    def test_pointer_arith(self):
        self.assertRejected("__global__ void k(float* a){ float* p = a + 1; }", "포인터")

    def test_buffer_passed_to_function(self):
        self.assertRejected("__device__ float f(float v){return v;}\n__global__ void k(float* a){ f(a); }", "a[인덱스]")

    def test_printf(self):
        self.assertRejected('__global__ void k(float* a){ printf("x"); }', "printf")

    def test_double(self):
        self.assertRejected("__global__ void k(float* a){ double d = 1.0; a[0] = d; }", "double")

    def test_dynamic_shared(self):
        self.assertRejected("__global__ void k(float* a){ extern __shared__ float s[]; }", "extern")

    def test_float_atomic_value_used(self):
        self.assertRejected("__global__ void k(float* a, float* b){ float o = atomicAdd(&a[0], 1.0f); b[0] = o; }",
                            "단독 문장")

    def test_shared_float_atomic(self):
        self.assertRejected("__global__ void k(float* a){ __shared__ float s[4]; atomicAdd(&s[0], 1.0f); }",
                            "__shared__ float")

    def test_atomic_on_const(self):
        self.assertRejected("__global__ void k(const int* a, int* b){ atomicAdd(&a[0], 1); }", "const")

    def test_warp(self):
        self.assertRejected("__global__ void k(float* a){ a[0] = __shfl_down_sync(0xffffffffu, a[0], 1); }",
                            "__shfl_down_sync")

    def test_global_var(self):
        self.assertRejected("float g = 1.0f;\n__global__ void k(float* a){ a[0] = g; }", "전역 변수")

    def test_constant_mem(self):
        self.assertRejected("__constant__ float c[4];\n__global__ void k(float* a){ a[0] = c[0]; }", "__constant__")

    def test_long_literal(self):
        self.assertRejected("__global__ void k(int* a){ a[0] = 1L; }", "64비트")

    def test_no_buffer(self):
        self.assertRejected("__global__ void k(int n){ }", "버퍼")

    def test_dim3_bare(self):
        self.assertRejected("__global__ void k(int* a){ a[0] = threadIdx; }", ".x/.y/.z")

    def test_include(self):
        self.assertRejected('#include "x.h"\n__global__ void k(int* a){ }', "#include")


class Translates(unittest.TestCase):
    def test_reserved_names_renamed(self):
        src = "__global__ void k(const float* input, float* output, int length){" \
              " int i = threadIdx.x; if (i < length) output[i] = input[i]; }"
        g = glsl(src)[0]
        self.assertIn("amu_u_input", g)
        self.assertIn("amu_u_output", g)
        self.assertIn("amu_u_length", g)
        self.assertIn("amu_buf_amu_u_input.v[i]", g)
        self.assertNotIn(" input[", g)
        cl = opencl(src)[0]
        self.assertTrue(clang_ok(cl))

    def test_casts(self):
        src = "__global__ void k(float* a, int n){ int i = threadIdx.x;" \
              " a[i] = (float)i / (float)(n - 1) + (float)-i + (float)a[i] * 2.0f;" \
              " unsigned int u = (unsigned int)(a[i]); a[i] += (float)u; }"
        g = glsl(src)[0]
        self.assertIn("float(i)", g)
        self.assertIn("float((n - 1))", g)
        self.assertIn("float(- i)", g)
        self.assertIn("float(amu_buf_a.v[i])", g)
        self.assertIn("uint((amu_buf_a.v[i]))", g)
        self.assertNotIn("(float)", g)
        self.assertTrue(clang_ok(opencl(src)[0]))

    def test_define_with_builtins(self):
        src = "#define GID (blockIdx.x * blockDim.x + threadIdx.x)\n" \
              "__global__ void k(float* a, int n){ int i = GID; if (i < n) a[i] = sqrtf(a[i]); }"
        g = glsl(src)[0]
        self.assertIn("#define GID (int(gl_WorkGroupID.x) * int(gl_WorkGroupSize.x) + int(gl_LocalInvocationID.x))", g)
        self.assertIn("sqrt(amu_buf_a.v[i])", g)
        self.assertTrue(clang_ok(opencl(src)[0]))

    def test_multiple_kernels_and_device_fn(self):
        src = "__device__ float sq(float x){ return x * x; }\n" \
              "__global__ void k1(float* a){ a[threadIdx.x] = sq(a[threadIdx.x]); }\n" \
              "__global__ void k2(int* b, int v){ atomicMax(&b[0], v); }\n"
        gs = glsl(src)
        self.assertEqual(len(gs), 2)
        self.assertIn("float sq(float x)", gs[0])
        self.assertIn("atomicMax(amu_buf_b.v[0], v)", gs[1])
        for cl in opencl(src):
            self.assertTrue(clang_ok(cl))

    def test_float_atomic_cas(self):
        src = "__global__ void k(float* acc, const float* v, int n){" \
              " int i = threadIdx.x; if (i < n) atomicAdd(&acc[0], v[i]); }"
        g = glsl(src)[0]
        # float atomic target is declared once as uint storage: no aliased descriptors
        self.assertIn("buffer AmuB0 { uint v[]; } amu_buf_acc;", g)
        self.assertEqual(g.count("binding = 0)"), 1)
        self.assertIn("atomicCompSwap(amu_buf_acc.v[", g)
        self.assertIn("amu_buf_v.v[i]", g)
        cl = opencl(src)[0]
        self.assertIn("amu_atomic_add_f32_global(&acc[0], v[i])", cl)
        self.assertTrue(clang_ok(cl))

    def test_float_atomic_buffer_must_be_atomic_only(self):
        with self.assertRaises(amudacc.AmuError) as cm:
            glsl("__global__ void k(float* acc){ atomicAdd(&acc[0], 1.0f); acc[1] = 2.0f; }")
        self.assertIn("atomicAdd로만", cm.exception.msg)

    def test_buffers_are_instance_named(self):
        src = "__device__ float f(float x, float y){ return x * y; }\n" \
              "__global__ void k(const float* x, float* y){ int i = threadIdx.x; y[i] = f(x[i], y[i]); }"
        g = glsl(src)[0]
        self.assertIn("} amu_buf_x;", g)
        self.assertIn("amu_buf_y.v[i] = f(amu_buf_x.v[i], amu_buf_y.v[i]);", g)
        self.assertIn("float f(float x, float y)", g)  # device-function params untouched
        self.assertNotRegex(g, r"\{ float x\[\]; \};")

    def test_define_referencing_buffer(self):
        src = "#define AT(i) a[(i)]\n__global__ void k(float* a){ int i = threadIdx.x; a[i] = AT(i) * 2.0f; }"
        g = glsl(src)[0]
        self.assertIn("#define AT(i) amu_buf_a.v[(i)]", g)

    def test_helpers_only_when_used(self):
        g = glsl("__global__ void k(float* a){ a[threadIdx.x] = 1.0f; }")[0]
        self.assertNotIn("amu_roundf", g)
        g2 = glsl("__global__ void k(float* a){ a[threadIdx.x] = roundf(a[threadIdx.x]); }")[0]
        self.assertIn("float amu_roundf(float x)", g2)
        self.assertNotIn("amu_fmodf", g2)

    def test_atomic_sub_and_cas(self):
        src = "__global__ void k(int* a){ atomicSub(&a[0], 2); atomicCAS(&a[1], 0, 5); atomicExch(&a[2], 7); }"
        g = glsl(src)[0]
        self.assertIn("atomicAdd(amu_buf_a.v[0], -(2))", g)
        self.assertIn("atomicCompSwap(amu_buf_a.v[1], 0, 5)", g)
        self.assertIn("atomicExchange(amu_buf_a.v[2], 7)", g)
        self.assertTrue(clang_ok(opencl(src)[0]))

    def test_math_mapping(self):
        src = "__global__ void k(float* a){ int i = threadIdx.x;" \
              " a[i] = fmaxf(expf(a[i]), rsqrtf(fabsf(a[i]) + 1.0f)) + roundf(a[i]) + fmodf(a[i], 3.0f); }"
        g = glsl(src)[0]
        for s in ("max(exp(amu_buf_a.v[i])", "inversesqrt(abs(amu_buf_a.v[i])", "amu_roundf(amu_buf_a.v[i])",
                  "amu_fmodf(amu_buf_a.v[i], 3.0f)"):
            self.assertIn(s, g)
        cl = opencl(src)[0]
        for s in ("fmax(exp(a[i])", "rsqrt(fabs(a[i])", "round(a[i])", "fmod(a[i], 3.0f)"):
            self.assertIn(s, cl)
        self.assertTrue(clang_ok(cl))

    def test_shared_memory_static_cap(self):
        ok = amudacc.Compiler("#define TILE 16\n__global__ void k(float* a){ __shared__ float t[TILE][TILE + 1];"
                              " t[0][0] = a[0]; a[1] = t[0][0]; }", "t.acu", [])
        ok.prepare(ok.kernels[0])
        self.assertEqual(ok.kernels[0].shared_bytes, 16 * 17 * 4)
        with self.assertRaises(amudacc.AmuError) as cm:
            c = amudacc.Compiler("__global__ void k(float* a){ __shared__ float big[4097]; big[0] = a[0]; }", "t.acu", [])
            c.glsl(c.kernels[0])
        self.assertIn("16384", cm.exception.msg)
        with self.assertRaises(amudacc.AmuError) as cm:
            c = amudacc.Compiler("__global__ void k(float* a, int n){ __shared__ float s[N]; s[0] = a[0]; }", "t.acu", [])
            c.glsl(c.kernels[0])
        self.assertIn("정적으로 계산", cm.exception.msg)
        d = amudacc.Compiler("__global__ void k(float* a){ __shared__ float s[N]; s[0] = a[0]; a[0] = s[0]; }",
                             "t.acu", [("N", "64")])
        d.prepare(d.kernels[0])
        self.assertEqual(d.kernels[0].shared_bytes, 256)

    def test_lint_catches_undeclared(self):
        bad = ("#version 450\n#extension GL_GOOGLE_cpp_style_line_directive : enable\n"
               "layout(std430, set = 0, binding = 0) buffer B0 { uint v[]; } amu_buf_a;\n"
               "void main() { float q = float(v[0]); }\n")
        self.assertTrue(any("'v'" in p for p in glsl_lint.lint(bad)))

    def test_samples(self):
        sd = os.path.join(HERE, "..", "..", "samples", "cuda_like")
        for f in sorted(os.listdir(sd)):
            if not f.endswith(".acu"):
                continue
            with open(os.path.join(sd, f)) as fh:
                src = fh.read()
            c = amudacc.Compiler(src, f, [])
            for k in c.kernels:
                self.assertEqual(glsl_lint.lint(c.glsl(k)), [], f)
                self.assertTrue(clang_ok(c.opencl(k)), f)


if __name__ == "__main__":
    unittest.main(verbosity=2)
