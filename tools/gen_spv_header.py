import hashlib
import sys

spv_path = "adapters/llama/ggml-ameva/ggml_mul_mat_q5_0_vec.spv"
header_path = "adapters/llama/ggml-ameva/ggml_mul_mat_q5_0_vec_spv.h"

with open(spv_path, "rb") as f:
    b = f.read()

sha = hashlib.sha256(b).hexdigest()
words = [int.from_bytes(b[i:i+4], "little") for i in range(0, len(b), 4)]

lines = [
    "#ifndef AC_GGMLMULMATQ50VEC_SPV_H",
    "#define AC_GGMLMULMATQ50VEC_SPV_H",
    "",
    "#include <stdint.h>",
    "#include <stddef.h>",
    "",
    "static const uint32_t kGgmlMulMatQ50VecSpv[] = {",
]

for i in range(0, len(words), 6):
    lines.append("    " + ", ".join(f"0x{w:08x}u" for w in words[i:i+6]) + ",")

lines.extend([
    "};",
    "",
    "static const size_t kGgmlMulMatQ50VecSpvByteSize = sizeof(kGgmlMulMatQ50VecSpv);",
    f'static const char* const kGgmlMulMatQ50VecSpvSha256 = "{sha}";',
    "",
    "#endif /* AC_GGMLMULMATQ50VEC_SPV_H */",
    "",
])

with open(header_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Header generated successfully! Size: {len(b)} bytes, SHA: {sha}")
