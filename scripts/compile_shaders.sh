#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHADERS_DIR="${ROOT_DIR}/src/shaders"

echo "=== AMUDA Vision Shader Compilation ==="

# 1. Compile normal pointwise_conv.comp
glslangValidator -V "${SHADERS_DIR}/pointwise_conv.comp" -o "${SHADERS_DIR}/pointwise_conv.spv"

# 2. Create mutated shader (output = normal_result + 0.25)
sed 's/C\[row \* dims\.N + col\] = acc;/C[row * dims.N + col] = acc + 0.25;/' \
    "${SHADERS_DIR}/pointwise_conv.comp" > "${SHADERS_DIR}/pointwise_conv_mutated.comp"

glslangValidator -V "${SHADERS_DIR}/pointwise_conv_mutated.comp" -o "${SHADERS_DIR}/pointwise_conv_mutated.spv"

export SHADERS_DIR
python3 - << 'EOF'
import os, struct, hashlib

shaders_dir = os.environ.get('SHADERS_DIR', 'src/shaders')

def convert(spv_path, var_name, out_h):
    with open(spv_path, 'rb') as f:
        data = f.read()
    sha256 = hashlib.sha256(data).hexdigest()
    words = struct.unpack('<' + str(len(data)//4) + 'I', data)
    lines = []
    guard = f"AC_{var_name.upper()}_SPV_H"
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}\n")
    lines.append("#include <stdint.h>")
    lines.append("#include <stddef.h>\n")
    lines.append(f"static const uint32_t k{var_name}Spv[] = {{")
    chunk = []
    for i, w in enumerate(words):
        chunk.append(f"0x{w:08x}u")
        if len(chunk) == 6 or i == len(words) - 1:
            lines.append("    " + ", ".join(chunk) + ",")
            chunk = []
    lines.append("};\n")
    lines.append(f"static const size_t k{var_name}SpvByteSize = sizeof(k{var_name}Spv);")
    lines.append(f"static const char* const k{var_name}SpvSha256 = \"{sha256}\";\n")
    lines.append(f"#endif /* {guard} */\n")
    with open(out_h, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Generated {out_h} ({len(data)} bytes, sha256: {sha256})")

convert(os.path.join(shaders_dir, 'pointwise_conv.spv'), 'PointwiseConv', os.path.join(shaders_dir, 'pointwise_conv_spv.h'))
convert(os.path.join(shaders_dir, 'pointwise_conv_mutated.spv'), 'PointwiseConvMutated', os.path.join(shaders_dir, 'pointwise_conv_mutated_spv.h'))
EOF

echo "Shader compilation and header generation complete!"
