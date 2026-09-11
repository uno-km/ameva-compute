#!/usr/bin/env python3
import argparse
import hashlib
import os
import struct

def main():
    parser = argparse.ArgumentParser(description="Embed SPIR-V binary as C header")
    parser.add_argument("--input", required=True, help="Path to .spv binary file")
    parser.add_argument("--output", required=True, help="Path to output .h file")
    parser.add_argument("--var", required=False, default=None, help="Variable name (e.g. Conv3x3Relu)")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = f.read()

    var_name = args.var
    if not var_name:
        base = os.path.basename(args.input)
        if base.endswith(".spv"):
            base = base[:-4]
        var_name = "".join(word.capitalize() for word in base.split("_"))

    sha256 = hashlib.sha256(data).hexdigest()
    words = struct.unpack("<" + str(len(data) // 4) + "I", data)
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

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"Generated {args.output} ({len(data)} bytes, sha256: {sha256})")

if __name__ == "__main__":
    main()
