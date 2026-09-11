import struct
with open("src/shaders/tts/leaky_relu.spv", "rb") as f:
    data = f.read()

words = struct.unpack(f"<{len(data)//4}I", data)
out = [
    "#ifndef AC_TTS_LEAKY_RELU_SPV_H",
    "#define AC_TTS_LEAKY_RELU_SPV_H",
    "",
    "#include <stdint.h>",
    "#include <stddef.h>",
    "",
    "#define AC_LEAKY_RELU_SPV_SHA256 \"e7cf4d0a0bfc0fa21a56469a59c94cf0ce3a119b37a5ba646d6b4bdcb702afed\"",
    "",
    f"static const uint32_t kLeakyReluSpv[{len(words)}] = {{"
]
for i in range(0, len(words), 6):
    chunk = words[i:i+6]
    out.append("    " + ", ".join(f"0x{w:08x}u" for w in chunk) + ",")
out.append("};")
out.append("")
out.append(f"static const size_t kLeakyReluSpvByteSize = sizeof(kLeakyReluSpv);")
out.append("")
out.append("#endif // AC_TTS_LEAKY_RELU_SPV_H")
out.append("")
with open("src/shaders/tts/leaky_relu_spv.h", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Generated leaky_relu_spv.h successfully!")
