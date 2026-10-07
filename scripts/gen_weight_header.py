#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import struct
import hashlib

def make_header(w_path, b_path, out_h):
    with open(w_path, 'rb') as f:
        w_data = f.read()
    w_sha = hashlib.sha256(w_data).hexdigest()
    w_words = struct.unpack('<' + str(len(w_data)//4) + 'I', w_data)

    with open(b_path, 'rb') as f:
        b_data = f.read()
    b_sha = hashlib.sha256(b_data).hexdigest()
    b_words = struct.unpack('<' + str(len(b_data)//4) + 'I', b_data)

    lines = []
    lines.append('/**')
    lines.append(' * @file squeezenet_fire2_squeeze1x1_data.h')
    lines.append(' * @brief Extracted weights and biases for SqueezeNet v1.1 fire2/squeeze1x1 layer.')
    lines.append(' * Preserved as bit-exact uint32 arrays with verified SHA-256.')
    lines.append(' */')
    lines.append('#ifndef SQUEEZENET_FIRE2_SQUEEZE1X1_DATA_H')
    lines.append('#define SQUEEZENET_FIRE2_SQUEEZE1X1_DATA_H\n')
    lines.append('#include <stdint.h>')
    lines.append('#include <stddef.h>\n')
    lines.append(f'#define SQUEEZENET_FIRE2_WEIGHT_SHA256 "{w_sha}"')
    lines.append(f'#define SQUEEZENET_FIRE2_BIAS_SHA256 "{b_sha}"\n')
    lines.append('#define SQUEEZENET_FIRE2_CIN 64u')
    lines.append('#define SQUEEZENET_FIRE2_COUT 16u')
    lines.append('#define SQUEEZENET_FIRE2_WEIGHT_ELEMS 1024u')
    lines.append('#define SQUEEZENET_FIRE2_BIAS_ELEMS 16u\n')

    lines.append('static const union {')
    lines.append('    uint32_t u32[1024];')
    lines.append('    float f32[1024];')
    lines.append('} kSqueezeNetFire2WeightUnion = {')
    lines.append('    {')
    chunk = []
    for i, val in enumerate(w_words):
        chunk.append(f'0x{val:08x}u')
        if len(chunk) == 8 or i == len(w_words) - 1:
            lines.append('        ' + ', '.join(chunk) + ',')
            chunk = []
    lines.append('    }')
    lines.append('};\n')
    lines.append('#define kSqueezeNetFire2Weight (kSqueezeNetFire2WeightUnion.f32)\n')

    lines.append('static const union {')
    lines.append('    uint32_t u32[16];')
    lines.append('    float f32[16];')
    lines.append('} kSqueezeNetFire2BiasUnion = {')
    lines.append('    {')
    chunk = []
    for i, val in enumerate(b_words):
        chunk.append(f'0x{val:08x}u')
        if len(chunk) == 8 or i == len(b_words) - 1:
            lines.append('        ' + ', '.join(chunk) + ',')
            chunk = []
    lines.append('    }')
    lines.append('};\n')
    lines.append('#define kSqueezeNetFire2Bias (kSqueezeNetFire2BiasUnion.f32)\n')

    lines.append('#endif /* SQUEEZENET_FIRE2_SQUEEZE1X1_DATA_H */\n')

    with open(out_h, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'Generated {out_h} (w_sha={w_sha}, b_sha={b_sha})')

if __name__ == '__main__':
    make_header('tests/fixtures/squeezenet_fire2_squeeze1x1_weight.bin',
                'tests/fixtures/squeezenet_fire2_squeeze1x1_bias.bin',
                'tests/fixtures/squeezenet_fire2_squeeze1x1_data.h')