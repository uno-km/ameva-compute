#!/usr/bin/env python3
"""
NCNN Model Parser & Weight Extractor for SqueezeNet v1.1
Calculates exact offsets, dtypes, and SHA-256 for target 1x1 Conv layers.
"""

import sys
import struct
import hashlib
from pathlib import Path

def parse_ncnn(param_path: str, bin_path: str):
    with open(param_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    magic = lines[0]
    counts = lines[1].split()
    layer_count = int(counts[0])
    blob_count = int(counts[1])

    print(f"NCNN Magic: {magic}")
    print(f"Layers: {layer_count}, Blobs: {blob_count}")

    with open(bin_path, 'rb') as f:
        bin_data = f.read()

    file_size = len(bin_data)
    print(f"Binary file size: {file_size} bytes")

    offset = 0
    target_info = None

    # Parse layers
    for idx, line in enumerate(lines[2:2 + layer_count]):
        parts = line.split()
        ltype = parts[0]
        lname = parts[1]
        in_count = int(parts[2])
        out_count = int(parts[3])
        
        # params follow inputs and outputs
        param_dict = {}
        for p in parts[4 + in_count + out_count:]:
            if '=' in p:
                k, v = p.split('=', 1)
                param_dict[int(k)] = v

        if ltype == 'Convolution':
            num_output = int(param_dict.get(0, 0))
            kernel_w = int(param_dict.get(1, 0))
            kernel_h = int(param_dict.get(11, kernel_w))
            dilation = int(param_dict.get(2, 1))
            stride = int(param_dict.get(3, 1))
            bias_term = int(param_dict.get(5, 0))
            weight_data_size = int(param_dict.get(6, 0))

            # Check flag tag in NCNN bin
            # NCNN format:
            # tag (int32) if weight_data_size != 0
            # 0x00000000 = float32 raw
            # 0x01306B47 = float16
            # or direct float32 if old format
            flag_val = struct.unpack('<I', bin_data[offset:offset+4])[0]
            
            has_tag = False
            w_dtype = 'float32'
            tag_size = 0
            if flag_val == 0x00000000:
                has_tag = True
                tag_size = 4
                w_dtype = 'float32'
            elif flag_val == 0x01306B47:
                has_tag = True
                tag_size = 4
                w_dtype = 'float16'
            elif flag_val == 0x0002C056:
                has_tag = True
                tag_size = 4
                w_dtype = 'int8'
            else:
                # Direct float32 raw (no tag)
                has_tag = False
                tag_size = 0
                w_dtype = 'float32'

            w_elem_size = 2 if w_dtype == 'float16' else (1 if w_dtype == 'int8' else 4)
            weight_offset = offset + tag_size
            weight_bytes = weight_data_size * w_elem_size
            offset = weight_offset + weight_bytes

            # Bias
            bias_offset = 0
            bias_bytes = 0
            if bias_term == 1:
                bias_offset = offset
                bias_bytes = num_output * 4 # bias is always float32 in NCNN
                offset += bias_bytes

            print(f"[{idx}] Conv '{lname}': K=[{kernel_w},{kernel_h}], S={stride}, D={dilation}, Cout={num_output}, W_size={weight_data_size}, W_dtype={w_dtype}, W_off={weight_offset}, B_off={bias_offset}")

            if kernel_w == 1 and kernel_h == 1 and stride == 1 and dilation == 1 and target_info is None:
                target_info = {
                    'index': idx,
                    'name': lname,
                    'type': ltype,
                    'kernel': [kernel_w, kernel_h],
                    'stride': [stride, stride],
                    'dilation': [dilation, dilation],
                    'group': 1,
                    'output_channels': num_output,
                    'input_channels': weight_data_size // num_output,
                    'weight_elements': weight_data_size,
                    'bias_elements': num_output if bias_term == 1 else 0,
                    'weight_offset': weight_offset,
                    'weight_bytes': weight_bytes,
                    'weight_dtype': w_dtype,
                    'bias_offset': bias_offset,
                    'bias_bytes': bias_bytes,
                    'next_layer_offset': offset,
                    'weight_tag': hex(flag_val) if has_tag else 'NONE'
                }

        elif ltype == 'InnerProduct':
            num_output = int(param_dict.get(0, 0))
            bias_term = int(param_dict.get(1, 0))
            weight_data_size = int(param_dict.get(2, 0))
            flag_val = struct.unpack('<I', bin_data[offset:offset+4])[0]
            tag_size = 4 if flag_val in (0, 0x01306B47, 0x0002C056) else 0
            w_elem_size = 2 if flag_val == 0x01306B47 else (1 if flag_val == 0x0002C056 else 4)
            offset += tag_size + weight_data_size * w_elem_size
            if bias_term == 1:
                offset += num_output * 4

        elif ltype == 'BatchNorm':
            channels = int(param_dict.get(0, 0))
            offset += channels * 4 * 4 # slope, mean, var, bias

        elif ltype == 'Scale':
            scale_data_size = int(param_dict.get(0, 0))
            bias_term = int(param_dict.get(1, 0))
            offset += scale_data_size * 4
            if bias_term == 1:
                offset += scale_data_size * 4

    print(f"\nFinal parsed offset: {offset} / {file_size} bytes (Match: {offset == file_size})")

    if target_info:
        print("\n--- Target 1x1 Conv Layer Selected ---")
        for k, v in target_info.items():
            print(f"  {k}: {v}")

        w_slice = bin_data[target_info['weight_offset']:target_info['weight_offset'] + target_info['weight_bytes']]
        w_sha = hashlib.sha256(w_slice).hexdigest()
        target_info['weight_sha256'] = w_sha
        print(f"  WEIGHT_SHA256: {w_sha}")

        if target_info['bias_elements'] > 0:
            b_slice = bin_data[target_info['bias_offset']:target_info['bias_offset'] + target_info['bias_bytes']]
            b_sha = hashlib.sha256(b_slice).hexdigest()
            target_info['bias_sha256'] = b_sha
            print(f"  BIAS_SHA256: {b_sha}")
        else:
            target_info['bias_sha256'] = 'NONE'

        # Write extracted raw weight and bias files for test usage
        out_dir = Path(param_path).parent
        w_out = out_dir / f"{target_info['name'].replace('/', '_')}_weight.bin"
        b_out = out_dir / f"{target_info['name'].replace('/', '_')}_bias.bin"
        with open(w_out, 'wb') as f:
            f.write(w_slice)
        with open(b_out, 'wb') as f:
            f.write(b_slice)
        print(f"Extracted weight to: {w_out} ({len(w_slice)} bytes)")
        print(f"Extracted bias to: {b_out} ({len(b_slice)} bytes)")

    return target_info

if __name__ == '__main__':
    p = '/data/data/com.termux/files/home/ncnn/examples/squeezenet_v1.1.param'
    b = '/data/data/com.termux/files/home/ncnn/examples/squeezenet_v1.1.bin'
    parse_ncnn(p, b)
