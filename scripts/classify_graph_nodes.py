# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import json
from collections import Counter

with open(r'docs/evidence/vision/vision_graph_nodes.json', 'r') as f:
    nodes = json.load(f)

print("Total nodes:", len(nodes))
# 273 compute nodes:
# LayerNorm: 55
# Linear: 109
# GELU: 27
# ScaledDotProductAttention: 27
# Add: 55
# Sum = 273

# Remaining 340 nodes breakdown:
# getitem: 166
# reshape: 56
# getattr: 28
# permute: 28
# unbind: 27
# transpose: 27
# mul: 3
# floordiv: 2
# placeholder: 1
# get_attr: 1
# output: 1

# Let's inspect where reshape, permute, unbind, transpose are used:
print("\n--- RESHAPE / PERMUTE / TRANSPOSE / UNBIND NODES ---")
for n in nodes:
    target = str(n['target'])
    op = n['op']
    name = n['name']
    if any(k in target or k in op for k in ['reshape', 'permute', 'unbind', 'transpose']):
        print(f"ID {n['id']}: name={name}, op={op}, target={target}, inputs={n['input_count']}")

