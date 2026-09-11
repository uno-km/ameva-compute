import subprocess
import json
import collections

base_cmd = ["ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "u0_a30@100.106.251.21"]

param_files = ["encoder.ncnn.param", "dp.ncnn.param", "flow.ncnn.param", "decoder.ncnn.param"]
model_dir = "/data/data/com.termux/files/home/ncnn-vits-piper-en_US-lessac-high-fp16-patched"

all_nodes = []
global_index = 0
op_counter = collections.Counter()
subgraph_stats = {}

for pf in param_files:
    res = subprocess.run(base_cmd + [f"cat {model_dir}/{pf}"], capture_output=True, text=True, check=True)
    lines = res.stdout.strip().splitlines()
    if not lines:
        continue
    
    # First line: magic (7767517)
    # Second line: layer_count blob_count
    magic = lines[0].strip()
    header_parts = lines[1].strip().split()
    layer_count = int(header_parts[0])
    blob_count = int(header_parts[1])
    
    subgraph_name = pf.replace(".ncnn.param", "")
    subgraph_nodes = []
    
    for l in lines[2:]:
        parts = l.strip().split()
        if not parts:
            continue
        op_type = parts[0]
        node_name = parts[1]
        input_count = int(parts[2])
        output_count = int(parts[3])
        
        idx = 4
        input_blobs = parts[idx : idx + input_count]
        idx += input_count
        output_blobs = parts[idx : idx + output_count]
        idx += output_count
        params = parts[idx:]
        
        is_ameva = (op_type == "AmevaLeakyReLU")
        current_backend = "AMEVA_VULKAN" if is_ameva else "CPU_NCNN"
        
        node_info = {
            "global_index": global_index,
            "subgraph": subgraph_name,
            "node_name": node_name,
            "op_type": op_type,
            "input_count": input_count,
            "output_count": output_count,
            "input_blobs": input_blobs,
            "output_blobs": output_blobs,
            "current_backend": current_backend,
            "is_ameva": is_ameva
        }
        all_nodes.append(node_info)
        subgraph_nodes.append(node_info)
        op_counter[op_type] += 1
        global_index += 1
        
    subgraph_stats[subgraph_name] = {
        "layer_count": layer_count,
        "blob_count": blob_count,
        "parsed_nodes": len(subgraph_nodes)
    }

total_nodes = len(all_nodes)
ameva_nodes = sum(1 for n in all_nodes if n["is_ameva"])
cpu_nodes = total_nodes - ameva_nodes

print("================================================================================")
print("AMEVA TTS GRAPH DISSECTION REPORT")
print("================================================================================")
print(f"TOTAL_GRAPH_NODE_COUNT: {total_nodes}")
print(f"CURRENT_AMEVA_NODE_COUNT: {ameva_nodes}")
print(f"CURRENT_CPU_NODE_COUNT: {cpu_nodes}")
print(f"SUBGRAPH_BREAKDOWN:")
for sg, st in subgraph_stats.items():
    print(f"  {sg}: {st['parsed_nodes']} nodes (Header says: {st['layer_count']} layers, {st['blob_count']} blobs)")

print("\n--- OP TYPE FREQUENCY ACROSS ENTIRE NEURAL GRAPH ---")
for op, cnt in op_counter.most_common():
    print(f"  {op:<25}: {cnt:>4} nodes ({cnt/total_nodes*100:>5.1f}%)")

with open("tts_graph_dissection.json", "w") as f:
    json.dump({
        "total_nodes": total_nodes,
        "ameva_nodes": ameva_nodes,
        "cpu_nodes": cpu_nodes,
        "subgraph_stats": subgraph_stats,
        "op_distribution": dict(op_counter),
        "nodes": all_nodes
    }, f, indent=2)

print("\nDetailed graph saved to tts_graph_dissection.json")
