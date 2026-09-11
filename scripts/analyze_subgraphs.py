import json
import collections

with open("tts_graph_dissection.json") as f:
    data = json.load(f)

nodes = data["nodes"]

for sg in ["encoder", "dp", "flow", "decoder"]:
    sg_nodes = [n for n in nodes if n["subgraph"] == sg]
    counter = collections.Counter([n["op_type"] for n in sg_nodes])
    print(f"\n=== SUBGRAPH: {sg} (Total nodes: {len(sg_nodes)}) ===")
    for op, cnt in counter.most_common():
        print(f"  {op:<30}: {cnt:>3} ({cnt/len(sg_nodes)*100:>5.1f}%)")
