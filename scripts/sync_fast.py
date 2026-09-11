import base64
import subprocess
import os

files = [
    "adapters/tts/ncnn/ameva_tts_counters.h",
    "adapters/tts/ncnn/ameva_tts_counters.cpp",
    "adapters/tts/ncnn/ameva_leaky_relu.h",
    "adapters/tts/ncnn/ameva_leaky_relu.cpp",
]

ssh_base = ["ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10", "u0_a30@100.106.251.21"]

for rel_path in files:
    with open(rel_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    remote_path = f"/data/data/com.termux/files/home/ameva-compute-tts/{rel_path}"
    remote_dir = os.path.dirname(remote_path)
    
    proc = subprocess.Popen(ssh_base + [f"mkdir -p {remote_dir} && base64 -d > {remote_path}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate(input=b64.encode("ascii"), timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to sync {rel_path}: {stderr.decode()}")
    print(f"Transferred {rel_path} ({len(data)} bytes)")

print("ALL_FILES_TRANSFERRED_SUCCESSFULLY")