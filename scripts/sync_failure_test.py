import base64
import subprocess
import os

files = [
    "tests/tts/test_tts_failure_injection.cpp",
    "CMakeLists.txt"
]

ssh_base = ["ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "u0_a30@100.106.251.21"]

for rel_path in files:
    with open(rel_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    remote_path = f"/data/data/com.termux/files/home/ameva-compute-tts/{rel_path}"
    remote_dir = os.path.dirname(remote_path)
    b64_file = remote_path + ".b64"

    subprocess.run(ssh_base + [f"mkdir -p {remote_dir} && : > {b64_file}"], check=True)

    chunk_sz = 2000
    for i in range(0, len(b64), chunk_sz):
        chunk = b64[i:i+chunk_sz]
        subprocess.run(ssh_base + [f"printf '%s' '{chunk}' >> {b64_file}"], check=True)

    subprocess.run(ssh_base + [f"base64 -d {b64_file} > {remote_path} && rm -f {b64_file}"], check=True)
    print(f"Transferred {rel_path} ({len(data)} bytes)")

print("FAILURE_TEST_TRANSFERRED")
