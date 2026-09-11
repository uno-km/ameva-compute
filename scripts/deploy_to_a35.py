import base64
import subprocess
import sys

with open("tts_sync.tar.gz", "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("ascii")

print(f"Encoded {len(b64_data)} characters.")

# 1. Clear remote file
subprocess.run(["ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "u0_a30@100.106.251.21", ": > ~/tts_sync.b64"], check=True)

# 2. Send in chunks of 32k
chunk_size = 32768
total_chunks = (len(b64_data) + chunk_size - 1) // chunk_size
for i in range(0, len(b64_data), chunk_size):
    chunk = b64_data[i:i+chunk_size]
    subprocess.run(["ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "u0_a30@100.106.251.21", f"cat << 'EOF' >> ~/tts_sync.b64\n{chunk}\nEOF"], check=True)
    print(f"Sent chunk {i//chunk_size + 1}/{total_chunks}")

# 3. Decode and extract on A35
cmd = "base64 -d ~/tts_sync.b64 > ~/tts_sync.tar.gz && rm -rf ~/ameva-compute-tts && mkdir -p ~/ameva-compute-tts && tar -xzf ~/tts_sync.tar.gz -C ~/ameva-compute-tts && rm -f ~/tts_sync.b64 ~/tts_sync.tar.gz && ls -la ~/ameva-compute-tts"
res = subprocess.run(["ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "u0_a30@100.106.251.21", cmd], capture_output=True, text=True, check=True)
print("DEPLOY_SUCCESS:")
print(res.stdout)
