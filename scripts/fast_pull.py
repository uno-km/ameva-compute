import tarfile
import http.server
import socketserver
import threading
import subprocess
import time
import os

tar_name = "adapters_update.tar.gz"
with tarfile.open(tar_name, "w:gz") as tar:
    tar.add("adapters/tts/ncnn", arcname="adapters/tts/ncnn")
print(f"Created {tar_name} ({os.path.getsize(tar_name)} bytes)")

PORT = 8089
class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

socketserver.TCPServer.allow_reuse_address = True
server = socketserver.TCPServer(("0.0.0.0", PORT), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
print("HTTP server running on port 8089...")

time.sleep(1)

cmd = (
    "curl -s -f http://100.101.241.92:8089/adapters_update.tar.gz -o ~/adapters_update.tar.gz && "
    "tar -xzf ~/adapters_update.tar.gz -C ~/ameva-compute-tts/ && "
    "rm -f ~/adapters_update.tar.gz && "
    "ls -la ~/ameva-compute-tts/adapters/tts/ncnn"
)

res = subprocess.run(["ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "u0_a30@100.106.251.21", cmd], capture_output=True, text=True)
print("STDOUT:\n", res.stdout)
print("STDERR:\n", res.stderr)
print("RETURNCODE:", res.returncode)

server.shutdown()
if os.path.exists(tar_name):
    os.remove(tar_name)
print("PULL_AND_EXTRACT_COMPLETED")