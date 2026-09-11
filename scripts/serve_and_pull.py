import http.server
import socketserver
import threading
import subprocess
import time
import os

PORT = 8089

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

server = socketserver.TCPServer(("0.0.0.0", PORT), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
print(f"HTTP server running on port {PORT}...")

# Wait 1s
time.sleep(1)

cmd = "curl -s -f http://100.101.241.92:8089/tts_sync.tar.gz -o ~/tts_sync.tar.gz && rm -rf ~/ameva-compute-tts && mkdir -p ~/ameva-compute-tts && tar -xzf ~/tts_sync.tar.gz -C ~/ameva-compute-tts && rm ~/tts_sync.tar.gz && ls -la ~/ameva-compute-tts"

print("Executing curl and extract on A35...")
res = subprocess.run(["ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "u0_a30@100.106.251.21", cmd], capture_output=True, text=True)

print("STDOUT:")
print(res.stdout)
print("STDERR:")
print(res.stderr)
print("CODE:", res.returncode)

server.shutdown()
