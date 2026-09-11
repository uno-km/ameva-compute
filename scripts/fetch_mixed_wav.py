import subprocess
import base64
import hashlib
import os

print("Fetching wave1a_mixed_en1.wav via SSH base64...")
res = subprocess.run(["ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "u0_a30@100.106.251.21", "base64 ~/wave1a_mixed_en1.wav"], capture_output=True, text=True, check=True)

wav_bytes = base64.b64decode(res.stdout)
sha256 = hashlib.sha256(wav_bytes).hexdigest()
print(f"Downloaded {len(wav_bytes)} bytes, SHA-256: {sha256}")

dest_paths = [
    r"C:\Users\GAME\Desktop\tts_wavs\wave1a_mixed_en1.wav",
    r"C:\Users\GAME\Desktop\uno-km\dev\tts_wavs\wave1a_mixed_en1.wav",
    r"C:\Users\GAME\.cache\ameva-audits\tts\wave1a\wave1a_mixed_en1.wav"
]

for p in dest_paths:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(wav_bytes)
    print(f"Saved to: {p}")

print("DOWNLOAD_AND_DELIVERY_SUCCESS")
