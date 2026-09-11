import base64
import subprocess
import sys

s = '''
import shutil
src = '/data/data/com.termux/files/home/ameva-compute-tts/adapters/tts/ncnn/'
dst = '/data/data/com.termux/files/home/sherpa-ncnn/sherpa-ncnn/csrc/'
for f in ['ameva_tts_counters.h', 'ameva_tts_counters.cpp', 'ameva_leaky_relu.h', 'ameva_leaky_relu.cpp']:
    shutil.copy(src + f, dst + f)
p = dst + 'ameva_leaky_relu.cpp'
with open(p) as fp:
    c = fp.read().replace('../../src/shaders/tts/leaky_relu_spv.h', 'leaky_relu_spv.h')
with open(p, 'w') as fp:
    fp.write(c)
print('COPIED_AND_PATCHED_FOR_SHERPA')
'''

b = base64.b64encode(s.encode()).decode()
cmd = (
    f"echo {b} | base64 -d | python3 && "
    "ninja -C ~/sherpa-ncnn/build-vulkan sherpa-ncnn-offline-tts && "
    "rm -f ~/sherpa-ncnn/build-vulkan/bin/sherpa-ncnn-offline-tts-mixed && "
    "cp ~/sherpa-ncnn/build-vulkan/bin/sherpa-ncnn-offline-tts ~/sherpa-ncnn/build-vulkan/bin/sherpa-ncnn-offline-tts-mixed && "
    "sha256sum ~/sherpa-ncnn/build-vulkan/bin/sherpa-ncnn-offline-tts-mixed"
)

res = subprocess.run(
    ["ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "u0_a30@100.106.251.21", cmd],
    capture_output=True,
    text=True
)

print("STDOUT:\n", res.stdout)
print("STDERR:\n", res.stderr)
print("RETURNCODE:", res.returncode)
if res.returncode != 0:
    sys.exit(1)
