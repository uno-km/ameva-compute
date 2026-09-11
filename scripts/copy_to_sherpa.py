import base64, subprocess
s = '''
import shutil
src = '/data/data/com.termux/files/home/ameva-compute-tts/adapters/tts/ncnn/'
dst = '/data/data/com.termux/files/home/sherpa-ncnn/sherpa-ncnn/csrc/'
for f in ['ameva_tts_counters.h', 'ameva_tts_counters.cpp', 'ameva_leaky_relu.h', 'ameva_leaky_relu.cpp']:
    shutil.copy(src + f, dst + f)
p = dst + 'ameva_leaky_relu.cpp'
with open(p) as fp: c = fp.read().replace('../../src/shaders/tts/leaky_relu_spv.h', 'leaky_relu_spv.h')
with open(p, 'w') as fp: fp.write(c)
print('COPIED_AND_PATCHED')
'''
b = base64.b64encode(s.encode()).decode()
subprocess.run(['ssh', '-p', '8022', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no', 'u0_a30@100.106.251.21', f'echo {b} | base64 -d | python3'], check=True)