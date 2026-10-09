#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# vision_fa_sweep.sh PREFIX "IDX ..." [ameva_vision_parity args...]: one vision run per forced flash-attention variant
# (AMEVA_FA_VARIANT), FA GPU time per dispatch and parity; summary in ~/abi-lab/PREFIX.out, ~/abi-lab/PREFIX.done at end
P=$1; IDX=$2; shift 2
cd ~/abi-lab
rm -f $P.done
for i in $IDX; do
  AMEVA_PROFILE=2 AMEVA_FA_VARIANT=$i bash vision_parity_run.sh $P-$i "$@"
  echo "== variant $i: $(grep -o "AMEVA_FA_VARIANT=[0-9]*: .*" vis-$P-$i.log | head -1 | sed "s/.*flash attention //")"
  head -1 vis-$P-$i.log
  grep -E "kind=\"FLASH_ATTN_EXT\"" vis-$P-$i.log | sed "s/AMEVA_PROFILE //; s/share=.*per_dispatch_us=/us=/"
  grep -E "^(cpu-ref|cpu-model|ameva) |vs cpu-ref|^caption (D|I)|AMEVA_VISION|^rc=|Abort|FAIL|error" vis-$P-$i.log \
      | sed "s/fa=enabled splits=1 unsupported_ops=0 chunks=1 tokens=64 //"
done > $P.out 2>&1
echo done > $P.done
