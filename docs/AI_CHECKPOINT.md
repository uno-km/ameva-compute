# Current Engineering Checkpoint

**Timestamp**: 2026-10-09T21:40:00+09:00
**Branch**: `perf/opencl-mali` (local, 10 commits ahead of origin; not pushed)
**Goal**: Stage 3, BitNet b1.58 on the AMUDA runtime (Stage 1 Diffusion and Stage 2 STT closed).

### Completed
- `51f940b`: 9809842 did not compile from a clean checkout (AMEVA_K_NORM missing); committed the kernel headers the fleet build used (S25 src-stt4, shaders identical by SHA-256).
- `860b80e`: raw-Vulkan `VulkanBitNetEngine` removed; `AcBitNetEngine` (adapters/bitnet/src/core) on ac_runtime, one batch per token, kernels in src/shaders/bitnet_*.comp (GEMV now int8 activations, exact integer sum). Core: missing tensors / unsupported types / non-gpt2 tokenizers refused, full context is an error, byte-level detokenizer fixed, logits / reset API.
- `2b8efe3`: integer-dot (S25, S20) and emulated (Mali) GEMV, work splits and LM-head load width measured on 3 devices; parity gate relative to the CPU float-order floor.

### Verified (bitnet-2b i2_s, 110 teacher-forced positions, each device's CPU reference, default gate)
- S25 Adreno 830: KLD mean 8.76e-4, arg-max 110/110, GPU 36.0 ms/token (CPU path 252.6).
- S21 Mali-G78: 1.87e-3, 109/110 (flip at 0.069-logit lead), 66.2 ms (CPU 127-136).
- S20 Adreno 650: 1.45e-3, 109/110, 103.8 ms (CPU 161-208).
- Floor: CPU `-ffp-contract=fast` vs reference KLD mean 1.34e-3, 108/110. GPU positions 0-1 KLD ~1e-12.
- CLI text on GPU coherent on all three. Details: docs/design/bitnet-amuda-engine.md.

### Remaining
1. A35, A53: offline all session; run `fleet_parity` (CPU save + GPU load) and `final` there.
2. Mali GEMV is ALU-bound (S21 32.6 ms / 498 MB): OpenCL `arm_dot` path for BitNet.
3. LM head 626 MB f16 per token (S25 16.9, S21 22.9, S20 35.9 ms): near bandwidth; smaller head = product decision.
4. Tokenizer is greedy longest match, not BPE merges (prompts may split differently from llama.cpp).
5. Push of perf/opencl-mali awaits the user.

### Commands
- Build (S25): `bash scripts/termux/build_bitnet_adapter.sh --kernels --out DIR`; run with `LD_LIBRARY_PATH=DIR`.
- Parity: `ac_bitnet_parity -m M --save ref.bin --backend cpu -p P -n 96 -c 512`, then `--load ref.bin --backend gpu`.
- Profile: `AC_BITNET_PROFILE=1`; variants `AC_BITNET_GEMV`, `AC_BITNET_GEMV_SHAPE`, `AC_BITNET_LMHEAD_SHAPE`.
- S25 sleeps between ssh polls: `am startservice --user 0 -a com.termux.service_wake_lock com.termux/com.termux.app.TermuxService` (unlock after).

### Known failures
- None open in the verified scope. Device trees: S25 ~/abi-lab/bn6, S21/S20 ~/abi-lab/bn6out (refs in bn2/bn2out).
