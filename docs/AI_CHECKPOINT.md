# Current Engineering Checkpoint

**Timestamp**: 2026-10-09T18:05:00+09:00  
**Branch**: `perf/opencl-mali`  
**Current Goal**: Stage 3, BitNet b1.58 native integration. (Stage 1 Diffusion & Stage 2 STT CLOSED).  

### Verified (Ground Truth)
- **Stage 1 (Diffusion)**: CLOSED. SD 1.x, SDXL Q4_0, SDXS, Z-Image-Turbo, LoRA runtime & baking verified (67 artifacts).
- **Stage 2 (STT - Whisper large-v3-turbo q5_0 on AMEVA_STRICT=1)**: 100% CLOSED across 5-device fleet:
  - **S25** (Adreno 830): `-fa` total 25.25s (encode 9.85s/run), `-nfa` total 35.94s (encode 15.32s).
  - **S21** (Mali-G78): `-fa` total 87.11s (encode 39.20s/run).
  - **A35** (Mali-G68 MP5): `-fa` total 106.21s (encode 42.49s/run). OpenCL ops verified: NORM 20/20, GET_ROWS 23/23, FLASH_ATTN_EXT 1048/1048, IM2COL 88/88, CPY/CONT/ADD/MUL/GELU/SCALE/MUL_MAT 100% PASS.
  - **S20** (Adreno 650): `-fa` total 171.16s (encode 75.77s/run). Untruncated full transcript verified.
  - **A53** (Mali-G68 MP4): `-fa` total 255.63s (encode 111.29s/run). Untruncated full transcript verified. Currently in power-save doze.
- **Transcript Integrity**: All 5 devices produced identical, untruncated full JFK 1-min transcripts matching CPU baseline.
- **Kernels**: `GET_ROWS` q4_0/q5_0 layout indexing bug resolved (23/23 PASS), `mm_f16a` multi-head attention verified.

### Next Stage: Stage 3 (BitNet 1.58-bit)
1. Commit and push Stage 2 STT codebase (`perf/opencl-mali`).
2. Replace legacy Vulkan shader dispatch in `termux-bitnet` with native AMUDA `ac_runtime`.
3. Wire `src/shaders/bitnet_gemv_i2_s.comp` (1.58-bit ternary GEMV) into `adapters/bitnet/`.
4. Implement Zero-Overhead Token Chain (`RMSNorm` -> `RoPE` -> `Attention_Decode` -> `SwiGLU` -> `GEMV`) via AMUDA Batch API.
5. Deploy and benchmark across S25, S21, A35, S20 (and A53 upon wake).
