# Whisper Tiny STT Operator Inventory & AMEVA Offload Matrix

## 1. Subsystem Breakdown

| Subsystem | Operator | Shape / Tensor Dimensions | Dtype | Contiguity | Call Count (per 7s) | Proposed Backend | AMEVA Support Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Audio Frontend** | WAV Decode | 112,000 samples @ 16kHz | S16_LE | Contiguous | 1 | CPU (miniaudio) | CPU_ONLY_BY_POLICY |
| | STFT / FFT | Window 400, Hop 160 | FP32 | Contiguous | 700 frames | CPU | CPU_ONLY_BY_POLICY |
| | Mel Filterbank | 80 Mel bins x 201 FFT bins | FP32 | Contiguous | 1 | CPU | CPU_ONLY_BY_POLICY |
| | Log Scaling | 80 x 3000 (padded) | FP32 | Contiguous | 1 | CPU | CPU_ONLY_BY_POLICY |
| **Encoder Stems** | Conv1D Layer 1 | [384, 80, 3], stride 1, pad 1 | FP16/FP32 | Contiguous | 1 | CPU / GPU Track 2 | NOT_SUPPORTED |
| | GELU Activation | [384, 3000] | FP32 | Contiguous | 2 | CPU / GPU Track 2 | NOT_SUPPORTED |
| | Conv1D Layer 2 | [384, 384, 3], stride 2, pad 1 | FP16/FP32 | Contiguous | 1 | CPU / GPU Track 2 | NOT_SUPPORTED |
| | Position Embedding | [1500, 384] | FP32 | Contiguous | 1 | CPU | CPU_ONLY_BY_POLICY |
| **Encoder Blocks (x4)** | LayerNorm | [1500, 384], eps=1e-5 | FP32 | Contiguous | 8 (2 per block) | CPU / GPU Track 2 | PARTIAL (RMSNorm exists) |
| | **attn_q Projection** | **[384, 384] @ [1500, 384]** | **FP16/FP32** | **Contiguous** | **4** | **AMEVA Vulkan** | **SUPPORTED_NOT_INTEGRATED** |
| | **attn_k Projection** | **[384, 384] @ [1500, 384]** | **FP16/FP32** | **Contiguous** | **4** | **AMEVA Vulkan** | **SUPPORTED_NOT_INTEGRATED** |
| | **attn_v Projection** | **[384, 384] @ [1500, 384]** | **FP16/FP32** | **Contiguous** | **4** | **AMEVA Vulkan** | **SUPPORTED_NOT_INTEGRATED** |
| | Self-Attention SDPA | 6 heads x [1500, 64] | FP32 | Non-contiguous | 4 | CPU / FlashAttn | NOT_SUPPORTED |
| | **attn_out Projection**| **[384, 384] @ [1500, 384]** | **FP16/FP32** | **Contiguous** | **4** | **AMEVA Vulkan** | **SUPPORTED_NOT_INTEGRATED** |
| | Residual Add | [1500, 384] | FP32 | Contiguous | 8 | CPU | SUPPORTED_NOT_INTEGRATED |
| | **mlp_fc1 (FFN Up)** | **[1536, 384] @ [1500, 384]** | **FP16/FP32** | **Contiguous** | **4** | **AMEVA Vulkan** | **SUPPORTED_NOT_INTEGRATED** |
| | GELU Activation | [1500, 1536] | FP32 | Contiguous | 4 | CPU | NOT_SUPPORTED |
| | **mlp_fc2 (FFN Down)** | **[384, 1536] @ [1500, 1536]**| **FP16/FP32** | **Contiguous** | **4** | **AMEVA Vulkan** | **SUPPORTED_NOT_INTEGRATED** |
| **Encoder Output** | Final LayerNorm | [1500, 384] | FP32 | Contiguous | 1 | CPU | PARTIAL |
| **Decoder Blocks (x4)** | Self-Attention | 6 heads x [N, 64] | FP32 | KV-Cache | N steps | CPU | NOT_SUPPORTED |
| | Cross-Attention | [N, 384] x [1500, 384] | FP32 | KV-Cache | N steps | CPU | NOT_SUPPORTED |
| | Decoder MatMuls | [384, 384], [1536, 384] | FP16/FP32 | Contiguous | 6 x N steps | CPU / GPU | SUPPORTED_NOT_INTEGRATED |
| **Logits Head** | Output Projection | [51865, 384] @ [1, 384] | FP16/FP32 | Contiguous | N steps | CPU / GPU | SUPPORTED_NOT_INTEGRATED |

## 2. Wave 1 Offload Candidate Selection

1. **Target Operator**: `encoder.attn_q` / `encoder.attn_k` / `encoder.attn_v` Projection MatMul
2. **Matrix Dimensions**: $M = 1500, K = 384, N = 384$
3. **Arithmetic Type**: SGEMM (FP32 accumulation, FP16 or FP32 weights)
4. **Rationale**:
   - High time contribution (encoder accounts for 4,375 ms out of 7,805 ms total on A53).
   - Clean 2D contiguous dense row-major buffer layout with zero stride irregularities.
   - Matches existing AMUDA / AMEVA Vulkan SGEMM dispatch pipeline without requiring specialized graph refactoring.
