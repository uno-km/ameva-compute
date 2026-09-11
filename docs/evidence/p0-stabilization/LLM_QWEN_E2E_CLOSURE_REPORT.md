# Qwen2.5-0.5B AMEVA GPU Backend E2E Inference Closure Report
**Final Audit Verdict**: `PASS_WITH_SCOPE_LIMITS`

---

## 1. Executive Summary & Verification Matrix

| Audit Metric | Verdict / Value | Ground Truth Evidence & Invariant Notes |
| :--- | :---: | :--- |
| **DYNAMIC_AMEVA_PLUGIN_LOAD** | **PASS** | Dynamic library `libggml-ameva.so` successfully discovered and loaded via GGML plugin loader |
| **AMEVA_BACKEND_SELECTED** | **PASS** | `AMEVA0` accelerator selected by llama.cpp scheduler and assigned to supported graph nodes |
| **GGML_CPU_BACKEND_AVAILABLE** | **PASS** | Standard CPU backend available and active for handling non-offloaded model tensors |
| **SEPARATE_GGML_VULKAN_PLUGIN_LOAD** | **NOT_ASSERTED** | Separate `libggml-vulkan.so` registration/usage is not asserted; AMEVA operates its own Vulkan compute pipeline |
| **GENERATED_TOKEN_PROVENANCE** | **SEALED** | Token sequence `[52854, 16032, 151645]` uniquely generated; prompt echo unambiguously excluded |
| **PROMPT_ECHO_EXCLUDED** | **true** | Prompt string `"Reply with exactly one word: Tokyo"` isolated; assistant output contains only `"Tokyo"` |
| **SEMANTIC_COMPLETION** | **PASS** | Single-word directive satisfied and properly terminated on `<|im_end|>` (Token ID: 151645) |
| **BACKEND_PROCESS_TERMINATION_CALL_COUNT** | **0** | Zero calls to `exit`, `_Exit`, `abort`, or `quick_exit` across entire `adapters/llama/ggml-ameva` codebase |
| **BACKEND_ERROR_PROPAGATION** | **PASS** | `GGML_STATUS_FAILED` propagates to `llama_decode` fail (`-3`), yielding CLI process exit code 1 |
| **NORMAL_LIFECYCLE_BALANCED** | **true** | Context 1/1, Kernel 1/1, Buffer 665/665, Event 660/660, Staging 660/660 balanced upon exit |
| **FORCED_FAILURE_LIFECYCLE_BALANCED** | **true** | Context 1/1, Kernel 1/1, Buffer 5/5, Event 0/0, Staging 0/0 balanced upon early failure |
| **SILENT_RUNTIME_FALLBACK_COUNT** | **0** | Zero runtime fallback on operations accepted by AMEVA (`SILENT_FALLBACK_COUNT=0`) |
| **CPU_PREASSIGNED_OR_RETAINED_OPS** | **PRESENT** | Non-supported tensor types (`F32: 121, Q8_0: 13, Q4_K: 12, Q6_K: 12`) retained on CPU by design |
| **FULL_MODEL_GPU_EXECUTION** | **false** | Not a full-model GPU execution; hybrid CPU/GPU topology |
| **HYBRID_CPU_AMEVA_EXECUTION** | **true** | Model execution divided between host CPU and AMEVA Vulkan GPU |
| **REMOTE_PUSH_STATUS** | **HOLD** | Changes isolated to local worktree branch `fix/p0-stabilization`; remote push prohibited |
| **LLM_QWEN_E2E_CLOSURE** | **PASS_WITH_SCOPE_LIMITS** | Closed within strictly defined and verified execution boundaries |

---

## 2. Provenance and Source-Binary Correspondence

* **EVIDENCE_COMMIT**: `e46c0fb580e922a80576240022b1b65bba2c7e5d`
* **TESTED_SOURCE_COMMIT**: `de9183f7f212cb5cb74e4b262810b3fbf6504d3f`
* **BINARY_SOURCE_COMMIT**: `de9183f7f212cb5cb74e4b262810b3fbf6504d3f`
* **SOURCE_BINARY_CORRESPONDENCE**: `ESTABLISHED_FOR_de9183f`
* **LOCAL_SOURCE_MANIFEST_SHA256**: `8bce0957b61f92f5ce4adccb73d26ce8cfe1a938f45e83f2bc679e91469c649f`
* **REMOTE_SOURCE_MANIFEST_SHA256**: `8bce0957b61f92f5ce4adccb73d26ce8cfe1a938f45e83f2bc679e91469c649f`
* **SOURCE_MANIFEST_MATCH**: `true`

> [!NOTE]
> `e46c0fb` is the subsequent commit containing raw test execution logs. Direct source-to-binary correspondence is established specifically for `de9183f`, which was built and executed on Samsung Galaxy S25.

### Binary and Model Hashes
* `LLAMA_CLI_SHA256`: `f56bb5335627c26c0e40eeb65fce9eea2bda52e3cec810a40ac3bd68ad6315cd`
* `LIBGGML_AMEVA_SHA256`: `4b22f017cf6a5eddf0fb367d293f4b24529e702ae425036461d89a21477e40a1`
* `LIBGGML_BASE_SHA256`: `12a07f532c39672461030504b0afc91a4a35c2069ab816a41abf0c7ac04b4a2b`
* `LIBGGML_CPU_SHA256`: `51c2144030963eae66bd742e5656adb64c2a5674ac41fc8c16cad14ede18d9d8`
* `MODEL_SHA256`: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`

---

## 3. Dataflow Path and Execution Scope Clarification

### 3.1 Actual Q5_0 Dataflow Execution Path
The current implementation utilizes a hybrid dequantization and execution path:
$$\text{Q5\_0 Model Weight} \xrightarrow{\text{CPU Dequantization}} \text{F32 Staging Host Buffer} \xrightarrow{\text{Vulkan Dispatch}} \text{AMEVA GPU F32 SGEMM}$$

* **Empirical Verification**:
  * `Q5_0_CPU_DEQUANT_COUNT`: 660
  * `F32_GPU_SGEMM_DISPATCH_COUNT`: 660
  * `Q5_0_CPU_DEQUANT`: `true`
  * `Q5_0_GPU_NATIVE`: `false`
  * `GPU_COMPUTE_PORTION`: `F32_SGEMM_ONLY`
* **Precise Technical Description**:
  Q5_0 입력 660건은 CPU에서 F32로 역양자화된 후, 660건의 F32 SGEMM이 AMEVA Vulkan GPU 셰이더에서 실행되었다. Q5_0 GPU-native dequantization 또는 Q5_0-native MatMul은 본 검증 범위에 포함되지 않는다.

### 3.2 Dispatch Count and Model Tensor Distribution
* **Total Tensors in GGUF**: 291
  * `F32`: 121
  * `Q5_0`: 133 (1 `token_embd.weight` + 132 weights across 24 transformer blocks: 12 blocks $\times$ 5, 12 blocks $\times$ 6)
  * `Q8_0`: 13
  * `Q4_K`: 12
  * `Q6_K`: 12
* **Scheduler Assignment**:
  해당 타입(`F32`, `Q8_0`, `Q4_K`, `Q6_K`)의 MUL_MAT 및 비지원 연산은 GGML 스케줄러에 의해 CPU 백엔드에 유지되었으며, AMEVA가 수락한 Q5_0 MUL_MAT에서의 런타임 silent fallback은 0건이었다.
* **Dispatch Totals**:
  * `PROMPT_PHASE_AMEVA_DISPATCH_COUNT`: 390
  * `GENERATION_PHASE_AMEVA_DISPATCH_COUNT`: 270
  * `TOTAL_AMEVA_DISPATCH_COUNT`: 660 (`LLAMA_SCHEDULER_AMEVA_NODE_COUNT = 660`)

---

## 4. Empirical Evidence: Token Generation & Error Propagation

### 4.1 Contrast Prompt Token Decoding
* **Input Prompt**: `"Reply with exactly one word: Tokyo"`
* **Decoding Log Snippet**:
  ```text
  0.11.984.207 D slot process_toke: id  0 | task 0 | n_decoded = 1, n_remaining = 7, next token: 52854 'Tok'
  0.15.214.112 D slot process_toke: id  0 | task 0 | n_decoded = 2, n_remaining = 6, next token: 16032 'yo'
  0.17.912.912 D slot process_toke: id  0 | task 0 | stopped by EOS
  0.17.912.913 D slot process_toke: id  0 | task 0 | n_decoded = 3, n_remaining = 5, next token: 151645 ''
  0.17.912.981 D Parsed message: {"role":"assistant","content":"Tokyo"}
  ```
* **Metrics**:
  * Prompt Processing: 3,153.68 ms / 36 tokens (11.42 t/s)
  * Token Generation: 5,928.72 ms / 3 tokens (0.51 t/s)
  * Process Return Code: 0

### 4.2 Control Failure Run (`AMEVA_FAIL_STAGE=dispatch`)
* **Behavior**: Backend returns `GGML_STATUS_FAILED` directly without calling `exit(1)`.
* **Decoding Log Snippet**:
  ```text
  ggml-ameva [FAIL_FAST]: AMEVA_FAIL_STAGE=dispatch triggered, returning GGML_STATUS_FAILED
  0.02.554.203 D srv          next: received error result, stopping further processing
  0.02.560.282 E srv  update_slots: decode() failed: Compute error.
  ...
  Error: Compute error.
  ```
* **Process Exit Code**: 1 (`CLI_PROCESS_EXIT_CODE=1`, verified in `s25_qwen_e2e_forced_fail_raw.log`)

---

## 5. Scope Boundary Disclosures

### 5.1 검증된 범위 (Verified In-Scope)
1. Samsung Galaxy S25 환경에서 Qwen2.5-0.5B의 실제 llama.cpp 종단 추론 성공
2. 생성 응답 `"Tokyo"` (`Tok` + `yo`) 및 EOS (`<|im_end|>`) 수렴 확인
3. Q5_0 CPU dequantization 후 AMEVA Vulkan F32 SGEMM 660회 성공
4. AMEVA 수락 연산의 silent runtime fallback 0건
5. 정상 및 강제 실패 경로의 리소스 카운터 균형
6. Backend 내부 프로세스 종료 호출 제거 및 CLI RC=1 오류 전파

### 5.2 검증되지 않았거나 주장하지 않는 범위 (Out-of-Scope / Not Asserted)
1. Q5_0-native GPU dequantization
2. Q5_0-native GPU MatMul
3. 전체 모델의 GPU-only 실행
4. 모든 비지원 연산의 CPU 미사용
5. Evidence commit `e46c0fb`와 실행 바이너리의 직접 correspondence
