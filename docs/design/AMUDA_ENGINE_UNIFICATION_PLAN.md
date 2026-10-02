# AMUDA 엔진 통합 설계·분석·계획서 v0.1

> **상태 (2026-10-01 갱신): 보류.** 사용자 결정에 따라 우선순위를 "엔진 통합"에서 "CUDA처럼 쓸 수 있는 GPU 프로그래밍 POC"로 바꿨다. 진행 문서는 `docs/design/AMUDA_CUDA_LIKE_POC.md`다. 이 문서는 POC가 성립한 뒤 엔진 통합을 다시 검토할 때 쓴다.

- 작성: 2026-10-01 (KST), Claude (Cowork 세션)
- 범위: `ameva-compute`(AMUDA) + termux 7개 엔진(llama.cpp, vision, whisper/STT, diffusion, TTS, BitNet, train)
- 입력: brain `203243d2…` 산출물 6건과 transcript 1,392 step, `dev/ameva`, `dev/termux` 소스

## 근거 표기 규칙

| 표기 | 의미 |
| :--- | :--- |
| **[코드]** | 이번 세션에서 소스나 아티팩트를 직접 열어서 확인함 |
| **[로그]** | 이전 세션 transcript나 빌드 로그 원문에서 확인함 |
| **[보고서]** | 이전 세션 보고서에 있는 수치. 이번에 재현하지 않음 |
| **[추정]** | 계산값이나 경험치. 실측으로 대체하기 전까지는 약속이 아님 |
| **[미검증]** | 확인이 필요한 가설 |

---

## 0. 한 페이지 요약

1. **통합의 주축은 `ggml-ameva` 백엔드 하나다.** 7개 엔진 중 4개(llama.cpp, vision=mtmd, whisper.cpp, stable-diffusion.cpp)가 ggml 위에서 돈다. BitNet(llama.cpp TQ2_0 경유)과 TTS(ggml 기반 구현 경유)까지 끌어올 수 있으면 7개 중 6개를 백엔드 하나로 덮는다. op 하나를 추가하면 여러 엔진이 동시에 득을 본다.
2. **"`libggml-ameva.so` 한 개로 전부"는 불가능하다.** ggml 백엔드 vtable(`ggml-backend-impl.h`)은 ABI가 안정되어 있지 않다. 현재 llama.cpp(build 8471, libggml 0.9.8), sd.cpp(leejet/ggml, 0.19.0), whisper.cpp(커밋 미고정)는 서로 다른 ggml이다. 그래서 **소스는 1개, 빌드는 엔진별 N개, 버전 차이는 shim으로 흡수**하는 구조가 맞다.
3. **현재 프로토타입은 빌드가 성공해도 CPU보다 느릴 공산이 크다 [추정].** `MUL_MAT`(F32, Q5_0)만 지원하고, Q5_0은 CPU에서 역양자화하며, 노드마다 버퍼 생성 → 디스패치 → fence 대기를 반복한다. LoRA 벤치마크에서 GPU를 2.8배 느리게 만든 것과 같은 "작은 단위 + 호출당 오버헤드" 구조다.
4. **커널 25~30종을 백지에서 쓰지 않는다.** ggml-vulkan의 GLSL 셰이더(MIT)를 AMUDA 런타임 위로 이식해 정확도를 먼저 확보한다. AMUDA 고유 작업은 quirk 격벽, Zero-Silent-Fallback 거버넌스, UMA, 모바일 특화 디코드 GEMV에 집중한다.
5. **이길 상대가 이미 있다.** ggml-vulkan(사용자의 Adreno 패치 포함)과 Qualcomm의 ggml-opencl이 존재한다. ggml-ameva가 실측에서 이들을 이기지 못하면 별도 백엔드를 둘 이유가 약해진다. 그래서 Phase 2 끝에 중단/전환 기준을 둔다(§8).
6. **지금 당장 할 일:** 미커밋 Sprint 1~4 자산 커밋 → 침묵 폴백 6건 제거 → llama.cpp `5e6a37cb` 트리 기준으로 ggml-ameva를 CMake 재빌드 → 기준선(CPU / ggml-vulkan / ggml-opencl) 실측.

---

## 1. 경과 정리 (채팅 기록 + brain)

| 시점 (KST) | 단계 | 결과 | 근거 |
| :--- | :--- | :--- | :--- |
| 09-29 | 종합 전수분석 / 세부 개발 계획서 | 계획서는 "AMUDA = ameva-runtime 내부 엔진 전용"으로 방향을 바꾸면서 폐기 | [보고서] |
| 09-30 23:20 | Sprint 1~3 | `libac_runtime.so` 빌드(S20/S21/A35), Python ctypes 바인딩 재작성 35/35(A35), FP16 셰이더 4종, FP16 matmul E2E 4/4 게이트(A35) | [보고서] |
| 10-01 11:06 | Sprint 4 1단계 | FP16 셰이더 10종, `ops.py` 고수준 API | [보고서] |
| 10-01 14:16 | Sprint 4 Stage 1 | 13종 SPIR-V, i2_s GEMV cosine 1.0, `AmudaBackend` 등록, 1024² matmul 1.54x | [보고서] |
| 10-01 ~15:00 | LoRA 학습 벤치 | CPU 1,922 ms vs GPU 5,424 ms(GPU가 2.8배 느림). `GPU_OFFLOAD_THRESHOLD` 4096 → 262144 | [보고서] [코드: amuda_backend.py:56] |
| 10-01 15:11 | 방향 지시 | BitNet 보류, llama.cpp/whisper(ggml-ameva) 우선 | [로그] |
| 10-01 19:42 | A35에서 ggml-ameva 빌드 | **실패**: ggml 헤더 이중 정의 | [로그: task-1386.log] |
| 10-01 19:43 | 중단 | 빌드 스크립트는 고쳤지만 재실행 전에 이전 세션 모델 API가 429(할당량 소진, 약 168시간 후 리셋)와 503으로 멈춤. 그 뒤 "종합 현황" 문서만 생성됨 | [로그: step 1364~1407] |

---

## 2. 정정 사항

### 2.1 이 세션 앞부분(제 브리핑)의 오류

| 잘못 말한 것 | 실제 |
| :--- | :--- |
| `libtermux_bitnet_a53.so` = "ARM Cortex-A53 최적화" | **Galaxy A53 기기용 빌드**. `probe_fleet.py`의 `a53` = Galaxy A53 (Exynos 1280) |
| 플릿에 S22 포함 | 사용자가 지정한 플릿은 S25/S21/S20/A35/A53. `probe_fleet.py`에는 s22, a7이 있고 s21은 "사용자 지시로 제외"라고 되어 있음 → 현재 기준 확정 필요 |
| "CI/CD 자동화 완료" | 확인하지 않은 주장이었음. 이번에도 CI 설정은 감사하지 않음 |
| "포렌식 보고서 170편+" | foundation README의 171편은 `05_infraindex` 기계번역·수집 검증 보고서임 |
| "$150", "Play Store 가능" | 근거 없음. 철회 |
| 기기 연결을 확인하겠다고 하고 안 함 | 이번에 시도함. 결과는 §3.6 |

### 2.2 기존 문서·README와 실물의 불일치 (이번에 발견)

- **termux-llamacpp 배포물에 Vulkan 백엔드가 없다 [코드].** `releases/termux-llamacpp-android-arm64.tar.gz`(build 8471, 커밋 `5e6a37cb`)의 lib에는 `libggml-base`, `libggml-cpu`, `libggml`, `libllama`, `libmtmd`만 있고 `libggml-vulkan.so`는 없다. from-source 경로도 `-DGGML_VULKAN=OFF`다. 따라서 ameva-runtime README의 "Vulkan 25/25 offload, S25 35.8 t/s"는 이 배포물로는 재현되지 않는다. 당시 어떤 바이너리로 측정했는지 추적해야 한다.
- **ggml-ameva 프로토타입이 검증된 llama.cpp는 배포용 llama.cpp가 아니다 [코드].** `CMakeLists.txt`의 `LLAMA_CPP_DIR` 기본값은 `~/BitNet_ms/3rdparty/llama.cpp`(Microsoft BitNet 저장소에 들어 있는 포크)다. 그래서 brain 보고서의 "LLM Backend Adapter Complete, 25/25 레이어"는 termux-llamacpp(`5e6a37cb`) 기준이 아니다.
- **S20 SoC 표기가 서로 다르다.** ameva-runtime README의 TTS 표는 "Exynos 990 / Mali-G77"이라고 하지만, Sprint 1~3 보고서와 `probe_fleet.py`는 SM-G986N, Snapdragon 865 / Adreno 650이라고 한다. SM-G986N이면 Adreno 650이 맞다.
- **"1024² 1.54x"는 같은 정밀도끼리의 비교가 아니다.** GPU는 FP16 입출력(FP32 누산), CPU NumPy는 FP32다. 형변환 비용 포함 여부도 적혀 있지 않다.
- **종합 전수분석 보고서의 수치는 이 계획서의 근거로 쓰지 않는다.** "핵심 엔진 80% 완성 / 잔여 10%", "모바일 AI 시장 $25B", "300만 LOC"가 해당한다. 특히 "잔여 10%"는 같은 세션이 나중에 낸 "커널 25~30개 필요, 현재 4개" 판단과 모순된다.

---

## 3. 현재 상태 (Ground Truth)

### 3.1 엔진별 현황

| 엔진 | 업스트림 / 고정 상태 | 현재 GPU 경로 | ggml | AMUDA 연결 |
| :--- | :--- | :--- | :---: | :--- |
| **LLM** (termux-llamacpp) | llama.cpp `5e6a37cb`, build 8471, libggml 0.9.8 [코드] | 배포 바이너리는 CPU 전용 [코드] | O | ggml-ameva 프로토타입(BitNet 포크 기준), 현재 빌드 실패 |
| **Vision** (termux-vision) | 자체 엔진 없음. `lib/vlm.js`가 `llama-cli` / `llama-mtmd-cli`를 subprocess로 호출 [코드] | llama.cpp를 그대로 따라감 | O | llama.cpp가 통합되면 자동으로 편입됨. 다만 CLIP 계열 op가 추가로 필요 |
| **STT** (termux-stt) | whisper.cpp **HEAD clone, 커밋 미고정** [코드], vosk, sherpa-onnx | `GGML_VULKAN=ON`을 시도하고, 실패하면 `2>/dev/null`로 숨긴 채 CPU 빌드 [코드] | O | `adapters/whisper`(1,624 LOC, MUL_MAT만). 빌드에 필요한 ggml 트리가 저장소에 없음(`third_party/whisper.cpp`는 파일 3개) [코드] |
| **Diffusion** (termux-diffusion) | sd.cpp `50d6405`(2026-08-26), ggml = leejet/ggml `8e800ce`, `GGML_VERSION 0.19.0` [코드] | ggml-vulkan + Adreno 830 패치 7항목 [코드] | O | 없음 |
| **TTS** (termux-tts) | sherpa-ncnn(NCNN Vulkan), MNN Vulkan, onnxruntime [코드] | NCNN/MNN Vulkan | X | ameva-compute에 HiFi-GAN 부분 구현 [보고서] |
| **BitNet** (termux-bitnet) | 자체 C++ 코어 1,555 LOC + `VulkanBitNetEngine` 1,974 LOC, 셰이더 8종 [코드] | 자체 raw Vulkan | X | AMUDA 쪽 대응 셰이더 8/8 작성, 엔진과의 연결 0 [보고서] |
| **Train** (termux-train) | Python NumPy / Vulkan / Amuda 백엔드 [코드] | `AmudaBackend` (Python ctypes) | X | 연결은 됐지만 실사용 크기에서 CPU보다 느림 [보고서] |

### 3.2 ameva-compute 자산

- **C ABI `ac_runtime`**: context / buffer / kernel / dispatch / event와 batch API(`ac_runtime_batch_begin/dispatch/submit/abort/destroy`)가 있다 [코드: `include/ac_runtime.h:393-442`].
- **셰이더 3묶음** [코드]
  - Sprint 4의 13종. FP16 입출력에 연속 메모리를 가정한다.
  - vision(SqueezeNet) 계열: conv1, conv3x3, conv10, pointwise, maxpool, avgpool, concat2, relu.
  - ggml-ameva 전용 3종: `ggml_mul_mat`(F32), `q5_0_dequant`, `ggml_mul_mat_fused_q5_0`.
- **중요한 한계: 13종은 ggml 텐서 규약을 따르지 않는다.** ggml 규약은 F32 activation, `nb[]` stride 기반 비연속 뷰, broadcast(`ggml_can_repeat`), ggml 양자화 블록 포맷이다. 알고리즘은 재사용할 수 있지만 ggml 백엔드에 그대로 꽂을 수는 없다. "커널 13종 준비됨"과 "ggml 백엔드 준비됨"은 다른 말이다.
- **`ops.py`(443 LOC)**: 호출할 때마다 버퍼 3개를 만들고 부수며, FP32 → FP16 변환을 한다 [코드]. 커널 캐싱은 들어가 있다.

### 3.3 ggml-ameva 프로토타입 분석 [코드]

| 항목 | 현재 | 문제 |
| :--- | :--- | :--- |
| `supports_op` | `GGML_OP_MUL_MAT`만. dst/src1은 F32, src0은 F32 또는 Q5_0 | 나머지는 전부 CPU로 감 |
| Q5_0 처리 | CPU에서 F32로 역양자화한 뒤 GPU SGEMM (`q5_0_cpu_dequant_count`) | 메모리 트래픽이 4배 이상 늘어 디코드에 치명적. `fused_q5_0` 셰이더는 있지만 `graph_compute`에서 쓰지 않음 |
| `graph_compute` | 노드마다 임시 버퍼 생성 → map/flush → `ac_runtime_dispatch` → `event_wait`(5초 타임아웃) → invalidate → destroy | 노드당 GPU 왕복 동기화가 1회씩 발생 |
| caps | `async` / `events` / `buffer_from_host_ptr` 모두 false | UMA 장점을 반만 씀 |
| 버퍼 타입 | `is_host = true` (UMA) | **좋음.** CPU 백엔드가 같은 버퍼를 복사 없이 읽을 수 있음 |
| 제한값 | `get_max_size` 512 MiB, alignment 64를 하드코딩 | 기기의 `maxStorageBufferRange`, `minStorageBufferOffsetAlignment`에서 읽어야 함 |
| 실패 처리 | 실패하면 `GGML_STATUS_FAILED` 반환. 실패 주입 훅 10종 | **좋음.** doctrine에 부합 |

**규모 추정 [추정]:** Qwen2.5-0.5B는 24레이어 × MUL_MAT 7개(Q, K, V, O, gate, up, down)에 output 1개를 더해 토큰당 약 170개의 MUL_MAT 노드가 생긴다. 지금 구조라면 토큰 하나에 submit + fence 왕복이 약 170회 일어난다. 왕복 1회 비용을 기기별로 실측하면(Phase 0) 이 구조의 상한이 바로 나온다.

### 3.4 빌드가 끊긴 원인 [로그] [코드]

1. **직접 원인:** 스크립트가 whisper.cpp의 `ggml.h`, `ggml-backend.h`, `ggml-cpu.h`만 A35로 복사하고 `ggml-alloc.h`는 빠뜨렸다. 그래서 `ggml-backend.h` 안의 `#include "ggml-alloc.h"`가 Termux 시스템의 `/usr/include/ggml-alloc.h`로 떨어졌고, 그 파일이 시스템 `ggml.h`를 다시 끌어오면서 `enum ggml_status`, `ggml_type` 등이 이중 정의됐다.
2. **근본 원인:** 백엔드를 PC에 있는 `dev/whisper.cpp`의 ggml 헤더로 빌드하려 했다. 하지만 `llama-cli`(`5e6a37cb`)에 로드할 백엔드는 **반드시 그 llama.cpp의 ggml 헤더로** 빌드해야 한다. `CMakeLists.txt`에는 이미 `LLAMA_CPP_DIR`을 쓰는 `ggml-ameva` 타깃이 있다(250행 부근). 임시 스크립트 대신 이 타깃을 쓰고, `LLAMA_CPP_DIR`을 `5e6a37cb` 트리로 바꾸면 된다.
3. **구조적 위험:** A35에 `/usr/include/ggml.h`가 있다는 것은 Termux 쪽에 다른 버전의 시스템 ggml이 깔려 있다는 뜻이다. 게다가 termux-llamacpp `install.sh`는 자기 `libggml*.so`를 `$PREFIX/lib`에 복사한다 [코드]. 서로 다른 ggml 버전이 같은 `$PREFIX/lib`에서 충돌할 수 있는 구조다.

### 3.5 Zero-Silent-Fallback 위반 (이번 감사에서 발견) [코드]

| # | 위치 | 내용 |
| :---: | :--- | :--- |
| 1 | `termux-train/termux_train/backend/amuda_backend.py:168-170` | matmul에서 GPU 예외가 나면 카운터만 올리고 `np.matmul` 결과를 반환 |
| 2 | 같은 파일 `:184`, `:196`, `:208` | add/sub/mul이 `except Exception: pass`로 NumPy를 반환. 카운터조차 없음 |
| 3 | `termux-stt/scripts/install_sherpa_onnx.sh` | 바이너리를 내려받지 않고 `touch`로 빈 파일을 만든 뒤 `chmod +x`, "installed"를 출력하는 **가짜 설치** |
| 4 | `termux-stt/scripts/install_whisper_cpp.sh` | Vulkan cmake 실패를 `2>/dev/null \|\|`로 숨기고 CPU 빌드로 진행. 커밋 미고정, `\|\| true` 다수 |
| 5 | `termux-llamacpp/termux_llamacpp/hardware.py` (opencl 분기) | `except Exception: pass` 뒤에 무조건 `("opencl", ngl)`을 반환. 바인딩이 실패해도 "opencl"이라고 보고함 |
| 6 | `termux-train/termux_train/backend/__init__.py:55` | amuda 초기화 실패 시 warning만 남기고 다음 백엔드로 넘어감. 로그는 남지만 doctrine상 폴백 |

추가 확인 필요 [미검증]: termux-llamacpp의 vulkan 분기는 시스템 Vulkan 드라이버(.so)는 검사하지만, 실행할 `llama-cli`에 Vulkan 백엔드가 들어 있는지는 검사하지 않는 것으로 보인다. 그렇다면 CPU 전용 바이너리에 `-ngl 999`를 넘겨도 GPU 오프로드가 실제로 일어나지 않는다.

### 3.6 플릿 상태

- **이 세션에서는 확인할 수 없었다.** ssh 별칭(`s25`, `a35` 등)은 PC의 Windows `~/.ssh/config`에 있는데, 이 세션 셸은 루프백만 있고 Tailscale 경로가 없다. 그래서 5대 모두 `Could not resolve hostname`으로 실패했다. 지금 켜져 있는지는 모른다.
- **마지막 기록 [보고서]:** A35, A53은 온라인. S20/S21은 sshd 중단 또는 오프라인. A53은 `libac_runtime`과 SPIR-V 빌드는 됐지만 NumPy(Python 3.13, libc++ ABI) 문제로 Python 테스트가 불가했다. 이것도 C++ 기반 검증 도구(`test-backend-ops`, `llama-bench`)를 주력으로 삼아야 하는 이유다.
- **GPU 패밀리:** Adreno 6xx(S20: 650), Adreno 8xx(S25: 830), Mali Valhall(A35: G68 MP5, A53: G68 MP4 [사양, 미검증], S21: G78).

### 3.7 미커밋 자산 [코드]

- `ameva-compute`의 `git status`는 72개 항목이다. 13종 셰이더, `ops.py`, `python/tests/`, 벤치 스크립트는 **untracked** 상태다.
- 공백·개행을 무시하면 실질 변경은 4개 파일(`CMakeLists.txt`, `pyproject.toml`, `__init__.py`, `core.py`, 합계 +1,095 / −94줄)이다. 나머지 51개 파일의 diff는 공백·개행(CRLF)만 다르다.
- **Sprint 3~4 성과물은 지금 이 PC 작업 트리에만 있다.**

---

## 4. 핵심 설계 판단

### 4.1 통합 축: ggml을 공통 실행 계층으로

```
  앱 / 에이전트 (ameva-runtime, orchestrator, aichain)
          │
  ┌───────┴─────────────────────────────────────────────────────┐
  │ llama.cpp      mtmd(vision)      whisper.cpp       sd.cpp    │  엔진: 각자 ggml 커밋 고정
  │ [BitNet: llama.cpp TQ2_0 경유 — 스파이크로 확인]              │
  │ [TTS: ggml 기반 구현 경유 — 평가]                             │
  └───────┬─────────────────────────────────────────────────────┘
          │ ggml-backend 인터페이스 (버전별 shim)
  ┌───────┴──────────────────────┐
  │ ggml-ameva                    │  supports_op 계약 / 그래프 → 단일 배치
  │                               │  strict 모드 / 노드 배치 리포트 / 카운터
  └───────┬──────────────────────┘
          │ AMUDA C ABI (ac_runtime: buffer, kernel, batch, event, error)
  ┌───────┴──────────────────────┐        ┌──────────────────────────────┐
  │ AMUDA 커널 라이브러리          │        │ ggml 밖 경로 (직접 호출)       │
  │ + Quirk DB                     │◄───────│ train: Python → ac_runtime   │
  │ (Adreno 6xx/8xx, Mali Valhall) │        │ TTS 잔여: HiFi-GAN 네이티브   │
  └───────┬──────────────────────┘        └──────────────────────────────┘
          │ /system/lib64/libvulkan.so 직결 (Termux llvmpipe 함정 회피)
     Adreno / Mali 실리콘           [OpenCL: Adreno 연구 트랙]
```

이 구조를 고르는 이유는 세 가지다. 백엔드 하나를 구현하면 엔진 4~6개가 따라온다. 정확도·성능 검증 도구(`test-backend-ops`, `llama-bench`, `llama-perplexity`)가 업스트림에 이미 있다. 그리고 엔진 쪽 코드를 거의 건드리지 않으므로 업스트림을 따라가기 쉽다.

### 4.2 ggml 버전 문제: 소스 1개, 빌드 N개

- **원칙:** 백엔드 `.so`는 호스트 엔진과 **정확히 같은 ggml 커밋 헤더**로 빌드한다.
- **shim:** `ameva_ggml_compat.h`에서 버전 차이를 `#if`로 흡수한다(`graph_compute` 시그니처, 새 op enum, device caps 필드 등).
- **빌드 산출물:** `libggml-ameva-<engine>-<ggmlcommit>.so`. 빌드 receipt에 엔진 커밋, ggml 커밋, AMUDA 커밋, 컴파일러를 기록한다(termux-llamacpp가 이미 쓰는 receipt 방식을 확장).
- **whisper.cpp는 즉시 커밋 고정.** 가능하면 llama.cpp `5e6a37cb`의 ggml과 같은 sync 지점의 whisper.cpp를 골라 백엔드 빌드를 공유한다. sd.cpp는 leejet/ggml 포크라 별도 빌드가 불가피하다.
- **설치 격리:** 엔진별 프리픽스 `~/.ameva/engines/<engine>-<commit>/{bin,lib}`, `RPATH=$ORIGIN/../lib`를 쓰고, `$PREFIX/lib`에 libggml을 복사하는 것은 금지한다.
- **로딩:** 호스트를 `GGML_BACKEND_DL=ON`으로 빌드한 뒤, 트리 밖 백엔드를 `GGML_BACKEND_PATH` 환경변수나 `ggml_backend_load()`로 올린다. 프로토타입은 이미 `ggml_backend_init` / `ggml_backend_score`를 export하고 있다 [코드]. `GGML_BACKEND_PATH`가 build 8471에서 동작하는지는 확인이 필요하다 [미검증].

### 4.3 graph_compute: 노드별 동기 → split 단위 단일 배치

- **현재:** 노드 1개 = dispatch 1회 + wait 1회.
- **목표:** split 하나를 다음 순서로 처리한다.
  1. `ac_runtime_batch_begin`
  2. 노드마다 `batch_dispatch`. 배리어는 자동 주입하고, view/reshape/permute/transpose는 연산 없이 넘긴다.
  3. `batch_submit` 1회, fence 1회.
- **임시 버퍼 제거:** 중간 텐서는 ggml-alloc이 AMEVA 버퍼 안에 배치하므로 임시 버퍼가 필요 없다. 지금 있는 임시 버퍼는 CPU 역양자화 스테이징 때문이고, GPU 역양자화로 바꾸면 사라진다.
- **디스크립터:** 노드마다 descriptor set을 할당하지 말고 `VK_KHR_push_descriptor`나 buffer device address를 검토한다. 기기별 지원 여부는 확인이 필요하다 [미검증].
- **커맨드 버퍼 재사용:** 디코드는 토큰마다 그래프 형상이 거의 같다. 그래프 형상 해시가 같으면 기록해 둔 커맨드 버퍼를 재사용하고 push constant만 갱신한다. 실험 항목으로 둔다.

### 4.4 UMA 활용

- **`is_host = true` 유지:** CPU 백엔드와 버퍼를 공유하므로 split 경계에서 복사가 0이다.
- **`buffer_from_host_ptr`:** `VK_EXT_external_memory_host`로 mmap된 GGUF 가중치를 그대로 GPU에 노출하면 모델 로드 시의 복사와 이중 메모리가 사라진다. 기기별 지원 여부와 `minImportedHostPointerAlignment`를 확인해야 한다 [미검증]. 지원하지 않는 기기에서는 복사 로드를 **명시적 경로**로 택하고, 그 사실을 로그에 남긴다.
- **메모리 타입 정책:** `ac_vk_buffer`는 기본값(`force_non_coherent=0`)에서 HOST_VISIBLE + COHERENT를 먼저 찾고, 없으면 non-coherent를 쓴다. `force_non_coherent=1`일 때만 COHERENT를 배제한다 [코드: 72~100행]. (v0.1 초안에는 "COHERENT 회피를 우선"이라고 잘못 적었다.) flush/invalidate 비용과 coherent 비용을 기기별로 측정한 뒤 정책을 정한다.

### 4.5 커널 조달 전략

| 옵션 | 내용 | 장점 | 단점 |
| :--- | :--- | :--- | :--- |
| A. 전부 자체 작성 | 25~30종을 새로 씀 | 완전한 통제 | ggml 레이아웃과 모든 양자화 포맷을 재구현해야 함. 이전 세션 추정 8~12주 |
| **B. ggml-vulkan GLSL 이식** | MIT 셰이더를 AMUDA 런타임으로 로드. push constant·spec constant 구조를 그대로 미러링 | ggml 레이아웃과 양자화 포맷이 이미 해결돼 있음. 사용자의 Adreno 830 패치 지식을 그대로 재사용 | 데스크톱 기준 튜닝이라 모바일 성능은 보장되지 않음. 출처 고지 필요 |
| C. Qualcomm ggml-opencl 참고 | Adreno 특화 커널 | Adreno에서 유리하다고 알려짐 [미검증] | OpenCL 경로 필요(AMUDA legacy OpenCL 엔진). Mali에는 무관 |

**권장:** B로 정확도를 먼저 확보한다. 차별화가 필요한 핫스팟(디코드 GEMV Q4_K/Q4_0/Q8_0, fused dequant, attention decode)만 A로 모바일 특화한다. C는 Adreno 연구 트랙으로 둔다. README의 "클린룸" 문구는 CUDA/NVIDIA에 대한 것이므로, MIT 코드를 도입하면 `THIRD_PARTY_NOTICES`에 출처를 기록한다.

### 4.6 supports_op는 계약이다

- `supports_op`가 true를 반환하는 (op, type, shape 조건) 조합은 **해당 GPU 패밀리에서 `test-backend-ops`를 통과한 것만** 허용한다.
- 조합은 테이블로 관리한다: `ops_contract.json`에 op, 허용 타입, 제약, 통과 기기, 통과 커밋을 적는다.
- Quirk DB가 기기별로 특정 조합을 끌 수 있게 한다. 예: Adreno 650(S20)에서 공유 메모리 32 KB를 넘는 커널은 false.

### 4.7 ggml 환경에서 Zero-Silent-Fallback의 정의

- ggml 스케줄러가 미지원 op를 CPU에 배치하는 것은 **폴백이 아니라 설계된 배치**다. 단, 반드시 보이게 한다.
  - 그래프마다(또는 N 토큰마다) "AMEVA 노드 수 / CPU 노드 수 / split 수"를 출력한다. 카운터는 이미 있다.
  - `AMEVA_STRICT=1`이면, CPU로 배치되는 노드가 하나라도 있을 때 그래프 실행 전에 해당 op 목록을 출력하고 중단한다.
- **금지:** GPU 커널이 실패한 뒤 같은 op를 CPU로 다시 실행하는 것. 실패는 곧 `GGML_STATUS_FAILED` + ac 에러 스택이다. 프로토타입은 이미 이렇게 동작한다.

### 4.8 엔진 간 GPU 메모리 (후반 과제)

- 지금 엔진들은 각각 별도 프로세스(subprocess CLI)로 돌기 때문에 저마다 VkDevice를 만든다. STT → LLM → TTS 파이프라인을 동시에 띄우면 메모리가 중복되고 GPU를 두고 경합한다.
- 장기적으로는 ameva-runtime이 엔진별 GPU 메모리 예산을 나눠 준다(파일 락 기반 레지스트리). 또는 상주 서버 하나(`llama-server` 형태)로 모달리티를 합친다. Phase 8 이후 과제다.

---

## 5. Op 커버리지 매트릭스 (초안)

> 이 표는 각 엔진 구조에 대한 지식으로 만든 **초안 [추정]**이다. Phase 0-7에서 엔진별 실제 그래프를 덤프해 확정한다.
>
> ● 필수 / ○ 모델·옵션에 따라 필요 / – 불필요

| Op | llama | vision | whisper | sd | BitNet* | 현재 AMUDA 자산 | Phase |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| VIEW/RESHAPE/PERMUTE/TRANSPOSE | ● | ● | ● | ● | ● | 연산 없음 | P1 |
| MUL_MAT F32/F16 | ● | ● | ● | ● | ● | `ggml_mul_mat`(F32), `matmul_f16`(연속 FP16) | P1 |
| MUL_MAT Q4_0 / Q8_0 | ● | ● | ● | ● | – | 없음 | P1 |
| MUL_MAT Q4_K / Q6_K / Q5_0 | ● | ● | ● | ○ | – | Q5_0 역양자화(CPU 경로) | P2 |
| GET_ROWS | ● | ● | ● | ● | ● | 없음 | P2 |
| RMS_NORM | ● | ○ | – | ○ | ● | `rmsnorm_f16` | P2 |
| ADD/SUB/MUL/DIV (broadcast) | ● | ● | ● | ● | ● | `elementwise_f16`(같은 shape만) | P2 |
| SCALE | ● | ● | ● | ● | ● | `ops.scale` | P2 |
| ROPE (norm/neox/mrope) | ● | ○ | – | ○ | ● | `rope` | P2 |
| SOFT_MAX (mask, scale) | ● | ● | ● | ● | ● | `softmax_f16` | P2 |
| GLU (SWIGLU/GEGLU) | ● | ○ | – | ○ | – | `swiglu_silu` | P2 |
| UNARY (SILU/GELU/GELU_QUICK/RELU/SQR) | ● | ● | ● | ● | ● | `silu`, `gelu_tanh`, `relu` | P2 |
| CPY / SET_ROWS (KV 기록, F32→F16) | ● | ● | ● | ● | ● | 없음 | P2 |
| CONT / DUP | ● | ● | ● | ● | ● | 없음 | P2 |
| FLASH_ATTN_EXT | ○ | ○ | ● | ○ | ○ | `attention_decode`(자체 레이아웃) | P2–P3 |
| NORM (LayerNorm) | – | ● | ● | ● | – | `layernorm_f16/f32` | P3 |
| IM2COL (conv1d/2d) | – | ● | ● | ● | – | 없음(vision conv는 NHWC 직접 conv) | P3 |
| GROUP_NORM | – | – | – | ● | – | 없음 | P5 |
| CONV_2D / CONV_2D_DW / CONV_TRANSPOSE | – | ○ | – | ● | – | `conv3x3`, `pointwise` | P5 |
| UPSCALE / PAD / CONCAT | – | ○ | – | ● | – | `concat2_channel` | P5 |
| TIMESTEP_EMBEDDING / ARANGE | – | – | – | ● | – | 없음 | P5 |
| MUL_MAT TQ2_0 / TQ1_0 | – | – | – | – | ● | `bitnet_gemv_i2_s`(포맷이 다름) | P6 |

\* BitNet은 llama.cpp 경유 흡수가 가능할 때의 기준이다.

---

## 6. 실행 계획

기간은 **1인 풀타임 + AI 페어 기준 [추정]**이다. Phase 1 실적이 나오면 다시 보정한다.

### Phase 0 — 기반 정리 (1주)

| # | 작업 |
| :---: | :--- |
| 0-1 | 미커밋 자산 커밋. `.gitattributes`(eol) 정규화 커밋과 기능 커밋을 분리 |
| 0-2 | §3.5의 침묵 폴백 6건 제거·수정 |
| 0-3 | whisper.cpp 커밋 고정, 엔진×ggml 버전 매트릭스 문서화 |
| 0-4 | 기준선 빌드: llama.cpp `5e6a37cb`를 (a) CPU (b) `GGML_VULKAN=ON` (c) `GGML_OPENCL=ON`(Adreno 기기만) 세 가지로, 엔진별 격리 프리픽스에 빌드 |
| 0-5 | **기준선 Vulkan이 실제 GPU에서 도는지 확인.** `/system/lib64/libvulkan.so`로 로드되는지 확인하고, llvmpipe로 돌면 기준선이 오염된 것이므로 다시 함 |
| 0-6 | 기준선 측정: `llama-bench` pp512/tg128 `-r 5`, 모델 2종(Qwen2.5-0.5B Q4_K_M, 같은 모델 Q4_0) |
| 0-7 | AMUDA copy/triad 커널로 기기별 GPU 실측 대역폭 측정 → roofline 상한 산정(디코드 t/s 상한 ≈ 실측 대역폭 ÷ 토큰당 읽는 가중치 바이트). submit+fence 왕복 1회 비용도 함께 측정 |
| 0-8 | 엔진별 실제 그래프 덤프로 op 목록 확정 → §5 갱신 |

**완료 게이트:** 접속 가능한 기기 × 3 백엔드 기준선 표가 채워진다. **이 표가 생기기 전에는 어떤 "가속" 주장도 하지 않는다.**

### Phase 1 — ggml-ameva 재건 (2~3주)

**작업**

- CMake `ggml-ameva` 타깃으로 `LLAMA_CPP_DIR=<5e6a37cb 트리>` 빌드. 임시 스크립트와 `-include ggml.h` / `GGML_API` 재정의 해킹은 폐기한다.
- `graph_compute`를 batch API 단일 submit으로 전환한다(§4.3).
- MUL_MAT F32/F16/Q8_0/Q4_0을 GPU 역양자화로 처리한다. CPU 역양자화 경로는 삭제한다.
- alignment / max_size를 device limits에서 읽는다.
- `AMEVA_STRICT`, 노드 배치 리포트를 추가한다.

**완료 게이트**

- `test-backend-ops -b AMEVA0 -o MUL_MAT`가 A35와 S25에서 최소 통과한다.
- `llama-cli --device AMEVA0`(greedy, temp 0)의 처음 N 토큰이 CPU 출력과 일치한다.
- `llama-perplexity` 차이가 허용 범위 안에 든다. 허용 범위는 ggml-vulkan과 CPU의 차이를 참고해 Phase 0에서 정한다.

### Phase 2 — LLM 디코드 경로 전체 오프로드 (4~6주)

**작업**

- §5의 P2 op 전부와 Q4_K / Q6_K / Q5_0.
- 디코드 그래프 split 수를 1~2로 줄인다.
- 커맨드 버퍼 재사용을 실험한다.

**완료 게이트**

- 모든 op가 접속 가능한 GPU 패밀리 전부에서 `test-backend-ops`를 통과한다.
- tg128 / pp512를 기준선 3종과 비교한다.
- **§8 기준으로 중단/전환을 판정한다.**

### Phase 3 — whisper.cpp (2~3주)

**작업:** 고정 커밋용 shim 빌드, IM2COL, NORM, GELU, FLASH_ATTN_EXT.

**완료 게이트:** 고정 오디오 세트(한국어·영어)에서 CPU와 같은 전사 결과가 나온다. 인코더 ms와 RTF를 기준선과 비교한다.

### Phase 4 — vision / mtmd (2~3주)

**작업:** CLIP/SigLIP 인코더 op(IM2COL conv2d, GELU_QUICK, 모델에 따라 mrope). termux-vision은 바이너리 경로만 엔진 프리픽스로 바꾸면 된다.

**완료 게이트:** 이미지 임베딩이 CPU 대비 cosine 기준을 만족한다(기준값은 Phase 1에서 확정). 고정 프롬프트 캡션이 일치한다.

### Phase 5 — sd.cpp (5~8주)

**작업**

- leejet/ggml shim 빌드. GROUP_NORM, CONV_2D, UPSCALE, PAD, CONCAT, TIMESTEP_EMBEDDING.
- `0001-qualcomm-adreno830-...patch`의 7항목을 AMUDA Quirk DB 항목으로 옮긴다.
- Z-Image Turbo Q4_0(3.68 GB) 대응: `maxStorageBufferRange`에 맞춘 버퍼 분할, VAE 타일링.

**완료 게이트:** 고정 시드 이미지의 PSNR/SSIM을 CPU 또는 ggml-vulkan과 비교한다. s/it와 피크 RSS를 기록한다.

### Phase 6 — BitNet 흡수 (스파이크 3일 → 2~4주)

**스파이크:** BitNet b1.58 2B-4T를 llama.cpp `5e6a37cb`가 TQ2_0/TQ1_0으로 로드할 수 있는지 확인한다 [미검증].

| 결과 | 방향 |
| :--- | :--- |
| 가능 | termux-bitnet을 termux-llamacpp 위의 얇은 래퍼로 만든다. AMUDA i2_s 커널은 TQ2_0 블록 포맷으로 개작한다 |
| 불가 | `VulkanBitNetEngine`을 `ac_runtime` batch API 위로 재호스팅한다 |

**완료 게이트:** bitnet.cpp CPU 대비 로짓을 비교한다. 활성값 양자화 방식이 달라 bit-exact는 아니므로 허용 오차를 미리 정한다.

### Phase 7 — TTS (평가 2주 → 결정)

**후보**

- **후보 1:** ggml 기반 TTS 구현으로 이전해 ggml-ameva를 재사용한다(Kokoro 등 지원 여부 평가).
- **후보 2:** Piper/Melo의 HiFi-GAN 보코더만 AMUDA 네이티브로 만든다(conv1d, conv_transpose1d, leaky_relu, ResBlock, 32 MB 타일링 지식 재사용). 앞단은 ONNX CPU로 유지한다.

**완료 게이트:** 기준 엔진 대비 파형 상관/MCD와 RTF를 비교한다.

### Phase 8 — Train (보류 가능)

- **경로 A:** GPU 상주 텐서(Python 객체가 AcBuffer를 소유), 여러 GEMM을 단일 배치로 묶기, FP32 누산과 손실 스케일링 정책, 침묵 폴백 제거.
- **경로 B:** ggml 학습 경로(ggml-opt, OUT_PROD, OPT_STEP_ADAMW). ggml-ameva에 backward op를 넣으면 공유할 수 있다.

### 합계 [추정]

| 범위 | 기간 |
| :--- | :--- |
| P0~P4: LLM + Vision + STT | 11~16주 |
| P5~P7: Diffusion + BitNet + TTS | 추가 11~18주 |

---

## 7. 측정 프로토콜

- **기준선 3종:** CPU(NEON, 스레드 수 고정), ggml-vulkan(같은 커밋, 실제 GPU 로드 확인), ggml-opencl(Adreno만).
- **반복:** 5회. 중앙값과 표준편차를 함께 기록한다.
- **열:** 측정 전에 쿨다운한다. 시작·종료 온도를 기록한다(`termux-battery-status` 또는 `/sys/class/thermal` 읽기 가능 여부를 Phase 0에서 확인). 충전 상태를 고정한다.
- **메모리:** 피크 RSS(`VmHWM`), Vulkan 할당/해제 카운터.
- **저장:** 기기, 엔진 커밋, ggml 커밋, AMUDA 커밋, 빌드 receipt를 묶어 JSON으로 `evidence/`에 저장한다.
- **표기 규칙:** "N배 가속"은 같은 정밀도, 같은 모델, 같은 커밋의 기준선 대비로만 쓴다.

---

## 8. 리스크와 중단/전환 기준

| 리스크 | 가능성 | 대응 |
| :--- | :---: | :--- |
| ggml 백엔드 인터페이스 변경 | 높음 | 커밋 고정, shim, 엔진별 빌드 검증 |
| 모바일 드라이버 버그 | 높음 | Quirk DB, 기기별 `test-backend-ops` 통과 조건 |
| ggml-vulkan / ggml-opencl보다 느림 | 중간~높음 [추정] | 아래 판정 기준 |
| 대형 모델 메모리 한계(LMK 종료) | 중간 | 버퍼 분할, external host memory, 엔진별 예산 |
| 1인 개발 대역폭 | 높음 | Phase 게이트, 범위 축소 우선순위 |
| 미커밋 자산 손실 | 현재 존재 | Phase 0-1 |

**Phase 2 종료 시 판정**

- **조건:** 접속 가능한 GPU 패밀리의 과반에서 ggml-ameva의 tg128이 max(CPU, ggml-vulkan, ggml-opencl)보다 낮다. 그리고 프로파일링으로 2주 안에 고칠 수 있는 원인을 특정하지 못한다.
- **전환:** 이 조건이면 AMUDA를 별도 실행 백엔드가 아니라 다음 두 가지로 바꾼다.
  - (a) Quirk DB + ggml-vulkan 패치셋
  - (b) strict 모드 / 카운터 / 메모리 예산 거버넌스 계층
- "안 된다"를 숨기지 않고 처리하는 방식이 이것이다.

---

## 9. 다음 3일 실행 항목

### Day 1

1. `ameva-compute` 커밋 2개: (a) `.gitattributes` + eol 정규화, (b) Sprint 3~4 기능 커밋.
2. `amuda_backend.py` 침묵 폴백 제거. `except`에서 예외를 다시 던진다(`AmudaComputeError`). 크기 기반 CPU 라우팅은 유지하되 카운터와 로그로 공개한다.
3. `install_sherpa_onnx.sh`: 실제 바이너리를 내려받고 sha256을 검증하거나, "미지원"을 명시하고 `exit 1`로 끝낸다.

### Day 2 — A35

4. llama.cpp `5e6a37cb` 소스 트리를 준비한다(PC에서 전송).
5. 기준선과 테스트 도구를 빌드한다.

   ```bash
   cd ~/llama.cpp && git rev-parse HEAD    # 5e6a37cb115dc1074e274ac004373f5661909695 이어야 함
   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
         -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DLLAMA_BUILD_TESTS=ON
   cmake --build build --target llama-cli llama-bench test-backend-ops -j4
   ```

6. ggml-ameva를 같은 트리의 헤더와 라이브러리로 빌드한다.

   ```bash
   cd ~/ameva-compute
   cmake -B build-ggml -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CPP_DIR=$HOME/llama.cpp
   cmake --build build-ggml --target ggml-ameva -j4
   ```

   `target_link_directories`가 `${LLAMA_CPP_DIR}/build/bin`을 보므로, `GGML_BACKEND_DL` 빌드에서 libggml이 실제로 어디에 생기는지 확인해야 한다.

7. `test-backend-ops -b AMEVA0 -o MUL_MAT`로 현 프로토타입의 실제 통과/실패를 기록한다.

### Day 3

8. 접속 가능한 기기에서 기준선 `llama-bench`(CPU, Vulkan)를 돌린다.
9. Qwen2.5-0.5B 디코드 그래프의 op와 split 수를 덤프한다.
10. 이 문서의 §5와 §6을 실측값으로 갱신한다.

---

## 10. 사용자 결정이 필요한 항목

1. **ggml-vulkan 셰이더(MIT) 이식을 허용할지 (§4.5).** 일정에 가장 큰 영향을 준다.
2. **현재 플릿 기준.** S25/S21/S20/A35/A53인지, `probe_fleet.py`의 S22·Tab A7 포함 목록인지.
3. **whisper.cpp 고정 커밋 선택 기준.** llama.cpp ggml과 맞출지, 최신 안정판을 쓸지.
4. **BitNet 흡수 방향.** llama.cpp 경유인지, 자체 엔진 재호스팅인지(Phase 6 스파이크 결과를 보고 결정).
