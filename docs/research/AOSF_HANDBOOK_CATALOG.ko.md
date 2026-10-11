<!--
  SPDX-License-Identifier: Apache-2.0 OR Commercial-License
  Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.

  [Architectural Governance]
  - Interface: Open Standard (Vulkan 1.1 / OpenCL 2.0 / POSIX / ggml ABI)
  - Core Engine: AMUDA / ac_runtime native engine
  - License Defense: Dual-License Model (Open Research / Commercial Enterprise)
-->

# AOSF 온디바이스 시스템 엔지니어링 집필 카탈로그 (근거 대조판)

- **문서 식별자:** AOSF-HANDBOOK-CATALOG-2026-10-11
- **작성자:** AMEVA Systems Architecture Group (초안: Claude Opus 5.5)
- **기준 시점:** 2026-10-11, ameva-compute `51e7da3` (브랜치 `perf/opencl-mali`)
- **용도:** 후속 에이전트와 연구원이 핸드북 챕터·기술 보고서를 집필할 때 쓰는 주제 목록, 확인된 사실, 근거 위치
- **라이선스:** Apache-2.0 OR Commercial-License

---

## 0. 근거 원칙

### 0.1 근거 등급

| 등급 | 뜻 | 집필 시 처리 |
| :--- | :--- | :--- |
| **[측정]** | 커밋 본문, 설계 문서, 인수인계서, 증거 로그에 수치와 측정 조건이 있습니다 | 사실로 서술하고 근거 위치를 밝힙니다 |
| **[부분]** | 일부 수치만 기록되어 있습니다 | 기록된 부분만 사실로 쓰고, 나머지는 "미측정"으로 표시합니다 |
| **[미확인]** | 조사한 저장소에서 근거를 찾지 못했습니다 | 원시 로그를 확보하기 전에는 사실로 쓰지 않습니다 |
| **[상충]** | 기록된 측정과 어긋납니다 | 집필 전에 정정합니다. 이미 게시된 문서는 날짜를 붙인 정정 공지로 고칩니다 |

### 0.2 조사 범위와 작성 주체

- 조사 대상: ameva-compute 커밋 이력과 `docs/`(설계 문서, 인수인계서, 체크포인트), ameva-runtime 커밋 이력,
  uno-km `docs/research_papers`의 백서 6편, termux-stt·termux-tts·termux-diffusion·termux-bitnet(키워드 검색만).
- 작성 주체는 커밋의 `Co-Authored-By` 표기로 판단했습니다(`git log --all` 기준). **Claude Opus 5.5** 표기 커밋은
  ameva-compute 67건, ameva-runtime 35건, uno-km 3건, termux-llamacpp 1건이며 모두 2026-10-03 이후입니다.
  2026-09의 커밋과 표기가 없는 커밋은 작성 주체를 특정하지 않습니다.
- uno-km 백서 6편 가운데 Opus 5.5 표기가 있는 것은 Mali 커맨드 버퍼 백서(`cbbdf03`) 한 편입니다.
  S20, BitNet 2편, SOTA TTS, 플릿 벤치마크 백서는 다른 세션의 작업입니다.
- 조사하지 않은 것: `docs/evidence/` 원시 로그 전수, Kotlin·Rust 바인딩 존재 여부, termux-* 저장소의 코드 내용.

---

## 1. 기존 8개 챕터 대조 결과

| 기존 챕터 | 판정 | 기록과 일치하는 내용 | 기록과 어긋나거나 근거가 없는 내용 |
| :--- | :--- | :--- | :--- |
| 01 Mali 커맨드 버퍼 누적 | **[측정]**, 정정 2건 | 원인(디스패치당 약 100 KB), 펜스·재기록·최대 4개 대기, 프로브 3159 → 121 MiB | ① RSS 883/862/920 MiB는 단일 모델 실행의 최고치이고, LSD 3.13 dB 패리티 통과는 4모델 실행(최고 1709~1799 MiB)의 결과입니다. 두 실행을 한 문장에 묶으면 안 됩니다. ② "메모리 상한 약 64 MB"는 측정값이 아니라 추정입니다 |
| 02 S20 Adreno 650 Denormal 붕괴 | **[미확인]·[상충]** | 없음 | 비정규화 수·레지스터 언더플로 원인에 대한 디스어셈블리·절제 실험 기록이 없습니다. 측정된 결함은 업스트림 ggml-vulkan b8471의 플래시 어텐션 텍스트 손상과 S20 Turnip의 `ErrorDeviceLost`입니다(CH-02). f16 비정규화 수 변환 검사는 S20(Turnip)에서 비트 일치로 통과했습니다(CH-15). ggml-ameva는 2.8.0부터 S20에서 GPU 기본 경로로 정상 텍스트를 냅니다. "Adreno 650 감지 시 자동 CPU 전환"은 침묵 폴백 금지 원칙과 충돌합니다. 42.9 t/s, "Vulkan 정상화 8.4 t/s"의 원자료는 찾지 못했습니다 |
| 03 BitNet 삼항 가중치 붕괴 | **[부분]** | AMUDA 런타임 BitNet 엔진(`860b80e`, `2b8efe3`), KLD 표, `ac_bitnet_sample` top-1 약 94.4 % | "ARM64 엔디안·비트 시프트 결함", "ReLU² 미지원이 원인", "NEON 인라인 어셈블리"는 기록이 없습니다. 17.7/13.3/8.4/7.6 t/s는 출처를 찾지 못했습니다(기록된 값은 교사 강제 ms/token: S25 36.0, S21 66.2, S20 103.8). 업스트림 PR #633은 **OPEN**(미병합)이며 2026-10-11 사용자 결정으로 동결되었습니다 |
| 04 S25 SOTA TTS 실패와 Kokoro 전환 | **[부분]** | Kokoro-82M on ggml-ameva: S25 t1 RTF 0.174, t3 LSD 3.14 dB(하한 3.52) | ChatTTS·CosyVoice 2·F5-TTS 실패와 562초는 termux-tts 백서(다른 세션)의 수치이며 이 기록에서 재측정하지 않았습니다. "가중치를 libggml-ameva.so로 재작성"은 부정확합니다(그래프는 `adapters/tts/kokoro/kokoro.cpp`가 ggml 연산으로 구성하고 AMEVA0 백엔드가 실행). "환각 루프 0건"은 비자기회귀 모델의 구조적 성질이지 측정 결과가 아닙니다. `f56b917`은 패리티 도구가 CPU 모델 사본을 싣지 않게 한 시험 도구 변경입니다 |
| 05 whisper.cpp Split-Mode 냉각 | **[미확인]·[상충]** | PR #4089 제출 사실 | PR #4089는 2026-10-11 현재 **OPEN**(미병합)입니다. 플릿 벤치마크 백서 102행의 "머지"는 사실과 다릅니다. "칩셋 열 스트레스 지표 408 % → 16 %"는 지표 정의와 원시 로그를 찾지 못했습니다. 24.8 %는 출처가 없습니다. AMEVA의 STT 작업은 분할 모드가 아니라 Whisper 그래프 전체를 AMEVA0에서 실행하는 설계입니다(CH-12) |
| 06 S21 6.0B DiT 확산 | **[미확인]**(이 기록 기준) | termux-diffusion 연구 보고서에 39.3 s/step, 820 MB, TAESD 14 → 1.2 s 서술 | 이 조사에서 원시 로그를 확인하지 않았습니다. 알려진 제약: `--stream-layers`는 `--params-backend diffusion=cpu`가 필요하고, sd-cli는 텍스트 인코더 1.48 GB를 이미지 완료까지 해제하지 않습니다 |
| 07 Bionic `pthread_cancel` 대체 | **[미확인]** | 없음 | 조사한 모든 저장소에서 `pthread_cancel` 문자열이 0건입니다. 실제로 측정·수정한 Bionic/Termux 링커 문제는 CH-07로 대체합니다 |
| 08 PyTorch/ONNX 배제와 AMUDA C ABI | **[부분]** | `src/shaders/`에 `softmax_f16.comp`, `rmsnorm_f16.comp`, `swiglu_silu.comp`, `attention_decode.comp` 존재 | "0ms 수준 초기화"는 자체 금지 표현이며 측정과 어긋납니다. 기기별 첫 실행은 GEMV 배치를 측정하고 파이프라인을 컴파일합니다(S25 첫 `ameva-run exec` 18 t/s, 두 번째 77 t/s). 바이너리 크기 "수 MB"는 측정 기록이 없습니다 |

---

## 2. 검증된 카탈로그

기존 챕터 01을 유지하고, Opus 5.5 작업분을 중심으로 측정 근거가 있는 주제를 추가했습니다.
각 항목의 "근거"에 적힌 커밋과 문서를 먼저 읽은 뒤 집필하십시오.

### CH-01 Mali GPU 드라이버의 디스패치당 커맨드 메모리 누적과 유계 재활용
*Per-Dispatch Command Memory on Mali Drivers and Bounded Command-Buffer Recycling*

- **대상:** Galaxy A35(Mali-G68 r38p1), A53(Mali-G68 r32p1), S21(Mali-G78 r38p0) · `ac_runtime`
  `src/backend/vulkan/ac_vk_dispatch.cpp`(`ac_vk_batch_flush`)
- **근거:** `084d513`, `f56b917`, `docs/AI_CHECKPOINT.md`, `tests/test_batch_graph_kat.cpp`(MEMORY 절),
  uno-km `cbbdf03`(AOSF-TR-2026-MALI-DRIVER-04)
- **증상:** Kokoro-82M t3(음소 446개, 26.05초, 디스패치 29,973회)를 AMEVA0에서 실행하면 A35에서 첫 실행 중 SIGKILL(rc 137).
  MemAvailable이 4초 안에 2.35 → 0.92 GB로 떨어졌습니다. A53은 같은 지점에서 sshd가 사라졌고, S21(8 GB)도 같은 방식으로 종료되었습니다.
  CPU F32·F16 실행은 A35에서 완료되었습니다.
- **원인 규명:**
  - Mali kbase는 GPU 메모리를 VmRSS에 계상하지만 smaps의 `/dev/mali0` 매핑은 Rss 0으로 보입니다. 매핑 개수를 세어 추적했습니다.
  - 기각: 디스크립터 풀(셋당 약 0.7 KB), 커맨드 버퍼 개수(`AMEVA_FLUSH_MAC=1e6`으로 플러시를 크게 늘려도 0.10 GB 증가).
  - 절제 프로브(A35, f32 `ameva_qmm` 30,000회): `vkCmdDispatch` 1회당 약 100 KB의 드라이버 메모리가 커맨드 버퍼에 잡히며,
    배리어 수·플러시 수·디스크립터 셋 재사용과 무관합니다. `vkFreeCommandBuffers`는 대부분을 시스템이 아닌 커맨드 풀로 돌려줍니다.
  - 기존 런타임은 플러시한 커맨드 버퍼를 그래프의 마지막 펜스까지 보유했습니다.
- **조치:** 플러시한 커맨드 버퍼마다 펜스를 두고, 완료된 버퍼는 이후 플러시에서 다시 기록합니다. 대기 중인 플러시는 최대 4개이며,
  넘으면 기록 스레드가 가장 오래된 플러시를 기다립니다. 60초 초과 시 `AC_ERR_LAUNCH_TIMEOUT`을 반환하고 배치를 중단합니다.
- **검증:**
  - 프로브 최고 RSS 3159 → 121 MiB.
  - `test_batch_graph_kat` MEMORY(디스패치 30,000회, 플러시 128, RSS 증가 < 48 MiB): 5대 모두 PASS(−1.5~1.3 MiB).
    같은 시험을 이전 런타임으로 A35에서 실행하면 FAIL(+2932.5 MiB).
  - t3 4모델 패리티(f16 AMEVA vs f32 CPU, 기준 1.5 × CPU f16 하한): A35 LSD 3.134 dB(최고 1765 MiB), A53 3.135 dB(1709 MiB), S21 3.13 dB(1799 MiB).
  - t3 단일 모델 `--gpu-only` 최고 RSS(`f56b917` 이후): A35 883, A53 862, S21 920 MiB.
  - S25 t2 시간: 이전·이후 런타임 모두 651.9~656.4 ms(같은 분 안에 교대 측정, 차이 없음).
- **남은 한계:** 펜스 대기 상한에서 도출되는 메모리 상한은 측정하지 않았습니다. A53은 4모델 패리티를 올릴 메모리가 부족해 2모델 실행을 씁니다.
  `084d513` 백엔드의 공식 배포는 사용자 결정 사항입니다.

### CH-02 업스트림 ggml-vulkan(b8471)의 텍스트 손상 실측과 공개 문구 정정
*Measured Text Corruption in Upstream ggml-vulkan b8471 and Correction of Published Claims*

- **대상:** 5대 전부 · ameva-runtime 2.7.8~2.7.9 llama 번들(`llama-cli-vulkan-android-arm64`, llama.cpp b8471, ggml 0.9.8)
- **근거:** ameva-runtime `38fe25f`, `d0fd5fd`, `28080b6`, `2ea73ab`, 2.7.9 릴리스 기록
- **측정 조건:** 2026-10-04, CPU 텍스트와 탐욕 디코딩 비교, `-c 2048 -t 4`, Qwen2.5-0.5B · Llama-3.2-1B · Gemma-2-2B(Q4_K_M)
- **결과:**
  - 플래시 어텐션 켜짐(GPU 백엔드의 llama.cpp 기본값): Adreno 830과 Mali 전 모델에서 손상된 텍스트(`!!!!`, `@@@@`).
    `GGML_VK_DISABLE_F16`, KV 타입(f16/f32), `--no-kv-offload`와 무관합니다.
  - `-fa 0`: S25의 0.5B는 CPU와 같은 텍스트, 1B·2B는 첫 토큰 이후 달라지는 일관된 텍스트(17/21/10 t/s, CPU 41/28/9).
    Mali에서는 `GGML_ASSERT(ctx->descriptor_set_idx < ctx->descriptor_sets.size())`(ggml-vulkan.cpp:6523)로 중단합니다.
  - S20(Turnip): 플래시 어텐션 유무와 관계없이 `ErrorDeviceLost`.
- **조치:** 2.7.9 라우터는 Adreno·Mali의 모든 Vulkan 계획에 `-fa 0`을 붙입니다(올바른 텍스트를 내거나, assert로 명시적으로 멈춥니다).
  2.8.0부터 LLM GPU 경로를 자체 백엔드 ggml-ameva로 교체했습니다(CH-03).
- **공개 문구 정정:** v2.7.8 릴리스 노트의 "Vulkan: PASS"는 토큰이 나오는 것만 본 판정이었고, 텍스트는 손상되어 있었습니다.
  릴리스 본문을 정정하고 README의 "35.80 t/s, CPU 대비 35.8배"(S25)와 "4.44 t/s, NEON 대비 +26.9 %"(A35)를 철회했습니다.
  2026-09-29의 "A35 OpenCL 17 t/s" 보고는 실제로 ggml-vulkan을 실행한 것이었습니다.
- **집필 시사점:** 기존 챕터 02의 S20 텍스트 붕괴는 이 측정된 결함과 대조해 다시 검토해야 합니다.

### CH-03 ggml-ameva를 ameva-runtime의 LLM GPU 경로로: 2.8.0과 2.8.1
*ggml-ameva as the LLM GPU Route of ameva-runtime 2.8.0 / 2.8.1*

- **대상:** 5대 · `adapters/llama/ggml-ameva/`, ameva-runtime llama 번들
- **근거:** `c8aa8fd`, `f2265d1`, `7e99154`, `docs/handover/20261005-1315.md`, `20261005-2310.md`, ameva-runtime `5b4c3ff`, `0ed878c`
- **내용:**
  - `ameva_mmvq`(`c8aa8fd`): int8 활성값과 정수 내적으로 토큰 생성. 형태별로 f32 배치와 비교해 10 % 이상 빠를 때만 채택.
    A35 12.5 → 19.0, S20 25.8 → 32.7 t/s.
  - 2.8.0 게시(2026-10-04, PyPI·npm·GitHub). 사용자 결정으로 S25·S20·A35는 GPU 기본, S21·A53은 CPU 기본.
  - **컨텍스트 이동 결함(`f2265d1`):** llama.cpp는 컨텍스트 이동 시 f16 K 캐시의 뷰에 ROPE를 제자리 실행하는데 ggml-ameva는 f32 ROPE만 받았습니다.
    AMEVA 버퍼의 텐서는 다른 백엔드로 넘길 수 없어 `ggml-backend.cpp:809`에서 중단했습니다. `ameva_rope.comp -DF16`으로 수정.
  - 검증: 이동 후 CPU 대비 KLD 2.6e-3~8.65e-3(ctx 256, 32위치), KV 셀 재사용 비트 동일, 2048토큰 문맥,
    `test-backend-ops` 2023~2025건과 전체 재배열 1766~1768건 통과(S25·S20·A53·S21, 이후 A35).
  - 2.8.1 게시(2026-10-06). A35의 `test-backend-ops`는 처음에 **거짓 통과**였습니다(CH-07).
- **속도(Qwen2.5-0.5B Q4_K_M, CPU / GPU t/s, 백엔드 `7e99154`, A35는 `c8aa8fd`):**

  | 기기 | 프롬프트 약 370토큰 | 생성 64토큰 |
  | :--- | :--- | :--- |
  | S25(코어 0-7) | 126~195 / 1081~1087 | 79~105 / 65~76 |
  | S20 | 27.5~27.7 / 180 | 9.3~9.9 / 18.7 |
  | A35 | 109~110 / 190 | 35~37 / 21.5 |
  | A53 | 50~55 / 67 | 22.6 / 14.6~14.7 |
  | S21 | 106~111 / 221~226 | 43~45 / 22.8~24.3 |

- **남은 한계:** 토큰 생성은 S20을 제외한 4대에서 CPU가 빠릅니다. Llama-3.2-1B는 모든 GPU 경로(ggml-ameva, ggml-vulkan)에서
  약 10토큰 이후 CPU의 탐욕 텍스트와 갈라지며 KLD는 약 2.4e-3입니다(근소 차이로 보고하고, 동일 텍스트로 보고하지 않습니다).

### CH-04 셰이더 구조에 대한 드라이버 컴파일러의 민감도: 실행되지 않는 루프 하나가 12배
*Driver Compiler Sensitivity to Shader Structure: a Never-Executed Loop Costs 12x on Mali r38*

- **대상:** S21(Mali-G78 r38p0), A35(Mali-G68 r38p1), S20(Turnip), 대조군 A53(r32p1)·S25(Adreno 830) · `ameva_qmv.comp`
- **근거:** `3a05203`
- **증상:** 2026-10-03 리팩터링 이후 기본 GEMV가 Mali r38 드라이버에서 약 10~12배 느려졌습니다.
- **원인 규명:** 토큰 생성 루프를 "UPL 단위 주 루프 + 잔여 루프"로 바꾸었고, 배포되는 UPL == 1 빌드에서는 잔여 루프가 한 번도 실행되지 않습니다.
  그런데 r38 Mali 컴파일러는 이 형태에 대해 약 12배 느린 코드를 생성했습니다. 처음에는 "Mali-G78 드라이버 병리", 이어서 "특수화 상수 문제"로
  잘못 판단했습니다. 같은 기기에서 이전 빌드와 비교하고, 변경 하나씩을 뺀 셰이더 변형을 빌드해 원인을 찾았습니다.
- **조치:** UPL == 1일 때 단일 루프. ROWS 변형은 모듈로 분리.
- **검증(S21, Qwen2.5-0.5B, 기본 배치, 2회):** 두 루프 2.39~2.46 t/s(그래프당 GPU 281~283 ms, q5_0 4864×896 2988~3020 us) →
  한 루프 14.1~14.9 t/s(40.1~40.5 ms, 228~235 us). A35 6061~6524 → 680 us. Turnip 약 1.6배. A53·S25 변화 없음.
  융합 에필로그는 원인이 아니었습니다(에필로그 없는 두 루프 2.40~2.58 t/s).
- **시사점:** `mali_qmv`의 "G78에서 6~11배 우위"는 이 결함이 만든 착시였습니다. 핫 셰이더를 바꾸면 모든 드라이버 계열에서 이전 빌드와 비교합니다.

### CH-05 Mali GEMV 스케일 읽기 경계의 한 워드 오차
*Off-by-One-Word Scale Bound in the Mali GEMV Decoder*

- **대상:** A53(실패 재현), S25(`test-backend-ops`) · `mali_qdecode.glsl`, `ggml-ameva.cpp`
- **근거:** `3e1404b`
- **원인:** 디스패치별 경계 `((s_off >> 2) + (N * s_row >> 2) - 1)`가 `s_off` 또는 `N * s_row`가 4로 나눈 나머지 2일 때 한 워드 짧아,
  마지막 행의 마지막 스케일 필드를 이전 워드에서 읽었습니다.
- **조치:** 바인딩 길이(`sw.length() - 1`)로 경계를 정했습니다.
- **검증:** A53 `ameva_qmm_check` 실패 18/516(K=544 N=21: q4_0·q5_0·q8_0·iq4_nl, K=256 N=37: q3_K·q6_K, 최대 상대오차 1.2e7) → 516/516.
- **함께 수정한 계약:** `AMEVA_REPACK_ALL=1`에서 양자화 텐서를 쓰거나 MUL_MAT src0 밖에서 읽는 연산은 `supports_op`이 거절합니다
  (첫 쓰기 시점에 재배열되므로). S25 `test-backend-ops` 1696/1949 → 1692/1692.

### CH-06 비캐시 GPU 매핑 위의 가중치 재배열: 모델 로드 81초 → 4초
*Weight Repacking on Uncached GPU Mappings*

- **대상:** A35·A53·S21(Mali), S20(Turnip) · ggml-ameva 재배열 코덱
- **근거:** `bf85cfb`
- **원인:** Mali와 Turnip에서 첫 HOST_VISIBLE | HOST_COHERENT 메모리 타입은 비캐시 매핑입니다. 재배열 코덱이 블록 필드를 작은 단위로
  흩어 읽고 쓰는 동작이 이 매핑 위에서 약 6~9 MB/s로 실행되었습니다.
- **증상:** Llama-3.2-1B Q4_K_M 로드 81~134초, Gemma-2-2B 182~294초(S25는 5초·12초).
- **조치:** 일반 메모리의 스테이징 사본(한 번에 최대 4 MiB 행)에서 재배열하고, 매핑에는 영역 단위 `memcpy`만 합니다.
- **검증:** A35 Llama-1B 로드 81 → 4초.
- **시사점:** 매핑된 AMEVA 버퍼를 CPU가 작게·흩어서 접근하는 코드는 모두 같은 문제를 가집니다.

### CH-07 Android Bionic·Termux 링커와 ggml ABI: 잘못된 라이브러리, 거짓 통과, SIGILL
*Bionic/Termux Linker and ggml ABI Traps* (기존 챕터 07 대체)

- **대상:** 5대 · ggml-ameva 플러그인, `ac_runtime`, ameva-runtime 번들
- **근거:** CLAUDE.md 종결 기록(2026-10-02), `8a2c63f`, `fdcc3e8`, ameva-runtime 2.8.1 기록, `docs/design/stt-whisper-ameva.md` 3절
- **사례:**
  1. **vtable 슬롯 밀림(`GGML_ASSERT(device)`, 2026-10-02 종결, 작성 주체 표기 없음):** 플러그인이 실행 측 ggml 0.9.8보다 새 헤더로 빌드되었습니다.
     `ggml_backend_i`에 슬롯이 추가되었는데 `GGML_BACKEND_API_VERSION`은 2 그대로라 로더가 거르지 못했고, `backend->device`가
     `event_wait` 슬롯(NULL)에서 읽혔습니다. 수정: 이름 지정 초기화, 로드 시점 ABI 탐침 `ameva_abi_check`(불일치 시 명시적 거부).
     검증: A35에서 재현, 양방향 거부, S25 번인 10/10.
  2. **거짓 통과(2026-10-06):** A35의 `test-backend-ops`가 rc=0으로 끝났지만 AMEVA 시험은 0건이었습니다. 기기의 실험용 ggml(`rt098`)이
     번들(b8471)과 `ggml_backend_i` 멤버 수가 달라 ABI 탐침이 백엔드를 거부했기 때문입니다. 번들 ggml로 재실행해 2024/2024 통과.
     이후 `verify_kv.sh`는 AMEVA 시험 0건을 `TBO_AMEVA_NOT_TESTED`로 표시합니다.
  3. **RUNPATH 순서(`8a2c63f`):** Termux clang 드라이버는 명령행의 `-rpath`보다 `$PREFIX/lib`를 앞에 둡니다. 플러그인이
     `$PREFIX/lib`에 남은 옛 `libac_runtime.so.2`에 결합되어 "cannot locate symbol", AMEVA0 미등록. 빌드 스크립트의 등록 검사는
     `LD_LIBRARY_PATH`로 출력 디렉터리를 노출해서 이를 보지 못했습니다. 링크 후 patchelf로 `$ORIGIN:$PREFIX/lib`.
  4. **적재 시점에만 드러나는 미해결 심볼(`fdcc3e8`):** 정의되지 않은 오류 코드와 빠진 소스 때문에 `libac_runtime.so.2`가 미해결 심볼을 가진 채
     빌드되었고, A53에서 적재할 때 "CANNOT LINK EXECUTABLE"로 실패했습니다. `AC_ERR_BACKEND_UNAVAILABLE = -111` 정의, `--no-undefined` 링크.
  5. **번들 CPU 백엔드의 SIGILL:** STT 번들의 `libggml-cpu.so`는 `MATMUL_INT8 = 1`로 빌드되어 i8mm가 없는 S21·A35 코어에서 첫 그래프에
     SIGILL로 종료합니다. 스케줄러가 CPU에 둔 노드 하나가 프로세스를 끝냅니다.
  6. **OpenCL 드라이버 접근:** Termux 기본 환경의 ocl-icd는 -1001(플랫폼 없음)을 반환합니다. 벤더 경로를 `LD_LIBRARY_PATH`에 넣으면 동작하지만
     `$PREFIX/lib`나 `/vendor/lib64`를 통째로 넣으면 시스템 바이너리가 쓰는 라이브러리를 가립니다(S25 `Xzs_Construct`, A53 `egl_get_connection`).
     `ac_runtime`의 OpenCL 백엔드는 `android_load_sphal_library`로 벤더 링커 네임스페이스를 통해 드라이버를 적재하므로 벤더 경로가 필요 없습니다.
  7. **명령 소유권 충돌:** `llama-cli`와 `libggml*.so`를 런타임 번들(b8471), termux-* 설치기, Termux apt `llama-cpp`(b9590) 세 곳이 소유합니다.
- **기록 없음:** `pthread_cancel` 대체 구현은 조사한 저장소에 없습니다.

### CH-08 확장 없는 Mali의 정수 내적 에뮬레이션과 빗나간 세 예측
*Integer-Dot Emulation on Mali without the Extension, and Three Predictions that Missed*

- **대상:** S21(Mali-G78 r38p0), A53(Mali-G68 r32p1): Vulkan에 `VK_KHR_shader_integer_dot_product`가 없습니다
- **근거:** `550552e`, `4099597`, `e5e5bff`, `6bae494`, `5440d88`, `194718a`, CLAUDE.md 종결 기록(2026-10-05)
- **에뮬레이션(`550552e`):** 가중치 4개(부호 없음, 최대 63)와 활성값 4개(u = q + 128)의 내적을 32비트 곱셈 두 번으로 계산합니다.
  `(W & 0x00FF00FF) * (u2 | u0 << 16) + ((W >> 8) & 0x00FF00FF) * (u3 | u1 << 16)`의 상위 절반을 취하고,
  Σwq = Σwu − 128Σw로 복원합니다. w ≤ 63, u ≤ 255에서 자리올림과 오버플로가 없어 정확합니다.
  이득은 작았습니다: A53 14.4 → 14.6, S21 21.8~22.8 → 22.9~23.1 t/s.
- **빗나간 예측(모두 연산 수 추론에 근거):**

  | 실험 | 예측 | 측정 |
  | :--- | :--- | :--- |
  | 프롬프트 커널의 정수 내적 에뮬레이션 | f32 대비 약 1.25배 시간 | 22~25배 느림(내부 루프에 누산기 32개가 살아 있음) |
  | 16비트 부동소수 타일 | 2배 | 1.06~1.28배 |
  | 한 쿼리 행 어텐션(키 블록 배리어 제거) | 2~3배 빠름 | Mali-G78·Adreno 650에서 2~3배 느림 |

- **절제 측정(`AMEVA_BUILD_QMV_ABLATE`, S21):** GEMV는 로드 약 30 %, 디코딩 약 40 %, 곱 약 30 %. 출력 투영은 로드 75~90 %.
  `AMEVA_FA_SPLIT_KEYS`는 효과가 없어 배리어 깊이는 비용이 아니었습니다.
- **결론:** Vulkan에서 곱을 싸게 만드는 방법으로는 두 기기의 토큰 생성이 CPU를 넘지 못합니다. OpenCL 경로로 전환했습니다(CH-09).
  새 커널군을 만들기 전에 그 근거가 되는 가정을 절제 빌드·환경 스위치·라벨 프로파일로 먼저 측정합니다.

### CH-09 Mali OpenCL 블록 정수 경로와 그래프 전체의 OpenCL 실행
*Block-Integer MUL_MAT on Mali OpenCL and Whole-Graph Execution through Translated GLSL*

- **대상:** S21, A53, A35 · `src/backend/opencl/ac_cl_backend.cpp`(`AC_BACKEND_OPENCL`), `scripts/glsl_to_opencl.py`, `ameva_mmqb.comp`
- **근거:** `465b3f4`, `0bf3801`, `42c7df1`, `0b32b7a`, `ea55564`, `cc463f6`, `e509fdb`, `8726dd5`, `795f268`, `29b5362`, `e3a3af8`,
  `docs/design/opencl-mali-int8.md`, `docs/design/opencl-prompt-path.md`
- **측정된 장치 특성:**
  - Mali Valhall에서 희소한 자원은 로드입니다. L1 상주 16바이트 로드 22.9 G/s(S21, A53 7.0), 패킹 내적 258 G/s(A53 84): 로드 1개가 내적 11~12개 값.
    캐시에 없는 메모리의 로드는 12~21배 느립니다(S21 약 17 GB/s, A53 약 9 GB/s).
  - 표준 `dot_4x8packed_ss_int`와 벤더 `arm_dot_acc`는 S21에서 같은 시간입니다. A53 드라이버(r32p1)에는 표준 내장 함수가 없습니다.
  - 작업 항목의 행 수와 로컬 크기에 따라 같은 커널이 3배 이상 달라지며 최적점이 기기마다 다릅니다(S21 1×4, A53 2×8).
- **블록 정수 형식:** 256원소 블록마다 활성 스케일 하나, 블록 안에서는 정수로 누적. Vulkan f32 타일 대비 커널 시간 S21 1.6~2.2배, A53 2.2~2.9배(교대 측정).
  첫 게시의 S21 "1.8~2.2배"는 Vulkan 커널이 두 배 느리던 시기의 수치와 비교한 것이어서 날짜를 붙여 정정했습니다(`0bf3801`).
- **기각한 구조:**
  - Vulkan↔OpenCL 메모리 공유(AHardwareBuffer + `cl_arm_import_memory`): A53·A35는 정상, S21(r38p0)은 Vulkan 쓰기 후 OpenCL이 64바이트 단위의
    오래된 줄을 읽습니다. 동기화 방법 10가지가 모두 효과가 없었습니다.
  - 연산 오프로드(`AMEVA_OFFLOAD`, CPU가 가중치 보유): 그래프 전체 GPU 실행보다 3.5~6배 느렸습니다.
- **그래프 전체 OpenCL 실행(`8726dd5`):** 커널을 손으로 옮기지 않고 `glslangValidator -E` 출력을 OpenCL C로 번역합니다. 195개 변형이 Mali-G68
  r32p1·r38p1에서 빌드됩니다(Adreno 830 컴파일러는 다중 행 q2_K·q6_K 변형 10개를 −6으로 거부). 번역기 함정: 공유 벡터의 성분 쓰기(`v->x = x`)는
  읽기-수정-쓰기가 되어 호출들이 서로 덮어씁니다. 스칼라 저장으로 바꾸기 전에는 M ≥ 8 사례 144건이 실패했습니다.
- **CPU 유휴 시 Mali OpenCL 감속:** 대기 스레드가 잠들면 같은 커널이 수십 ms 단위로 빠른 모드와 느린 모드를 오갑니다(GPU 클럭은 최대).
  아무 클러스터의 코어 하나만 바쁘면 사라집니다. A53 생성 11.9 → 19.2~20.0 t/s, 프롬프트 95 → 110. 그래서 OpenCL 대기의 기본값은 스핀입니다(`AMEVA_CL_WAIT=spin`).
- **검증:** A53·A35에서 `ameva_qmm_check` 516/516, `test-backend-ops` 2025/2025·1768/1768, 배치·한 토큰씩 KLD 4.0e-3~5.7e-3, 2048토큰 1.8e-3~1.9e-3.
- **모델 수준(Qwen2.5-0.5B, 교대 측정, 텍스트 CPU와 동일):**
  A53 프롬프트 CPU 55 / Vulkan 67 / OpenCL 110 t/s, 생성 21.2~22.6 / 14.6 / 20.0.
  A35(코어 0-2) 프롬프트 Vulkan 189 / OpenCL 91~93, 생성 Vulkan 18.9 / OpenCL 22.5.
  A53 Llama-3.2-1B 프롬프트(블록 커널) 45.9~46.1 → 57.4~57.9 t/s(CPU 47.2~47.9).
- **컴파일러 함정 두 가지(A53 이분 탐색):** 핫 루프 전에 디코딩해 루프 내내 살아 있는 벡터는 레지스터를 잡아먹습니다(19.7 → 26.4 ms).
  Mali OpenCL 컴파일러가 펼치지 않는 짧은 상수 루프는 배열을 메모리에 둡니다. 가중치 2행 디코드 루프를 풀어 쓰자 q5_K 67 → 24 ms, q6_K 62 → 28 ms(`e3a3af8`).
- **남은 한계:** CPU를 넘은 것은 A53 프롬프트뿐입니다. 생성은 토큰마다 가중치 373 MB를 읽는 메모리 대역 한계에 있고 CPU도 같은 하한입니다.
  번역 타일 커널은 A35에서 같은 GLSL의 Vulkan 실행보다 2.1~2.9배 느리며 원인은 분리되지 않았습니다. 블록 커널은 A35 Vulkan, S25, S20에서 타일보다 느립니다.

### CH-10 컨텍스트 이동 후 KLD 판정: 커널 결함이 아니라 f16 반올림 잡음
*Context-Shift KLD Is Dominated by f16 Rounding Noise*

- **대상:** A53 · `ameva_llm_parity --kv-ops`, ameva-runtime `tools/fleet/lab/parity_long.sh`
- **근거:** `795f268`, ameva-runtime `docs/handover/evidence/20261007-opencl-graph/a53_parity_shift_*.txt`
- **증상:** 새 OpenCL 경로가 이동 후 판정(`kld_mean <= 0.01`)에서 FAIL. int8 커널 결함처럼 보였습니다.
- **분리 측정:** 기준을 f32 KV 캐시의 CPU로 바꾸면 f16 캐시 CPU 자신이 1.40e-2(ctx 256, 32위치), 1.69e-2(ctx 512, 128위치)입니다.
  ctx 512 / 128위치에서는 게시된 2.8.1 Vulkan f32 커널 3.35e-2, OpenCL 3.11e-2, OpenCL f32 전용 3.32e-2로 모두 기준을 넘고 서로 차이가 없습니다.
  32위치에서는 KLD 0.2인 위치 하나가 통과 여부를 결정합니다. 이동 후 모델의 퍼플렉시티는 220~350입니다.
- **결론:** 32위치 이동 실행의 통과·실패를 커널 변경의 증거로 읽지 않습니다. `REFKV=f32`로 측정하고 GPU 거리를 f16 CPU 자신의 거리와 비교합니다.
  릴리스 기준 재정의는 사용자 결정 사항으로 남아 있습니다.

### CH-11 긴 그래프와 큰 제출: 제출 예산과 기록기 비용
*Long Graphs and Large Submissions: Dispatch Budgets and Recorder Cost*

- **근거:** `9836997`, `f74b427`
- **S20 GPU 정지(`9836997`):** Whisper 인코더 스템의 7.4e9곱 f16 MUL_MAT이 `ameva_mm`에서 4e9곱 제출 두 번으로 나뉘어 각각 수 초가 걸렸고,
  Adreno 650 드라이버는 그런 제출을 완료하지 않았습니다. `ameva_mm`은 결과 원소당 호출 하나에 스칼라 로드라서 Adreno 650 1.4e9곱/s,
  Mali-G68 MP4 0.4e9곱/s입니다. 디스패치·제출 예산에서 이 커널의 곱 하나를 32(정확 f16 디코더는 128)로 계산하도록 바꾸었습니다.
- **기록기의 이차 비용(`f74b427`):** `batch_record`가 디스패치마다 `reserve(size + count)`를 호출해 매번 목록 전체를 재할당·복사했습니다.
  디스패치 약 19,000회인 Kokoro 26초 발화의 텍스트 그래프에서 디스패치당 약 120 us → 기하 증가로 약 25 us(S25, `AMEVA_PROFILE` record_ms).
  TTS만이 아니라 모든 긴 그래프에 해당합니다.

### CH-12 Whisper large-v3-turbo를 AMEVA 단독으로
*Whisper large-v3-turbo on AMEVA Alone* (기존 챕터 05의 AMEVA 부분 대체)

- **대상:** S25, S21, A35(전체 그래프), A35 화자분리 동시 실행 · 수정 없는 whisper.cpp에 `GGML_BACKEND_PATH` 플러그인
- **근거:** `docs/design/stt-whisper-ameva.md`, Opus 5.5 커밋 `fa4781d`, `e6bc0ec`, `9836997`, `4c521a2`, `1b89a69`
  (STT 2단계는 여러 세션의 공동 작업입니다. `aed3e39`, `e58d508`, `9809842` 등은 다른 세션 커밋입니다)
- **결함과 수정:**
  - GET_ROWS q4_0/q5_0(`fa4781d`): ggml은 원소 j를 `qs[j]`의 하위 니블, j+16을 상위 니블에 두는데 커널은 교차로 읽었습니다.
    배치 인덱스의 `nb[2]/nb[3]`도 무시했습니다. S25 `test-backend-ops` GET_ROWS 12/23 → 23/23. 잘린 전사문의 원인이었습니다.
  - 잘못된 전제의 되돌림(`e6bc0ec`): "GGUF가 conv 가중치 행을 채운다"는 이전 커밋의 전제는 틀렸습니다(정렬 채움은 텐서 사이에 있음).
    그 변경은 순열된 K 뷰를 받아들여 오류 없이 잘못된 보폭으로 읽을 수 있었습니다.
  - whisper.cpp는 요청한 장치가 없으면 오류 없이 CPU로 돌아갑니다("no GPU found"). 모든 측정은 로그의 백엔드 줄을 확인합니다.
  - whisper.cpp는 업로드 후에 버퍼를 가중치로 표시하고, ggml-ameva는 첫 쓰기 시점에 표시된 버퍼만 재배열해서 모든 MUL_MAT이 범용 커널로 갔습니다.
- **측정(S25, Adreno 830, `jfk.wav` 11초, 전사문 CPU와 동일):**

  | 모델 | 경로 | 인코딩 ms | 합계 ms |
  | :--- | :--- | ---: | ---: |
  | turbo f16 | 번들 CPU | 75741 | 79621 |
  | turbo f16 | 번들 ggml-vulkan | 26943~27207 | 33167~34169 |
  | turbo f16 | AMEVA strict, 그래프 전체 | 23084~23226 | 27956~28979 |
  | turbo q5_0 | 번들 ggml-vulkan | 33987~34338 | 37825~38394 |
  | turbo q5_0 | AMEVA, `AMEVA_REPACK_ALL=1` | 9954~10108 | 13731~14241 |

- **남은 한계:** S21·A35는 전사문이 정확하지만 S21에서 잘 빌드된 CPU 바이너리가 AMEVA f16보다 4~8배 빠릅니다. A35 화자분리는 Whisper와 동시 실행 시
  구간이 동일하고 시간이 9.1 → 35.5초로 늘었습니다(A55 코어 3개 공유). termux-stt의 `install_sherpa_onnx.sh`는 빈 파일을 바이너리로 만들고,
  `download_models.sh`는 자리표시 텍스트 파일을 씁니다(발견 사항).

### CH-13 BitNet b1.58을 AMUDA 런타임으로, 그리고 CPU 자체 편차 기반 판정
*BitNet b1.58 on the AMUDA Runtime and a Floor-Relative Parity Gate* (기존 챕터 03의 측정 가능한 부분)

- **대상:** S25(Adreno 830), S21(Mali-G78), S20(Adreno 650, Turnip); 이후 `ac_bitnet_sample`은 5대
- **근거:** `860b80e`, `2b8efe3`, `docs/design/bitnet-amuda-engine.md`, `docs/AI_CHECKPOINT.md`
- **구조 변경:** 자체 인스턴스·장치·파이프라인을 가진 raw-Vulkan `VulkanBitNetEngine`을 제거하고 GPU 층을 `ac_runtime`으로만 실행합니다.
  토큰·층당 디스패치 13회, 명시적 배리어 8개.
- **수치 계약(W1.58A8):** RMS norm 후 벡터별 int8 absmax 양자화(0에서 먼 쪽으로 반올림, ±127). i2_s GEMV는 정확한 정수합 후 스케일 곱으로
  `ggml_vec_dot_i2_i8_s`와 같은 활성값에서 비트 동일. 코드 3은 양쪽 모두 +2. RoPE 계수는 호스트에서 CPU 식으로 계산(GPU sin/cos는 [−π, π] 밖에서 미지정).
  CPU가 따로 반올림하는 곳은 `precise`로 표시, CPU 기준은 `-ffp-contract=off`.
- **판정 기준의 근거:** CPU 경로끼리도 다릅니다. `-ffp-contract=fast` 빌드는 KLD 평균 1.34e-3, 최대 5.89e-2, arg-max 108/110. 스칼라 빌드 9.44e-4.
  절대 기준(최대 KLD ≤ 1e-3)은 올바른 GPU도 실패시킵니다. 기준: KLD 평균 ≤ 2.7e-3, 기준 쪽 1·2위 차가 0.5 logit 이상인 위치에서 arg-max 불일치 0.
- **결과(110위치, 기기별 자체 CPU 기준):** S25 정수 내적 KLD 8.76e-4, 110/110, 36.0 ms/token. S21 에뮬레이션 1.87e-3, 109/110, 66.2 ms.
  S20 정수 내적 1.45e-3, 109/110, 103.8 ms. S21·S20의 불일치 한 곳(위치 13, 차 0.069)은 CPU FMA 빌드도 뒤집는 위치입니다.
  `ac_bitnet_sample`(온도 1.0/1.5): 5대에서 top-1 확률 약 94.4 %, 표본 텍스트는 기기마다 다름.
- **남은 한계:** 토크나이저가 BPE 병합이 아닌 탐욕 최장 일치입니다. LM 헤드(토큰당 f16 626 MB)가 S25·S21의 최대 비용입니다.
  업스트림 PR #633은 OPEN이며 사용자 결정으로 동결되었습니다.

### CH-14 Kokoro-82M을 ggml-ameva로: 연산 합성, 출력 판독 함정, 실시간 계수
*Kokoro-82M on ggml-ameva* (기존 챕터 04의 AMEVA 부분)

- **대상:** S25, S21, S20, A35, A53 · `adapters/tts/kokoro/kokoro.cpp`, `tests/ameva_tts_parity.cpp`
- **근거:** `5bb49d7`, `2f1a7e2`, `2aa9315`, `8e6371a`, `docs/design/tts-kokoro-ameva.md`
- **방식:** 모델 내려받기 없이 각 기기가 자기 int8 ONNX를 약 2초에 GGUF로 변환합니다. 그래프 두 개를 백엔드 하나에서 실행하며,
  모든 노드를 사전에 `ggml_backend_supports_op`으로 검사하고 지원하지 않는 노드 하나면 연산 이름과 함께 실패합니다(스케줄러·보조 백엔드 없음).
- **ggml에 없는 연산의 합성:** 양방향 LSTM(입력 투영 한 번 + 스텝별 배치 MUL_MAT), ConvTranspose1d(탭별 MUL_MAT + 겹쳐 더하기),
  InstanceNorm+AdaIN(채널당 그룹 하나인 GROUP_NORM), STFT(20탭 CONV_2D), atan(|u| ≤ tan(π/8) 홀수 다항식, 최대 오차 1.9e-8 rad).
- **ggml-ameva 추가:** SIN·COS(`ameva_sincos`: π/2 단위 축약을 융합해 1e5~1e6 rad의 SineGen 위상 정확도 유지), CUMSUM(ggml-cpu 순서, 비트 동일),
  쌍선형 UPSCALE, SNAKE 연쇄 융합(5개 연산 → 디스패치 1회).
- **출력 판독 함정:** 합성 편의를 위해 bias를 제자리 덧셈으로 바꾸자 F0 텐서가 conv 노드의 뷰가 되었고, `ggml_set_output` 후 읽은 값이 약 1e-9였습니다.
  ggml-alloc이 부모 메모리를 재사용했습니다. 오디오는 정상이었고 **F0 지표만 조용히 무의미해져 "쓰레기 대 쓰레기"로 판정을 통과**했습니다.
  출력은 별도 버퍼로 복사해서 읽습니다.
- **f16 선형층 경로:** f16 가중치의 M > 1 MUL_MAT은 GEMV 지향 `ameva_qmm`으로 가서 S20에서 약 14 GFLOP/s였습니다. 선형층을 1×1 CONV_2D로 표현해
  타일 컨볼루션 커널로 보내자 S20 t1 합성 7476 → 6133 ms.
- **결과(AMEVA0 F16, t1 7.55초):** S25 1313 ms(RTF 0.174), S21 3194 ms(0.423), S20 6017 ms(0.797). t3(26.05초): S25 RTF 0.176, S20 0.788.
  로그 스펙트럼 거리 1.48~1.49 dB(하한 3.16), F0 상대오차 1.1e-4(하한 2.0e-3), 모든 토큰의 지속시간이 기준과 같음.
  t3는 3.14 dB(하한 3.52). 비교: 현재 제품 경로(sherpa-onnx int8 CPU, S20) RTF 1.61.
- **남은 한계:** S20 GPU 시간의 82 %가 CONV_2D입니다. Mali-G68에서 컨볼루션 커널이 약 63 GFLOP/s라 A53·A35는 컨볼루션만으로 실시간을 넘습니다.

### CH-15 열 행렬 없는 컨볼루션 커널과 기기별 f16 비정규화 수 검사
*Convolution without a Column Matrix, and an On-Device f16 Subnormal Check*

- **대상:** A35, S20, S25 · `ameva_conv.comp`, `GGML_OP_CONV_2D`
- **근거:** `ea0a336`, `109ba54`, `docs/design/diffusion-vae-conv.md` 7~8절
- **커널:** `ameva_qgemm` 타일에 활성값을 원본 텐서에서 직접 스테이징합니다(IM2COL 열 행렬 없음).
- **f16 디코더 선택:** 프로세스마다 첫 f16 컨볼루션 그래프 전에 1×1 컨볼루션으로 **모든 f16 비정규화 수, 양·음의 0, 정규 값 2048개**를 커널에 통과시켜
  4096개 결과가 CPU 변환과 비트 단위로 같을 때만 내장 `unpackHalf2x16`을 씁니다. 아니면 정확 디코더. 이 검사는 A35, S20(Turnip), S25에서 내장 변환을 선택했습니다.
  (기존 챕터 02의 "Adreno 650 비정규화 수 붕괴"와 관련해 이 기록에서 측정된 유일한 비정규화 수 데이터입니다. 단, Turnip이며 Qualcomm 독점 드라이버가 아닙니다.)
- **검증:** `ameva_conv_check` 12사례 × 커널 선택 5종 비트 비교 A35·S20 12/12, `test-backend-ops -o CONV_2D` A35·S20 1568/1568,
  `ameva_qmm_check` 516/516(A35, S20, S25). 19.33 GFLOP 컨볼루션: A35 271~288 ms(ggml-vulkan MUL_MAT 대비 4.10~4.34배), S20 159~182 ms.
- **방법론 교훈(Mali-G68):** 절제하지 않은 기준 커널과 비교하면서 네 가설이 연속으로 틀렸습니다. 기준 커널에도 같은 절제를 적용하자,
  새 커널의 출력 단계(평면 쓰기)가 273 ms 중 45 ms임이 드러났습니다. 절제 목록에 출력 단계를 포함합니다.

### CH-16 비전 인코더 정합성과 침묵 폴백 탐지, 긴 쿼리용 플래시 어텐션 타일
*SigLIP Encoder Parity, Silent-Fallback Detection, and Rows-per-Invocation Flash-Attention Tiles*

- **대상:** 5대 · `tests/ameva_vision_parity.cpp`, `ameva_flash_attn`
- **근거:** `61896f7`, `ebccab6`, `docs/design/vision-siglip-ameva.md`
- **침묵 폴백:** llama.cpp `clip.cpp`는 `MTMD_BACKEND_DEVICE`가 없거나 틀리면 경고만 내고 기본 GPU 또는 CPU로 갑니다. 거절된 연산은 스케줄러가 CPU로 보냅니다.
  시험 도구는 열린 백엔드, 완료된 GPU 연산, CPU에 남은 연산, 적재된 라이브러리를 확인합니다.
- **판정 기준:** CPU Q8_0 경로 자신이 F32 기준에서 상대 L2 1.92e-2(최소 토큰 코사인 0.99873)입니다. 고정 코사인 0.999 기준은 CPU 자신을 실패시킵니다.
  기준: AMEVA 상대 L2 ≤ 1.5 × 하한. 하한 대비 비율 0.03~1.17로 5대 모두 기준 안이며, 캡션은 Vulkan에서 동일, OpenCL에서는 토큰 4(차 0.024)에서 달라집니다.
- **플래시 어텐션 타일(`ebccab6`):** 스테이징한 K/V vec4 하나가 여러 행을 처리하는 RPI 변형. SigLIP-base 어텐션(1024×1024, d64) 층당 GPU ms:
  S25 38.3 → 22.3, S20 135.1 → 112.9, S21 126.8 → 126.5, A53 254.0 → 163.6. `test-backend-ops FLASH_ATTN_EXT` 1048/1048(S25, S20, S21).
  Qwen2.5-0.5B 프롬프트(368토큰, S25) 1108~1123 → 1194~1220 t/s.

### CH-17 실기기 측정 방법론: 수치를 믿을 수 있게 만드는 조건
*Fleet Measurement Methodology*

- **근거:** 체크포인트·인수인계서의 측정 조건 기록, ameva-runtime `tools/fleet/lab/`
- **조건에 따라 결과가 뒤집힌 사례:**
  - CPU 코어 집합: A35의 ssh 세션은 A55 코어만 받습니다. S25는 코어 4개와 8개에서 CPU와 GPU의 우열이 바뀝니다(생성 CPU 79~105 t/s 대 16~49 t/s).
  - GPU 클럭: S21 GPU 거버너 상태로 2~3배 차이. 같은 S21에서 AMEVA Whisper 두 실행이 2배 달랐습니다. Adreno 열 상한.
  - 절전: S25(및 A35·A53)는 ssh 폴링 사이에 잠들어 작업이 약 4배 느려집니다. Termux 웨이크락으로 막습니다. A35는 유휴 ssh 작업을 CPU 약 1 %로 제한합니다.
  - CPU 유휴 상태가 Mali OpenCL 커널을 최대 2배 느리게 합니다(CH-09).
  - S25 t2의 4 % 차이는 세션 드리프트였습니다(같은 분 안에 교대 측정하자 사라짐).
- **규칙:** 같은 분 안에 교대 측정. 코어 집합(`Cpus_allowed_list`)과 GPU 클럭을 함께 기록. 프롬프트 처리와 토큰 생성을 분리 보고
  (생성만 담은 표가 GPU의 프롬프트 우위를 가린 적이 있습니다). PASS는 기준 출력과 비교한 경우에만 씁니다.
- **원격 실행:** 긴 실행에서 Termux sshd가 A35·A53·S20·S21에서 종료된 적이 있습니다. 원격 대기 루프 금지, `nohup` + 완료 표지 + 일회성 확인.

### CH-18 릴리스 무결성, 라우터, 거짓 통과 방지
*Release Integrity, Routing, and False-Pass Prevention*

- **근거:** ameva-runtime `70c02ec`, `854bb4a`, `1e91dc8`, `cb4cf69`, `2ea73ab`, `d0fd5fd`, `28080b6`, `999077a`, `0ed878c`; termux-llamacpp `0377c8c`
- **내용:**
  - 2.7.8: STT 번들 배치와 핀, ELF RUNPATH, 체크섬 부속 파일.
  - 2.7.9: 엔진 번들 격리(번들의 어떤 파일도 `$PREFIX/lib`에 두지 않음), 설치된 파일 읽기 전용, 시험이 호스트 설치를 읽거나 쓰지 않도록 수정.
  - 라우터: CPU 경로는 GPU 장치를 제시하지 않음, Mali의 적응 경로는 CPU, 정수 내적이 없는 Mali의 강제 Vulkan은 플래시 어텐션 끔, Adreno 600 계열은 GPU 기본.
  - termux-llamacpp: 명시적 OpenCL 요청에 다른 백엔드를 실행하지 않음(`0377c8c`).
  - 2.8.1 후보 문서에서 2.8.0 노트의 두 곳을 정정하고, 실행하지 않은 A35를 "미실행"으로 표시했습니다.
- **게시 상태(2026-10-06):** 2.8.1은 PyPI·npm·GitHub에 게시, GitHub `latest`는 v2.7.3 유지, 플릿 운영 설치는 2.7.9(승격은 사용자 결정).

---

## 3. 다른 세션 작업: 집필 전 원자료 확보가 필요한 주제

아래는 Opus 5.5 표기가 없는 작업입니다. 커밋이나 문서는 존재하지만 이 조사에서 원시 측정을 확인하지 않았습니다.

| 주제 | 존재하는 근거 | 집필 전 확보할 것 |
| :--- | :--- | :--- |
| S20 Adreno 650 독점 드라이버의 Vulkan 결함과 OpenCL·NEON 경로 | uno-km S20 백서 | 독점 드라이버 버전별 재현 로그, 비정규화 수 가설의 절제 실험(예: CH-15 방식의 비트 검사), CH-02 결함과의 구분 |
| ChatTTS·CosyVoice 2·F5-TTS 온디바이스 실패 | uno-km SOTA TTS 백서, termux-tts `engine_sota.py` | 562초 실행의 원시 로그와 측정 조건 |
| whisper.cpp 분할 모드(PR #4089) | termux-stt CHANGELOG, termux-bitnet RELEASE_NOTES | "408 % → 16 %" 지표의 정의와 측정 방법, 24.8 %의 원자료, PR 상태(OPEN) |
| S21 Z-Image Turbo 6.0B, TAESD | termux-diffusion `docs/research/s21_z_image_turbo_vulkan_research_report.md` | 단계당 39.3초, 820 MB, TAESD 1.2초의 원시 로그 |
| ggml-ameva 확산 E2E(K2/K3, SDXL, Z-Image MMDiT, GPU LoRA 병합) | ameva-compute `c93f05b`, `d31bfba`, `f333149`, `bf0ec99`, `177193d` | 각 커밋의 증거 경로 |
| ac_runtime Phase 0~4A(I2_S CPU 검증, Vulkan 코어, 비일관 메모리, 불투명 C ABI, GGUF 경계 판독기) | ameva-compute `1987f63`~`1b7202b`, `42cd737` | `docs/evidence/` 해당 묶음 |
| SqueezeNet 비전 체인(2026-09) | ameva-compute `20a108e`~`16229d9` | 해당 증거 로그 |
| Q5_0 네이티브 GEMV, 배치 그래프 엔진, MMQ·KV 양자화·연산 융합 | `5baa824`, `a6eee0f`, `4a932dc`, CLAUDE.md 종결 기록 | 인수인계서 `20261002-*`, `20261003-*` |

---

## 4. 게시된 문서의 정정 후보 (사용자 결정 사항)

**처리 상태(2026-10-11):** 사용자 지시로 1~3번을 정정했습니다(uno-km `387fa00`: 백서 md 2편, 색인 README, Labs #65·#47에 날짜를 붙인 정정 공지,
운영 사이트 반영 확인). 4번은 게시된 파일에 없고 마스터 프롬프트에만 있습니다. 나머지 게시물의 감사 결과는
[LABS_CONTENT_AUDIT_2026-10-11.ko.md](LABS_CONTENT_AUDIT_2026-10-11.ko.md)에 있습니다.

1. `uno-km/docs/research_papers/whitepaper_real_device_fleet_benchmarks_modalities.md` 102행: "PR #4089 머지" → 2026-10-11 현재 OPEN.
2. 같은 문서 156행: "세계 최초" 표현(검증 불가능한 최상급 표현).
3. `whitepaper_s20_adreno650_vulkan_defect_and_opencl_bypass.md` 3절: 비정규화 수 원인은 측정 근거가 제시되지 않았습니다.
   5.2~6절의 "Adreno 650 감지 시 자동 CPU" 설명은 2.8.0 이후 S20 GPU 기본 경로와 어긋납니다.
4. 기존 마스터 프롬프트 챕터 08: "0ms 수준의 초기화 지연시간"은 자체 컴플라이언스 규칙 위반이며 측정과 다릅니다.

---

## 5. 집필 원칙과 챕터 템플릿 (개정)

### 5.1 원칙

1. 모든 수치에는 근거 위치(커밋, 문서와 절, 증거 로그 경로)와 측정 조건(기기, 코어 집합, GPU 클럭, 모델, 반복 수)을 붙입니다.
2. 근거 등급이 [미확인]·[상충]인 서술은 사실로 쓰지 않습니다. 쓰려면 원자료를 먼저 확보합니다.
3. 기각된 가설을 숨기지 않습니다. 가설 → 측정 → 기각/채택의 순서로 씁니다(CH-01, CH-04, CH-08, CH-10이 본보기입니다).
4. 실패와 느린 결과를 같은 비중으로 씁니다(예: Mali 토큰 생성은 대부분 CPU보다 느림).
5. 금지 표현: "100 %", "완벽한", "보장", "0ms", "완전 박멸", "기적처럼", "세계 최초", "압도적", "초고속". 대신 "자동화 회귀 시험으로 검증됨",
   "기준 안", "측정 범위에서", "유계화"를 씁니다.
6. 작성 주체를 기록할 때는 커밋 표기를 따르고, 표기가 없으면 특정하지 않습니다.
7. 이미 게시된 문서의 오류는 날짜를 붙인 정정 공지로 고칩니다.

### 5.2 챕터 템플릿

```markdown
<!-- SPDX 헤더 (문서 최상단) -->

# [국문 제목]
### [English Subtitle]

- 문서 식별자: AOSF-TR-YYYY-[CATEGORY]-[NN] 또는 HANDBOOK-CH-[NN]
- 작성자: AMEVA Systems Architecture Group
- 대상 하드웨어: (기기, SoC, GPU, 드라이버 버전)
- 대상 엔진/커밋: (커밋 해시, 파일 경로)
- 근거 등급: [측정] / [부분]
- 라이선스: Apache-2.0 OR Commercial-License

## 1. 문제 정의
## 2. 원인 규명 (가설, 절제 실험, 기각된 가설 포함)
## 3. 설계와 조치 (상태도·파이프라인 Mermaid, 제약 조건 수식)
## 4. 구현 세부 (실제 코드, 명시적 오류 반환 경로)
## 5. 실기기 검증 (기기별 표, 수정 전/후, 측정 조건)
## 6. 결론과 남은 한계
## 7. 근거 색인 (커밋, 문서, 증거 로그 경로, 원시 로그 위치)
```

---

## 6. 실행 지침

- "CH-NN 써줘"를 받으면: 해당 항목의 "근거"를 먼저 읽고 수치를 재확인한 뒤 5.2 템플릿으로 집필합니다.
- 결과물은 먼저 이 저장소의 `docs/research/`에 초안으로 둡니다. uno-km 포털과 Labs 게시는 사용자 승인 후에 합니다
  (포털 빌더가 `docs/benchmarks.html`에 공유 자산을 중복 삽입하는 문제가 있으므로 커밋 전에 복원합니다).
- 3절 주제를 집필하라는 요청을 받으면, 원자료를 먼저 확보하고 그 결과를 보고한 뒤 진행합니다.

---

## 7. 안드로이드 시스템 엔지니어링 핸드북 보강 제안 (2026-10-11)

현재 핸드북(ameva-foundation `handbook/`, Labs 1.1~8.3, 3.4·3.5·7.5, 부록 A·B)은 AOSP 일반 구조가 중심입니다.
아래는 이 프로젝트가 실기기에서 직접 측정한 결과로 쓸 수 있는 장입니다. 일반 교재에는 없는 내용이라는 점이 이 핸드북의 차별점이 됩니다.

### 7.1 측정 근거가 있는 신규·보강 장

| 위치(제안) | 주제 | 담을 측정 사실 | 근거 |
| :--- | :--- | :--- | :--- |
| 2.5 (신규) | GPU 드라이버 메모리의 가시성과 측정 | Mali kbase는 GPU 메모리를 VmRSS에 계상하지만 smaps의 `/dev/mali0` 매핑은 Rss 0. 매핑 수로 추적하는 방법. 디스패치당 약 100 KB, lmkd와의 상호작용 | CH-01, `084d513` |
| 3.4 (보강) | 통합 메모리(UMA)의 캐시 속성 | Mali·Turnip의 첫 HOST_VISIBLE·HOST_COHERENT 메모리는 비캐시. 흩어진 작은 쓰기가 6~9 MB/s. "제로 카피" 설명에 이 조건을 더함 | CH-06, `bf85cfb` |
| 3.7 (신규) | 모바일 GPU 컴퓨트 기능 노출의 실측 지형 | DDK r32p1·r38p0·r38p1, Vulkan 1.1과 1.3, 정수 내적 노출 차이(A53은 Vulkan에 없고 OpenCL에 있음), Mali 서브그룹 16 고정·공유 메모리 32 KB, Adreno 서브그룹 64, Turnip과 독점 드라이버 | CH-09, `docs/design/opencl-mali-int8.md` |
| 3.8 (신규) | GPU 제출 단위와 드라이버의 완료 한계 | Adreno 650에서 4e9곱 제출이 끝나지 않은 사례, 처리량 기반 제출 예산, 기록기의 이차 비용 | CH-11, `9836997`, `f74b427` |
| 3.9 (신규) | 셰이더 컴파일러와 성능 이식성 | 실행되지 않는 루프 하나로 Mali r38에서 12배 감속, Turnip 1.6배. Mali OpenCL 컴파일러가 짧은 루프를 펼치지 않아 배열이 메모리에 남는 사례. 절제 측정 방법 | CH-04, CH-08, CH-09 |
| 3.10 (신규) | Vulkan과 OpenCL의 메모리 공유 | AHardwareBuffer + `cl_arm_import_memory`: A53·A35에서는 정상, S21(r38p0)에서는 64바이트 단위로 오래된 값을 읽음. 동기화 방법 10가지 모두 실패 | CH-09, `cc463f6` |
| 7.6 (신규) | Termux에서 벤더 GPU 드라이버 적재 | ocl-icd의 −1001, `LD_LIBRARY_PATH`가 시스템 라이브러리를 가리는 문제(`Xzs_Construct`, `egl_get_connection`), `android_load_sphal_library`와 벤더 링커 네임스페이스, Termux clang의 RUNPATH 순서, `linker64 --list` | CH-07, `8a2c63f` |
| 7.5 (보강) | 플러그인 ABI 안정성 | 같은 API 버전에서 vtable 슬롯이 밀린 사례, 이름 지정 초기화, 적재 시점 ABI 탐침, 시험 0건 거짓 통과 | CH-07 |
| 1.2 (보강) 또는 부록 C | ARM64 CPU 기능 탐지와 바이너리 배포 | HWCAP·HWCAP2(i8mm, dotprod, fp16), i8mm 빌드가 S21·A35에서 SIGILL로 종료한 사례 | CH-07 |
| 2.4·5.4 (보강) | 전원 상태가 연산 처리량에 미치는 영향 | CPU 유휴 시 Mali OpenCL 최대 2배 감속(스핀 대기로 해결), ssh 작업 중 S25 절전으로 약 4배 감속, Termux 웨이크락, A35 ssh 세션의 코어 집합(A55만) | CH-09, CH-17 |
| 부록 C (신규) | 실기기 측정 프로토콜 | 같은 분 안 교대 측정, 코어 집합·GPU 클럭 기록, 프롬프트와 생성 분리, 출력 비교 없는 PASS 금지, 종료 코드는 성공의 증거가 아님, 거짓 통과 탐지 | CH-17, CH-18 |
| 부록 D (신규) | 온디바이스 추론의 수치 정합성 판정 | CPU 자체 편차 기반 판정(BitNet KLD 하한, 비전 상대 L2 하한, TTS LSD 하한), f16 KV 컨텍스트 이동 잡음, 출력 뷰 판독 함정 | CH-10, CH-13, CH-14, CH-16 |

### 7.2 외부 문헌 인용이 필요한 일반 주제

이 프로젝트에서 측정하지 않은 내용이므로 공식 문서를 인용해 써야 합니다.

- 갱신 가능한 그래픽 드라이버(Android 10 이후)와 기기별 드라이버 버전 차이
- Android Dynamic Performance Framework(성능 힌트, 열 여유 API)와 장시간 추론
- NNAPI의 지원 중단 이후 온디바이스 추론 경로(비공개 초안 3.6과 연결)

### 7.3 기존 장의 정리 과제 (감사 결과)

- 코드 셀 32편이 리터럴 값을 assert하고 "PASS"를 출력합니다. "예시(시뮬레이션), 측정 아님"으로 표시합니다.
- 영문판 23편 끝의 상투 부록("Extended Engineering Appendix")을 삭제합니다.
- 3.4의 제로 카피 설명에 비캐시 매핑 조건(흩어진 CPU 접근이 6~9 MB/s)을 더합니다. #45의 "제로 카피 전송 오버헤드 ≈ 0"도 같은 조건이 빠져 있습니다.
- 상세 목록: [LABS_CONTENT_AUDIT_2026-10-11.ko.md](LABS_CONTENT_AUDIT_2026-10-11.ko.md)
