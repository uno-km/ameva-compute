# AMEVA 인수인계서: 2.8.1 대비 비교, Mali 어텐션 회귀 복구, Vulkan 정수 내적 강제 호출, 폰별 자동 경로

- 작성: 2026-11-09 13:30 (KST). 직전 문서: `20261005-2310.md`
- 브랜치: `perf/opencl-mali` (HEAD `01d9f21`, 푸시 완료). 기본 브랜치 `amuda2-phase0-i2s-proof`보다 앞서 있습니다.

## 0. 2026-11-09 오후 추가분 (최신, 연구 노트 N-20261109-gap-closure-round1)

- 커밋(로컬, 푸시 안 함): `0c1db84` 자동 경로·자체 검사(아래 8장의 미커밋 4개 파일), `e0af61f` 재배열 배치 MUL_MAT 거절 수정·분할 도구·실험 행렬, `eeb4483` 드라이버 측 커널 조사 도구.
- `AMEVA_REPACK_ALL=1` 회귀(5대 공통 100건 실패) 수정. 수정 후 S25 4054/4054, S21 4053/4053, S20 4052/4052. **A35·A53은 수정 후 미측정.**
- 분할 실행(`tests/ameva_split_gen.cpp`): S25에서 GPU 프롬프트와 CPU 생성 속도를 함께 얻습니다. 메모리는 가중치 두 벌. 상태 이전은 무손실(S20 `xfer` 검사).
- 실험 행렬: S25(0.5B만 유효), S20(441 MHz), A35. 공식 ggml Vulkan은 장치 손실 또는 깨진 출력. A53·S21은 행렬 도중 Termux 종료. A35도 마지막 Gemma 회차 중 접속이 끊겼습니다.
- `VK_KHR_pipeline_executable_properties`: Turnip(IR3 디스어셈블리 포함)과 Adreno 830은 제공, Mali r38p0은 미제공. 제품 커널 264개 조사 결과: Turnip 넘침 21개, Adreno 830 0개.
- S21 행렬 완료(15:10~16:40): 강제 정수 내적 Vulkan이 프롬프트 최고, 0.5B·1.5B 생성은 CPU 4스레드가 최고. 같은 설정끼리 GPU 텍스트가 달라지는 현상(공유 튜닝 파일, RQ-63). S21은 분할 실행(가중치 두 벌) 중 세 번 끊겼습니다(메모리 압박 추정, 미확인).
- 남은 일: S25 전경(코어 0-7) 재측정(1.5B·Gemma), S21 4스레드 분할 재측정, A53 행렬, A35·A53 수정 후 재배열 시험, Mali R1(Mali Offline Compiler 설치 권한 대기), 비 LLM 모달리티 회귀, 긴 대화 텍스트.
- 기기 작업 파일: 각 폰 `~/abi-lab/gap/`(`mx.log`, `split.log`, `fix.summary`), S20·S25 `~/abi-lab/re/`(`re_dump`, `kout/`).

## 1. 요약

- 목표: 배포판 2.8.1(백엔드 `7e99154`) 대비 전체 속도 개선과, 폰별 최적 경로 도출입니다.
- 현재 위치: 어텐션 회귀를 복구해 푸시했습니다. 폰별 자동 경로와 Vulkan 강제 호출 자체 검사는 **미커밋**이며 5대에서 정합성을 확인했습니다. 배포된 것은 없습니다.
- 다음 담당자가 먼저 할 일: 7장의 결정 사항을 사용자에게 확인하고, 8장의 미커밋 변경을 검토해 커밋합니다. 그다음 4장의 실험 행렬을 채웁니다.

## 2. 완료·검증된 작업

### 2.1 Mali Vulkan 생성 회귀의 원인과 복구 (`01d9f21`)
- 원인: `ebccab6`(2026-10-09)가 `ameva_flash_attn.comp`를 RPI 배열 구조로 바꿨습니다. RPI 1 변형의 SPIR-V가 35,336 → 41,788바이트로 늘었고, S21 Qwen 0.5B에서 FLASH_ATTN_EXT GPU 시간이 271 → 471 ms(1.74배)가 되었습니다. Mali에서 느려지는 내부 기제는 미규명입니다.
- 16행 교차 배치(`359cc68`)는 원인이 아니었습니다. 직전 커밋 `4dcd92c`에서 이미 느립니다.
- 복구: RPI 1, DUNROLL 1 변형을 `shaders/ameva_flash_attn_r1.comp`(2.8.1 소스)로 빌드합니다. d64 모듈의 SHA-256이 2.8.1과 같습니다(`a5da21b3...`).
- 교체 시험(생성 t/s, 2.8.1 / 수정 전 / 2.8.1 어텐션 적용): S21 Qwen 0.5B 18.0 / 14.7 / 18.2, S21 Qwen 1.5B 8.3 / 6.7 / 9.0, A35 Qwen 0.5B 15.7 / 13.3 / 15.8, A35 Qwen 1.5B 6.8 / 5.7 / 7.1. S25·S20은 변화가 없습니다.

### 2.2 2.8.1 대비 최종 비교 (같은 시간대 5회차 교대, 354토큰 입력, 64토큰 생성)
근거: `evidence/ladder/rel_20261109b/` (git 제외, PC에만 있음), 도구 `scripts/fleet/release_ab.py`.

| 폰 | 경로 | 프롬프트 | 생성 | 코어 |
|---|---|---|---|---|
| S25 | Vulkan | +1~+15 % | 0~+6 % | 0-7 |
| S20 | Vulkan | −1~+12 % | −2~+18 % | 0-7 |
| A53 | Vulkan | −1~+10 % | +5~+12 % | 0-2 |
| A53 | OpenCL | +12~+148 % | +30~+53 % | 0-2 |
| A35 | Vulkan | −7~+16 % | 0~+6 % | 0-2 |
| A35 | OpenCL | −39~+16 % (Qwen 0.5B만 −39 %) | +15~+25 % | 0-2 |

S21은 수정 빌드로 2.8.1과 같은 시간대에 재지 못했습니다(측정을 중단). 2.4절의 Vulkan·강제·OpenCL 비교만 같은 시간대입니다.

### 2.3 크기 사다리 결과
- 토큰당 시간은 모델 크기에 대해 대체로 직선입니다(시간 = a + b·크기). 도구 `scripts/fleet/ladder_fit.py`, 근거 `evidence/ladder/20261108b/`.
- S21 Gemma 2 2B 생성의 GPU/CPU는 입력 15토큰에서 1.19, 354토큰에서 0.93입니다. 문맥 비용은 GPU 0.103, CPU 0.029 ms/토큰입니다(두 점 추정). 어텐션 복구는 OpenCL을 2~4 %만 개선했으므로 이 문맥 비용의 원인은 미규명입니다.
- **S25 코어 8개(`0-7`) GPU 대 CPU**: 생성은 CPU가 1.3~2배 빠릅니다(Qwen 0.5B GPU 50.9 / CPU 99.3, Gemma 19.6 / 24.7, Qwen 3B 16.6 / 22.7). 프롬프트는 GPU가 2.3~5.8배 빠릅니다. 일부 회차에서 클럭 상한이 832 → 734 MHz로 내려갔습니다. 근거 `evidence/ladder/s25_8core/`. 코어 4개 조건의 "모든 크기에서 GPU 우세"는 CPU 조건이 불공정했던 결과입니다.

### 2.4 Vulkan 정수 내적 강제 호출 (S21 같은 시간대 5회차, 858 MHz, 코어 0-2,4-7)
| 모델 | Vulkan / 강제 / OpenCL 프롬프트 | Vulkan / 강제 / OpenCL 생성 |
|---|---|---|
| Qwen 0.5B | 216 / 383 / 174 | 18.0 / 20.9 / 22.4 |
| Llama 1B | 94 / 204 / 188 | 16.2 / 18.5 / 18.0 |
| Qwen 1.5B | 67 / 154 / 130 | 9.0 / 9.9 / 10.9 |
| Gemma 2B | 42 / 98 / 85 | 6.6 / 7.3 / 7.4 |

- S21 Mali-G78 r38p0(Vulkan 1.1, 확장 미제공): 드라이버가 수용합니다. `ameva_qmm_check` 516/516(147건이 해당 경로).
- A53 Mali-G68 r32p1: 커널 생성이 거부됩니다(372/516, 실제 모델 중단).
- A35는 확장을 공식 제공하므로 강제가 해당되지 않습니다.
- 선례는 웹 검색 2회로는 찾지 못했습니다. 조사 깊이가 얕아 "선례 없음"으로 단정하지 않습니다.

### 2.5 폰별 자동 경로와 자체 검사 (미커밋, 8장)
- `AC_VK_FORCE_INT_DOT`: 미설정이면 Mali-G78 + 드라이버 메이저 38에서만, 16개 값의 정답 검사(KAT)를 통과할 때 켭니다. `=1`은 모든 기기(검사 실패 시 오류), `=0`은 끕니다. 켜지거나 실패하면 stderr에 이유를 출력합니다.
- `AMEVA_BACKEND` 미설정 또는 `auto`: Mali-G68은 OpenCL, 그 외는 Vulkan입니다. OpenCL 컨텍스트 실패 시 stderr에 알리고 Vulkan으로 진행합니다.
- `ameva_qmm_check` 516/516: S21·S20·A35·A53·S25 자동 경로. 2회 측정 사이 A35·A53의 480/516은 검사 도구가 자동 선택을 몰라서 난 판정 오류이며 도구를 수정했습니다.

### 2.6 정확도 (`parity_long.sh`, 2048토큰 문맥, KVOPS=128, 자동 경로)
| 폰 (경로) | 일반 구간 kld_mean | Qwen 0.5B 이동 후 | Llama 1B 이동 후 |
|---|---|---|---|
| S21 (강제 Vulkan) | 1.9e-3~2.1e-3 | **2.9e-2 FAIL** | 1.7e-3 PASS |
| S25 (Vulkan) | 1.6e-3~2.0e-3 | **2.2e-2 FAIL** | 1.4e-3 PASS |
| S20 (Vulkan) | 1.7e-3~1.9e-3 | **1.9e-2 FAIL** | 1.6e-3 PASS |
| A53 (OpenCL) | 1.8e-3~1.9e-3 | **2.3e-2 FAIL** | 1.6e-3 PASS |
| A35 (OpenCL) | 1.7e-3~2.2e-3 | **2.5e-2 FAIL** | 1.6e-3 PASS |

- CPU 자신의 한 토큰씩 실행 kld_mean은 1.8e-3~2.3e-3입니다. 캐시 재사용은 전부 0입니다.
- S21 Qwen 1.5B는 일반 구간 6.0e-3(CPU 5.8e-3), 이동 후 2.4e-2입니다.
- Qwen 이동 후 FAIL은 경로와 무관하게 5대 전부에서 나옵니다. 기준 재정의는 사용자 결정 대기입니다(7장). 긴 대화 텍스트(`text_soak.sh`)는 미측정입니다.

### 2.7 Mali-G68 블록 커널 기본값 (`5d5f0fc`)
G68은 `q4_K=2x8,q6_K=2x8,q5_K=2x8`, 그 밖의 Mali는 `q4_K=2x4,q6_K=1x8,q5_K=2x8`입니다. A53 Gemma 프롬프트 28.2~28.5 → 30.7~31.6 t/s(코어 0-7, 3회차).

## 3. 중단 지점

사용자 지시로 추가 측정을 멈췄습니다(2026-11-09 13:30).
- 5대 모두 실행 중인 측정이 없고, 절전 방지 잠금을 해제했습니다.
- 연구 전체의 인수인계는 ameva-foundation `03_core/docs/research/RESEARCH_HANDOVER_20261109.md`에 있습니다.

## 4. 다음 작업 (우선순위 순)

1. **미커밋 변경 검토 및 커밋** — 8장. 완료 기준: 푸시 후 키트 재빌드, 5대 `ameva_qmm_check` 516/516.
2. **실험 행렬 완성** — 폰 5대 × 모델(0.5B, 1.5B, 2B). 행: CPU만, 공식 ggml Vulkan, 공식 ggml OpenCL(Adreno만), 아뮤다 Vulkan, 아뮤다 OpenCL, GPU+CPU 분할(`-ngl` 층 분할, 프롬프트 GPU·생성 CPU). 추가 축: CPU 스레드 2·4·6·8, 짧은·긴 문맥, 5분 이상 지속 속도와 발열, 메모리, 정확도, 앱 전경·배경 코어 조건. 완료 기준: 폰별 최적 경로 표. 현재 미측정 칸: 공식 Vulkan·OpenCL 전부, GPU+CPU 분할 전부, S25 OpenCL 실제 모델, A35 공정 CPU, A35·A53 전경 코어 조건의 GPU 비교.
3. **S25 생성**: 프롬프트 GPU·생성 CPU 분할 구현 검토(RQ-25 계열).
4. **S21을 수정 빌드로 2.8.1과 같은 시간대 비교**, 긴 대화 텍스트 측정(`text_soak.sh`, `soak_scan.py`).
5. **커널 컴파일 결과 분석**: Arm Mali Offline Compiler로 RPI 배열 구조와 스칼라 구조, OpenCL과 Vulkan을 비교합니다. 필요 시 OpenCL 프로그램 바이너리 디스어셈블로 넘어갑니다.
6. **미규명 항목**: S20 Qwen 0.5B 생성 −2~−7 %, A35 Qwen 0.5B OpenCL 프롬프트 −39 %, S21 OpenCL 생성 이득의 주원인, GPU 문맥 비용.
7. **Qwen2.5 1.5B의 GPU 생성이 크기 대비 느린 현상**(RQ-56 후보).

## 5. 환경과 명령

- PC에서 키트: `BUILD_DIR=abi-lab/out-auto bash scripts/fleet/hw_study_fleet.sh kit`, `bash scripts/fleet/hw_study_fleet.sh push <dev>`. 기기의 키트 경로는 `~/abi-lab/hwkit/current`이고, `src/llm_plugin_ab2.sh`, `src/ladder_run.sh`는 키트를 올린 뒤 따로 복사해야 합니다.
- S21 빌드 폴더: `~/abi-lab/src-fa`(소스), `~/abi-lab/out-auto`(현재 산출물), `~/abi-lab/out-v281`(2.8.1 백엔드). 기기 배포본 2.8.1: `~/abi-lab/hwkit/v281`.
- 사다리: `LADDER_VARIANTS="v281:$V:vulkan cur:$K/bin:vulkan curcl:$K/bin:opencl cpu:$K/bin:cpu" CAP_TARGET=<kHz> MODELS="..." bash $K/src/ladder_run.sh vulkan`. 변형 넷째 칸에 `VAR=값;VAR=값`을 줄 수 있습니다.
- 클럭 상한: S25 832000, S21 858000, S20 587000, A35 949000, A53 897000.
- 정확도: 2.8.1 번들이 설치된 폰(S21·S20·S25)은 번들의 옛 런타임이 새 플러그인을 막습니다. `~/abi-lab/plong2.sh`를 `BLX=$HOME/abi-lab/hwkit/current/rt KVOPS=128 bash ~/abi-lab/plong2.sh <backend dir> <tag> 2048 <모델...>` 형태로 쓰십시오. A35·A53은 `parity_long.sh` 그대로 동작했습니다.

## 6. 알려진 문제와 주의사항

- **클럭 관문**: 모델마다 다시 읽으면 내려간 상한이 기준이 됩니다. `CAP_TARGET`을 고정하십시오.
- **코어 집합**: 앱이 배경이면 S25는 `0-1,4-5`, A35·A53은 `0-2`입니다. GPU 대 CPU 판정은 전경(`top-app`)에서만 하십시오.
- **파일 전송**: 휴대폰 → PC는 약 1 MB/s입니다. `timeout`이 짧으면 tar가 잘리므로 압축과 sha256 확인을 쓰십시오. `git archive`로 받은 스크립트는 CRLF일 수 있어 줄바꿈 변환이 필요합니다.
- **A53**: 측정 중 Termux가 네 번 종료되었고, 원인은 미규명입니다. A35에서 Qwen 3B는 메모리 부족으로 실행 불가에 가깝습니다(가용 640 MB).
- **A35의 `curl`**: 라이브러리 불일치로 동작하지 않습니다. Python `urllib`을 쓰십시오.
- **튜닝 파일**: GEMV 배치 조정 결과가 변형별로 다르면 생성 텍스트가 달라질 수 있습니다. `AMEVA_QMV_TUNE_FILE`을 변형별로 분리하십시오.

## 7. 사용자 협의가 필요한 결정

1. Qwen 0.5B 계열의 컨텍스트 이동 후 정확도 기준(현행 0.01, 5대 전부 초과). 제안: "배포된 Vulkan 경로보다 나쁘지 않을 것".
2. 강제 Vulkan과 OpenCL 자동 선택을 기본값으로 배포할지 여부. 강제 호출은 Vulkan 규격 밖 동작입니다.
3. A35의 경로: 현재 자동 선택은 OpenCL입니다(Qwen 0.5B 프롬프트 −39 %).
4. S25 생성에서 CPU가 빠른 문제의 처리(분할 또는 경로 선택).
5. 리버스 엔지니어링 범위: 공식 도구 → 우리 커널 디스어셈블 → 드라이버 분석 순으로 제안했습니다.

## 8. 미커밋 변경 목록

이번 작업분(커밋 대상):
- `adapters/llama/ggml-ameva/ggml-ameva.cpp`: `AMEVA_BACKEND` 자동 선택
- `src/backend/vulkan/ac_vk_context.cpp`: 정답 검사, 드라이버 허용 목록
- `src/backend/vulkan/ac_vk_int_dot_probe_spv.h`: 새 파일
- `tests/ameva_qmm_check.cpp`: 자동 선택 인식

이번 작업과 무관한 기존 변경(건드리지 않았습니다): `scripts/termux/amuda_poc.sh`, `scripts/termux/lab/*.sh` 21개(줄바꿈 차이로 보임), 서브모듈 `third_party/llama.cpp`, `third_party/whisper.cpp`.
