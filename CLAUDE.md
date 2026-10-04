# AMEVA-COMPUTE & AMUDA Master Engineering Specification (CLAUDE.md)

> 본 문서는 Claude Code CLI 및 모든 에이전트가 본 프로젝트에서 반드시 100% 준수해야 하는 절대적 엔지니어링 행동 강령, 철학, 어조 규약, 실기기 테스팅 하네스 가이드라인입니다.

---

## 1. 어조 및 커뮤니케이션 프로토콜 (Strict & Dignified Tone)

1. **엄격·근엄·진지(엄근진) 및 품격 있는 정중한 어조**:
   - 가볍거나 장난스러운 말투, 과도한 감탄사/추임새, 격양된 어조, 자극적인 수사를 철저히 배제한다.
   - 차분하고 점잖으며 엄격·근엄·진지한 최고급 엔지니어링 표준 존댓말(-하십시오, -습니다)을 항시 유지한다.
2. **기술적 객관성 및 비과장(Zero-Hype) 원칙**:
   - 뜬구름 잡는 찬사, 과장된 마케팅적 수사, 허세, 자극적 표현, 허위·부풀리기 표현을 전면 금지한다.
   - 시스템 아키텍처, 성능 지표, 코드 무결성, 보안 한계 및 경계 조건을 냉정하고 정확한 사실(Ground Truth)에 기반하여 분석·보고한다.
3. **오픈소스 컴플라이언스(OpenSSF/CNCF/Apache-2.0) 준수**:
   - 문서 및 코드 작성 시 '100%', 'Unforgeable', 'Guaranteed', 'Zero-ReDoS', '0ms', 'Zero-Heap' 등 절대적·법적 위험 표현을 일절 배제한다.
   - 대신 'Validated by automated checks', 'Cryptographically authenticated', 'Low allocation footprint' 등 중립적·검증 가능한 기술 표준 용어만을 사용한다.
4. **Just Chat Mode (저챗 모드)**:
   - 사용자가 명시적으로 "저챗!" 이라고 외치거나 저스트 채팅 모드를 요구하는 경우에만 친근한 일상 대화 모드로 즉시 전환한다. 코딩/명령 실행을 멈추고 필터링 없이 진실과 깊이 있는 대화에 집중한다.

---

## 2. 핵심 엔지니어링 철학 및 행동 강령

### [철학 1] Zero-Silent-Fallback & Fail-Fast
- 미구현, 에러, 지원하지 않는 하드웨어 발생 시 절대로 침묵 폴백(Silent Fallback)하거나 가짜 데이터(Fake Data)를 반환하지 않는다.
- `try-except: pass`로 에러를 삼키는 행위를 엄격히 금지한다.
- 요청한 백엔드나 기능이 동작하지 못할 경우 즉각 명확한 에러 코드(`AMU_ERR_*`), 실패 원인, 스택 트레이스를 분출하고 안전 종료한다.

### [철학 2] Anti-Deception Code Enforcement (안티 기만 코드)
- 빈 함수(Stub), 상수만 리턴하는 가짜 함수, 의사 디커플링(Pseudo-decoupling) 래핑을 일절 금지한다.
- 모든 API는 100% C ABI 네이티브 직결(`ac_runtime`)로 동작해야 한다.

### [철학 3] Jira Development & Strict Tracking
- 모든 코드 수정, 버그 패치, 리팩토링, 테스트 실행은 사전에 승인된 태스크 기반으로 수행한다.
- 실시간으로 '무엇을 수정했고, 무엇을 검증했는지' 상세히 기록한다.

---

## 3. 실기기 원격 SSH 테스팅 하네스 (Fleet Harness)

사용자 환경은 5대의 실제 Android Termux 기기와 Tailscale 망으로 직결되어 있습니다.
터미널 실행 권한을 가진 Claude Code는 핑계를 대지 말고 직접 `ssh` 또는 `run_poc_fleet.ps1`을 실행하여 실기기 검증을 수행해야 합니다.

### 기기 플릿 구성
- **s25**: Samsung Galaxy S25 (Snapdragon 8 Elite / Adreno 830, Android 16) — `ssh s25`
- **s21**: Samsung Galaxy S21 (Exynos 2100 / Mali-G78, Android 15) — `ssh s21`
- **s20**: Samsung Galaxy S20 (Snapdragon 865 / Adreno 650, Mesa Turnip / OpenCL, Android 13) — `ssh s20`
- **a35**: Samsung Galaxy A35 (Exynos 1380 / Mali-G68, Android 16) — `ssh a35`
- **a53**: Samsung Galaxy A53 (Exynos 1280 / Mali-G68) — `ssh a53`

### 검증 명령어
```powershell
# 전체 플릿 5종 커널 자동 빌드 및 실기기 벤치마크
powershell -ExecutionPolicy Bypass -File scripts\fleet\run_poc_fleet.ps1 -InstallDeps

# 로그 확인 위치
evidence/poc_cuda_like/<timestamp>/
```

---

## 4. 현재 진행 중인 최우선 과제

- **[최우선] ggml-ameva를 ameva-runtime llama 번들의 네이티브 GPU 경로로 내장** (사용자 결정 2026-10-04):
  업스트림 llama.cpp Vulkan 백엔드에 의존하지 않는다. ameva-runtime 2.7.9 번들의 ggml-vulkan(b8471)은 플래시 어텐션을
  켜면 손상된 텍스트를 내고, 끄면 Mali에서 디스크립터 세트 assert로 중단한다(ameva-runtime `RELEASE_NOTES.md` Known issues 1).
  번들의 ggml(b8471, 0.9.8)에 맞춰 `libggml-ameva.so`와 `libac_runtime.so.2`를 빌드해 번들 `lib/`에 넣고,
  런타임 라우터에 AMEVA 경로(`GGML_BACKEND_PATH`, `--device AMEVA0`)를 추가한다. 판정은 5대 기기에서 CPU와 같은 텍스트이며 수치를 기록한다.
  계획과 진행 상태는 ameva-runtime `docs/handover/LATEST.md`.
- **[다음] 2026-10-03 개발분 실기기 검증**: MMQ(int8 정수 내적, `AMEVA_MMQ`), 양자화 KV 캐시(q8_0/q4_0),
  연산 융합(`AMEVA_FUSE`), 파이프라인 캐시(`AMEVA_PIPELINE_CACHE`). 로컬 검사(구문·셰이더 컴파일)만 통과한 상태이다.
  런북과 판정 기준은 `docs/handover/LATEST.md` §3. 기기 생성 커널 헤더(spirv-val 포함)로 저장소 헤더를 교체한다.
  MMQ 수용 기준(사용자 결정): CPU 대비 KLD 평균 ≤ 0.01.
- **[그다음]** 실측 결과에 따른 MMQ 기본값 정책, 융합 확대(QKV 합치기, ROPE+SET_ROWS), 양자화 KV 타입 확대.
- 현재 기준선(S25 Adreno 830, Qwen2.5-0.5B Q4_K_M, 커밋 a6eee0f): 디코드 59~67 t/s(CPU 35~50), 프롬프트 368토큰 287 t/s(CPU 59).
  측정 시 `Cpus_allowed_list`와 Adreno `max_gpuclk`를 함께 기록한다(조건에 따라 2~3배 차이).

### 종결 기록: Q5_0 native GEMV / 레이어 단위 GPU 그래프 실행 (2026-10-02~03 종결)
- 가중치 재배열(12개 양자화 타입) + 전용 GEMV(`ameva_qmv`)·타일 GEMM(`ameva_qgemm`)으로 대체했고, 그래프 전체를 배치 1회로 실행한다
  (RMS_NORM, ROPE, FA, SwiGLU 등 GPU 실행). 검증: `ameva_qmm_check`, `test-backend-ops`, 실모델 3종 정합성(인수인계서 참조).

### 종결 기록: ggml-ameva S25 `GGML_ASSERT(device)` (2026-10-02 종결)
- 원인: `libggml-ameva.so`가 실행 측 llama.cpp b8471(ggml 0.9.8)보다 새 ggml 헤더로 빌드되었다.
  ggml-backend-impl.h의 `ggml_backend_i`/`ggml_backend_buffer_i`에 슬롯이 추가되었는데도 `GGML_BACKEND_API_VERSION`은 2 그대로여서 로더가 거르지 못한다.
  그 결과 `backend->device`가 `iface.event_wait` 슬롯(NULL)에서 읽혔다.
- 수정: 이름 지정(designated) vtable 초기화, 로드 시점 ABI 탐침(`ameva_abi_check`, 불일치 시 명시적 거부), 장치/버퍼 타입 연결,
  `scripts/termux/build_ggml_ameva.sh`(정확한 ggml 지정 및 등록 검증).
- 검증: A35에서 S25 조건 재현(E1 assert), E2–E4(정상 및 양방향 거부), S25 설치본 번인 10/10, 출력 일치.
