# Mali OpenCL 프롬프트 경로 실모델 연결 설계 (안 B 최소 범위)

- 작성: 2026-10-05 (사용자 승인: 안 B 최소 범위 = 프롬프트 경로 우선 OpenCL화, 브랜치 `perf/opencl-mali`)
- 선행 문서: `docs/design/opencl-mali-int8.md`(커널 프로토타입과 실측). 이 문서는 그 6장 3번(B의 최소 범위)을 실행 설계로 구체화합니다.
- 목표: S21(Mali-G78)·A53(Mali-G68)에서 블록 정수 누적 타일을 **실제 모델 프롬프트에 연결하여 프롬프트 t/s와 CPU 대비 KL 발산**을 측정한다. 판정 기준은 Vulkan 경로와 같다(`kld_mean <= 0.01`, `verify_kv.sh`).

## 1. 현재 구조 (현물 확인)

- `libac_runtime.so.2`(ggml-ameva가 링크하는 층)는 **Vulkan 전용**이다(`include/ac_runtime.h`: `backend_kind`는 `AC_BACKEND_VULKAN`이어야 함).
- ac_runtime 하위 `amuda_rt`에는 **레거시 OpenCL 엔진**(`src/amuda/`, `libamuda`)과 어댑터(`src/amuda_rt/amuda_rt_opencl.c`, `AMU_BACKEND_OPENCL`)가 있으나, ggml-ameva가 쓰는 상위 AC 경로에는 OpenCL 백엔드가 없다.
- 프로토타입 OpenCL 커널(`adapters/llama/ggml-ameva/opencl/ameva_cl_q4k.cl`)은 ac_runtime을 거치지 않는 독립 벤치(`tests/ameva_cl_bench.c`)로만 돈다.
- 백엔드가 프롬프트 그래프에서 GPU로 수용하는 연산(`ggml-ameva.cpp` `supports_op`): MUL_MAT, ADD/SUB/MUL/DIV, SCALE, UNARY, GLU, RMS_NORM, ROPE, SET_ROWS, GET_ROWS, CPY/CONT/DUP, FLASH_ATTN_EXT, 그리고 뷰 계열.

## 2. 설계 원칙: 수학 검증과 실행 이식을 분리한다

블록 정수 누적 커널에는 두 개의 독립된 질문이 있다. 섞으면 둘 다 불명확해진다.

1. **수학(품질)**: 활성값 스케일을 32개 단위가 아니라 256개 블록마다 하나로 두면(ggml의 Q8_K) 모델 품질이 떨어지는가? 프로토타입 난수 데이터에서는 역양자화 대비 거리가 블록 1.2e-3 대 단위 6.8e-4였다(단위마다 크기를 다르게 만든, 블록에 불리한 데이터). **실모델 가중치·활성값에서의 KL 발산은 아직 없다.**
2. **실행(속도)**: 그 커널을 OpenCL로 실모델 그래프에서 돌리면 프롬프트 t/s가 Vulkan f32 경로보다 빠른가? 커널 시간은 1.6~2.9배 빨랐으나 **그래프 전체의 t/s는 아직 없다.**

질문 1은 OpenCL 없이, 지금 있는 Vulkan/CPU 경로에서 **싸게** 답할 수 있다. 질문 2만 OpenCL AC 백엔드라는 큰 작업이 필요하다.

## 3. 단계

### 단계 A — 블록 양자화의 수학을 실모델에서 검증 (OpenCL 불필요, 먼저)

- 활성값을 256개 블록마다 하나의 스케일로 int8 양자화하는 경로를 **기존 백엔드(Vulkan) 또는 CPU 참조**에 선택 기능으로 추가한다. CPU 백엔드가 q4_K 가중치에 곱할 때 쓰는 Q8_K와 같은 형식이므로 큰 위험은 없을 것으로 **예상**하지만, 재기 전에는 결과가 아니다.
- `ameva_llm_parity`로 CPU(f16 활성, 참조) 대비 KL 발산을 잰다. Qwen2.5-0.5B, Llama-3.2-1B. 기준 `kld_mean <= 0.01`.
- 산출물: "블록 단위 활성 양자화가 모델 품질 기준 안인가"에 대한 수치. 기준을 넘으면 안 B 전체를 재검토한다(단위 스케일을 유지하되 로드를 줄이는 다른 형태로).
- 비용: 커널 1종(활성 양자화)과 파서 옵션. OpenCL 백엔드 불필요.

### 단계 B — OpenCL AC 백엔드의 프롬프트 경로 최소 구현 (속도)

단계 A가 기준을 통과하면 착수한다. 프롬프트 t/s는 그래프 전체가 한 큐에서 돌아야 정직하게 나온다(연산마다 Vulkan↔OpenCL을 오가는 안 A는 "그래프 한 번 제출" 구조를 깨므로 기각됨, 선행 문서 6장).

- **AC OpenCL 백엔드**: Vulkan 백엔드가 구현하는 AC 표면(장치/버퍼/커널 로드/디스패치, `AcDeviceLimits`)을 OpenCL로 미러링한다. `amuda_rt`의 레거시 OpenCL 엔진을 재사용할지, 새 CL 경로를 둘지는 그 표면의 적합성으로 정한다(레거시 엔진의 커널 로드·버퍼 수명·동기화가 AC 계약과 맞는지 먼저 확인).
- **프롬프트 경로 연산 이식**: 프롬프트 그래프가 GPU로 돌리는 연산 중 프롬프트에 실제로 쓰이는 것부터. 최소 집합은 MUL_MAT(블록 정수 타일), RMS_NORM, ROPE, GLU(SwiGLU), ADD/MUL/SCALE, CPY/CONT, FLASH_ATTN_EXT. GET_ROWS·SET_ROWS는 토큰당 1회로 CPU에 둘 수 있는지 확인한다.
- 수용 못 하는 연산은 **거절하지 않고 실패한다**(철학 1, Fail-Fast). 프롬프트 경로가 완성되기 전에는 OpenCL 백엔드를 기본값으로 두지 않는다.

### 단계 C — 실모델 측정과 판정

- `verify_kv.sh`와 같은 틀로 S21·A53에서 CPU 대비 프롬프트 t/s와 KL 발산, `test-backend-ops`(OpenCL 백엔드), 응답 텍스트를 잰다.
- 판정: 프롬프트 t/s가 CPU를 넘고 `kld_mean <= 0.01`. A53 기준선은 1B 프롬프트 GPU(Vulkan) 27 대 CPU 47~48 t/s이므로, 블록 정수 타일의 커널 2.9배가 그래프 t/s로 얼마나 전달되는지가 관건이다.
- 토큰 생성 경로의 OpenCL 실행은 이 결과를 본 뒤에 정한다.

## 4. 위험

- **그래프 오버헤드가 커널 이득을 먹는다**: 커널이 2.9배라도 제출·동기화·타입 변환이 그래프 시간을 지배하면 t/s 이득이 작을 수 있다. 단계 A의 수학이 통과해도 단계 B의 t/s가 CPU를 못 넘을 수 있다. 그 경우 정직하게 "커널은 빠르나 그래프는 못 넘었다"로 보고한다.
- **레거시 OpenCL 엔진의 적합성**: `libamuda`가 AC 계약(버퍼 수명, 비동기 디스패치, 파이프라인 캐시)을 만족하지 못하면 OpenCL 백엔드를 새로 써야 하고 범위가 커진다. 단계 B 착수 전에 이 적합성부터 확인한다.
- **기기별 커널 조합**: 작업 항목당 행 수·로컬 크기가 기기마다 다르고 3배까지 차이 난다(선행 문서 4.3). GEMV 레이아웃처럼 기기에서 측정해 고르는 선택 단계가 백엔드에 필요하다.
- **드라이버별 수치**: OpenCL 커널의 변환·반올림이 드라이버마다 다를 수 있다. `ameva_qmm_check`에 해당하는 OpenCL 정합성 검사를 두 기기에서 돌린다.

## 5. 다음 한 걸음

단계 A부터 착수한다(OpenCL 없이, 위험 가장 낮고 안 B 전체의 전제를 가른다). 구체적으로:
- 블록 단위 활성 양자화를 기존 경로에 선택 기능으로 추가하고 `ameva_llm_parity`로 S21·A53에서 CPU 대비 KL 발산 측정.
- 결과가 기준 안이면 단계 B(레거시 OpenCL 엔진 적합성 확인 → AC OpenCL 백엔드 → 프롬프트 연산 이식)로 넘어간다.

이 설계는 2.8.1 게시가 끝난 뒤 본격 구현한다(사용자 결정: 2.8.1 우선).
