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

#### 단계 A 결과 (2026-10-06, 통과)

구현: ggml-ameva에 `AMEVA_ACT_FQ`(shaders/ameva_act_fq.comp). 양자화 가중치 MUL_MAT의 활성값을 블록 단위로
int8 반올림·복원(d = amax/127, ties-to-even, CPU의 quantize_row_q8_K / q8_0과 같은 규칙)한 뒤 f32 커널이 읽는다.
`256` = 256-원소 수퍼블록 타입(q4_K·q6_K…)만 Q8_K, `cpu` = CPU 백엔드 선택(수퍼블록 256, 나머지 32), `32` = 전부 32.
f32 커널만 쓰도록 `AMEVA_MMQ=0 AMEVA_MMVQ=0`을 요구(아니면 abort). 기존 커널 SPIR-V 194개 불변, `act_fq` 1개 추가,
`ameva_qmm_check` 516/516(S25 빌드). 측정: ameva-runtime `tools/fleet/lab/actfq_matrix.sh`(parity_long.sh, 문맥 256,
16위치, CPU 대비). 근거: ameva-runtime `docs/handover/evidence/20261006-stageA-actfq/`.

| CPU 대비 kld_mean (기준 0.01) | 활성 f32 (off) | Q8_K (256) | CPU 방식 (cpu) |
| :--- | :---: | :---: | :---: |
| S21 Qwen2.5-0.5B 배치 / 토큰 단위 | 4.21e-3 / 4.19e-3 | 5.10e-3 / 4.96e-3 | 4.71e-3 / 4.79e-3 |
| S21 Llama-3.2-1B 배치 / 토큰 단위 | 2.32e-3 / 2.31e-3 | 3.55e-3 / 2.30e-3 | 3.55e-3 / 2.30e-3 |
| A53 Qwen2.5-0.5B 배치 / 토큰 단위 | 4.21e-3 / 4.22e-3 | 5.10e-3 / 4.27e-3 | 4.71e-3 / 4.29e-3 |
| A53 Llama-3.2-1B 배치 / 토큰 단위 | 2.32e-3 / 2.31e-3 | 3.55e-3 / 2.80e-3 | 3.55e-3 / 2.80e-3 |

- **모든 조합이 기준 안**(최대 5.10e-3, 기준의 절반). top-1 93.75~100 %(off와 같음). CPU 자신의 배치·토큰 단위 경로 사이
  거리가 Qwen 4.76e-3, Llama 3.30e-3이므로, 블록 int8 활성의 GPU 결과는 CPU의 자기 경로 간 차이와 같은 수준이다.
- f32 활성 대비 변화는 +0.0~1.2e-3. perplexity(16위치)는 Qwen 9.32~9.48(CPU 9.50~9.53), Llama 18.40~18.87(CPU
  18.86~19.71)로 CPU의 두 경로 범위와 겹친다.
- 배치 결과는 S21과 A53에서 비트 단위로 같다(같은 f32 타일 커널). 토큰 단위는 기기별 GEMV 레이아웃 차이로 조금 다르다.
- Llama는 전부 수퍼블록 타입이라 `256`과 `cpu`가 같다. Qwen은 q5_0·q8_0(K=896)도 32블록으로 맞춘 `cpu`가 CPU에 더 가깝다.
- 예상과 다른 점: CPU의 Q8_K와 같은 반올림을 넣었는데도 배치 KLD가 f32 활성보다 조금 올랐다(하강 예상). 원인(CPU 배치
  경로의 다른 활성 처리, 연산별 오차의 상쇄 변화)은 확인하지 못했다. 판정에는 영향 없음.
- **결론: 블록 정수 누적 커널이 계산하는 수학은 실모델에서 기준을 통과한다. 단계 B(OpenCL 실행)로 진행한다.**

### 단계 B — OpenCL AC 백엔드의 프롬프트 경로 최소 구현 (속도)

단계 A가 기준을 통과하면 착수한다. 프롬프트 t/s는 그래프 전체가 한 큐에서 돌아야 정직하게 나온다(연산마다 Vulkan↔OpenCL을 오가는 안 A는 "그래프 한 번 제출" 구조를 깨므로 기각됨, 선행 문서 6장).

- **AC OpenCL 백엔드**: Vulkan 백엔드가 구현하는 AC 표면(장치/버퍼/커널 로드/디스패치, `AcDeviceLimits`)을 OpenCL로 미러링한다. `amuda_rt`의 레거시 OpenCL 엔진을 재사용할지, 새 CL 경로를 둘지는 그 표면의 적합성으로 정한다(레거시 엔진의 커널 로드·버퍼 수명·동기화가 AC 계약과 맞는지 먼저 확인).
- **프롬프트 경로 연산 이식**: 프롬프트 그래프가 GPU로 돌리는 연산 중 프롬프트에 실제로 쓰이는 것부터. 최소 집합은 MUL_MAT(블록 정수 타일), RMS_NORM, ROPE, GLU(SwiGLU), ADD/MUL/SCALE, CPY/CONT, FLASH_ATTN_EXT. GET_ROWS·SET_ROWS는 토큰당 1회로 CPU에 둘 수 있는지 확인한다.
- 수용 못 하는 연산은 **거절하지 않고 실패한다**(철학 1, Fail-Fast). 프롬프트 경로가 완성되기 전에는 OpenCL 백엔드를 기본값으로 두지 않는다.

#### 단계 B 사전 측정: Vulkan과 OpenCL의 메모리 공유 (2026-10-06, S21에서 불가)

OpenCL 백엔드 전체 대신 "Vulkan 그래프 + MUL_MAT만 OpenCL"(같은 메모리를 두 API가 공유, 복사 없음)이 가능한지 먼저 쟀다.
가능하면 이식할 연산이 MUL_MAT 하나로 줄어든다.

- 공유 수단(현물 확인): Mali 드라이버는 Vulkan 메모리를 dma_buf로 내보내지 못하고 AHardwareBuffer(AHB)로만 내보낸다.
  OpenCL은 `cl_arm_import_memory_android_hardware_buffer`로 AHB를 가져온다. Vulkan 매핑 주소의 호스트 가져오기는 거부된다
  (`tests/ameva_vk_cl_interop_probe.c`, -6). AHB로 내보낼 수 있는 메모리 타입은 모두 host-visible이다(0x7, 0xB).
  가져온 메모리에 대한 `clEnqueueMapBuffer`·`clEnqueueWriteBuffer`는 -59(CL_INVALID_OPERATION)로 거부된다.
- 시험(`tests/ameva_vk_cl_gpu_handoff_probe.c`, 256 MB × 10라운드): 라운드마다 앞 절반에 새 값 e를 쓰고(생성), OpenCL 커널이
  `2·a + 라운드 + 1`을 뒤 절반에 쓰고, Vulkan 커널이 두 절반을 검사해 **공유되지 않은 별도 버퍼**에 판정을 쓴다.
  오류는 본 값으로 분류한다(OpenCL이 이전 라운드 입력을 읽음 / Vulkan이 이전 라운드 OpenCL 결과를 읽음 / Vulkan 입력 오류).
  생성 경로 5종: gpu(Vulkan 커널이 씀, 그래프의 활성값과 같은 경우, 호스트는 공유 메모리에 접근하지 않음), cpu, launder,
  staged(vkCmdCopyBuffer), compute(셰이더 복사).

| 기기 (GPU, 드라이버) | 결과 |
| :--- | :--- |
| A53 (Mali-G68 r1p1, r32p1) | 5개 경로 14/14 통과 (이전 판 포함 추가 12회 통과) |
| A35 (Mali-G68 r1p1, r38p1) | 5개 경로 11/11 통과 |
| S21 (Mali-G78 r0p1, r38p0) | **모든 경로·모든 동기화 변형에서 실패** (회당 1~115개 64바이트 라인) |

- S21의 실패는 거의 전부 **"OpenCL이 이전 라운드의 입력을 읽음"**이다. 값이 64바이트 라인 단위로 틀리고, 같은 라인이 이어지는
  라운드에서 반복되기도 한다. 반대 방향(OpenCL이 쓴 결과를 Vulkan이 읽음)은 결과 버퍼를 분리한 뒤 오류가 없었다.
- S21에서 효과가 없었던 동기화 변형(모두 각 3~12회): 호스트 대기 뒤 50 ms 지연, `VK_QUEUE_FAMILY_FOREIGN_EXT` 소유권 이전,
  매 라운드 재가져오기, `clEnqueueMigrateMemObjects`, 커널 시작의 device-scope acquire 펜스, 일반 CL 버퍼 map/unmap,
  host-cached 메모리 타입, dma_buf를 `CL_IMPORT_DMA_BUF_DATA_CONSISTENCY_WITH_HOST_ARM`으로 가져오기(OpenCL 시간 약 6배),
  volatile 로드, 외부 세마포어(Vulkan sync_fd → `cl_khr_external_semaphore_sync_fd`, 호스트 대기 없음).
- 지연이 효과가 없으므로 타이밍 경쟁이 아니라 S21의 OpenCL 작업이 다른 API가 쓴 라인을 무효화하지 않는 것으로 **추정**한다.
  같은 r38 계열 드라이버의 A35(G68)는 통과하므로 G78 또는 r38p0 고유 문제로 보이나, 원인은 확인하지 못했다.
- **결론: 두 API가 메모리를 공유하는 혼합 구조는 S21에서 결과가 틀린다. 단계 B는 API 간 메모리 공유 없이 구성한다.**
  A53·A35에서는 공유가 성립하지만, 기기별로 구조를 나누면 S21이 빠지므로 기본 설계로 삼지 않는다.

#### 단계 B 사전 측정 2: CPU가 가중치를 갖고 큰 MUL_MAT만 GPU로 보내는 구조 (2026-10-06, 기각)

공유 없이 가능한 다른 구조는 ggml 스케줄러의 op offload다(`-ngl 0`, 가중치는 CPU 메모리, GPU 백엔드의 `offload_op`가
true인 MUL_MAT은 가중치를 매 그래프 복사해 GPU에서 실행). OpenCL 백엔드를 이 위에 만들기 전에 구조의 비용을 기존 Vulkan
백엔드로 쟀다. 구현: `AMEVA_OFFLOAD=<행 수>`(ggml-ameva-device.cpp, 기본 꺼짐). 측정: ameva-runtime
`tools/fleet/lab/offload_bench.sh`(프롬프트 약 370토큰, 디코드 64토큰, 번갈아 2회, 2.8.1 번들). 반복 배치(repack)된 CPU
가중치는 host 버퍼가 아니어서 offload 대상이 아니므로 `-nr`이 필요하다.

| t/s (프롬프트 / 디코드) | CPU | CPU `-nr` | offload | GPU 전체 |
| :--- | :---: | :---: | :---: | :---: |
| A53 Qwen2.5-0.5B | 55.3 / 22.6 | 48.3 / 17.2 | 15.9 / 17.1 | 67.1 / 14.6 |
| A53 Llama-3.2-1B 프롬프트 | 47.7 | 26.5 | 6.5 | 27.1 |
| S21 Qwen2.5-0.5B 프롬프트 | 120.6 | 95.0 | 41.6 | 221.6 |
| S21 Llama-3.2-1B 프롬프트 | 53~68 | 25~33 | 7.8 | 46~50 |

- S21은 이 시간대에 발열로 흔들렸다(같은 모드의 두 회가 최대 2배 차이). 위 S21 값은 구조 비교용이며 기준선이 아니다.
- offload는 GPU 전체 경로보다 3.5~6배 느렸다. 재배열되지 않은 가중치를 매 그래프 복사해 범용 커널(`ameva_qmm`)이 처리하고,
  스케줄러가 인접 연산까지 GPU로 옮겨 분할이 195~338개가 된다(Qwen: GPU 그래프 213개, 그래프당 GPU 78~101 ms).
- `-nr`은 A53 CPU 디코드를 24 %, Llama 프롬프트를 44 % 낮춘다. 출력 텍스트는 모든 모드에서 같았다.
- **기각 이유**: 토큰 생성이 구조상 CPU에 남고(GPU 경로가 목표라는 원칙과 충돌), 그 CPU 경로도 `-nr` 때문에 느려진다.

#### 단계 B 개정 (2026-10-06): ac_runtime에 OpenCL 백엔드를 두고 그래프 전체를 그 위에서 실행

두 사전 측정에 따라, Mali의 정수 내적을 쓰려면 **한 API(OpenCL) 위에서 그래프 전체(프롬프트와 토큰 생성)를 실행**한다.
원래 단계 B의 "AC OpenCL 백엔드"와 같고, 범위가 토큰 생성까지 넓어진다.

- 장치 기능(실측, S21 r38p0·A53 r32p1 공통): OpenCL 3.0, 거친 단위 SVM(세밀 단위 없음), SPIR-V 1.0 IL, 기준 주소 정렬
  128바이트, 로컬 메모리 32 KB, 최대 워크그룹 512, 커널 인자 최대 1 KB, 서브그룹, `arm_dot_acc`(S21은 표준 내적도 있음).
- ac_runtime: `AC_BACKEND_OPENCL`. 버퍼는 SVM(바인딩 오프셋 = 포인터 + 오프셋), 커널은 OpenCL C 소스와 빌드 옵션,
  푸시 상수는 값으로 넘기는 구조체 인자, 배치는 in-order 큐(배리어는 큐 순서). 거친 단위 SVM은 호스트가 포인터에 직접
  쓰는 것을 허용하지 않으므로, 백엔드 중립의 호스트 접근 API(구간 쓰기·읽기·채우기)를 추가하고 ggml-ameva가 그것을 쓴다.
- ggml-ameva: 그래프 엔진과 정합성 시험(`ameva_qmm_check`, kvq, chunk, fuse, `test-backend-ops`)은 그대로 두고, 각 GLSL
  커널과 같은 바인딩·푸시 상수 배치의 OpenCL C 커널을 둔다(셰이더 27개 약 3,300줄, 서브그룹 연산 없음). MUL_MAT은 블록 정수
  커널(`opencl/ameva_cl_q4k.cl`)을 쓴다.
- 순서: (1) ac_runtime OpenCL 백엔드와 KAT, (2) 호스트 접근 API로 ggml-ameva 전환(Vulkan 결과 불변 확인), (3) 연산 커널
  이식과 시험 통과, (4) 블록 정수 MUL_MAT, (5) S21·A53 실모델 프롬프트·토큰 생성 t/s와 KLD.

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
