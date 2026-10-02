# AMUDA CUDA-like POC v0.1.2

- 작성: 2026-10-01, Claude (Cowork 세션). v0.1.1: 재검토 후 하드닝. v0.1.2: 첫 실기기 fleet 결과(20261001-214111) 반영, OpenCL 진단·샘플·trace 패치(§5, §6)
- 목표: Termux 안에서 CUDA처럼 커널을 쓰고, 컴파일하고, `<<<grid, block>>>` 방식으로 실행한다. 백엔드는 Vulkan과 OpenCL 두 가지다.
- 범위 밖: llama.cpp / whisper / sd.cpp 같은 엔진 통합. 이 작업은 `AMUDA_ENGINE_UNIFICATION_PLAN.md`로 보류한다.

## 1. 구성

| 경로 | 역할 |
| :--- | :--- |
| `tools/amudacc/amudacc.py` | CUDA 서브셋(.acu)을 커널마다 다음 두 가지로 번역한다. Vulkan용 GLSL 450 → SPIR-V, OpenCL용 OpenCL C 1.2. 결과는 둘 다 담긴 C 헤더 하나로 나온다. Python 표준 라이브러리만 써서 Termux에서도 돈다 |
| `include/amuda_rt.h`, `src/amuda_rt/*` | CUDA 런타임 비슷한 호스트 API다. Vulkan 백엔드는 기존 `ac_runtime`(AMUDA C ABI)을, OpenCL 백엔드는 기존 `libamuda`(legacy OpenCL 엔진)를 쓴다 |
| `samples/cuda_like/` | 샘플 5종: vector_add, saxpy, matmul_tiled(shared + syncthreads), reduce_sum(shared + float atomicAdd), histogram256(uint atomics). 모두 CPU 결과와 대조한다 |
| `scripts/termux/amuda_poc.sh` | 폰에서 소스 빌드 → 샘플 실행 → `AMU_RESULT`/`AMU_SUMMARY` 출력 |
| `scripts/fleet/run_poc_fleet.ps1` | PC에서 s25/s21/s20/a35/a53에 배포하고 실행한 뒤 로그를 `evidence/poc_cuda_like/<시각>/`에 모은다 |
| `scripts/fleet/parse_poc_logs.py` | 위 로그를 기기·백엔드·샘플별 표로 정리한다. 셰이더 컴파일 에러는 파일:줄까지 뽑는다 |
| `tests/amudacc/` | 번역기 단위 테스트 29개. `glsl_lint.py`는 생성된 GLSL의 미선언 식별자, 디스크립터 바인딩 중복, push constant 블록 수, `#line` 확장 순서를 정적으로 검사한다 |
| `tests/amuda_rt/` | 호스트 런타임 배선 테스트. 테스트 더블을 쓰며 GPU 실행은 하지 않는다 |
| `tools/audit/stub_scan.py` | 빈 함수, 상수만 반환하는 함수, `except: pass`, `\|\| true` 같은 실패 은폐 패턴을 찾는 스캐너 |

## 2. 커널 언어 (CUDA 서브셋)

| 지원 | 내용 |
| :--- | :--- |
| 커널 | `__global__ void name(...)`, `extern "C"`, `__launch_bounds__`(무시) |
| 파라미터 | `T*` / `const T*`(버퍼, `__restrict__` 허용), `T`(스칼라). T는 `float`, `int`, `unsigned int` |
| 내장 변수 | `threadIdx` / `blockIdx` / `blockDim` / `gridDim` 의 `.x/.y/.z` |
| 공유 메모리 | `__shared__ T name[N]...;` 정적 크기(정수·`#define`·`-D` 상수식). 커널당 합계 16384 B 이하만 허용한다. Vulkan이 모든 기기에 보장하는 최소 한도이기 때문이다 |
| 동기화 | `__syncthreads()`, `__threadfence()`, `__threadfence_block()` |
| 원자 연산 | int/uint: `atomicAdd/Sub/Exch/Min/Max/And/Or/Xor/CAS` (전역·공유 모두). float: 전역 버퍼 `atomicAdd`만 지원한다. 단독 문장으로만 쓸 수 있고, 그 버퍼는 해당 커널에서 `atomicAdd`로만 접근해야 한다. Vulkan에서는 uint 저장소에 CAS 루프로 구현하며, 디스크립터 별칭은 쓰지 않는다 |
| 함수 | `__device__` / `__forceinline__` 함수(값 전달), 프로토타입 |
| 전처리기 | `#define`(내장 변수 포함 가능), `#if` 계열, `#pragma unroll`(OpenCL만 반영) |
| 수학 | `sqrtf rsqrtf expf __expf exp2f logf log2f powf sinf cosf tanf tanhf asinf acosf atanf atan2f fabsf fminf fmaxf floorf ceilf truncf roundf fmodf fmaf __fdividef __saturatef isnan isinf` |

**미지원 — 쓰면 줄번호와 함께 컴파일 에러가 난다**

- **포인터 관련:** 포인터 연산·포인터 변수(버퍼는 `buf[i]`로만 접근), struct/union/typedef/template
- **타입:** double, long, half
- **메모리:** 전역 변수, `__constant__`, `extern __shared__`(동적 공유 메모리)
- **warp:** `__shfl*`, warp 투표 함수
- **기타:** printf, `#include`, 텍스처
- **GLSL 규칙상 주의:** 조건식은 bool이어야 한다. `if (n)` 대신 `if (n != 0)`을 쓴다. 이 규칙을 어기면 SPIR-V 컴파일 단계에서 원본 파일과 줄번호를 담은 에러가 난다.

**이름 처리:** GLSL이나 OpenCL 예약어와 겹치는 사용자 이름(`input`, `output`, `in`, `out`, `local`, `length` 등)은 생성 코드 안에서만 `amu_u_` 접두사로 바뀐다. 호스트 쪽 이름은 그대로다.

## 3. 사용법 (Termux)

```bash
pkg install clang python glslang vulkan-headers vulkan-loader-android
bash scripts/termux/amuda_poc.sh                       # 빌드 + 샘플 5종 × (vulkan, opencl)
AMUDA_BACKEND=vulkan build-poc/bin/matmul_tiled 1024   # 개별 실행
```

새 커널을 직접 만들 때:

```bash
python3 tools/amudacc/amudacc.py my_kernel.acu -o my_kernel.acu.h   # Vulkan+OpenCL 이미지
clang -O2 -I include -I . host.c -L build-poc/lib -lamuda_rt -Wl,-rpath,build-poc/lib -o host
```

```c
#include "my_kernel.acu.h"
amuContext* ctx;   amuInitFromEnv(&ctx);                 // AMUDA_BACKEND=vulkan|opencl, 기본 vulkan
amuBuffer* x;      amuMalloc(ctx, n * sizeof(float), &x);
amuMemcpyHtoD(x, 0, host_x, n * sizeof(float));         // 또는 amuHostPtr(x)로 직접 접근(UMA)
amuFunction* f;    amu_load_my_kernel(ctx, &f);
amu_launch_my_kernel(f, amuDim(blocks,1,1), amuDim(256,1,1), x, n);
amuDeviceSynchronize(ctx);
amuMemcpyDtoH(host_x, x, 0, n * sizeof(float));          // CPU가 여러 번 읽을 데이터는 복사해서 쓴다
```

`amuHostPtr`가 주는 매핑은 Mali/Adreno에서 uncached일 수 있다. CPU가 그 포인터로 반복 계산을 하면 메모리 매핑 속도를 재게 된다. 첫 fleet 실행에서 `cpu_ref_ms`가 비정상적으로 컸던 원인이다. 샘플은 v0.1.2부터 호스트 메모리에서 입력을 만들고 복사한다.

## 4. 실행 의미론

- **백엔드:** 요청한 백엔드로만 실행한다. 초기화가 실패하면 `AMU_ERR_BACKEND_UNAVAILABLE`을 돌려주고, CPU로 대신 실행하지 않는다. CPU 타입 Vulkan 장치(llvmpipe)는 `ac_runtime`이 거부한다.
- **실행 순서:** 한 컨텍스트 안의 launch는 발행 순서대로 실행된다(CUDA 기본 스트림과 같음). 다음 launch는 이전 launch가 끝난 뒤 디스패치된다.
- **메모리 가시성:** launch 전에 CPU가 쓴 내용은 커널이 볼 수 있다. non-coherent 메모리면 flush를 자동으로 한다. 커널 결과는 `amuDeviceSynchronize` 뒤에 보인다(invalidate 자동).
- **블록 크기:** Vulkan에서는 블록 크기를 specialization constant로 넘긴다. 모듈마다 블록 모양별로 파이프라인을 만들어 캐시하며, 최대 8가지 모양까지 가능하다.
- **타임아웃·장치 소실:** 60초 타임아웃이나 장치 소실이 나면 컨텍스트가 오염 상태가 되고, 이후 모든 호출이 같은 에러를 돌려준다.

## 5. 검증 현황

### 5.1 실기기 fleet 1차 (evidence/poc_cuda_like/20261001-214111)

| host | 기기 / SoC / Android | GPU (실제 실행 장치) | Vulkan 5종 | OpenCL 5종 |
| :--- | :--- | :--- | :--- | :--- |
| s25 | SM-S931N / SM8750 / 16 | Adreno 830 (Qualcomm 드라이버) | **5/5 PASS** | 5 ERROR: `ac_init` −104 |
| s21 | SM-G991N / Exynos 2100 / 15 | Mali-G78 | **5/5 PASS** (wall 이상치, 아래) | 5 ERROR: `ac_init` −104 |
| s20 | SM-G986N / SM8250 / 13 | Vulkan: **Mesa Turnip** Adreno 650 / OpenCL: Qualcomm Adreno | **5/5 PASS** | **5/5 PASS** |
| a35 | SM-A356N / s5e8835 / 16 | Mali-G68 | **5/5 PASS** | 5 ERROR: `ac_init` −104 |
| a53 | — | — | UNREACHABLE (scp exit 255) | UNREACHABLE |

- BUILD_ERROR 0건, 셰이더 컴파일 에러 0건(glslangValidator, 4대), 결과 불일치(FAIL) 0건. 오차는 vector_add/histogram 0, saxpy ≤ 8e-8, matmul ≤ 1.44e-6, reduce ≤ 1.3e-6.
- OpenCL ERROR는 4대 중 Android 15/16인 3대에서만 났다. 에러 위치는 샘플의 `amuInitFromEnv` 호출이다. Android 13인 s20은 통과했다. 가설은 Android 링커 namespace가 Termux 프로세스의 `/vendor/lib64/libOpenCL.so`·`libGLES_mali.so` 로드를 거부하는 것이다. 이 경우 legacy 엔진은 마지막 후보인 bare `libOpenCL.so`(Termux ICD 로더로 추정, 플랫폼 0개)로 넘어가 −104를 낸다. **아직 확인되지 않았다.** v0.1.2에서 실패 시 후보 경로별 dlopen 결과를 detail에 찍도록 했다.
- s21 Vulkan: GPU 시간은 0.9–5.4 ms인데 launch+sync wall 중앙값이 632–1612 ms다. 다른 3대의 Vulkan wall은 GPU 시간의 1.09–3.2배다. 최대치인 s25 saxpy는 매 회 y 복원 복사(16 MB)가 wall에 들어간 값이다. 원인은 **미확인**이다. 코드 쪽 sleep/polling은 없다. `ac_vk_event_wait`는 `vkWaitForFences` 한 번만 호출한다. v0.1.2에서 `AMUDA_TRACE=1` 단계별 시간 측정을 넣었다.
- s20 OpenCL(legacy 엔진)은 wall이 GPU 시간의 1.3–10배다(vector_add 14.9 ms 대 1.46 ms). launch마다 버퍼 sync(map/unmap) 비용이 드는 것으로 추정한다. **미측정**이며, v0.1.2 trace로 분해한다.
- s20의 Vulkan은 Qualcomm 드라이버가 아니라 Mesa Turnip으로 돌았다(장치 이름 `Turnip Adreno (TM) 650`으로 확인. 어떤 로더/ICD 설정 때문에 Turnip이 선택됐는지는 미확인). 같은 칩의 Qualcomm OpenCL과 비교하면 vector_add GPU 시간이 5.47 ms 대 1.46 ms다. 이것은 AMUDA 결함이 아니라 드라이버 차이다. 백엔드 비교를 할 때는 이 점을 표기해야 한다.

| 성능(GPU timestamp 중앙값) | vector_add 대역폭 | matmul 512³ |
| :--- | :--- | :--- |
| s25 Adreno 830 (Vulkan) | 61.5 GB/s | 65 GFLOPS |
| s21 Mali-G78 (Vulkan) | 36 GB/s | 49 GFLOPS |
| s20 Adreno 650 (Turnip Vulkan / Qualcomm OpenCL) | 9.2 / 34 GB/s | 13 / 18 GFLOPS |
| a35 Mali-G68 (Vulkan) | 11.7 GB/s | 10.8 GFLOPS |

### 5.2 정적·호스트 검증

| 항목 | 상태 | 근거 |
| :--- | :--- | :--- |
| 번역기 단위 테스트 29개 | 통과 | 클라우드 Python 3.11. PC VM Python 3.10에서도 같은 결과 |
| 생성 GLSL 정적 검사 (`glsl_lint`) | 샘플 5종과 테스트가 만드는 셰이더 전부 통과 | 미선언 식별자, 바인딩 중복(별칭), push constant 블록 1개, `#version`/`#extension`/`#line` 순서. 음성 대조로 미선언 `v`와 바인딩 중복을 실제로 잡는 것을 확인 |
| 샘플 5종 GLSL → SPIR-V | **실기기 4대에서 통과** | 폰의 glslangValidator(Termux), 위 5.1 |
| 샘플 5종 OpenCL C 출력 | 문법 검사 통과, 실기기 1대(s20) 실행 통과 | 클라우드 clang 18, s20 Qualcomm OpenCL |
| amuda_rt 호스트 배선 | 테스트 더블 기준 전부 통과, ASan/UBSan 경고 0 | 인자 패킹, 바인딩 순서, spec constant, 파이프라인 캐시, flush/invalidate 순서, 에러 전파, 타임아웃 오염 상태, 공유 메모리 한도 거부 |
| v0.1.2 샘플 호스트 5종 | 테스트 더블에서 ASan/UBSan 경고 0. 커널이 실행되지 않으므로 5종 모두 FAIL로 나오는 것이 정상이며, 실제로 FAIL로 나옴 | 가짜 PASS가 나오지 않는 것도 함께 확인 |
| v0.1.2 OpenCL 초기화 진단 | 엔진 스텁(−104)과 플랫폼 0개짜리 가짜 `libOpenCL.so`로 detail 출력 확인 | 실제 Android 링커 동작은 폰에서 확인해야 함 |
| `ac_vk_event.cpp` 패치 | 문법 검사만 통과 | 최소 Vulkan 스텁 헤더 사용. 실제 컴파일은 폰 빌드에서 |
| 공개 API 구현 여부 | `amuda_rt.h`의 함수 20개가 모두 정의되어 있고 내부 심볼은 export하지 않음 | `nm` 확인 |

## 6. 알려진 한계와 다음 단계

**v0.1.1 재검토에서 고친 것**

- **float atomicAdd의 디스크립터 별칭 제거.** v0.1은 같은 binding에 float 블록과 uint 블록을 함께 선언했다. 이것이 glslang에서 허용되는지 이 환경에서 확인할 수 없어서, uint 저장소 하나만 선언하고 CAS로 처리하는 방식으로 바꿨다.
- **버퍼를 인스턴스 이름이 있는 블록으로 변경.** 예: `amu_buf_x.v[i]`. v0.1은 버퍼 멤버 이름이 전역 스코프에 있어서, `__device__` 함수의 파라미터나 헬퍼 함수의 파라미터와 이름이 겹칠 수 있었다.
- **번역기 버그 수정.** float atomic의 값 식에 다른 버퍼를 쓰면, 그 버퍼 이름이 한정되지 않은 채 생성되는 버그가 있었다. 이번 수정 과정에서 테스트가 잡았고, 고친 뒤 회귀 테스트를 추가했다.
- **헬퍼 함수는 실제로 쓰일 때만 생성.**
- **`amuKernelImage.flags` 제거.** 쓰이지 않는 필드였다. 대신 `shared_bytes`를 넣었다. Vulkan은 16384 B를 넘으면 거부하고, OpenCL은 장치 local memory 크기와 비교해 미리 거부한다.
- **폰 스크립트: 샘플 하나가 실패해도 나머지 실행.** 셰이더 컴파일이 실패한 샘플만 `BUILD_ERROR`와 컴파일러 로그(`AMUDACC|` 접두사)로 남긴다.
- **`ac_vk_dispatch` 단일 dispatch 경로 보강.** 장치에서 호스트로 결과가 보이게 하는 barrier(`SHADER_WRITE → HOST_READ`)를 추가했다. `vkBegin/EndCommandBuffer`의 반환값도 이제 검사한다. 이 파일은 기존 코어 파일이며 수정 범위는 이 두 가지뿐이다. 폰 빌드 전에는 컴파일 검증을 하지 못했다. Vulkan 헤더가 이 환경에 없기 때문이다.

**v0.1.2 (fleet 1차 결과 반영)**

- **OpenCL 초기화 실패 진단.** `ac_init` 실패 시 legacy 엔진과 같은 순서로 후보 경로 6개를 다시 dlopen한다. 각각에 대해 `blocked by linker namespace` / `not found` / `opened <실제 경로>, clGetPlatformIDs rc, platforms` 중 하나를 detail에 남긴다. 함수 포인터 변환은 ISO C 규칙(-Wpedantic)에 맞게 `memcpy`로 한다.
- **폰 스크립트: OpenCL 실행에만 `LD_LIBRARY_PATH=/vendor/lib64:/vendor/lib64/egl`.** bionic 링커는 격리된 namespace에서도 `LD_LIBRARY_PATH`에 있는 디렉터리의 파일은 접근 가능으로 본다. 다만 vendor 라이브러리의 의존성(VNDK 등)이 풀리는지는 기기마다 다르다. 이 변수는 샘플 프로세스에만 적용한다. `--no-cl-ldpath`로 끌 수 있다.
- **`AMUDA_TRACE=1`.** launch마다 다음 시간을 stderr에 찍는다: 이전 launch drain, flush, dispatch(커맨드 기록+submit), wait. wait은 fence / timestamp 조회 / 해제로 나눈다. 그 밖에 to_host와 gpu_ms도 찍는다. 폰 스크립트는 백엔드별로 vector_add를 한 번 trace 실행하고, 결과를 `TRACE|` 줄로 남긴다.
- **`--wake-lock` / `-WakeLock`.** s21 지연 이상치를 A/B로 확인하기 위한 옵션이다.
- **샘플.**
  - 샘플 이름을 `main` 첫 줄에서 정한다. 이전에는 init 실패 시 `sample=?`로 찍혔다.
  - 입력은 호스트 메모리에서 만들고, `amuMemcpyHtoD`/`DtoH`를 쓴다. CPU 기준값도 호스트 메모리에서 계산한다.
  - 결과 복사 시간은 `d2h_ms`로 따로 기록한다. 이 값은 uncached 매핑을 읽는 비용의 지표다.
  - 리셋+launch를 쉼표 연산식 대신 함수로 바꿨다.
- **`ac_vk_event_destroy`(코어 파일).** 5초 대기가 타임아웃이어도 `is_signaled = true`로 표시한 뒤, 아직 GPU가 쓰는 command buffer, fence, query pool, descriptor pool을 해제하던 결함을 고쳤다. 이 결함은 정적 분석으로 찾았고 실기기에서는 나타나지 않았다. 이제 타임아웃이나 OOM이면 아무것도 해제하지 않고 에러를 돌려준다. 장치 소실 때만 해제한다. `amuda_rt`의 `amuDestroy`는 이 경우를 stderr에 알리고 의도적으로 누수시킨다.

**현재 한계**

- `ac_vk_batch_destroy`에도 같은 "타임아웃 후 해제" 패턴이 있다. 반환형이 `void`라서 고치려면 API를 바꿔야 한다. POC는 batch를 쓰지 않으므로 이번에는 고치지 않았다.
- OpenCL `LD_LIBRARY_PATH` 우회가 Android 15/16 Samsung 기기에서 실제로 통하는지는 **미확인**이다. 통하지 않으면 Termux 프로세스에서는 vendor OpenCL을 쓸 수 없다. 그 경우 OpenCL 백엔드는 Android 13 이하 기기 전용으로 남는다.

- 매 launch마다 이전 launch를 기다린다(fence 왕복 1회). 정확성을 위한 POC 선택이다. 다음 단계는 `ac_runtime_batch_*`로 여러 launch를 커맨드 버퍼 하나에 묶는 것이다.
- 매 dispatch마다 descriptor pool, command buffer, fence를 새로 만든다(`ac_vk_dispatch` 내부). 반복 launch 오버헤드의 주요 후보다.
- batch 경로(`ac_runtime_batch_*`)는 "마지막 바인딩이 출력 버퍼"라고 가정해서 barrier 생략 여부를 정한다. 출력이 마지막이 아닌 커널을 batch로 묶으면 필요한 barrier가 빠질 수 있다. POC는 batch를 쓰지 않는다. 쓰기 전에 고쳐야 한다.
- GLSL의 엄격한 타입 규칙에 걸리는 경우는 폰의 glslang에서 원본 줄번호와 함께 에러가 난다. 예: `int x = 어떤uint;` 또는 `if (정수)`. 번역기는 이 부분의 타입 추론을 하지 않는다.

**legacy OpenCL 엔진에서 발견한 문제 (이번에 고치지 않음)**

- `ac_buffer_free`가 원래 할당 포인터가 아니라 map으로 받은 포인터를 free한다. 드라이버가 map에서 다른 주소를 돌려주면 잘못된 free가 된다.
- Mali 가드가 1D 블록 32 미만을 일괄 거부한다.
- 전처리 구문에 남아 있는 `atomicAdd` 매크로가 무조건 `__global int*`로 캐스팅한다. amudacc 출력은 이 매크로를 쓰지 않는다.
- `-cl-fast-relaxed-math`가 강제된다.

**다음 후보**

- warp/subgroup 내장 함수(`__shfl_*` → `subgroupShuffle*`, 기기별 지원 확인 필요)
- half/fp16
- 동적 공유 메모리(spec constant 크기)
- Python 런처
- 런타임 JIT: 폰에서 .acu를 바로 로드
