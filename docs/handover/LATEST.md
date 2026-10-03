# AMEVA 인수인계서: MMQ, 양자화 KV 캐시, 연산 융합, 파이프라인 캐시

- 작성: 2026-10-03 20:40 (KST)
- 브랜치: `vision/v1-pointwise-conv`
  - 마지막 커밋 `a6eee0f`(사용자, 15:44)에는 타일 FA, FA 키 분할, QGEMM, 행 청크 처리, 분기 없는 디코더까지 포함되어 있습니다.
  - 이 문서의 작업은 그 이후의 **미커밋** 변경입니다.
- 이전 문서: `docs/handover/20261003-0722.md`
- **이번 작업은 실기기 검증 전입니다.**
  - 사용자 지시에 따라 실기기 검증은 별도 세션에서 진행합니다. 이 세션은 기기를 쓰지 않았습니다.
  - 모든 변경은 로컬에서만 확인했습니다: aarch64 C++ 구문·형 검사, 셰이더 컴파일.
  - §3 런북의 결과가 나오기 전까지는 "구현됨, 검증 대기" 상태입니다.

## 1. 요약

- **목표:** 사용자 결정(2026-10-03)에 따른 기능 개발입니다. 결정 사항은 아래와 같습니다.
  - (1) MMQ(int8 정수 내적) 도입. 기본 활성, `AMEVA_MMQ=0`으로 끔, 조건은 CPU 대비 KLD 평균 0.01 이하
  - (2) 양자화 KV 캐시(Q8_0/Q4_0)와 가중치 재배열의 공존
  - (3) 그 밖에 남은 개발: 연산 융합, 파이프라인 캐시
- **현재 위치:** 네 기능 모두 구현했고 로컬 검사를 통과했습니다. 실기기 검증용 시험·스크립트를 함께 작성했습니다.
- **다음 담당자가 먼저 할 일:** §3 런북을 S25에서 그대로 실행하고, 결과를 §3의 판정 기준에 맞춰 기록합니다.

## 2. 변경 내용

### 2.1 양자화 KV 캐시 (Q8_0 / Q4_0)

**설계 변경 (사용자 지시안과 다름. 사유 포함)**

사용자 지시는 "KV 버퍼를 호스트·비재배열 경로로 분리"였으나, llama.cpp b8471에서는 이 방식을 구현할 수 없습니다.
- KV 캐시와 가중치가 모두 장치의 기본 버퍼 타입으로 할당됩니다. 호스트 여부는 버퍼 타입 단위로 정해집니다.
- 분리하더라도 FA와 양자화를 CPU가 수행하게 되어, 층마다 GPU↔CPU 왕복이 생깁니다.

그래서 KV 캐시를 건드리는 연산 전부를 GPU에서 수행하도록 했습니다(llama.cpp b8471 소스로 확인).

| 연산 | 시점 | 구현 |
|---|---|---|
| SET_ROWS f32→q8_0/q4_0 | KV 쓰기 | 새 커널 `shaders/ameva_set_rows_q.comp` |
| FLASH_ATTN_EXT, K/V q8_0/q4_0 | 어텐션 | `ameva_flash_attn.comp`의 K/V 적재에 역양자화 추가. K와 V 타입을 서로 다르게 쓸 수 있습니다. |
| CPY q8_0/q4_0→f32 | 컨텍스트 시프트 | `ameva_cpy.comp`가 양자화 소스를 바이트 주소로 읽습니다. |
| CPY f32→q8_0/q4_0 | 컨텍스트 시프트 | `ameva_set_rows_q` 항등 모드 |

**양자화 공식은 ggml과 같습니다.**
- q8_0: d = amax/127, 반올림은 ties-to-even(ggml ARM `quantize_row_q8_0`와 같음), d는 f16 RNE로 저장
- q4_0: d = max/-8, q = min(15, trunc(x/d + 8.5))
- Vulkan의 나눗셈은 정확 반올림이 아니어서(2.5 ulp), 경계값이 이웃 단계로 갈 수 있습니다. 시험은 이를 허용 범위로 판정합니다.

**블록 경계 처리**
- 블록은 34/18바이트입니다. 이웃 블록과 공유하는 16비트 반쪽은 atomic 두 번(AND, OR)으로 씁니다.

**계약 (그 밖의 KV 타입은 명시적으로 거부)**
- SET_ROWS 대상과 CPY 대상·원본에 q8_0/q4_0을 허용합니다. 행은 32의 배수(블록 단위)여야 합니다.
- FA는 K/V에 f32/f16/q8_0/q4_0을 허용합니다. 양자화 K/V는 헤드 크기가 32의 배수여야 합니다.
- q4_1, q5_0, q5_1, iq4_nl KV는 거부합니다. 재배열이 켜져 있으면 llama.cpp 스케줄러가 명시적으로 중단합니다.

**FA 끔 + 양자화 K:** 지원하지 않습니다. K의 MUL_MAT 경로가 대상이 아니기 때문입니다. llama.cpp는 양자화 V에 FA를 요구합니다.

**도구 변경:** parity 도구에 `--kv q8_0|q4_0` 옵션을 추가했습니다(K와 V 같은 타입).

### 2.2 MMQ (int8 활성값 × 재배열 가중치, 정수 내적)

**런타임**
- `VK_KHR_shader_integer_dot_product`가 제공되면 장치 생성 시 확장과 기능을 켭니다.
- 지원 여부는 `AcDeviceLimits.integer_dot_product`와 `integer_dot_product_accelerated`(필드 추가)로 보고합니다.
- `struct_size`가 이전 크기인 호출자도 받아들입니다.

**커널**
- `ameva_quant_a8.comp`
  - 32원소 단위로 d = amax/127, q = round(x/d)(ties-to-even)를 계산합니다.
  - 반쪽(16원소)별 q 합도 함께 구합니다.
- `ameva_mmq.comp` (12타입)
  - 가중치를 `decode_i8`(`ameva_qdecode.glsl`)로 풉니다: w = sc_h × v + mc_h, v는 int8 바이트입니다.
  - 64×32 타일입니다. 스레드 0..63은 가중치를 디코드하고, 64..127은 활성값을 스테이징합니다.
  - `dotPacked4x8EXT`로 정확한 정수 내적을 계산합니다.

**정밀도:** 활성값만 int8로 반올림됩니다(원소당 오차 d/2 이하). 가중치는 원래 정밀도 그대로입니다. CPU 백엔드도 활성값을 q8_0/q8_K로 반올림합니다.

**적용 조건:** 재배열 가중치, M ≥ 8(QGEMM과 같은 기준), 장치 지원.

**환경 변수 `AMEVA_MMQ`**

| 값 | 동작 |
|---|---|
| 미설정 | 지원 시 켜고, stderr에 한 줄 출력 |
| 1 | 반드시 사용합니다. 미지원이면 중단(abort) |
| 0 | 끕니다(f32 QGEMM) |

- 활성값 int8은 엔진 스크래치 버퍼에 기록하며, FA 분할 부분합과 같은 버퍼를 씁니다.
- 계수기: `ggml_backend_ameva_get_mmq_mul_mat_count()`를 추가했습니다. qmv 계수기는 재배열 경로 전체를 셉니다.

### 2.3 연산 융합 (write-through)

인접 노드 쌍을 디스패치 하나로 계산합니다.

| 쌍 | 처리 |
|---|---|
| RMS_NORM → 1행 가중치 MUL | `ameva_rms_norm`이 y와 y·w를 함께 씁니다. |
| MUL_MAT(QMV, 1행) → N벡터 ADD | bias·residual을 에필로그에서 더합니다. |
| MUL_MAT(QMV, 1행) → SWIGLU(split) | 곱이 gate일 때와 up일 때를 모두 처리합니다. |

- 첫 노드의 결과도 그대로 씁니다. 백엔드는 그래프 분할 하나만 보므로, 중간 결과를 다른 곳에서 쓰는지 알 수 없기 때문입니다.
- 연산은 비융합 노드와 같은 f32 연산입니다. 결과 비트가 같아야 하며, `ameva_fuse_check`의 해시로 확인합니다.

**메모리 겹침 안전장치**
- 문제: ggml 할당기는 두 번째 노드의 출력을, 첫 노드에서 수명이 끝난 입력(활성값)의 메모리에 둘 수 있습니다. 융합 디스패치에서 이 경우 경쟁 조건이 생깁니다.
- 대책: 출력과 입력의 바이트 범위가 겹치면 융합하지 않습니다. 같은 원소를 같은 스레드가 읽고 쓰는 정확한 in-place 별칭만 허용합니다.

**예상 효과와 설정**
- 효과는 계산값이며 실측 전입니다. Qwen 0.5B는 토큰당 534 디스패치 중 약 193개, Llama 1B는 약 80개가 줄어듭니다(재배열 + 1행 기준).
- `AMEVA_FUSE=0`으로 끕니다.
- 계수기: `ggml_backend_ameva_get_fused_pair_count()`, 프로파일에서는 `fused_pairs`

### 2.4 파이프라인 캐시 (첫 토큰 지연)

- 측정된 문제: S25에서 커널 16개 생성에 1.75초가 걸렸습니다(2.5절).
- 런타임
  - 컨텍스트별 `VkPipelineCache`를 두고 모든 커널 생성에 사용합니다.
  - `ac_runtime_pipeline_cache_merge`: 헤더의 vendor, device, UUID가 다르면 `AC_ERR_INVALID_ARGUMENT`로 거부합니다.
  - `ac_runtime_pipeline_cache_get_data`
- 엔진
  - 시작 시 `$HOME/.cache/ameva/ggml-ameva-pipelines.bin`을 읽습니다.
  - 종료(teardown) 시 데이터가 늘었을 때만 저장합니다. 프로세스별 임시 파일을 쓴 뒤 rename합니다.
  - `AMEVA_PIPELINE_CACHE=0`이면 끄고, 경로를 주면 그 경로를 씁니다.
  - 기기나 드라이버가 다른 데이터는 한 줄 출력 후 쓰지 않고, 종료 시 새로 씁니다.

### 2.5 오늘 S25에서 측정한 사실 (Adreno 830, 상한 832 MHz, cpuset 0-1,4-5)

**2단위 동시 적재(UPL) 기각.** 디코드 t/s, 같은 조건에서 번갈아 2회 측정했습니다.

| UPL | Qwen 0.5B | Llama-1B |
|---|---|---|
| 1 | 59.2~60.2 | 41.9~42.1 |
| 2 | 46.7~46.9 | 41.5~41.7 |
| 4 | 28.3~28.6 | 31.0~31.1 |

- 기본값은 1입니다. 빌드 옵션 `AMEVA_BUILD_QMV_UPL`로만 바꿀 수 있습니다.
- 분기 제거 디코더는 측정 오차 범위 안이었습니다.

**S25의 GEMV 대역폭**
- q4_K 28~40 GB/s, 출력 투영 q6_K 약 57 GB/s
- 이전 보고의 "4~7 GB/s"는 S20·A35 수치였습니다.

**CPU 기록 비용**
- 커널을 처음 생성하는 비용이 평균에 섞여 커 보였습니다(110 ms × 16).
- 정상 상태는 노드당 약 7 µs입니다. S25 디코드의 병목은 GPU입니다.

**결함 1건 수정**
- 바인딩 한계를 64 KiB로 낮춘 시험에서 디스패치 용량을 과소 산정하는 결함이 드러났습니다(`AC_ERR_DEVICE_BUSY`).
- 용량은 이제 타임스탬프 풀이 없을 때 런타임 최대값입니다.

**같은 S25에서 다른 세션이 측정한 값 (out-g3, 16:33)**
- Qwen 0.5B: 디코드 67.4 t/s(CPU 34.9), 프롬프트 287 t/s(CPU 59.2)
- `test-backend-ops` 1692/1692(기본·전체 재배열 모두)
- 3모델 정합성: 모든 개별 로그가 PASS입니다(Gemma 디코드 20.4 대 CPU 12.0). 요약 파일의 rc=127은 래퍼 스크립트 문제입니다.

### 2.6 기타

- `test_batch_graph_kat`: 정렬 위반 시험을 기기 정렬값에 맞췄습니다(S20은 4바이트).
- `ac_vk_event.cpp`: 커밋된 파일에 `<string.h>`가 빠져 있어 추가했습니다(`strcmp`).
- `tbo_paths.sh`
  - `-o UNARY`는 아무 시험도 고르지 않았습니다(0/0). 시험 이름이 함수명이기 때문이며, `TANH`로 바꿨습니다.
  - MMQ 비교 실행을 추가했습니다: `AMEVA_REPACK_ALL=1 AMEVA_MMQ=0`
- `build_ameva_kernels.py`
  - `#include`를 따라가며 바인딩을 셉니다.
  - QGEMM·MMQ·FA 계열과 이름 있는 op 변형(`OPS_DEFINED`)을 추가했습니다.
  - `--spirv-val none` 옵션을 추가했습니다(검증 생략, 헤더 두 번째 줄에 기록).

## 3. 실기기 검증 런북 (별도 세션용)

**중요:** 저장소의 `ggml_ameva_kernels_spv.h`는 Windows 로컬에서 `--spirv-val none`으로 생성한 것입니다(spirv-val 미실행). 기기의 `g2_build.sh`가 spirv-val을 포함해 다시 생성합니다. 그 생성본을 저장소에 반영하십시오.

### 3.1 동기화와 빌드 (S25 기준, A35·S20·S21 동일)

```
# PC에서
tar czf - include src adapters/llama/ggml-ameva scripts tests tools CMakeLists.txt | ssh s25 'cd ~/abi-lab/src && tar xzf -'
# 기기에서 (다른 세션과 겹치지 않는 OUT 이름)
cd ~/abi-lab && for f in decode_bench.sh prompt_bench.sh tbo_paths.sh parity_three.sh llm_parity_run.sh llm_parity_setup.sh ab_decode.sh; do cp src/scripts/termux/lab/$f .; done
bash src/scripts/termux/lab/g2_build.sh $HOME/abi-lab/out-v1
OUT=$HOME/abi-lab/out-v1 bash llm_parity_setup.sh
```

**판정 (`g2_build.sh` 출력)**

| 줄 | 기준 |
|---|---|
| 커널 헤더 | `wrote ... (30 qmm, 24 qmv, 12 qgemm, 12 mmq, 3 fa variants, 12 op kernels)`, 오류 없음 |
| `AMEVA_QMM_CHECK` | PASS (516/516) |
| `through ameva_mmq` | 정수 내적 지원 기기에서는 0보다 커야 합니다. 그때는 `AMEVA_MMQ=0` 줄도 PASS여야 합니다. |
| `AMEVA_KVQ_CHECK` | PASS |
| `AMEVA_CHUNK_CHECK` | PASS |
| `fuse on` / `fuse off` | 둘 다 PASS |
| `AMEVA_FUSE_BITEXACT` | PASS |
| 배치 KAT | `AMUDA_BATCH_GRAPH_KAT=PASS` |

**미지원 기기:** `ggml-ameva: MMQ unavailable ...` 한 줄이 출력되고 `through ameva_mmq`가 0이면 정상입니다. S20 Adreno 650은 지원 여부가 미확인입니다.

### 3.2 적합성 시험

```
(OUT=$HOME/abi-lab/out-v1 nohup bash tbo_paths.sh v1 > /dev/null 2>&1 &)
cat ~/abi-lab/results/tbop_v1.txt    # TBOP_DONE 이 마지막 줄
```

- 판정: 모든 줄이 `rc=0`이고 `N/N tests passed`, FAIL 없음
- 새로 시험되는 범위
  - `SET_ROWS`·`CPY`·`FLASH_ATTN_EXT`의 q8_0/q4_0 케이스(기본 실행)
  - MMQ(`repack_all`)
  - 융합 패턴(`MUL_MAT_VEC_FUSION` 등, `repack_all`)

### 3.3 실모델 정합성

```
OUT=$HOME/abi-lab/out-v1 nohup bash parity_three.sh v1 &                                   # f16 KV, MMQ 기본
OUT=$HOME/abi-lab/out-v1 bash llm_parity_run.sh ~/.cache/termux-llamacpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf v1_kvq8 --kv q8_0
OUT=$HOME/abi-lab/out-v1 bash llm_parity_run.sh ~/.cache/termux-llamacpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf v1_kvq4 --kv q4_0
```

- 판정: `AMEVA_LLM_PARITY=PASS`(KLD 평균 0.01 이하). MMQ 조건은 사용자 기준입니다.
- 참고값: f16 KV일 때 KLD는 지금까지 Qwen 2.57e-3, Llama 2.40e-3, Gemma 5.40e-3였습니다.
- `--kv q8_0`은 CPU도 같은 q8_0 KV를 쓰므로 비교가 공정합니다.
- 프로파일 줄 `mmq_mul_mat=`이 0보다 큰지(배치 채점 경로), `fused_pairs=`가 0보다 큰지 확인합니다.

### 3.4 성능 (같은 조건에서 번갈아 측정, cpuset·클럭 기록)

```
M=~/.cache/termux-llamacpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
OUT=$HOME/abi-lab/out-v1 bash prompt_bench.sh $M v1_mmq gpu
AMEVA_MMQ=0 OUT=$HOME/abi-lab/out-v1 bash prompt_bench.sh $M v1_nommq gpu
bash ab_decode.sh $M 2 out-g3 out-v1                              # 융합 전후 디코드
AMEVA_FUSE=0 OUT=$HOME/abi-lab/out-v1 bash decode_bench.sh $M v1_nofuse gpu
# 파이프라인 캐시: 같은 명령을 두 번 실행해 첫 실행과 두 번째의 prompt eval time 비교
rm -f ~/.cache/ameva/ggml-ameva-pipelines.bin; OUT=$HOME/abi-lab/out-v1 bash decode_bench.sh $M pc1 gpu; OUT=$HOME/abi-lab/out-v1 bash decode_bench.sh $M pc2 gpu; ls -la ~/.cache/ameva/
```

**기록 항목**
- 프롬프트 t/s: MMQ 대 f32 QGEMM. 정수 내적이 accelerated인지 보고 줄과 함께 기록합니다.
- 디코드 t/s: 융합 전후
- 첫 토큰(prompt eval) 지연: 캐시 유무
- `AMEVA_PROFILE=1`의 `dispatches`/`graphs` 감소

**양자화 KV 실사용**

```
llama-completion --device AMEVA0 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 ...
```

중단 없이 생성되는지, 출력이 f16 KV와 같은 양상인지 확인합니다. `-ctk q4_0 -ctv q4_0`도 같은 방법으로 확인합니다.

## 4. 남은 과제 (우선순위 순)

1. **§3 실기기 검증 (전 기기)**
   - 실패하면 해당 기능의 환경 변수로 끈 상태와 비교해 원인을 분리합니다: `AMEVA_MMQ=0`, `AMEVA_FUSE=0`, `AMEVA_PIPELINE_CACHE=0`, `--kv f16`
2. **MMQ 타일 조정:** 실측 후 BM 64 등 공유 메모리 여유를 활용합니다(현재 약 4.5 KB 사용).
3. **융합 확대:** QKV GEMV 합치기, ROPE+SET_ROWS, 프롬프트 경로 에필로그. 1번 실측에서 이득이 확인된 뒤 진행합니다.
4. **양자화 KV 타입 확대:** q4_1, q5_0, q5_1, iq4_nl.
5. 이월 항목
   - FA 끔 경로(SOFT_MAX 등)
   - 정확도 판정 모드(`--ref-model`)
   - `$PREFIX/lib` 설치(승인 필요)
   - 배포기 RUNPATH(별도 저장소)
   - STT 배포
   - whisper.cpp 서브모듈 여부(결정 필요)

## 5. 환경과 명령

**로컬 검사 도구 (Windows, 저장소 밖)**
- 위치: `%TEMP%\claude\tools`
  - `glslang` 16.6.0(Khronos main-tot 바이너리)
  - zig 0.13.0(clang 18)
  - Vulkan-Headers 1.3.280
  - llama.cpp b8471 헤더: `%TEMP%\claude\llsrc\stage\pkg`
- 사용법
  - C++ 구문·형 검사: `bash $TEMP/claude/tools/syn.sh <파일...>` (aarch64-linux, `-Wall -Wextra`)
  - 커널 헤더 생성: `python scripts/build_ameva_kernels.py --glslang <glslang.exe> --spirv-val none`
- spirv-val은 공식 바이너리가 배포되지 않아 로컬에 없습니다.
- 주의: `zig cc`는 FileNotFound로 실패합니다. 반드시 `zig clang`에 libc·libc++ 경로와 매크로를 직접 넘기십시오(syn.sh 참조).

**새 환경 변수**

| 변수 | 의미 |
|---|---|
| `AMEVA_MMQ` | 0 / 1 |
| `AMEVA_FUSE` | 0 / 1 |
| `AMEVA_PIPELINE_CACHE` | 0 / 경로 |
| `AMEVA_BINDING_LIMIT` | 시험 전용, 바이트 단위(4096 이상) |
| `AMEVA_FA_SPLIT_KEYS` | 0 또는 64의 배수 |
| `AMEVA_BUILD_QMV_UPL` | 빌드 시 1/2/4 |

## 6. 알려진 위험과 주의사항

- **정수 내적 성능:** MMQ가 f32 QGEMM보다 빠르다는 보장은 없습니다.
  - Adreno·Mali에서 `dotPacked4x8EXT`가 accelerated가 아니면 에뮬레이션됩니다.
  - 느리면 기본값 정책(accelerated일 때만 켜기 등)을 다시 정해야 합니다.
- **GLSL 오버로드:** `dotPacked4x8EXT(uint,uint)`는 부호 없는 판입니다. MMQ는 `int` 인자 판(부호 있음)을 씁니다. 로컬 glslang에서 확인했습니다.
- **같은 기기를 여러 세션이 함께 쓰면** 디렉터리와 측정이 충돌합니다. 빌드 디렉터리와 태그를 분리하고, 측정 중에는 다른 GPU 작업이 없는지 확인하십시오(S20의 `sd-cli` 사례).
- **파이프라인 캐시는 파일을 씁니다:** `~/.cache/ameva`. 읽기 전용 HOME 등에서는 한 줄 경고만 출력하고 계속 동작합니다.
- **`AMEVA_REPACK_ALL=1`에서는 KV 형태 텐서도 재배열됩니다.** 그 결과 양자화 KV 연산은 "not supported"로 건너뜁니다. 시험 설정의 특성이며, 실사용과 무관합니다.

## 7. 사용자 협의가 필요한 결정

1. **MMQ 기본값 정책.** 정수 내적이 가속되지 않는 기기에서 MMQ가 느릴 경우, 기본값을 "accelerated일 때만"으로 바꿀지 정해야 합니다(실측 후).
2. **로컬 생성 헤더를 커밋할지 여부.** 기기 생성본(spirv-val 포함)으로 교체한 뒤 커밋하기를 권합니다.
3. 이월 항목: whisper.cpp, RUNPATH, 설치 시점, A53

## 8. 미커밋 변경 (a6eee0f 이후)

- 수정
  - `adapters/llama/ggml-ameva/`
    - `ggml-ameva.cpp`: 양자화 KV, MMQ, 융합, 파이프라인 캐시, 계측, 용량
    - `ggml-ameva-device.cpp`, `ggml-ameva-impl.h`, `ggml-ameva.h`: 계수기
    - `ggml_ameva_kernels_spv.h`: 로컬 생성, spirv-val 미실행
  - 셰이더: `ameva_cpy.comp`, `ameva_flash_attn.comp`, `ameva_qdecode.glsl`, `ameva_qmv.comp`, `ameva_rms_norm.comp`
  - 런타임: `include/ac_context.h`, `include/ac_runtime.h`, `src/backend/vulkan/ac_vk_context.cpp`, `ac_vk_pipeline.cpp`, `ac_vk_event.cpp`, `src/runtime/ac_runtime.cpp`
  - 스크립트: `scripts/build_ameva_kernels.py`, `scripts/termux/lab/g2_build.sh`
  - 시험: `tests/ameva_llm_parity.cpp`, `tests/ameva_qmm_check.cpp`
- 신규
  - 셰이더: `ameva_mmq.comp`, `ameva_quant_a8.comp`, `ameva_set_rows_q.comp`
  - 시험: `tests/ameva_kvq_check.cpp`, `ameva_fuse_check.cpp`, `ameva_chunk_check.cpp`
  - 스크립트: `scripts/termux/lab/tbo_paths.sh`, `ab_decode.sh`
  - 문서: `docs/handover/20261003-2040.md`
