# AMEVA 인수인계서: 재배열 가중치, QMV·QGEMM, 타일 Flash Attention

- 작성: 2026-10-03 07:22 (KST), 중간 갱신입니다. 작업은 계속 진행 중입니다.
- 이전 문서
  - `docs/handover/20261002-2248.md`: 재배열과 QMV 도입
  - `docs/handover/20261002-1740.md`: 그래프 엔진
- 브랜치: `vision/v1-pointwise-conv`. 마지막 커밋 `500b6e6` 이후 작업은 모두 **미커밋**이며, 커밋은 사용자가 합니다.
- 검증 기기
  - A35 (Mali-G68): 10/02 밤까지 사용했고, 10/03 새벽부터 ssh 연결 시간 초과입니다.
  - S20 (Adreno 650): 10/03에 검증했습니다.
  - S25, S21은 연결 거부, A53은 연결 시간 초과입니다.

## 1. 요약

- **목표:** 양자화 LLM의 디코드와 프롬프트 처리를 GPU에서 빠르게 합니다.
- **현재 위치:**
  - 가중치 재배열 + 디코드 GEMV(QMV) + 프롬프트 타일 GEMM(QGEMM) + 타일 Flash Attention이 들어갔습니다.
  - A35·S20에서 정확성 시험, ggml 적합성 시험, 실모델 정합성 검사를 통과했습니다.
- **S20 결과 (Qwen2.5-0.5B Q4_K_M)**

  | 항목 | AMEVA | 종전 경로 | CPU |
  |---|---|---|---|
  | 프롬프트 368토큰 | 121 t/s | 28.6 t/s | 47.4 t/s (CPU의 2.55배) |
  | 디코드 | 16.6~19.4 t/s | 8.4 t/s | 22.8 t/s (CPU보다 낮음) |

- **다음 담당자가 먼저 할 일**
  1. 접속되는 기기(A35·S25)에서 §4.1 검증을 반복합니다.
  2. 이어서 §4.2 이후의 성능 작업을 진행합니다.

## 2. 완료·검증된 작업

### 2.1 이 단계에서 추가·변경한 것

1. **공용 디코더** `shaders/ameva_qdecode.glsl`
   - 재배열 가중치 바인딩 0~2, 적재 헬퍼, `decode()`, `qdecode_init()`를 담습니다.
   - QMV와 QGEMM이 같은 디코더를 씁니다.
2. **QMV 프롬프트 경로(COLS=8) 회귀 수정**
   - 단위 루프 안의 행 경계 분기 때문에 Mali에서 2배 느렸습니다(190 ms 대 QMM 100 ms).
   - 분기를 루프 밖으로 옮겨 QMM과 같은 속도가 되었습니다.
   - M=1 경로는 분기가 있는 원래 형태가 더 빨라서 그대로 둡니다. 실측: 행 고정 방식 11.56, 원래 형태 12.41 t/s.
3. **QGEMM** `shaders/ameva_qgemm.comp`
   - 재배열 가중치용 타일 GEMM입니다.
     - 128스레드가 64행 × 32열 타일을 맡습니다.
     - 단위마다 공유 메모리에서 f32로 역양자화하고, 스레드당 4×4 외적 누산을 합니다.
   - M ≥ 8이면 QGEMM을 씁니다. `AMEVA_QGEMM_MIN_M`으로 바꿀 수 있습니다.
     - A35 교차점: q5_0은 M=8부터, q4_K는 M=12부터 QGEMM이 빠릅니다.
   - 128×32 타일·스레드당 8×4 시도는 Adreno 650에서 6배 느려 되돌렸습니다(§6).
4. **QMV 행당 레인 수:** 모든 K에서 16으로 고정했습니다(근거는 §2.3).
5. **타일 Flash Attention** `shaders/ameva_flash_attn.comp`. 이전 커널은 (쿼리 행, 헤드)당 워크그룹 하나였고, 이를 교체했습니다.
   - 같은 KV 헤드를 읽는 (쿼리 위치, 그룹 헤드) 행 BQ개를 128스레드가 맡습니다(GQA 헤드가 K/V 적재를 공유).
   - 키는 BK개 블록 단위로 처리합니다: 공유 메모리에 f32로 올리고(f16 정확 변환) 온라인 softmax를 적용합니다.
   - 모든 행에서 마스킹된 블록은 건너뜁니다.
   - 변형 3종: DMAX 64(BQ16, BK32), 128(16, 32), 256(8, 16). 공유 메모리는 각 32 KB 미만입니다.
   - D·DV가 4의 배수가 아니면 계약에서 거부합니다.
6. **배치 KAT 수정** `tests/test_batch_graph_kat.cpp`
   - '정렬 위반' 검사를 기기 정렬값의 절반 오프셋으로 만들도록 바꿨습니다.
   - Adreno 650은 정렬이 4바이트여서, 고정 오프셋 4가 합법 값이었습니다.
7. **도구·스크립트** (`scripts/termux/lab/`)
   - `prompt_bench.sh`, `tbo_run.sh`, `parity_three.sh`, `qgemm_crossover.sh`, `wait_cool.sh`
   - `decode_bench.sh`·`prompt_bench.sh`는 `Cpus_allowed_list`와 `gpu_max_clk`(Adreno)를 함께 출력합니다.
   - `llm_parity_setup.sh`는 출력 디렉터리를 `OUT`으로 받습니다(기본 `out-g2`).
8. **S20 실험 환경 신규 구성** (`~/abi-lab`)
   - b8471 헤더(`ggml-b8471`, `llama-b8471`)
   - `rt098` (0.9.8 링크)
   - `tbo/test-backend-ops`
   - 말뭉치

### 2.2 검증 결과

| 검사 | A35 (Mali-G68) | S20 (Adreno 650) |
|---|---|---|
| `ameva_qmm_check` (15타입 × 8형상 × 배치 4종, QGEMM 포함) | 480/480 | 480/480. `AMEVA_QMV_TPR` 4/8/16/32/64 모두 480/480 |
| `test-backend-ops` 기본 / `AMEVA_REPACK_ALL=1` (QGEMM 빌드) | 1691/1691 / 1691/1691 (QGEMM 전 빌드) | 1690/1690 / 1690/1690 |
| `test-backend-ops -o FLASH_ATTN_EXT` (타일 FA) | 미실행 | 824/824 |
| 배치 KAT | PASS | PASS (KAT 수정 후) |
| parity Qwen2.5-0.5B Q4_K_M | PASS, KLD 2.571e-3, 그리디 32/32 | PASS, KLD 2.569e-3, 그리디 32/32 |
| parity Llama-3.2-1B Q4_K_M (ctx 256) | — | PASS, KLD 2.398e-3, 그리디 10/32 (S25 이전 결과와 같은 양상) |
| parity Gemma-2-2B Q4_K_M (ctx 256, 헤드 256, softcap) | — | PASS, KLD 5.404e-3 (CPU 자체 6.881e-3), 그리디 32/32 |

- 로그 위치
  - S20: `~/abi-lab/results/parity3_s20g3.txt`, `parity_s20g3_*.log`, `tbo_*.log`, `tbo_fa_g3.log`
  - A35: `~/abi-lab/results/parity_q05_qgemm_a35.log`

### 2.3 성능

**측정 조건:** `Cpus_allowed_list`와 Adreno 클럭 상한을 반드시 함께 보십시오(§6).

**S20, Qwen2.5-0.5B Q4_K_M** (cpuset 0-7)

| 구성 | 디코드 t/s | 프롬프트 368토큰 t/s |
|---|---|---|
| CPU (`-t 4`) | 22.79 | 47.43 |
| AMEVA QMM (`AMEVA_REPACK=0`) | 8.37 | 28.63 |
| AMEVA QMV/QGEMM + 이전 FA | 15.70 | 93.44 |
| AMEVA + 타일 FA (tpr 자동=K별) | 16.62 | 121.06 |
| AMEVA + 타일 FA, tpr 16 / 8 | 17.9~18.9 / 19.41 | — |

- 위 행들은 정상 클럭에서 측정했습니다. 발열로 상한이 305 MHz로 떨어진 상태에서는 디코드 11.7, 프롬프트 71.5였습니다.

**S20 디코드 형상별 QMV 시간** (tpr 8 / 16 / 64)

| 형상 | tpr 8 | tpr 16 | tpr 64 |
|---|---|---|---|
| q8_0 151936×896 | 7.28 ms | 8.19 ms | 11.4 ms |
| q4_K 896×4864 | 336 µs | 356 µs | 611 µs |
| q6_K 896×4864 | 296 µs | 296 µs | 559 µs |
| q5_0 4864×896 | 241 µs | 245 µs | 660 µs |

- 출력 투영(q8_0)은 약 20 GB/s입니다.

**S20 프롬프트 내역** (368토큰, 타일 FA)
- QGEMM이 82%입니다. q5_0 4864×896이 회당 33 ms로 약 48 G FMA/s입니다.
- FA가 8%(회당 5.4 ms)입니다. 이전 FA는 A35에서 회당 35 ms였습니다.

**A35, Qwen2.5-0.5B Q4_K_M** (QGEMM 빌드, cpuset 0-7)

| 항목 | 결과 |
|---|---|
| 프롬프트 185토큰 | 74.2 t/s (QMM 21.1) |
| 프롬프트 368토큰 | 69.2 t/s (QMM 20.3) |
| 디코드 | 14.0~14.5 t/s |
| CPU 디코드 | 34.2 t/s |
| ggml-vulkan | 7.0 t/s, 출력이 `!!!!`로 깨짐 |

- 정합성 도구 기준 CPU batch는 88.5 t/s입니다.

**모델별 S20 정합성 실행의 디코드** (AMEVA / CPU, 상한 525 MHz 구간 포함)

| 모델 | AMEVA | CPU |
|---|---|---|
| Qwen 0.5B | 17.7 | 20.4 |
| Llama 1B | 6.8 | 13.9 |
| Gemma 2B | 5.5 | 5.8 |

- 배치 채점(프롬프트형): Qwen 76.5, Gemma 19.1(CPU 14.4)

## 3. 중단 지점

1. **QMV 행당 레인 수:** 엔진 기본값을 "K별 16/32/64"에서 "항상 16"으로 바꿨습니다.
   - S20 `out-g3`에 빌드되어 있고, 정확성 480/480은 통과했습니다.
   - 성능은 발열 상한 구간에서만 측정되었습니다.
   - 정상 클럭에서 `decode_bench.sh`로 다시 재야 합니다(이전 수동 측정: 16 고정 17.9~18.9 t/s).
2. **커널 헤더**는 S20 생성본(glslang 16.6.0)으로 저장소에 동기화되어 있습니다. 64×32 QGEMM, 타일 FA가 반영되어 있습니다.
3. **S20 GPU 상한**은 07:20 기준 525 MHz입니다. 정상 상한은 587 MHz입니다.

## 4. 다음 작업 (우선순위 순)

### 4.1 다른 기기에서 재검증

- 대상: A35, S25 (복구 시)
- 목적: tpr 16 기본값, 64×32 QGEMM, 타일 FA를 Mali와 Adreno 830에서 확인합니다.
- 명령 (기기에서)
  ```
  cd ~/abi-lab && bash src/scripts/termux/lab/g2_build.sh $HOME/abi-lab/out-g3
  (nohup bash tbo_run.sh > /dev/null 2>&1 &)     # OUT=$HOME/abi-lab/out-g3 로 지정
  OUT=$HOME/abi-lab/out-g3 nohup bash parity_three.sh <tag> &
  OUT=$HOME/abi-lab/out-g3 bash decode_bench.sh <gguf> <tag> gpu; ... prompt_bench.sh ...
  ```
  - S25에는 `~/abi-lab`이 이미 있습니다.
  - A35는 `out-g2`에 QGEMM 빌드(FA 이전)가 있습니다.
- 완료 기준: qmm_check 480/480, `test-backend-ops` 전체 통과, parity 3모델 PASS

### 4.2 디코드 GEMV

- 목표: K=4864 형상(q4_K 4~7 GB/s, q6_K 7~12 GB/s)의 효율을 출력 투영 수준(20 GB/s, S20)에 가깝게 높입니다.
- 후보
  - q4_K 스케일 해석 분기 제거
  - 행당 레인 8 (Adreno에서 3% 이득)
  - 단위 2개 동시 적재로 메모리 수준 병렬성 확대

### 4.3 프롬프트 GEMM

- 현재 S20 약 48 G FMA/s로 이론치의 약 16%입니다.
- 레지스터를 늘리는 방향(스레드당 8×4)은 Adreno에서 실패했습니다.
- 후보
  - 256스레드 × 4×4로 타일만 키우기
  - K 단계 64(배리어 절반)
  - 활성값을 int8로 양자화한 정수 내적(MMQ, `VK_KHR_shader_integer_dot_product`, A35 지원)
    - 정밀도는 CPU의 Q8 경로와 같아집니다. 정책 결정이 필요합니다(§7).

### 4.4 디스패치·배리어 감소

- 현재 토큰당 534 디스패치입니다.
- 연산 융합 후보: RMS_NORM+MUL, gate/up+SwiGLU, 잔차 ADD
- FA 디코드 키 분할: 긴 컨텍스트에서 KV 헤드당 워크그룹 하나라서 직렬 구간이 깁니다.

### 4.5 바인딩 한계를 넘는 원소별 연산의 청크 분할

- 현상: S20(128 MiB)에서 Gemma 배치 256의 logit SCALE/TANH(262 MB)가 CPU로 갑니다.

### 4.6 이월 과제

- 정확도 판정 모드(`--ref-model`)
- `$PREFIX/lib` 설치
- CLAUDE.md 4장 갱신
- STT 배포
- 배포기 RUNPATH

## 5. 환경과 명령

- 기기 작업 디렉터리 구조는 이전 문서 §5와 같습니다.
- S20 `~/abi-lab`

  | 경로 | 내용 |
  |---|---|
  | `out-g2` | QGEMM, 이전 FA |
  | `out-g3` | QGEMM 64×32, 타일 FA, tpr 16. 최신 |
  | `rt098` | ggml 0.9.8 링크 |
  | `tbo` | test-backend-ops |
  | `results` | 결과 로그 |

- 기기 셸
  - A35: zsh. 따옴표 없는 변수가 분리되지 않으므로 bash 스크립트로 실행하십시오.
  - S20: bash
- 실행 중인 라이브러리를 덮어쓰지 마십시오. `test-backend-ops` 등이 도는 중에는 다른 `OUT` 디렉터리에 빌드합니다.
- 환경 변수

  | 변수 | 의미 |
  |---|---|
  | `AMEVA_REPACK` | 재배열 사용 여부 |
  | `AMEVA_REPACK_ALL` | 모든 버퍼의 대상 텐서 재배열(시험용) |
  | `AMEVA_QMV_TPR` | QMV 행당 레인 수 4/8/16/32/64 |
  | `AMEVA_QGEMM_MIN_M` | QGEMM을 쓰기 시작하는 활성값 행 수 2~65535 |
  | `AMEVA_PROFILE` | 1·2 |
  | `AMEVA_TRACE` | 그래프별 기록 |

## 6. 알려진 문제와 주의사항

**측정 조건이 결과를 2~3배 바꿉니다.**
- ssh로 띄운 Termux는 cpuset이 바뀝니다.
  - A35에서 `0-2`(A55 3개)였다가 이후 `0-7`이 되었습니다.
  - CPU 디코드 6.15 → 34 t/s, AMEVA 기록 비용 디스패치당 64 → 약 8 µs로 달라졌습니다.
  - 이전 보고의 "A35에서 AMEVA가 CPU의 2.1배"는 이 조건 때문이었고, 정정했습니다. CPU(A78 4코어)가 디코드 34 t/s로 더 빠릅니다.
- Adreno 클럭 상한(`/sys/class/kgsl/kgsl-3d0/max_gpuclk`)은 연속 실행 약 1시간 후 305 MHz까지 내려갑니다.
- A/B 비교는 같은 조건에서 번갈아 측정하십시오.

**Adreno 650 레지스터 절벽**
- QGEMM에서 스레드당 vec4 누산기 8개와 역양자화 출력 8개(vec4)가 동시에 살아 있으면 6배 느려졌습니다.
- 셰이더 통계 확장이 없어 스필을 직접 확인하지는 못했습니다. Mali-G68도 `VK_KHR_pipeline_executable_properties`가 없습니다.

**Mali 컴파일러**
- COLS=8 단위 루프 안의 경계 분기가 2배 느린 코드를 만들었습니다(§2.1의 2번 항목).

**S20 장치 한계**
- `maxStorageBufferRange` 128 MiB, 오프셋 정렬 4바이트, 서브그룹 128입니다.

**non-host 버퍼(재배열)**
- 양자화 KV 캐시·순환 모델은 llama.cpp 스케줄러가 명시적으로 중단합니다. `AMEVA_REPACK=0`으로 실행하십시오.

**정합성 도구와 llama-completion의 기록 시간 차이**는 미해명입니다(이전 문서 §6).

## 7. 사용자 협의가 필요한 결정

1. **재배열 기본값 정책.** 양자화 KV·순환 모델 지원과의 관계입니다(이전 문서 §7).
2. **프롬프트 GEMM에 활성값 int8 양자화(MMQ)를 쓸지 여부**
   - 현재: f32 역양자화 + f32 FMA. CPU보다 정확합니다.
   - MMQ로 가면 속도는 크게 오를 수 있습니다. 대신 CPU와 같은 Q8 근사가 들어갑니다.
3. **디코드 성능 목표.**
   - A35처럼 CPU가 강하고 GPU 대역폭이 낮은 기기에서는 GPU 디코드가 CPU를 넘기 어렵습니다(유효 대역폭 GPU 5~9 GB/s, CPU 약 12.7 GB/s).
   - GPU 오프로드의 가치를 무엇으로 볼지 정해야 합니다: 프롬프트 처리 가속, CPU 해방, 전력.
4. 이월 항목: whisper.cpp 서브모듈 여부, 배포기 RUNPATH, 설치 시점, A53 보류

## 8. 미커밋 변경 (이 단계 추가분)

- 신규
  - `adapters/llama/ggml-ameva/shaders/`: `ameva_qdecode.glsl`, `ameva_qgemm.comp`
  - `scripts/termux/lab/`: `prompt_bench.sh`, `tbo_run.sh`, `parity_three.sh`, `qgemm_crossover.sh`, `wait_cool.sh`
  - `docs/handover/20261003-0722.md`
- 수정
  - `shaders/ameva_flash_attn.comp` (타일 방식으로 재작성), `shaders/ameva_qmv.comp` (공용 디코더, 루프 구조)
  - `ggml-ameva.cpp`: QGEMM 경로, FA 인코더·계약, tpr 16
  - `scripts/build_ameva_kernels.py`: include를 따라가는 파싱, QGEMM·FA 계열
  - `ggml_ameva_kernels_spv.h`: 재생성
  - `tests/`: `ameva_qmm_check.cpp` (QGEMM 형상), `ameva_mm_bench.cpp`, `test_batch_graph_kat.cpp`
  - `scripts/termux/lab/`: `decode_bench.sh`, `llm_parity_setup.sh`
- 이전 단계 목록은 `docs/handover/20261002-2248.md` §8, `docs/handover/20261002-1740.md` §8에 있습니다. 전체는 `git status --short`로 확인하십시오.
