# AMEVA 인수인계서: 2026-10-03 개발분 실기기 검증, 정수 내적 토큰 생성(ameva_mmvq), 런타임 번들용 백엔드

- 작성: 2026-10-04 22:40 (KST). 직전 문서: `20261003-2215.md`, 런북은 `20261003-2040.md`
- 저장소: `ameva-compute`, 브랜치 `vision/v1-pointwise-conv` (원격에 푸시함)
- 관련: `ameva-runtime` 브랜치 `feat/ggml-ameva-bundle`의 `docs/handover/LATEST.md` (번들·게시 상태)
- 대상: llama.cpp b8471(ggml 0.9.8) + `libggml-ameva.so` + `libac_runtime.so.2`

## 1. 요약

- **ggml-ameva가 ameva-runtime llama 번들의 GPU 경로가 되었습니다**(2.8.0 후보, 미게시). 번들용 빌드는 `scripts/termux/build_runtime_backend.sh`.
- **2026-10-03 개발분(MMQ, 양자화 KV, 융합, 파이프라인 캐시)을 5대에서 실측했습니다**(2.2절). 속도에 기여하는 것은 MMQ뿐이었습니다.
- **토큰 생성에 정수 내적 커널 `ameva_mmvq`를 추가했습니다**(`c8aa8fd`). 기기·가중치 모양별 측정에 포함되고, f32 레이아웃보다 10 % 넘게 빠를 때만 선택됩니다. 행당 스레드 4·2·1 레이아웃도 측정 후보에 넣었습니다.
  - 생성 속도: A35 12.5 → 19.0 t/s, S20 25.8 → 32.7, A53 13.7 → 14.4 (Qwen2.5-0.5B). S25·S21은 측정 편차 안.
- **GPU가 CPU보다 느린 구간이 남아 있습니다.** Mali 3대의 토큰 생성(CPU의 0.5~0.7배)과 1B 모델 프롬프트(0.6~0.8배)입니다. 다음 과제는 4장.
- **미완 검증**: `c8aa8fd`의 `test-backend-ops` 전체, A35의 Gemma KL 발산, S20·S21의 런타임 스크래치 시험(3장).

## 2. 완료·검증된 작업

### 2.1 런타임 번들용 백엔드 (`8a2c63f`, `8f511dc`, `bf85cfb`)

- **RUNPATH**: Termux clang은 `$PREFIX/lib`를 `-rpath`보다 앞에 둡니다. 플러그인이 `$PREFIX/lib`의 옛 `libac_runtime.so.2`를 불러와 "cannot locate symbol"로 실패했습니다. `build_ggml_ameva.sh`가 patchelf로 `$ORIGIN`을 앞에 둡니다.
- **`build_runtime_backend.sh`**: 빌드, RUNPATH·의존 라이브러리 검사, `ameva_qmm_check`, 배치 KAT, 번들의 `llama-completion`으로 등록 확인, `backend-manifest.json` 작성.
- **가중치 로드**: 재배열 코덱이 GPU 매핑 메모리에 직접 썼습니다. Mali·Turnip은 그 매핑이 캐시되지 않아 6~9 MB/s였습니다. 스테이징 복사본에서 재배열한 뒤 구역 단위로 복사합니다.
  - A35 로드 시간: Qwen 11.5 → 1.2초, Llama-1B 81 → 2.6초, Gemma-2B 182 → 8.1초.
  - 매핑된 AMEVA 버퍼를 CPU가 작은 단위로 흩어서 읽고 쓰는 코드는 같은 문제를 냅니다.

### 2.2 2026-10-03 개발분 실기기 검증 (5대, 백엔드 `0bcbebd`)

측정: ameva-runtime `tools/fleet/lab/perf_matrix.sh`. Qwen2.5-0.5B·Llama-3.2-1B Q4_K_M, `-c 2048 -t 4`.

| 기능 | 결과 | 판정 |
| :--- | :--- | :--- |
| MMQ (프롬프트, `AMEVA_MMQ`) | f32 타일 커널 대비 프롬프트 t/s: S25 412 → 1066, S20 133 → 238, A35 95 → 190 (Qwen). A53·S21은 `VK_KHR_shader_integer_dot_product`가 없어 동작하지 않음 | 지원 기기에서 켜는 현재 기본값 유지 |
| 연산 융합 (`AMEVA_FUSE`) | 5대 모두 생성 t/s 차이가 측정 오차 안(S20 25.4 / 25.3, A53 13.5 / 13.6, A35 14.8 / 15.0) | 정확성은 유지, 속도 기여 없음. 확대는 보류 |
| 양자화 KV (`-ctk`/`-ctv`) | q8_0: CPU의 q8_0과 같은 문장, 속도 차이 없음. q4_0: GPU·CPU 모두 문장이 무너짐(Qwen 0.5B) | 메모리 절감 용도. 타입 확대는 낮은 우선순위 |
| 파이프라인 캐시 | 캐시 파일 유무에 따른 실행 시간 차이 0.0~0.8초(S20·S21·A53) | 유지 |
| 적합성 | `ameva_qmm_check` 516/516, 배치 KAT, `test-backend-ops` 1949/1949·전체 재배열 1692/1692(S25) | 통과 |
| 정확도 | CPU 대비 KL 발산 Qwen 2.57e-3, Llama 2.40e-3, Gemma 5.85e-3, 5대 동일(기준 0.01) | 통과 |

MMQ 정수 내적은 S20(Turnip)에서 "accelerated: no"로 보고되지만 f32보다 1.8배 빨랐습니다. 보고 값으로 기본값을 가르지 않습니다.

### 2.3 토큰 생성 시간의 구성 (`AMEVA_PROFILE=1`, `2`)

- Mali 3대에서 GPU 시간의 76~84 %가 1행 MUL_MAT입니다. 그중 출력 투영(q8_0 151936×896)이 전체의 20~25 %입니다.
- 토큰당 GPU 시간(Qwen): A53 75 ms, A35 67 ms, S20 40 ms, S21 36 ms, S25 12.6 ms.
- CPU의 명령 기록(encode)은 토큰당 2~9 ms입니다(S25 3.2 ms, S21 최대 8.7 ms).
- S25에서 대기 시간이 GPU 시간보다 길게 나온 실행이 있습니다(대기 20 ms / GPU 12.6 ms; 다른 실행은 11.7 / 13.2). 다른 4대는 대기 ≈ GPU입니다.

### 2.4 `ameva_mmvq`와 레이아웃 후보 확장 (`c8aa8fd`)

- **커널** `shaders/ameva_mmvq.comp`: `ameva_qmv`의 작업 분할 + int8 활성값(`ameva_quant_a8`) + `dotPacked4x8EXT`. 12타입 × rows 4·2·1 = 36개 모듈.
- **선택**: 가중치 모양별 측정에 f32 레이아웃과 함께 들어갑니다. 가장 빠른 f32 레이아웃보다 10 % 넘게 빠를 때만 선택합니다(결과가 활성값 반올림만큼 다르기 때문).
- **양자화 재사용**: 같은 활성값을 쓰는 연속 MUL_MAT(Q·K·V, gate·up)은 양자화를 한 번만 합니다.
- **f32 레이아웃**: 행당 스레드 4·2·1 추가(13 → 22개). 1이면 스레드 하나가 행 전체를 계산하고 축약이 없습니다.
- **환경 변수**: `AMEVA_MMVQ=auto|0|1`, `AMEVA_MMVQ_ROWS`, `AMEVA_MMVQ_TPR`(1일 때만), `AMEVA_QMV_TPR=1|2|4|8|16|32|64`.
- **튠 파일 형식 2**: f32 선택과 int8 선택을 함께 저장합니다. 형식 1 파일은 다시 측정합니다.
- **커널 헤더**: S25에서 spirv-val을 거쳐 생성한 것으로 교체했습니다(이전 저장소 헤더의 "로컬 생성" 문제 해소).

**생성 속도 (64토큰, t/s, 번갈아 3회·S21은 5회): `0bcbebd` → f32 후보만(`AMEVA_MMVQ=0`) → 기본**

| 기기 | Qwen2.5-0.5B | Llama-3.2-1B | CPU (Qwen / Llama) |
| :--- | :--- | :--- | :--- |
| A35 Mali-G68 r38p1 | 12.3~12.7 → 14.5~14.6 → 19.0 | 6.9~7.4 → 7.1~7.5 → 11.9 | 36.6 / 18.5 |
| S20 Adreno 650 Turnip (587 MHz) | 25.7~26.0 → 27.0~27.1 → 32.6~32.9 | 13.5 → 13.5 → 18.8 | 28~31 / 21.7 |
| S25 Adreno 830 | 832 MHz에서 차이가 편차 안(72~75) | 525 MHz: 30.9 → 30.8~31.1 → 34.3~35.0 | 33~49 / 19~27 |
| A53 Mali-G68 r32p1 (확장 없음) | 13.7 → 14.4 → 14.4 | 6.3 → 7.0 → 7.0 | 22.6 / 13.4 |
| S21 Mali-G78 r38p0 (확장 없음) | 21.3~22.4 → 22.1~22.8 → 같음 | 12.2~12.9 → 12.0~13.2 → 같음 | 46.2 / 17~22 |

- A35에서 선택된 int8 레이아웃은 대부분 1행 × 4스레드였고, 가장 빠른 f32 레이아웃의 0.43~0.77배 GPU 시간이었습니다(q4_K 896×4864: 244 대 562 µs).
- S25는 14개 모양 중 4개만 int8을 선택했습니다.

**검증**
- `ameva_qmm_check` 516/516(기본 정책), 516/516(`AMEVA_MMVQ=1`, int8 경로 234건), 배치 KAT.
- `test-backend-ops -o MUL_MAT`(S25): 기본·전체 재배열·전체 재배열+`AMEVA_MMVQ=1` 각 187/187.
- CPU 대비 KL 발산(Qwen / Llama / Gemma, 기준 0.01):
  - 기본 정책: S25 2.56e-3 / 2.71e-3 / 6.06e-3, S20 3.12e-3 / 2.86e-3 / 6.36e-3, A35 2.96e-3 / 2.90e-3 / 미측정, A53·S21 2.57e-3 / 2.40e-3 / 5.85e-3.
  - 전 곱셈 강제(`AMEVA_MMVQ=1`): S25 2.58e-3 / 2.79e-3 / 6.25e-3, S20 2.68e-3 / 2.86e-3 / 6.33e-3.
- 그리디 텍스트: Qwen·Gemma는 CPU와 같고, Llama는 이전과 같이 10토큰 이후 분기합니다.

## 3. 중단 지점

- 진행 중인 작업은 없습니다.
- **S20·S21 접속 불가(22:06경부터)**: ameva-runtime 스크래치 시험의 `-b vulkan` 단계 중 두 기기의 Termux 세션이 끝났습니다. 원인 미확인. 상세와 재시험 방법은 ameva-runtime `docs/handover/LATEST.md` 3장.
- A35는 KL 발산 측정 중(Gemma의 CPU 실행 구간) 세션이 끝났다가 돌아왔습니다.
- 기기의 `~/abi-lab/out-*`, `src-*`, `perf/`, `ab/`는 실험 산출물입니다. 지워도 됩니다.

## 4. 남은 개발 과제 (전체, 우선순위 순)

### 4.1 검증 부채

1. `c8aa8fd`의 `test-backend-ops` 전체(기본 + `AMEVA_REPACK_ALL=1`), S25. 약 2시간.
2. A35의 Gemma-2-2B KL 발산, S20·S21의 스크래치 `-b vulkan` 재시험과 원인 확인.
3. `scripts/termux/lab/tune_verify.sh`를 새 후보(행당 스레드 4·2·1, `ameva_mmvq`)에 맞게 갱신. 레이아웃을 하나씩 고정한 `ameva_qmm_check`는 13개 레이아웃 기준으로만 실행되었습니다.
4. Gemma-2-2B 등 2B 이상 모델의 속도 실측(오늘은 0.5B·1B만 양쪽 지표로 측정).

### 4.2 성능

1. **확장 없는 기기의 정수 내적(S21, A53)**: 두 기기의 Vulkan 드라이버에는 `VK_KHR_shader_integer_dot_product`가 없습니다. 32비트 정수 곱 한 번으로 2쌍의 8비트 곱 합을 얻는 방식(중간 16비트 필드)으로 `ameva_mmq`·`ameva_mmvq`와 같은 계산을 하는 커널을 추가합니다.
   - 근거: 확장이 있는 Mali(A35)에서 int8 레이아웃이 f32의 0.43~0.77배 시간, 프롬프트는 2.0배.
   - 주의: q8_0 가중치는 곱 합이 16비트를 넘으므로 활성값 7비트 또는 곱별 추출이 필요합니다. 수용 기준은 KL 발산 0.01 이하.
2. **Mali의 1B 프롬프트**(CPU의 0.6~0.8배): QGEMM·MMQ 타일을 Mali에서 측정해 조정합니다(MMQ 타일 BM 64 포함).
3. **출력 투영**(토큰 시간의 20~30 %): 151936행 × 896. 레이아웃 측정은 앞 8192행만 봅니다. 전체 행 기준 측정과 전용 분할을 검토합니다.
4. **CPU 기록 시간**(토큰당 2~9 ms): 토큰마다 같은 그래프를 다시 기록합니다. 기록 재사용 또는 기록 경로 단축.
5. **S25의 대기 시간**: 대기가 GPU 시간보다 길게 나온 실행의 원인을 확인합니다(다른 작업 부하인지, 제출·펜스 대기 지연인지).
6. **S21 거버너**: GPU 클럭이 부하에 따라 낮게 유지됩니다(상관 0.83). 엔진이 제어하지 못합니다. 측정 시 다회 반복이 필요합니다.
7. **FA 끔 경로**: `-fa 0`이면 1.2~14배 느립니다(SOFT_MAX 등 분리 연산). 런타임은 FA를 켠 채 실행합니다.
8. **융합 확대**(QKV 합치기, ROPE+SET_ROWS): 실측에서 융합의 속도 기여가 없어 보류합니다. 디스패치 수가 병목이 되는 경우에만 재검토합니다.
9. **mali_qmv 정리**: 128행 모양 일부에서만 선택됩니다. 더 많은 모델로 측정한 뒤 제거 여부를 정합니다.

### 4.3 기능

1. 양자화 KV 타입 확대(q4_1, q5_0, q5_1, iq4_nl): 속도 이득이 없어 낮은 우선순위입니다.
2. `GET_ROWS`(임베딩 q6_K)는 거절되어 CPU에서 실행됩니다(토큰당 1회).
3. 정확도 판정 모드(`--ref-model`), 이월.
4. STT·디퓨전의 AMEVA 경로: whisper.cpp 서브모듈 여부 결정, STT 배포. 이월.
5. 런타임 쪽: `llama-server`·termux-llamacpp가 플러그인을 쓰게 하기, 기기별 기본 경로(ameva-runtime 문서).

## 5. 환경과 명령

- **번들용 빌드(S25)**: ameva-runtime `docs/RELEASE_AND_MIGRATION.md` 2.2a.
  ```
  git -c core.autocrlf=false -c core.eol=lf archive --format=tar.gz -o ameva-<커밋>.tgz HEAD src include adapters/llama scripts tests
  # 기기: 압축을 풀고
  bash scripts/termux/build_runtime_backend.sh --ggml-dir ~/abi-lab/ggml-b8471 \
       --bundle-lib "$(readlink -f ~/.local/share/ameva/current/llamacpp)/lib" --commit <커밋> --out ~/abi-lab/out-rc
  ```
- **셰이더를 바꾼 개발 빌드**: ameva-runtime `tools/fleet/lab/dev_build.sh`(커널 헤더를 spirv-val과 함께 재생성). 생성된 헤더를 저장소에 반영합니다.
- **측정**: ameva-runtime `tools/fleet/lab/`(`perf_matrix.sh`, `ab_modes.sh`, `parity_new.sh`). 측정 도구도 `LD_LIBRARY_PATH=<번들 lib>:<백엔드 디렉터리>`로 실행합니다.
- **로컬 검사(Windows)**: `bash $TEMP/claude/tools/syn.sh <파일>`(aarch64 구문·형 검사). 셰이더 컴파일과 spirv-val은 기기에서 합니다.

## 6. 알려진 위험과 주의사항

- **속도 보고**: 프롬프트 처리와 토큰 생성을 따로 보고합니다. 생성 속도만 담은 표는 GPU 경로의 이점을 빠뜨립니다.
- **측정 조건**: S25는 다른 작업 부하에 따라 CPU 수치가 14~64 t/s로 움직였고, S20은 클럭 상한이 587 → 441 MHz로 내려갔습니다. 비교는 한 세션에서 번갈아 측정합니다.
- **튠 파일**: 형식이 다른 빌드를 번갈아 실행하면 서로 파일을 다시 씁니다. `AMEVA_QMV_TUNE_FILE`을 빌드별로 따로 줍니다.
- **레이아웃 측정값은 측정 시점의 GPU 클럭에 따라 달라집니다.** 같은 기기에서 세션마다 선택이 다를 수 있습니다(S20: r4t4 / r2t4).
- **ggml-vulkan 비교 실행**은 Mali·Turnip에서 중단되고, 중단 처리기가 디버거를 붙입니다. 반복 실행하지 않습니다.
- **장시간 연속 측정** 뒤 기기 3대에서 Termux 세션이 끝났습니다(원인 미확인). 기기당 측정을 짧게 나눕니다.

## 7. 사용자 협의가 필요한 결정

1. `ameva_mmvq` 기본 정책: 현재 "확장이 있으면 측정해 10 % 넘게 빠를 때 선택"입니다. 수용 기준(KL 발산 0.01 이하)은 통과했습니다. 끄려면 `AMEVA_MMVQ=0`.
2. 4.2-1(확장 없는 기기의 정수 내적) 착수 여부와 우선순위.
3. `vision/v1-pointwise-conv`를 기본 브랜치(`amuda2-phase0-i2s-proof`)로 병합할지.
4. 이월: whisper.cpp 서브모듈 여부, STT 배포.

## 8. 미커밋 변경

- 없음(이 문서, `LATEST.md`, `CLAUDE.md` 4장 갱신을 함께 커밋).
