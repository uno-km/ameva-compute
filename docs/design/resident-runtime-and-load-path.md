# 상주 런타임과 로드 경로 — 설계·분석서 v0.1 (2026-11-16)

지시(2026-11-16): UMLoad(arXiv 2611.04253) 검토 후속.
- (1) 모달리티별 로드 비용을 먼저 측정한다.
- (2) 구조를 갈아엎지 않고 **상주 런타임"도"** 쓸 수 있는 물꼬를 연다.
- (3) GPU·CPU 최대 부하에서 Termux와 상주 런타임이 죽지 않게 한다.
- (4) TTS는 소리를 내지 않는다. WAV를 GPU Whisper(우리 백엔드)와 파형 분석으로 검증하고, CPU 기준 실행은 하지 않는다.

근거 표기: [코드] 저장소 코드, [실측] 기기 측정, [기록] 인수인계서·메모, [추정] 검증 전.

## 1. 현재 구조 (Ground Truth)

### 1.1 실행 단위
- 모달리티마다 별도 CLI 프로세스가 뜨고, 호출마다 다음을 처음부터 반복한다 [코드]:
  1. VkDevice 생성
  2. 모델 mmap
  3. GPU 버퍼 할당
  4. 업로드
  5. 실행
  6. 해제
- 근거는 `AMUDA_ENGINE_UNIFICATION_PLAN.md` 4.8과 ameva-runtime `adapters/*`이다. 상주하는 것은 llama-server뿐이며 LLM 전용이다.
- 설치 번들(S21 기준 `~/.local/share/ameva/current`, 2.7.9/2.8.1)의 GPU 경로 [실측]:

| 모달리티 | 번들 실행 파일 | GPU 경로 | 우리 백엔드 경로 |
|---|---|---|---|
| LLM | llama-cli / llama-completion / llama-server (b8471) | ggml-ameva (2.8.1) | 제품 |
| STT | whisper-cli (ggml 0.25.1) | 업스트림 ggml-vulkan | 실험 빌드 (`stt_build.sh`, `AMEVA_REPACK_ALL=1`) |
| TTS | sherpa-ncnn-offline-tts | ncnn Vulkan | Kokoro 실험 빌드 (`adapters/tts/kokoro`) |
| 확산 | sd-cli-vulkan | 업스트림 ggml-vulkan | 실험 빌드 (sd.cpp 50d6405) |

### 1.2 가중치 로드 경로 [코드]
1. GGUF를 mmap한다.
2. `ac_runtime_buffer_create`가 `vkAllocateMemory`로 HOST_VISIBLE 메모리를 잡고 영구 매핑한다. 드라이버가 페이지를 0으로 채운다.
3. `set_tensor`가 CPU에서 AMEVA 레이아웃으로 재배열한다(스테이징 4 MiB 단위).
4. 재배열한 내용을 매핑 영역에 구간 단위로 쓴다.

UMA인데도 개별 GPU 방식으로 복사하는 구조이며, 재배열 연산이 추가로 붙는다.

### 1.3 첫 계측 [실측, S21, Qwen2.5-0.5B Q4_K_M, 페이지 캐시 웜, 백엔드 = 이 작업 트리]
`AMEVA_LOAD_PROFILE=2`(이번에 추가한 계측) 결과. llama.cpp 보고 load time은 2,332 ms다.

| 단계 | ms | 양 | 비고 |
|---|---|---|---|
| 프로세스 시작 → 백엔드 첫 이벤트 | 476 | — | exec, 링크, 백엔드 등록 |
| context (Vulkan 장치, 파이프라인 캐시) | 27 | — | |
| alloc (`vkAllocateMemory` + 매핑) | **764** | 678 MiB | 888 MiB/s. 드라이버의 페이지 0 채우기가 지배적이다 [추정] |
| src_touch (원본 페이지 첫 접근) | 2 | 373 MiB | 캐시 웜. 콜드는 루트 없이 캐시를 비울 수 없어 따로 표기한다 |
| repack_cpu | 196 | 373 MiB | 1.9 GB/s |
| mapped_write | 49 | 373 MiB | 7.6 GB/s (스테이징 수정 후) |
| 첫 그래프 시각(시작 기준) | 2,620 | — | |

UMLoad가 지적한 "페이지 0 채우기" 비용이 우리에게도 로드의 약 1/3을 차지한다. 이 비용은 **상주 런타임에서는 한 번만 낸다**.

### 1.4 플릿 측정 [실측 2026-11-16, 4대 × 3모델 × 3회, 모두 rc=0, AMEVA0]
- 조건: `-p` 44토큰, `-n 32 -c 1024 -ngl 99`, `AMEVA_LOAD_PROFILE=2`, 1회차는 캐시 콜드에 가깝고 2·3회차는 웜.
- S21은 첫 측정 도중 Termux가 종료되었다(1.6절). 사용자가 재기동한 뒤 단독으로 다시 측정했다(9회, rc=0).
- ls = llama load time ÷ 요청 전체 시간(프로세스 시작부터 끝까지).

| 기기 | 모델 | 전체 ms (웜) | ls | alloc ms | repack_cpu ms | mapped_write ms | 프롬프트 ms | 생성 t/s |
|---|---|---|---|---|---|---|---|---|
| S25 (코어 0-1,4-5) | Qwen 0.5B | 2,973~4,152 | 0.55~0.64 | 414~546 | 471~1,161 | 196~219 | 59~68 | 65~80 |
| | Llama 1B | 7,161~7,768 | 0.72~0.74 | 784~965 | 3,026~3,339 | 450~504 | 120~122 | 48~50 |
| | Gemma 2B | 12,551~14,064 | 0.76~0.77 | 1,607~2,096 | 5,422~6,239 | 857~909 | 240~245 | 24~26 |
| S20 | Qwen 0.5B | 3,013~3,061 | 0.33 | 32~44 | 213 | 79~80 | 262 | 27.9 |
| | Llama 1B | 5,088~5,120 | 0.45 | 41~65 | 1,171~1,174 | 167~170 | 578 | 21.6 |
| | Gemma 2B | 9,460~9,486 | 0.45 | 89~115 | 2,504 | 355~360 | 1,198 | 10.1 |
| A35 | Qwen 0.5B | 8,031~8,148 | 0.47~0.48 | 259~263 | 1,086~1,088 | 309~311 | 557~561 | 21.9 |
| | Llama 1B | 15,576~15,599 | 0.65 | 400~401 | 6,012~6,013 | 613~616 | 609~614 | 13.8 |
| | Gemma 2B | 39,709~40,699 | 0.75 | 1,626~1,696 | 13,002~13,098 | 1,441 | 1,218~1,258 | 6.3 |
| S21 (재측정, 단독) | Qwen 0.5B | 6,314~6,392 | 0.36 | 368~371 | 418 | 112~113 | 296 | 15.2~15.5 |
| | Llama 1B | 9,987~10,389 | 0.45~0.47 | 531~558 | 2,081~2,333 | 221~240 | 616~622 | 10.7~11.0 |
| | Gemma 2B | 20,397~20,429 | 0.52~0.54 | 1,886~1,906 | 4,493~4,871 | 451~489 | 1,291~1,294 | 5.4 |
| A53 | Qwen 0.5B | 14,395~14,423 | 0.51 | 255~259 | 1,095~1,096 | 267~270 | 2,712~2,767 | 16.6 |
| | Llama 1B | 20,857~20,956 | 0.61 | 388~395 | 6,088~6,093 | 535~538 | 2,424~2,429 | 10.9 |
| | Gemma 2B | 41,684~44,755 | 0.71~0.73 | 2,576~3,423 | 12,974~13,050 | 1,208~1,217 | 2,197~2,199 | 5.1 |

해석:
1. **요청 하나에서 로드가 전체 시간의 33~77%를 차지한다.** 상주 런타임은 이 몫을 두 번째 요청부터 없앤다.
2. **CPU 재배열이 로드의 최대 항목이다.** A35·A53은 약 170 MB/s(A55 코어 3개로 제한된 ssh cpuset)로 1B 6초, 2B 13초다. S25(`/moderate` cpuset)도 Gemma가 5~6초다. 1.3절의 "repack 8%"는 S21 0.5B 한 번의 측정값이며 일반화할 수 없었다. 따라서 **RQ-L2(오프라인 재배열 파일)의 우선순위를 높인다**. 재배열은 설치 시 한 번만 하고, 로드 때는 복사만 한다. 상주가 아닌 호출당 경로와 첫 상주 기동 모두에 효과가 있다.
3. alloc은 1회차가 크고(A53 Qwen 2,472 ms vs 255 ms) 웜에서 작아진다. 드라이버 풀이나 페이지 재사용 효과로 보인다 [추정].
4. Gemma의 src_touch가 A35 3.9~4.6초, A53 2.3~4.4초다. 2.2 GB 파일이 페이지 캐시에 다 남지 못하는 메모리 크기에서는 파일 I/O가 다시 나타난다.
5. Gemma 2B load의 약 절반(A35 30초 중 15초)은 AMEVA_LOAD 단계 밖에서 쓰인다. llama.cpp 쪽 파일 읽기, CPU_Mapped 버퍼, KV 초기화 등이며 아직 분해하지 못했다(다음 계측 대상).

### 1.5 STT·TTS [실측 2026-11-16, 우리 백엔드(AMEVA0, `AMEVA_STRICT=1`), CPU 추론 없음, 소리 재생 없음]

**STT**
- 조건: whisper-cli(STT 번들) + `out-stt-load` 플러그인, large-v3-turbo **f16**, jfk 11초 오디오, 3회 실행. 전사는 6회 모두 같은 내용이었다.
- 결과:

| 기기 | load ms | encode ms | 전체 ms | alloc | plain_write |
|---|---|---|---|---|---|
| S21 | 2,289~3,813 | 82,114~174,579 | 87,777~183,619 | 431~1,145 ms / 1,851 MiB | 218~288 ms / 1,550 MiB |
| A35 | 7,520~8,428 | 149,724~149,769 | 163,510~164,427 | 683~1,041 ms | 1,335~1,486 ms |

- 해석: STT는 로드가 아니라 **인코더 계산이 시간의 95% 이상**이다. f16 가중치의 M>1 MUL_MAT이 GEMV용 커널로 실행되는 기존 결함이 원인이다(기록 `ameva-f16-mulmat-path`). f16은 재배열 대상이 아니므로 repack 단계도 0이다.
- 따라서 STT의 우선 과제는 상주 런타임이 아니라 f16 타일 GEMM이나 q5_0 모델 사용이다.
- 원인을 아직 밝히지 못한 두 가지:
  - S21의 encode가 회차마다 2배 흔들린다.
  - A35는 CPU 시간이 실행 시간과 같다(대기가 바쁜 대기일 가능성).

**TTS**
- 조건: Kokoro f16, `ameva_tts_parity --gpu-only`, 문장 t1(18단어)·t2(6단어), S21·A35.

| 항목 | 결과 |
|---|---|
| 합성 시간 | t1 6.3~6.9초(RTF 0.84~0.91), t2 3.1~3.4초 |
| 첫 그래프까지 | 0.40~1.33초. 로드 비율 약 10~20% |
| 반복 일치 | 같은 기기 2회가 비트 단위로 같음(최대 차이 0) |
| 기기 간 차이 | S21 대 A35: SNR 36.9 dB(t1), 41.8 dB(t2) |
| 파형 | 클리핑 0, NaN/Inf 0. t1: RMS −27.0 dBFS, 피크 0.338, 무음 비율 26~28%. t2: RMS −28.0 dBFS, 피크 0.354, 무음 비율 50~52% |
| **GPU Whisper 받아쓰기** | t1, t2 모두 **WER 0.000**(두 기기 모두) |

- 기준 문장의 출처: 테스트 토큰을 Kokoro `tokens.txt`로 음소열로 풀면 "ðə kwˈɪk bɹˈaʊn fˈɑːks … ɡɹˈæfɪks pɹˈɑːsɛsɚ", "ɡˈʊd ˈiːvnɪŋ. ðə tˈɛst ɪz kəmplˈiːt"이다. 기준 문장은 이 음소열에서 정했으므로 Whisper 출력과 독립이다.
- WAV: 스크래치 `tts_wav/{a35,s21}_kokoro_{t1,t2}.r{1,2}.wav`(24 kHz). 사람이 들어 보는 확인은 남아 있다.

### 1.6 S21 Termux 종료 (2026-11-16 약 22:50) — 생존성 사례
- 상황: S21(가용 2.3 GB)에서 두 작업이 동시에 돌았다.
  - LLM 측정(Gemma 2B, GPU 버퍼 약 2.2 GB)
  - STT 측정(whisper f16, GPU 버퍼 1.85 GB)
- 결과: 합계 수요가 가용량을 넘었고, Termux 전체(sshd 포함)가 사라졌다. 기기는 재부팅되지 않았다(uptime 7시간 32분 유지). lmkd가 앱 단위로 종료한 정황과 맞는다.
- 원인: 두 작업을 한 기기에 배치한 오케스트레이션 오류다.
- 대책:
  - 4절의 입장 제어가 두 번째 작업을 거절했어야 한다. guard는 지금 `ameva-run` 경로에만 있고, 실험용 직접 실행에는 없다.
  - 측정 배치 규칙: 한 기기에서는 한 번에 GPU 작업 하나만 돌린다.

## 2. UMLoad 기법의 적용 가능성

### 2.1 호스트 메모리 가져오기(Vulkan) — 이 플릿에서는 불가 [실측 2026-11-16]

| 기기 | GPU / 드라이버 | `VK_EXT_external_memory_host` |
|---|---|---|
| S25 | Adreno 830 (독점) | 없음 |
| S21 | Mali-G78 r38p0 | 없음 |
| S20 | Turnip (Mesa 26.2.3) | 없음 (llvmpipe만 있음: CPU 장치) |
| A53 | Mali-G68 r32p1 | 없음 |
| A35 | Mali-G68 | 미확인 (vulkaninfo 없음; `tools/lab/vk_ext.c`로 확인 예정) |

따라서 UMLoad의 "복사 없는 로드"는 Vulkan으로는 할 수 없다.

### 2.2 남는 경로
1. **OpenCL 호스트 메모리 가져오기**(RQ-L1): Mali는 `cl_arm_import_memory`(host·dma_buf), Adreno는 `cl_qcom_ext_host_ptr`. OpenCL 경로(`AMEVA_BACKEND=opencl`)에서 재배열 파일을 mmap한 뒤 가져오면 복사가 없어진다. 확장 지원과 정렬 조건을 먼저 확인한다.
2. **오프라인 재배열 파일**(RQ-L2): 재배열을 로드 시점에서 설치 시점으로 옮긴다. 1.4절의 플릿 측정에서 repack_cpu는 로드의 최대 항목이다(A35·A53 1B 6초, 2B 13초). **상주 런타임 다음 순위로 채택한다.**
3. **상주 런타임**(3절): alloc·context·exec 비용과 모델 업로드를 요청당 0회로 만든다. 측정상 가장 큰 몫이며 커널 수정이 필요 없다. **채택.**

GPU를 이용한 페이지 0 채우기와 페이지 회수는 커널 드라이버를 고쳐야 하므로 순정 기기에서는 할 수 없다. 더는 검토하지 않는다.

## 3. 상주 런타임 — "물꼬 하나"

### 3.1 원칙
- 기존 CLI 경로(호출당 프로세스)는 그대로 둔다. 상주 경로는 **명시적으로 선택할 때만** 쓴다(`ameva-run serve` / `ameva-run call`).
- 상주 서버가 없는데 `call`을 하면 **오류로 끝낸다**(`AMEVA-RES-E001`). 호출당 실행으로 조용히 돌아가지 않는다.
- 엔진 코드는 바꾸지 않는다. 엔진이 원래 가진 상주형 실행 파일(llama-server 등)을 쓰고, 없는 엔진만 상주 모드를 추가한다.

### 3.2 구성 (ameva-runtime `ameva_runtime/resident/`)
```
ameva-run serve  ──  supervisor (Python, 1개 프로세스)
                      ├─ guard: 메모리·온도·CPU 감시, 작업자 일시정지·종료 (4절)
                      ├─ 제어 소켓: $PREFIX/var/run/ameva/resident.sock (Unix, JSON 한 줄)
                      └─ workers (모달리티별 1개, 모델 상주)
                           llm : llama-server (번들, ggml-ameva) — 127.0.0.1 임의 포트
                           stt : whisper-server (단계 2, 우리 플러그인)
                           tts : kokoro 상주 모드 (단계 2)
                           diff: sd-server (단계 3)
ameva-run call <modality> ...  ──  소켓 → supervisor → 작업자
```
- supervisor는 작업자와 별도 프로세스다. 작업자가 죽어도 supervisor는 남아서 사유를 기록하고, 정책에 따라 다시 띄운다.
- 작업자 실행 환경(`GGML_BACKEND_PATH`, `LD_LIBRARY_PATH`, 장치)은 기존 라우터의 계획(`SmartRouter`, `resolve_inference_environment`)을 그대로 쓴다. 경로 결정 로직을 두 벌로 만들지 않는다.
- 프로세스 수를 최소로 유지한다. 셸 래퍼 없이 직접 exec한다(Android 자식 프로세스 한도 32에 대비).

### 3.3 단계
| 단계 | 내용 | 판정 |
|---|---|---|
| R1 | supervisor, guard, 소켓 프로토콜, LLM 작업자(llama-server) | 상주 2회째 요청에 alloc·context가 0회. 출력이 호출당 실행과 같음(greedy) |
| R2 | STT(whisper-server, 우리 플러그인), TTS(Kokoro 상주) | 같은 기준. TTS는 WAV 검증(5절) |
| R3 | 확산(sd-server), 비전 | 같은 기준 |

### 3.4 R1 구현과 검증 [실측 2026-11-16, S20, ameva-runtime 작업 트리]
- 코드:
  - `python/ameva_runtime/resident/{guard,workers,supervisor}.py`
  - CLI `ameva-run serve | call | resident`
  - `core.execute`(호출당 경로)에도 guard 적용
- 단위 시험 `tests/test_resident.py`: S20 12/12 통과(PC에서는 POSIX 전용 7건을 건너뛰고 나머지 통과). 기존 `test_cpu_route_is_cpu`, `test_llm_gpu_backend`도 통과.
- 실기기 E2E(Qwen2.5-0.5B, AMEVA0, 번들 2.8.1 llama-server):

| 항목 | 결과 |
|---|---|
| 같은 요청(44토큰 프롬프트, 32토큰 생성)의 시간 | 호출당 2,954~3,040 ms, 상주 1,610~1,625 ms. **1.84배 단축** |
| 출력 | 3회 모두 상주 출력이 호출당 출력과 일치(greedy) |
| 상주 기동 | 2.6~4.2초(최초 1회), wakelock 획득 |
| 듀티 700:300 | 출력 동일, 생성 29.3 → 16.5 t/s, 정지 누적 605 ms. Turnip에서 SIGSTOP이 GPU 결과를 바꾸지 않음(Mali는 미확인, RQ-L3) |
| 메모리 guard 실동작 | 900 MiB를 잡는 프로세스로 MemAvailable을 5,045 MiB(하한 5,247 MiB 아래)로 떨어뜨렸다. 작업자만 종료되었고 supervisor는 생존, 사유가 기록되었다. 압박이 계속되는 동안 재기동은 입장 제어가 거절(`AMEVA-RES-E-MEM`)했고, 압박이 풀린 뒤 다음 요청에서 자동으로 재기동되어 정상 응답했다 |
| 정지 후 호출 | `AMEVA-RES-E001`(호출당 실행으로 넘어가지 않음) |

## 4. 생존성 — "최대 부하에서 죽지 않게"

### 4.1 무엇이 죽이는가 [실측 2026-11-16, 읽기 전용 조사]
- Termux 권한으로는 `settings`, PSI, `/proc/swaps`, 시스템 logcat을 모두 읽을 수 없다. 원인을 직접 관측할 수 없다.
- 정황:
  - **LMK(메모리 부족에 따른 강제 종료)** [기록]: Kokoro t3에서 A35 MemAvailable이 2.35 GB에서 0.92 GB로 4초 만에 떨어진 뒤 rc 137로 끝났다. A53은 같은 지점에서 sshd까지 사라졌다. lmkd는 앱 단위(cgroup)로 죽이므로 **Termux 전체가 같이 죽는다**. 5.5 GB 기기(A35·A53)의 평소 가용량은 1.9~2.6 GB다.
  - **자식 프로세스 감시** [실측]: A35는 `persist.sys.fflag.override.settings_enable_monitor_phantom_procs=true`다. S20은 Termux uid 프로세스가 33개로 한도 32를 넘는다.
  - **절전·잠자기** [실측]: A35 화면 상태가 약 60초마다 doze(4↔3)를 오간다. S25·A35·A53은 ssh 사이에 잠들어 작업이 약 4배 느려진다 [기록].
  - **cpuset 제한** [실측]: S25·A35·A53의 ssh 셸은 `/moderate`(제한 코어)에 있다. 죽이지는 않지만 느려진다.
  - 온도: S25·S20만 읽을 수 있다. Samsung `gpu_tmu`는 S21·A35·A53에서 읽힌다.

### 4.2 대책 (모두 루트 불필요)
| 위협 | 대책 | 구현 위치 |
|---|---|---|
| LMK | **입장 제어**: 요청 시작 전 예상 최대 메모리와 MemAvailable 하한을 비교해 부족하면 거절(`AMEVA-RES-E-MEM`). **감시 종료**: 실행 중 MemAvailable이 하한(기본 600 MB, 기기별 조정) 아래로 내려가면 작업자를 먼저 끝내고 사유를 남긴다. lmkd가 Termux 전체를 죽이기 전에 작업자 하나만 희생한다 | guard |
| CPU·온도 포화 | **슬라이싱**: 작업자에 SIGSTOP/SIGCONT 듀티 사이클(예: 800 ms 실행 / 200 ms 정지)을 건다. 온도(`gpu_tmu`, thermal zone, kgsl temp)가 상한을 넘을 때나 사용자가 CPU 상한을 지정할 때 켠다. 엔진을 고치지 않고 모든 엔진에 적용된다. 이미 제출된 GPU 작업은 정지 중에도 끝까지 실행된다 | guard |
| 잠자기 | `am startservice -n com.termux/.app.TermuxService -a com.termux.service_wake_lock`으로 wakelock을 잡고, 종료 시 `service_wake_unlock`으로 놓는다(5대 모두 `am` 있음). 실패는 경고로 기록한다 | supervisor |
| 프로세스 한도 | 래퍼 셸 없이 직접 exec하고 작업자는 모달리티당 1개 | supervisor |
| 작업자 사망 | supervisor가 사유(시그널·rc·guard 판단)를 기록하고 지수 백오프로 재기동한다. 연속 3회 실패하면 그 모달리티를 정지 상태로 두고 `call`에 오류를 반환한다 | supervisor |
| Termux 앱 자체 종료 | 프로세스 안에서는 막을 수 없다. 재기동은 `~/.termux/boot`(Termux:Boot, 부팅 시)와 `termux-job-scheduler`(Termux:API 앱이 있는 기기, A35만 확인)로 한다. 선택 사항이다 | 문서 |

### 4.3 엔진 안의 슬라이싱 (2차)
외부 SIGSTOP 방식으로 부족하면 엔진 내부에서 슬라이싱한다. ggml-ameva의 그래프 제출을 N 노드 단위로 나누고 `AMEVA_DUTY`로 사이에 쉬게 한다. 근거가 생길 때만 한다.

## 5. 측정 계획 (지시 1번)

### 5.1 항목
- 요청 하나의 "프로세스 시작 → 첫 결과"를 다음으로 나누어 잰다.
  - **백엔드 내부**: `AMEVA_LOAD_PROFILE`(context, alloc, repack_cpu, mapped_write, src_touch)
  - **엔진 보고**: llama `load time`, whisper `load time`, sd `loading tensors`
  - **실행**: 프롬프트 처리, 생성 또는 인코드·디코드
- 지표: **로드 비율 = 로드 / 전체**. 이 비율이 상주 런타임의 기대 이득의 상한이다.
- 조건 기록: 다음을 함께 기록한다.
  - `Cpus_allowed_list`
  - GPU 클럭
  - MemAvailable
  - 첫 회(캐시 콜드에 가까움)인지 반복(웜)인지

### 5.2 작업량
| 모달리티 | 기기 | 모델 | 요청 |
|---|---|---|---|
| LLM | 5대 | Qwen2.5-0.5B, Llama-3.2-1B, Gemma-2-2B (Q4_K_M) | 프롬프트 약 64토큰, 생성 32토큰 |
| STT | S21·A35·S25 | large-v3-turbo (f16 / q5_0) | jfk 11 s |
| TTS | Kokoro 자산이 있는 기기 | kokoro-f16 | t1·t2 문장 → WAV |
| 확산 | 실험 빌드가 있는 기기 | sdxs 또는 SD1.x | 1장, 최소 해상도 |

### 5.3 TTS 검증 (소리 재생 금지)
1. Kokoro(우리 GPU 경로)로 WAV를 만든다.
2. **우리 GPU Whisper**(ggml-ameva, `AMEVA_STRICT=1`)로 받아 적고 입력 문장과의 WER을 낸다.
3. 파형을 분석한다(`tts_wav_check.py` 확장):
   - 길이
   - RMS·피크·클리핑 비율
   - 무음 구간 비율
   - 스펙트럼 중심
   - NaN·Inf
   - 같은 입력의 반복 생성이 일치하는지
4. CPU 기준 실행은 하지 않는다(지시).

## 6. 역할 분담
- 메인(Opus): 설계, 계측 코드, 상주 런타임과 guard 구현, 결과 분석과 조치.
- 서브(Sonnet/Haiku): 기기 인벤토리, 배포, 측정 실행, 로그 수집과 1차 요약.

## 7. 연구 질문
- RQ-L1: Mali `cl_arm_import_memory` / Adreno `cl_qcom_ext_host_ptr`로 mmap한 재배열 파일을 복사 없이 쓸 수 있는가? 토큰 간 시간 손실은 얼마인가?
  **결과 (2026-11-17, S21 Mali-G78 r38p0·S20 Adreno 650, 64 MB, 각 2회, 프로브 `tests/ameva_cl_import_file_probe.c`)**: 페이지 캐시 복사 없는 가져오기는 **성립하지 않았다.**
  - S21 `clImportMemoryARM(HOST_ARM)`: 익명 mmap은 성공하고 GPU가 호스트 페이지를 직접 읽는다(쓰기 후 가시성 확인). 파일 mmap(MAP_PRIVATE, 읽기·쓰기)은
    성공하지만 가져온 뒤 `smaps`가 `Private_Clean 64 MB → Private_Dirty·Anonymous 64 MB`로 바뀐다. 드라이버가 쓰기 가능으로 고정하며 페이지를 복사(COW)한 것이다
    (생성 42~45 ms, 같은 크기의 `COPY_HOST_PTR`는 9.4~9.8 ms로 오히려 느림). 파일 mmap(읽기 전용)은 `-6`(CL_OUT_OF_HOST_MEMORY)로 거부된다.
  - `USE_HOST_PTR`: S21·S20 모두 생성은 성공하지만 호스트 쓰기가 GPU에 보이지 않는다(공유 아님, 드라이버가 복사). S20 생성 19~27 ms.
  - S20(Adreno)은 `cl_arm_import_memory`가 없다. `cl_qcom_ion_host_ptr` 경로(dma-buf 가져오기)는 이번 프로브 범위 밖이어서 **미측정**이다.
  - 비용 규모: 64 MB 기준 `COPY_HOST_PTR` 9.2~24 ms, 익명 가져오기 7~22 ms, `pread`로 파일을 익명 버퍼에 읽기 S21 30~31 ms·S20 35 ms(콜드 146 ms).
    2 GB 모델이면 복사 구간이 약 0.3~0.8 s로, 측정된 CPU 재배열(A35·A53 Gemma 2B 13 s)보다 한 자릿수 작다.
  - 결론: 가져오기로 줄일 수 있는 항목은 "GPU 버퍼로의 복사"뿐이고 그 크기가 작다. **재배열 파일(RQ-L2)이 로드 단축의 본체이고, OpenCL 가져오기는 보류한다.**
    미측정: A53(Mali r32p1, ssh 무응답), A35, S25, Adreno ION 경로. 이 결과는 S21 한 기기의 드라이버 동작이다.
- RQ-L2 구현 `AMEVA_REPACK_CACHE=<dir>` + `AMEVA_REPACK_MODEL=<gguf>` (`ggml-ameva-repack-cache.h`, 선택 경로, 미설정 시 기존과 동일):
  모델당 캐시 파일 1개(키 = GGUF 크기와 3지점 1 MiB 표본 해시). 헤더에 포맷·레이아웃 버전·**레이아웃 자체 검사값**(합성 텐서를 현재 코드로 재배열한 해시)을 기록하고,
  텐서별로 이름·타입·형상·GGUF 표본 지문을 확인한다. 없으면 첫 로드에서 기록하고 첫 그래프에서 완성(tmp→fsync→rename), 있으면 재배열 없이 복사.
  불일치(오래된 레이아웃·다른 모델·잘림·변조)는 `AMU_ERR_REPACK_CACHE_*`로 중단하며, 재생성은 `AMEVA_REPACK_CACHE_REBUILD=1`로 명시한다.
  한계: 파일 안의 블롭 비트 오류는 읽을 때 검사하지 않는다. ggml-ameva 재배열을 지나는 텐서만 해당(Kokoro 경로 미확인, whisper는 재배열이 업로드 후라 기록되지 않음).
  검증(2026-11-17, S21, 단위 시험 `ameva_repack_cache_check` 23/23): 캐시 없음·생성·읽기 출력 텍스트 동일(Qwen 5회, Gemma 7회 `cmp`),
  잘린 파일과 모델 경로 누락은 명시적 오류(rc 134).
  측정(S21, 코어 0-2,4-7, 교대 A/C 3쌍, 등급 B: 기기 1대, 잡음 1~3 s):
  | 모델 | 캐시 없음 A 벽시계 | 읽기 C 벽시계 | 첫 그래프까지 A → C | 재배열(CPU) → 캐시 읽기 |
  | :--- | :--- | :--- | :--- | :--- |
  | Qwen2.5-0.5B Q4_K_M (373 MiB) | 5.18, 5.53 s | 3.29, 5.35 s | 3.3, 3.1 → 1.6, 3.1 s | 0.23~0.33 s → 0.16~0.56 s (이득 판정 불가) |
  | Gemma-2-2B Q4_K_M (1623 MiB) | 23.0, 20.4, 18.9 s | 16.3, 15.7, 16.5 s | 16.1, 14.4, 12.8 → 10.2, 9.8, 10.3 s | 4.9~6.4 s → 2.4~3.1 s |
  Gemma에서 첫 그래프까지는 쌍별로 5.9 / 4.6 / 2.5 s 짧아졌다(평균 약 4.3 s). A가 라운드마다 빨라지는 경향이 있어 가장 따뜻한 쌍(2.4~2.5 s)이 하한에 가깝다.
  캐시 읽기 1.6 GiB가 2.4~3.1 s(0.5~0.7 GB/s)인 것은 캐시 파일 페이지가 교대 실행 때문에 밀려나 플래시에서 읽은 영향으로 추정하나 분리해 재지 않았다.
  캐시를 써도 첫 그래프까지 약 10 s가 남는다: GPU 메모리 할당 2.8~3.0 s(2180 MiB, 7회, 드라이버 제로화), 계측 단계 밖 약 4~5 s(원인 미확인).
  A35·A53(재배열 약 170 MB/s)·S25는 ssh 무응답으로 미측정이어서 가장 큰 이득이 예상되는 기기의 결과가 없다.
- RQ-L2: alloc 단계의 MiB/s가 기기·드라이버별로 얼마인가? 페이지 0 채우기가 지배적이라는 추정이 맞는가? (같은 크기를 두 번 할당하는 미세 벤치)
- RQ-L3: 상주 작업자에서 SIGSTOP 듀티 사이클이 GPU 작업(Mali·Adreno 작업 시간 제한)에 부작용을 내는가?
- RQ-L4: MemAvailable 하한의 기기별 적정값(lmkd 임계 직전).
