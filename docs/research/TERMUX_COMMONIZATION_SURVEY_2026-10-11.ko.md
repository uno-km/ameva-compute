# termux-* 6종 공통화·품질 전수조사 (2026-10-11)

- **대상:** `dev/termux`의 termux-stt, termux-tts, termux-bitnet, termux-llamacpp, termux-vision, termux-diffusion (Python 패키지 `termux_*`)
- **방법:** AST 정규화 해시로 함수·클래스 중복 측정(함수 1,303개, 클래스 350개), 파일 단위 줄 유사도(difflib), 정규식 패턴 집계, 설치기·로더 코드 직접 확인.
  코드는 수정하지 않았습니다. 기기 실행은 하지 않았습니다.
- **한계:** 정적 분석입니다. "미참조 함수"는 프레임워크 콜백(`do_GET` 등)과 동적 호출을 구분하지 못해 오탐이 섞입니다.
  패턴 집계는 코드 존재만 보여 주며 실제 악용 가능 여부는 호출 경로를 따로 확인해야 합니다. 확인하지 못한 항목은 "미확인"으로 적었습니다.

---

## 0. 결론

1. **공통 계층은 이미 3개 있고, 서로 겹칩니다.** `ameva-device-sdk`(env·hardware·storage), `ameva-component-sdk`(제어 계약·예외·락·상태),
   `ameva-runtime`(platform·installer·router). 새 모듈을 만들기보다 이 셋의 책임을 정리하는 쪽이 맞습니다.
2. **`ameva-device-sdk`는 6개 패키지 어디에서도 import하지 않습니다**(`ameva_device` 참조 0건). 반면 같은 일을 하는 코드가 패키지마다 복사되어 있습니다.
3. 공통화 대상은 크게 세 부류입니다: ① 그대로 옮겨도 되는 것(복사본), ② 파라미터만 다른 것(로그 태그, 경로), ③ 알고리즘이 갈라진 것(도메인 때문에 달라야 하는 것과 우연히 달라진 것이 섞임).
   `engine.py`·`cli.py`·`server.py`는 유사도 0.03~0.13이라 공통화 대상이 아닙니다.
4. 보안·안정성 측면에서 공통화보다 먼저 손볼 것이 있습니다: 설치기의 무결성 검증 부재, 검증 없는 `extractall`, 침묵 `except: pass` 98건, 삭제 명령 부재.

---

## 1. 현재 공통 계층과 사용 현황

| 계층 | 위치 | 제공 | 6종 사용 현황 |
| :--- | :--- | :--- | :--- |
| `ameva-device-sdk` | `dev/ameva/ameva-device-sdk` (Python+Node, 557줄) | `is_termux/android/proot`, 메모리·CPU 기능·스레드·Vulkan 탐지, 모델 경로, SHA-256, 디스크 여유, 원자적 교체, 원자적 다운로드 | **0개** |
| `ameva-component-sdk` | wheel 0.1.1 | `ComponentControl`(ABC), `BaseOrchestratorAdapter`, `AmevaError`(비밀 마스킹), 락, 상태, 하트비트, 매니페스트, 설치기, CLI 지원 | 6개 모두 상속·import. 단 **stt·tts는 `pyproject.toml`에 의존성 선언 없음**(stt는 `adapter.py` 최상단에서 무조건 import) |
| `ameva-runtime` | `python/ameva_runtime` | `platform.py`(is_termux 등), `installer.py`의 `safe_extract_tar`, 예외, 라우터 | bitnet·diffusion은 선언, llama는 폴백 import(`_AMEVA_PLATFORM_AVAILABLE`), 나머지는 부분 사용 |

`is_termux()`만 봐도 6개 패키지에서 **본문이 7종**입니다. 같은 기기에서 도구마다 다르게 판정할 수 있다는 뜻입니다.
예: llama는 `/data/data/com.termux/files/usr`를 보고, bitnet은 `TERMUX_APP_PID`와 `PREFIX`를 보지 않고, diffusion은 예외가 나면 환경 변수만 봅니다.

---

## 2. 중복 측정

### 2.1 파일 단위 (6개 모두에 있는 파일)

| 파일 | 줄 수(stt/tts/bitnet/llama/vision/diff) | 평균 유사도 | 판정 |
| :--- | :--- | :--- | :--- |
| `cluster.py` | 219 / 219 / 219 / 219 / 219 / 340 | **0.89** | 복사본. diff만 확장 |
| `__main__.py` | 7~12 | 0.74 | 복사본 |
| `adapter.py` | 234 / 204 / 234 / 44 / 201 / 80 | 0.31 | 골격 공통 + 도메인 차이 |
| `exceptions.py` | 230 / 122 / 190 / 374 / 526 / 292 | 0.38 | 이름은 같고 내용 분기 |
| `hardware.py` | 482 / 596 / 710 / 1096 / 502 / 1912 | 0.24 | 분기가 큼 |
| `downloader.py` | 80 / 62 / 458 / 914 / 74 / 220 | 0.24 | 분기가 큼 |
| `installer.py` (4종) | 890 / 312 / 573 / 996 (tts·bitnet·vision·diff) | 0.21 | 단계는 같고 구현이 다름 |
| `cli.py` (4종), `engine.py` (4종), `server.py` (2종), `config.py` (2종) | | 0.05~0.13 | **공통화 비대상** |

`control/component.py`는 termux-playwright·termux-train까지 합쳐 8개 패키지에 총 3,285줄입니다.

### 2.2 함수 단위

- AST가 같은 함수(6줄 이상, 2개 패키지 이상): **12종**. 대표:
  `parse_cluster_rpc_spec`(44줄, 6개 패키지), `check_cluster_license`(19줄, 6), `component_info`(19줄, 6), `stop_instance`(25줄, 5),
  `verify_rpc_cluster_health`(13줄, 5), `drain_instance`(11줄, 5), `deactivate_model`(9줄, 5), `is_valid_elf`(19줄, 4).
- 같은 이름이 3개 이상 패키지에 있으나 본문이 다른 함수: **43종**. 본문 종류 수: `is_termux` 7, `download_model` 8, `list_models` 9, `resolve_model_path` 8,
  `doctor_lite` 6, `ensure_wakelock` 6, `_resolve_ameva_runtime` 5, `verify_file_sha256` 5.
- 같은 이름 클래스가 3개 이상 패키지에 있는 것: 14종(`AmevaTermuxError`, `HardwareProfile`, `ModelNotFoundError`, `ClusterConnectionError` 등).

### 2.3 왜 다른가 (원인 분류)

| 원인 | 예 | 공통화 가능성 |
| :--- | :--- | :--- |
| ① 복사 후 문자열만 수정 | `ensure_wakelock`: 로그 태그(`[AMEVA-STT-CLUSTER]` 등)만 다름. `setup_cluster_guard_tunnels` 6종도 같은 이유 | 높음. 태그를 인자로 |
| ② 시기가 달라 계층이 다름 | llama·bitnet은 `ameva_runtime.platform`을 쓰고 없으면 자체 구현으로 폴백. stt·tts·vision은 자체 구현만 | 높음. 한쪽으로 통일 |
| ③ 필드 이름만 다름 | `ram_total_mb` / `total_ram_mb`, `cpu_count` / `cpu_cores`, `soc_model` / `soc_name`, `has_vulkan` / `vulkan_available` | 높음. 정규 이름 + 별칭 |
| ④ 도메인이 필요로 하는 차이 | diffusion: NPU·OpenCL·`recommended_ngl`·cmake 플래그. llama: `recommended_preset`. bitnet: `is_proot` | 확장 필드로 분리 |
| ⑤ 작성자별 알고리즘 차이 | `download_model` 8종(재시도·재개·검증 유무가 다름), `verify_file_sha256` 5종 | 사양을 먼저 정한 뒤 하나로. 가장 어려움 |
| ⑥ 인코딩 손상 | llama `hardware.py`의 한국어 docstring이 깨져 있음(mojibake) | 정리 대상 |

`HardwareProfile`은 6종에 필드가 합쳐 **33개**인데 **6개 모두에 있는 필드는 `has_fp16` 하나**입니다(나머지는 ③의 이름 차이 포함).

---

## 3. 공통화 제안

새 패키지를 만들지 않고, 기존 셋의 책임을 다음과 같이 나눕니다.

| 계층 | 책임 | 옮길 것 |
| :--- | :--- | :--- |
| **L0 `ameva-device-sdk`** (표준 라이브러리만, Python 3.8+, Node 동등) | 기기·파일·바이너리 사실 | ① `is_termux` 등 환경 판정 1종. ② `HardwareProfile` 핵심 필드 + `extras` 사전. ③ 다운로드(재개·재시도·SHA-256·원자적 교체)와 `safe_extract`(`ameva_runtime.safe_extract_tar`를 이동). ④ 바이너리·so 해석기(후보 경로, 버전·해시 검사, 가장 최신 선택). ⑤ 설치 원장과 삭제. ⑥ `ensure_wakelock` |
| **L1 `ameva-component-sdk`** | 제어 계약 | ① 같은 본문인 `stop_instance`·`drain_instance`·`component_info`·`model_status` 기본 구현(Mixin). ② `cluster.py` 단일화(RPC 스펙 파싱·상태 점검·라이선스 가드). ③ 예외 기반 클래스(`AmevaError` 위로 통일, 현재 bitnet만 `RuntimeError` 계열) |
| **L2 `ameva-runtime`** | GPU 런타임·라우터 | `platform.py`는 L0를 import하도록 얇게 |
| **패키지에 남김** | 도메인 | `engine.py`, `cli.py`의 도메인 서브명령, `server.py`, 모델 레지스트리, `HardwareProfile` 확장 필드, 설치 단계 중 도메인 부분 |

규모(현재 줄 수): `cluster.py` 1,435, `hardware.py` 5,298, `downloader.py` 1,808, `exceptions.py` 1,734, `installer.py`(4종) 2,771, `control/component.py` 3,285.
줄이는 양은 약속하지 않습니다. 계약과 시험을 먼저 정한 뒤 측정합니다.

**공통화하지 않을 것:** 추론 엔진 호출부, 서버(llama 1,954줄 / bitnet 473줄), 도메인 CLI, 모델별 후처리. 억지로 합치면 분기가 인자로 새어 나와 오히려 복잡해집니다.

---

## 4. 항목 2~6 조사 결과

### 4.1 버그·안드로이드 정책·보안 (항목 2)

| 항목 | stt | tts | bitnet | llama | vision | diff | 판단 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `except …: pass` (침묵 폴백) | 24 | 17 | 8 | 11 | 6 | 32 | **합계 98건.** 프로젝트 철학 1(침묵 폴백 금지)과 충돌. 건별 분류 필요 |
| `tarfile/zipfile.extractall` | 4 | 2 | 0 | 0 | 1 | 1 | **stt 설치기·허브 4곳은 경로 검증 없음 확인.** diffusion은 검증이 있으나 `startswith(realpath(staging))`이라 접두사가 같은 형제 경로를 막지 못하고 심볼릭 링크도 다루지 않음. tts·vision은 미확인 |
| 설치기 파일의 sha256/hashlib 언급 | 0 | 0 | 0 | - | 1 | 0 | 받은 바이너리·tarball의 무결성 검증 흔적이 없음(다운로더 파일에는 stt 2, bitnet 5, diff 3, llama 17). **공급망 위험** |
| 외부 `http://` URL | 1 | 0 | 0 | 1 | 2 | 2 | stt 1건은 SVG 네임스페이스 문자열. 나머지는 미확인 |
| `/sdcard`·`/storage/emulated` 직접 경로 | 0 | 0 | 0 | 0 | 0 | 11 | diffusion이 DCIM·Pictures에 직접 기록. Android 11 이후 범위 지정 저장소 정책과 `termux-setup-storage` 권한에 의존. 권한이 없을 때의 동작 미확인 |
| `LD_LIBRARY_PATH` 설정 | 11 | 8 | 2 | 10 | 2 | 15 | 합계 48. 과거에 시스템 라이브러리를 가려 `Xzs_Construct`·`egl_get_connection` 오류를 낸 패턴. 호출별 값 확인 필요 |
| `$PREFIX/lib` 관련 언급 | 2 | 3 | 3 | 1 | 5 | 10 | 읽기와 쓰기를 구분하지 않은 집계. 쓰기라면 다른 소유자와 충돌 |
| `shell=True`, `os.system`, `verify=False`, `eval/exec`, `pickle`, 하드코딩 비밀, `su/sudo`, `chmod 777` | 0 | 0 | 0 | 0 | 0 | 0 | 해당 없음 |

버전 관리 결함:
- `setup.py` 버전이 `pyproject.toml`과 어긋납니다: stt 1.5.0↔2.0.5, tts 1.5.4↔1.6.0, bitnet 2.0.1↔2.1.0, diffusion 1.8.0↔2.0.1. 빌드 경로에 따라 잘못된 버전의 패키지가 만들어집니다.
- stt·tts의 `ameva-component-sdk` 미선언 의존(위 1절).
- `requires-python`: 3개는 `>=3.10`, 3개는 `>=3.8`(정리 필요).

### 4.2 자원 정리 (항목 3)

- llama `cli.py:86`, `server.py:882`: `open(..., "a")`를 `with` 없이 열었습니다. 닫는 경로는 확인하지 못했습니다.
- `daemon=True` 스레드 12곳, `Popen` 6곳, stt의 `mkdtemp/mkstemp` 2곳: 종료·정리 경로 미확인.
- 위 `except: pass`가 정리 실패를 가릴 수 있어 누수 진단을 어렵게 합니다.
- 메모리 누수는 정적 분석으로 확정할 수 없습니다. 실기기에서 반복 실행 후 RSS 추이를 재야 합니다(`test_batch_graph_kat`의 MEMORY 시험 방식 참고).

### 4.3 죽은 코드·불필요 요소 (항목 4)

참조 횟수 1(정의뿐)인 함수: stt 15, tts 30, bitnet 19, llama 10, vision 24, diff 32. **오탐이 많습니다**(`do_GET/do_POST`, ABC 재정의, 프로토콜로 동적 호출되는 `start_instance` 등).
오탐을 제외하고 실제로 의심되는 것:

| 대상 | 규모 | 비고 |
| :--- | ---: | :--- |
| `setup_cluster_guard_tunnels` | 44줄 × 6개 | 6개 패키지 모두 정의만 있고 호출 없음 |
| vision `vlm/pipeline.py` | 76줄 | 어디서도 import되지 않는 모듈 |
| vision `execute_tvrp`, `_is_vulkan_failure` | 51줄, 31줄 | 참조 없음 |
| bitnet `verify_file_sha256` | 27줄 | **정의만 있고 설치기가 쓰지 않음**(4.1의 무결성 부재와 연결) |
| diffusion `workflow.py`의 `handle_*` | 106·49·44줄 | 디스패치 표로 호출될 가능성이 있어 미확정 |
| 추적 대상 `.so` | 0개 | git에는 `.so`가 없음. `releases/` 디렉터리에 있음(4.4) |

### 4.4 예전 버전·so 사용 (항목 5)

- bitnet `engine.py:106~120`은 `libtermux_bitnet.so`를 **후보 11곳에서 먼저 발견된 것**으로 씁니다. 순서는 `$PREFIX/lib` → `/data/data/com.termux/files/usr/lib` →
  `~/.local/share/ameva/current/bitnet/lib` → `/usr/local/lib` → `/usr/lib` → 패키지 내부·빌드 디렉터리입니다.
  **낡은 사본이 `$PREFIX/lib`에 남아 있으면 새 파일보다 먼저 로드됩니다.** 해시·버전·ABI 비교 코드는 후보 목록 부분에서 보이지 않습니다.
- 저장소 내 보관본: `releases/legacy_archives/libtermux_bitnet_a53.so`(120,760 B, 2026-10-01), `termux-bitnet/releases/libtermux_bitnet.so`(130,704 B, 2026-10-01),
  `termux-diffusion/releases/libomp-android-arm64.so`(1.2 MB, 2026-09-18). 어떤 것이 현행인지 가리키는 매니페스트는 확인하지 못했습니다.
- "항상 최신을 쓰게 하는" 방식 제안: 설치 시 `~/.local/share/ameva/releases/<버전>/`에 격리하고 `current` 링크를 바꾸는 방식(ameva-runtime 2.7.9의 번들 격리 구조와 동일).
  로더는 `current`만 보고, 후보 목록 방식은 폐기합니다. so마다 SHA-256과 ABI 버전을 매니페스트에 두고 불일치하면 명시적으로 거부합니다
  (`ameva_abi_check`와 같은 원칙, 조용한 폴백 금지).

### 4.5 이미 설치된 파일 덮어쓰기·삭제 (항목 6)

- **삭제(uninstall) 명령은 6종 어디에도 없습니다.** CLI 서브명령 목록:
  stt `app benchmark demo diarize doctor install listen live models transcribe web` /
  tts `compare create delete doctor inspect install list profile speak synth` /
  bitnet `benchmark chat doctor download info install models run serve` /
  llamacpp `doctor download find hardware install list models presets run serve stop` /
  vision `benchmark canny detect-face doctor download info install list model remove vlm` /
  diffusion `clear-cache doctor download generate inspect install models run web workflow`.
  모델 단위 삭제(`remove`, `delete`, `clear-cache`)만 일부에 있고, **설치한 바이너리·so·링크·원장을 지우는 명령은 없습니다.**
- 설치 시 덮어쓰기 동작은 패키지별로 확인하지 못했습니다. ameva-runtime은 `ameva-run install`이 남의 명령을 만나면 거부하고 `--force`가 필요하며 덮어쓴 파일을 백업하지 않는다는 기록이 있습니다.
- 제안: L0에 **설치 원장**(설치한 파일 경로·SHA-256·소유 패키지·버전)을 두고, 모든 설치기가 원장에 기록하게 합니다.
  `termux-<x> uninstall [--purge-models] [--dry-run]`은 원장에 있는 것만 지우고, 원장에 없는 파일(과거 설치분, 다른 소유자)은 발견만 하고 지우지 않으며 `--adopt`로 편입합니다.
  덮어쓸 때는 기존 파일을 해시로 비교하고 다르면 백업 후 교체, 같은 이름이 `$PREFIX/lib`에 남은 낡은 사본이면 경고합니다.

---

## 5. 단계별 계획 (위험이 낮고 효과가 큰 순)

| 단계 | 내용 | 검증 |
| :--- | :--- | :--- |
| **P0 안전** (구조 변경 없음) | ① 설치기에 SHA-256 검증 추가(bitnet 포함). ② stt·diffusion·tts·vision의 `extractall`을 `safe_extract`로 교체. ③ `setup.py` 버전 4건 정정, 의존성 선언 보완. ④ `except: pass` 98건을 건별로 분류(로그+재발생 / 허용 사유 주석) | 패키지별 단위 시험 + 5대 기기 설치·`doctor` |
| **P1 L0 채택** | `ameva-device-sdk`에 `safe_extract`·다운로드·바이너리 해석기 보강 후 6개 패키지가 `is_termux`부터 import(본문 7종 → 1종). 필드 이름 정규화와 별칭 | 골든 시험: 5대 기기에서 기존 판정값과 동일한지 비교 |
| **P2 L1 정리** | `cluster.py`를 `ameva_component`로 이동(복사 5벌 제거), 제어 Mixin, 예외 기반 통일 | 기존 `control` 프로토콜 시험, 실기기 `activate/start/stop` |
| **P3 하드웨어** | `HardwareProfile` 핵심 + `extras`, 6개 `hardware.py` 축소 | 기기별 프로필 덤프 전후 비교 |
| **P4 수명주기** | 설치 원장, `uninstall`, 버전 격리 + `current` 링크, so 매니페스트 | 신규 설치·업그레이드·삭제·낡은 사본 존재 상태의 5대 기기 시나리오 |

모든 단계는 **동작 보존**이 원칙이고, 단계마다 패키지 시험과 실기기 확인을 통과해야 다음으로 넘어갑니다.
배포(PyPI·npm)는 이 조사 범위 밖이며 별도 승인 사항입니다.

---

## 6. 결정이 필요한 사항

1. **SSOT 위치:** 기기 사실은 `ameva-device-sdk`로 모으고 `ameva-runtime/platform.py`가 그것을 쓰게 하는 안을 권합니다(device-sdk가 표준 라이브러리만 쓰고 Node 동등 구현이 있기 때문). 다른 의견이 있으면 알려 주십시오.
2. **Python 하한:** `>=3.8` 3종을 3.10으로 올릴지.
3. **P0 착수 여부:** 구조 변경 없이 바로 가능하고 보안상 가치가 가장 큽니다. 승인하시면 패키지별 브랜치로 진행합니다.
