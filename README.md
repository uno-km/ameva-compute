# ameva-compute

> **Powered by the AMUDA (Ameva Mobile Unified Device Architecture) Core Engine**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Termux%20%7C%20Android%20ARM64-green.svg)](#)
[![Engineering Doctrine](https://img.shields.io/badge/Doctrine-Zero--Silent--Fallback-red.svg)](#)

`ameva-compute`는 ARM 모바일 SoC(Snapdragon, Exynos)의 GPU 연산 자원을 Vulkan 1.3 / SPIR-V로 직접 바인딩하여, 엔비디아(NVIDIA) CUDA와 유사한 프로그래밍 모델(SIMT, 3차원 Grid/Block 인덱싱, 로컬 메모리 동기화, 제로카피 버퍼)로 병렬 연산 커널을 작성·디스패치하는 **저수준 GPGPU 컴퓨팅 파운데이션 런타임**입니다. 원래 Termux 환경에서 개발되었으나, Termux는 개발/검증 환경일 뿐 배포 대상이 아닙니다 — 이 저장소는 코어 엔진(C ABI)만 제공하며, 실제 배포 채널은 하단 "생태계 구성"을 참고하십시오.

---

## 📦 생태계 구성 (Ecosystem Layout)

이 저장소(`ameva-compute`)는 세 계층 중 **코어 엔진**만 담당합니다. 다른 계층을 찾는 경우 아래 저장소를 참고하십시오.

| 계층 | 저장소 | 용도 |
|---|---|---|
| 코어 엔진(C ABI) | `ameva-compute` (본 저장소) | `ac_runtime`, `amuda.h`, Vulkan/OpenCL 디스패치 |
| 번들 런타임 | [`ameva-runtime`](https://github.com/uno-km/ameva-runtime-releases) | llama.cpp 등을 포함한 PyPI/npm 설치형 패키지 |
| 앱용 SDK | Android AAR (`android/`) | 코어 엔진을 인앱으로 임베드하는 Kotlin API |

## 🚀 핵심 아키텍처 특성 (Core Features)

1. **UMA(Unified Memory Architecture) 제로카피 매핑**:
   * 모바일 AP의 공유 물리 LPDDR 메모리를 하드웨어 포인터 수준에서 맵핑하여 호스트-디바이스 복사를 생략한다. 실측 오버헤드는 워크로드와 기기에 따라 다르며, 측정값은 `evidence/`를 참고한다.
2. **제조사 하드웨어 쿼크(Quirks) 내장 격벽**:
   * **ARM Mali (Galaxy A35 등)**: 서브그룹 크기 16 환경에서 정수 나눗셈 `loadstride_b = 0`으로 인한 GPU 무한 루프 버그 방어 (`ShouldEnforceMediumMatMulKernel`).
   * **Qualcomm Adreno (Galaxy S20, S25 등)**: 서브그룹 제어 pNext 구조체 결함 우회 및 FP16 Shared Memory 리덕션 강제 라우팅.
3. **CUDA 소스 레벨 호환 헤더 (`include/amuda.h`)**:
   * `__global__`, `threadIdx.x`, `blockIdx.x`, `__syncthreads()`, `atomicAdd` 지원.
4. **엄격한 무(無)폴백 (Zero-Silent-Fallback)**:
   * 하드웨어 오류 발생 시 조용히 CPU로 전환하거나 가짜 데이터를 반환하지 않고, 즉시 명시적 에러 코드(`AcResult`) 및 원인 스택 트레이스 분출.

---

## 🛠️ 빠른 시작 (Quick Start)

### C / C++
```c
#include "amuda.h"

int main() {
    AcContext* ctx = NULL;
    if (ac_init(&ctx) != AC_SUCCESS) return 1;

    AcBuffer* buf = NULL;
    ac_buffer_alloc(ctx, 1024 * sizeof(float), &buf);

    // 커널 디스패치 및 연산
    // ...

    ac_buffer_free(ctx, buf);
    ac_destroy(ctx);
    return 0;
}
```

### Python
```python
import ameva_compute as amuda
import numpy as np

ctx = amuda.init()
print(ctx.device_info)

# UMA 제로카피 버퍼 생성
a = np.ones(100000, dtype=np.float32)
buf_a = ctx.buffer(a)

# SIMT 커널 디스패치
# ...
```

---

## ⚠️ 검증된 범위와 한계 (Verified Scope & Limitations)

본 문서의 다른 어떤 claim보다 이 항목을 먼저 읽을 것을 권장한다. 측정되지 않은 성능은 주장하지 않는다(CLAUDE.md 철학 1).

* **디코드(토큰 생성) 단계는 메모리 대역 한계이며, 기기에 따라 GPU가 CPU를 넘지 못한다.** 프롬프트(배치) 단계만 GPU가
  일관되게 유리하다. 두 수치는 항상 분리해서 보고한다.
* **디퓨전(stable-diffusion.cpp 어댑터) GPU 경로가 CPU보다 느린 경우가 측정되었다** (S21, 256x256, ~3배). 모델/해상도별로
  재측정 전에는 "GPU 가속"을 일반화해서 말하지 않는다.
* **STT 두 엔진은 GPU 가속 범위가 다르다.** `AmevaSpeech`(whisper.cpp)는 일부 matmul 모양만 GPU에서 실행되고 나머지는
  ggml 스케줄러가 배치하는 곳(통상 CPU)에서 실행된다. `AmevaSenseVoice`(sherpa-onnx)는 **AMEVA GPU 경로를 전혀 쓰지 않는
  순수 CPU/ONNX Runtime 엔진**이다.
* **모든 claim은 구체적 기기·커밋·측정 조건과 함께만 유효하다.** 플릿별 수치와 조건은 `evidence/`와 릴리스 노트를 근거로
  한다. "GPU 가속"이라는 표현만으로는 어떤 경로가 실행되었는지 증명되지 않는다 — 프로세스 정상 종료는 GPU 실행의 증거가
  아니다.

---

## ⚖️ 라이선스 및 상표권 고지 (Legal Disclaimer)

* 본 소프트웨어는 **Apache License, Version 2.0** 라이선스 하에 배포됩니다.
* **상표권 고지**: `CUDA`는 NVIDIA Corporation의 등록 상표입니다. `ameva-compute`는 엔비디아의 소프트웨어, 드라이버, 또는 바이너리를 전혀 참조하지 않는 독립적인 클린룸(Clean-Room) 자체 구현체이며, NVIDIA Corporation과 어떠한 제휴, 후원, 또는 보증 관계도 없습니다.
