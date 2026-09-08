# ameva-compute

> **Powered by the AMUDA (Ameva Mobile Unified Device Architecture) Core Engine**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Termux%20%7C%20Android%20ARM64-green.svg)](#)
[![Engineering Doctrine](https://img.shields.io/badge/Doctrine-Zero--Silent--Fallback-red.svg)](#)

`ameva-compute`는 안드로이드 터먹스(Termux) 환경에서 하드웨어 칩셋(SoC)의 GPU 연산 자원을 직접 바인딩하여, 개발자가 엔비디아(NVIDIA) CUDA와 유사한 프로그래밍 모델(SIMT, 3차원 Grid/Block 인덱싱, 로컬 메모리 동기화, 제로카피 버퍼)을 통해 고성능 병렬 연산 커널을 작성하고 디스패치할 수 있도록 설계된 **초경량 저수준 GPGPU 컴퓨팅 파운데이션 런타임**입니다.

---

## 🚀 핵심 아키텍처 특성 (Core Features)

1. **UMA(Unified Memory Architecture) 제로카피 직결**:
   * 모바일 AP(Snapdragon, Exynos)의 공유 물리 LPDDR 메모리를 하드웨어 포인터 수준에서 맵핑하여 호스트-디바이스 복사 오버헤드를 $0\mu\text{s}$에 수렴.
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

## ⚖️ 라이선스 및 상표권 고지 (Legal Disclaimer)

* 본 소프트웨어는 **Apache License, Version 2.0** 라이선스 하에 배포됩니다.
* **상표권 고지**: `CUDA`는 NVIDIA Corporation의 등록 상표입니다. `ameva-compute`는 엔비디아의 소프트웨어, 드라이버, 또는 바이너리를 전혀 참조하지 않는 독립적인 클린룸(Clean-Room) 자체 구현체이며, NVIDIA Corporation과 어떠한 제휴, 후원, 또는 보증 관계도 없습니다.
