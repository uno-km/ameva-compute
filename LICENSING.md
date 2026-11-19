# Licensing

AMEVA Compute (AMUDA: `ac_runtime`, the `ggml-ameva` backend, the engines and adapters, the Android SDK, the tools and
scripts of this repository) is offered under two licenses. You use it under one of them:

| Use | License |
| :--- | :--- |
| Noncommercial: personal study, research, experiment and testing, hobby projects, private entertainment; use by charitable, educational, public research, public safety or health, environmental protection and government organizations | [PolyForm Noncommercial License 1.0.0](LICENSE), free of charge |
| Commercial: any use with an anticipated commercial application, by a company or by an individual (a paid app, a product, a service, internal use at a company) | a commercial license from the licensor, see [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md) |

The SPDX identifier in the source files is `PolyForm-Noncommercial-1.0.0 OR LicenseRef-AMEVA-Commercial`.

Required Notice: Copyright (c) 2026 Uno Kim / AMEVA Foundation (https://github.com/uno-km)

## Earlier versions

Versions of this repository published before this change carried `Apache-2.0` or `Apache-2.0 OR Commercial-License`
in their files. The licenses granted with those versions remain in force for those versions; this change applies to
the code from this commit on.

## Third-party components

Code that is not ours keeps its own license. It is not covered by the table above:

| Component | Where | License |
| :--- | :--- | :--- |
| llama.cpp, whisper.cpp, ggml | `third_party/llama.cpp`, `third_party/whisper.cpp` (submodules) | MIT |
| stable-diffusion.cpp | `third_party/stable-diffusion.cpp` | MIT |
| Khronos OpenCL headers | `include/CL/` | Apache-2.0 |
| patches to the components above | `patches/` | the license of the component they patch |
| sherpa-onnx prebuilts in the Android SDK | fetched at build time (`android/scripts/fetch_sherpa_onnx.sh`) | Apache-2.0 (ONNX Runtime: MIT) |

Model files are not part of this repository; each model has its own license.
