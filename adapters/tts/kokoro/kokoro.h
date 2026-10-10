// SPDX-License-Identifier: Apache-2.0
// Kokoro-82M v0.19 text-to-speech on one ggml backend (ggml-cpu or AMEVA0), from the GGUF written by
// adapters/tts/kokoro/convert_onnx_to_gguf.py. Input: phoneme ids of tokens.txt (without the pad id 0 at both ends,
// which is added here); output: 24 kHz mono float samples.
//
// The whole network runs as two ggml graphs on the backend the model was loaded on:
//   1. text side: PL-BERT (ALBERT), the duration encoder, the duration LSTM and projection, and the text encoder;
//   2. after the host rounds the durations (round half to even, at least 1 frame, as the export does) and lays the
//      frames out: frame alignment (GET_ROWS), the shared LSTM, F0 / energy predictors, the decoder, the harmonic
//      source (SineGen), its STFT, the iSTFTNet generator and the inverse STFT.
// Every node of both graphs must be supported by the backend; an unsupported one fails the call with its op and name
// (there is no scheduler and no other backend, so nothing can move to the CPU). The host does: style row lookup
// (voices.bin), the SineGen noise (kokoro_noise: splitmix64 + Box-Muller, also in reference_onnx.py), duration
// rounding, the frame index list and the inverse-STFT window sum (a function of the length only).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ggml_backend;
struct kokoro_model;

struct kokoro_options {
    int voice = 0;
    float speed = 1.0f;
    uint64_t seed = 1234;
    // when non-empty: frame counts per token (pads included) used instead of the predicted ones, so that two
    // backends can be compared on the same alignment
    std::vector<int32_t> durations;
};

struct kokoro_result {
    std::vector<float> audio;          // 24 kHz
    std::vector<float> dur_raw;        // sum of sigmoid(duration_proj) / speed per token
    std::vector<int32_t> durations;    // frames per token (predicted or given)
    std::vector<float> f0;             // F0 curve, 2 values per frame
    double ms_text = 0, ms_synth = 0, ms_host = 0;  // graph 1, graph 2 (both incl. upload/download), host prep
    int nodes_text = 0, nodes_synth = 0;
};

// backend: an initialized ggml backend; the model's weights are placed in its default buffer type.
kokoro_model* kokoro_load(const char* gguf_path, ggml_backend* backend, std::string* err);
void kokoro_free(kokoro_model* m);
int kokoro_sample_rate(const kokoro_model* m);
int kokoro_n_voices(const kokoro_model* m);
std::string kokoro_weight_type(const kokoro_model* m);
bool kokoro_synthesize(kokoro_model* m, const std::vector<int32_t>& phonemes, const kokoro_options& opt,
                       kokoro_result* out, std::string* err);

// the SineGen noise: n standard normal values for `seed` (identical to reference_onnx.py kokoro_noise)
void kokoro_noise(uint64_t seed, size_t n, float* out);
