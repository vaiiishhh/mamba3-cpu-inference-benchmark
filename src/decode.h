#pragma once

#include "kernels.h"

#include <array>
#include <cstddef>
#include <vector>

namespace mamba {

struct Config {
    std::size_t model, heads, width, states, rank, groups, angles, model_width;

    void validate() const;
    std::size_t inner() const { return heads * width; }
    std::size_t packed() const;
};

struct Weights {
    explicit Weights(const Config &c);

    std::vector<float> in, out, dt_bias, b_norm, c_norm, b_bias, c_bias, skip;
    std::vector<float> value_scale, gate_scale, output_scale;
};

struct State {
    explicit State(const Config &c);

    std::vector<float> matrix, phase, prev_b, prev_x;
};

struct Scratch {
    explicit Scratch(const Config &c);

    std::vector<float> packed, inner, norm_b, norm_c, b, c, current, previous, readout;
};

// Phase durations in seconds. Intended for single-worker diagnostic runs.
using PhaseTimes = std::array<double, 5>;

void operator_step(const Config &c, const Weights &w, const float *projected,
                   State &state, Scratch &scratch, float *output, HeadKernel kernel,
                   int threads = 1, PhaseTimes *profile = nullptr);
void mixer_step(const Config &c, const Weights &w, const float *hidden,
                State &state, Scratch &scratch, float *output, HeadKernel kernel,
                int threads = 1, PhaseTimes *profile = nullptr);
void project_input(const Config &c, const Weights &w, const float *hidden, float *projected);

} // namespace mamba
