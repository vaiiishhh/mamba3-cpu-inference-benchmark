#include "decode.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace mamba {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float pi = 3.14159265358979323846f;

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu(float x) { return x * sigmoid(x); }

float softplus(float x) {
    return x > 20.0f ? x : std::log1p(std::exp(x));
}

float decay_parameter(float x) {
    const float positive = x >= 0.0f ? 1.0f + x : 1.0f / (1.0f - x);
    return -std::max(positive, 1e-4f);
}

void rms_norm(const float *input, const float *weight, std::size_t rows,
              std::size_t width, float *output) {
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < width; ++i) sum += input[i] * input[i];
        const float inv = 1.0f / std::sqrt(sum / static_cast<float>(width) + 1e-5f);
        for (std::size_t i = 0; i < width; ++i) output[i] = input[i] * inv * weight[i];
        input += width;
        output += width;
    }
}

void linear(const float *input, const float *weights, std::size_t rows,
            std::size_t cols, float *output) {
    for (std::size_t i = 0; i < rows; ++i) {
        const float *row = weights + i * cols;
        float lanes[8]{};
        std::size_t j = 0;
        for (; j + 8 <= cols; j += 8) {
            for (std::size_t lane = 0; lane < 8; ++lane) lanes[lane] += row[j + lane] * input[j + lane];
        }
        float sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
                    ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        for (; j < cols; ++j) sum += row[j] * input[j];
        output[i] = sum;
    }
}

void rotate(float *values, std::size_t left, std::size_t right, float cosine, float sine) {
    const float a = values[left];
    const float b = values[right];
    values[left] = a * cosine - b * sine;
    values[right] = a * sine + b * cosine;
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

} // namespace

void Config::validate() const {
    if (model != 2 && model != 3) throw std::invalid_argument("unsupported model");
    if (!heads || !width || !states || !groups || heads % groups ||
        heads > 1024 || width > 1024 || states > 1024 || model_width > 4096 ||
        heads * width * states > 16000000) {
        throw std::invalid_argument("invalid decode dimensions");
    }
    if ((rank != 1 && (model != 3 || rank != 4)) || angles > states / 2 ||
        (model == 3 && states % 2) || (model == 2 && (angles || model_width))) {
        throw std::invalid_argument("invalid rank, rotary, or mixer configuration");
    }
    if (model_width * (packed() + inner()) > 30000000) {
        throw std::invalid_argument("mixer weight dimensions exceed the fixture budget");
    }
}

std::size_t Config::packed() const {
    return 2 * inner() + 2 * rank * groups * states + (model == 3 ? 3 * heads + angles : 2 * heads);
}

Weights::Weights(const Config &c)
    : in(c.packed() * c.model_width), out(c.model_width * c.inner()),
      dt_bias(c.model == 3 ? c.heads : 0), b_norm(c.model == 3 ? c.states : 0, 1.0f),
      c_norm(b_norm), b_bias(c.model == 3 ? c.heads * c.rank * c.states : 0),
      c_bias(b_bias), skip(c.heads, 1.0f),
      value_scale(c.model == 3 ? c.rank * c.inner() : 0, 1.0f),
      gate_scale(value_scale), output_scale(value_scale) {}

State::State(const Config &c)
    : matrix(c.inner() * c.states), phase(c.heads * c.angles),
      prev_b(c.model == 3 ? c.heads * c.rank * c.states : 0),
      prev_x(c.model == 3 ? c.inner() : 0) {}

Scratch::Scratch(const Config &c)
    : packed(c.model_width ? c.packed() : 0), inner(c.inner()),
      norm_b(c.model == 3 ? c.rank * c.groups * c.states : 0), norm_c(norm_b),
      b(c.model == 3 ? c.heads * c.rank * c.states : 0), c(b),
      current(c.rank * c.inner()), previous(c.model == 3 ? c.rank * c.inner() : 0),
      readout(current) {}

void operator_step(const Config &cfg, const Weights &w, const float *input,
                   State &state, Scratch &tmp, float *output, HeadKernel kernel,
                   int threads, PhaseTimes *profile) {
    if (profile && threads != 1) {
        throw std::invalid_argument("phase profiling requires a single worker");
    }
    const auto h_count = cfg.heads;
    const auto width = cfg.width;
    const auto n = cfg.states;
    const auto rank = cfg.rank;
    const auto groups = cfg.groups;
    const auto angles = cfg.angles;
    const bool mamba3 = cfg.model == 3;
    const std::size_t bc_size = rank * groups * n;
    const float *z = input;
    const float *x = z + cfg.inner();
    const float *input_b = x + cfg.inner();
    const float *input_c = input_b + bc_size;
    const float *input_dt = input_c + bc_size;
    const float *input_a = input_dt + h_count;
    const float *input_trap = input_a + h_count;
    const float *input_angle = mamba3 ? input_trap + h_count : nullptr;

    if (mamba3) {
        const auto start = profile ? Clock::now() : Clock::time_point{};
        rms_norm(input_b, w.b_norm.data(), rank * groups, n, tmp.norm_b.data());
        rms_norm(input_c, w.c_norm.data(), rank * groups, n, tmp.norm_c.data());
        if (profile) (*profile)[1] += seconds_since(start);
    }

#if defined(_OPENMP)
#pragma omp parallel for if(threads > 1) num_threads(threads) schedule(static)
#else
    (void)threads;
#endif
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(h_count); ++index) {
        const auto h = static_cast<std::size_t>(index);
        const std::size_t group = h / (h_count / groups);
        const std::size_t hp = h * width;
        const std::size_t rp = h * rank * width;
        const std::size_t rn = h * rank * n;
        const float *x_head = x + hp;
        float *current = tmp.current.data() + rp;
        float *readout = tmp.readout.data() + rp;
        const float *b = input_b + group * n;
        const float *c = input_c + group * n;
        const float *prev_b = nullptr;
        float *previous = nullptr;

        auto start = profile ? Clock::now() : Clock::time_point{};
        const float dt = mamba3 ? softplus(input_dt[h] + w.dt_bias[h]) : input_dt[h];
        const float a = mamba3 ? decay_parameter(input_a[h]) : input_a[h];
        const float alpha = std::exp(a * dt);
        float gamma = dt;
        if (mamba3) {
            const float mix = sigmoid(input_trap[h]);
            gamma *= mix;
            const float beta = (1.0f - mix) * dt * alpha;
            float *rot_b = tmp.b.data() + rn;
            float *rot_c = tmp.c.data() + rn;
            float *phase = angles ? state.phase.data() + h * angles : nullptr;
            for (std::size_t r = 0; r < rank; ++r) {
                const std::size_t source = (r * groups + group) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    rot_b[r * n + i] = tmp.norm_b[source + i] + w.b_bias[rn + r * n + i];
                    rot_c[r * n + i] = tmp.norm_c[source + i] + w.c_bias[rn + r * n + i];
                }
            }
            for (std::size_t i = 0; i < angles; ++i) {
                phase[i] += std::tanh(input_angle[i]) * dt * pi;
                const float cosine = std::cos(phase[i]);
                const float sine = std::sin(phase[i]);
                const std::size_t left = rank == 1 ? 2 * i : i;
                const std::size_t right = rank == 1 ? left + 1 : i + n / 2;
                for (std::size_t r = 0; r < rank; ++r) {
                    rotate(rot_b + r * n, left, right, cosine, sine);
                    rotate(rot_c + r * n, left, right, cosine, sine);
                }
            }
            b = rot_b;
            c = rot_c;
            prev_b = state.prev_b.data() + rn;
            previous = tmp.previous.data() + rp;
            for (std::size_t r = 0; r < rank; ++r) {
                for (std::size_t p = 0; p < width; ++p) {
                    const std::size_t i = r * width + p;
                    const float scale = rank == 1 ? 1.0f : w.value_scale[rp + i];
                    current[i] = (x_head[p] * scale) * gamma;
                    previous[i] = (state.prev_x[hp + p] * scale) * beta;
                }
            }
        } else {
            for (std::size_t p = 0; p < width; ++p) current[p] = x_head[p] * gamma;
        }
        if (profile) {
            (*profile)[1] += seconds_since(start);
            start = Clock::now();
        }
        kernel({width, n, rank, alpha, state.matrix.data() + hp * n,
                b, prev_b, c, current, previous, readout});
        if (profile) {
            (*profile)[2] += seconds_since(start);
            start = Clock::now();
        }
        for (std::size_t p = 0; p < width; ++p) {
            if (rank == 1) {
                output[hp + p] = (readout[p] + w.skip[h] * x_head[p]) * silu(z[hp + p]);
            } else {
                float result = 0.0f;
                for (std::size_t r = 0; r < rank; ++r) {
                    const std::size_t i = r * width + p;
                    const float value = x_head[p] * w.value_scale[rp + i];
                    const float gate = silu(z[hp + p] * w.gate_scale[rp + i]);
                    result += (readout[i] + w.skip[h] * value) * gate * w.output_scale[rp + i];
                }
                output[hp + p] = result;
            }
        }
        if (mamba3) {
            std::copy_n(b, rank * n, state.prev_b.data() + rn);
            std::copy_n(x_head, width, state.prev_x.data() + hp);
        }
        if (profile) (*profile)[3] += seconds_since(start);
    }
}

void project_input(const Config &c, const Weights &w, const float *hidden, float *projected) {
    linear(hidden, w.in.data(), c.packed(), c.model_width, projected);
}

void mixer_step(const Config &c, const Weights &w, const float *hidden,
                State &state, Scratch &tmp, float *output, HeadKernel kernel,
                int threads, PhaseTimes *profile) {
    auto start = profile ? Clock::now() : Clock::time_point{};
    project_input(c, w, hidden, tmp.packed.data());
    if (profile) (*profile)[0] += seconds_since(start);
    operator_step(c, w, tmp.packed.data(), state, tmp, tmp.inner.data(), kernel, threads, profile);
    if (profile) start = Clock::now();
    linear(tmp.inner.data(), w.out.data(), c.model_width, c.inner(), output);
    if (profile) (*profile)[4] += seconds_since(start);
}

} // namespace mamba
