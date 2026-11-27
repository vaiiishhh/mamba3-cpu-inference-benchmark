#include "kernels.h"

#include <stdexcept>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define MAMBA_SIMD_NEON
#elif defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
#include <immintrin.h>
#define MAMBA_SIMD_AVX2
#endif

namespace mamba {
namespace {

#if defined(MAMBA_SIMD_NEON)
using Vector = float32x4_t;
constexpr std::size_t lanes = 4;
constexpr const char *backend = "neon";

inline Vector load(const float *data) { return vld1q_f32(data); }
inline void store(float *data, Vector v) { vst1q_f32(data, v); }
inline Vector broadcast(float value) { return vdupq_n_f32(value); }
inline Vector multiply(Vector a, Vector b) { return vmulq_f32(a, b); }
inline Vector multiply_add(Vector sum, Vector a, Vector b) { return vfmaq_f32(sum, a, b); }
inline float reduce(Vector v) { return vaddvq_f32(v); }
#elif defined(MAMBA_SIMD_AVX2)
using Vector = __m256;
constexpr std::size_t lanes = 8;
constexpr const char *backend = "avx2-fma";

inline Vector load(const float *data) { return _mm256_loadu_ps(data); }
inline void store(float *data, Vector v) { _mm256_storeu_ps(data, v); }
inline Vector broadcast(float value) { return _mm256_set1_ps(value); }
inline Vector multiply(Vector a, Vector b) { return _mm256_mul_ps(a, b); }
inline Vector multiply_add(Vector sum, Vector a, Vector b) { return _mm256_fmadd_ps(a, b, sum); }
inline float reduce(Vector v) {
    const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 pairs = _mm_hadd_ps(halves, halves);
    return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
}
#else
constexpr const char *backend = nullptr;
#endif

#if defined(MAMBA_SIMD_NEON) || defined(MAMBA_SIMD_AVX2)
template <std::size_t Rank, bool HasPrevious>
void update(const HeadStep &step) {
    const Vector alpha = broadcast(step.alpha);
    const std::size_t tail = step.states % lanes;
    const std::size_t vector_end = step.states - tail;
    for (std::size_t row = 0; row < step.width; ++row) {
        float *state_row = step.state + row * step.states;
        Vector current[Rank];
        Vector previous[Rank];
        Vector sums[Rank];
        for (std::size_t rank = 0; rank < Rank; ++rank) {
            const std::size_t input_index = rank * step.width + row;
            current[rank] = broadcast(step.current[input_index]);
            if constexpr (HasPrevious) {
                previous[rank] = broadcast(step.previous[input_index]);
            }
            sums[rank] = broadcast(0.0f);
        }

        // One load and store of the state tile serves every rank.
        for (std::size_t col = 0; col < vector_end; col += lanes) {
            Vector value = multiply(alpha, load(state_row + col));
            for (std::size_t rank = 0; rank < Rank; ++rank) {
                const std::size_t key_index = rank * step.states + col;
                value = multiply_add(value, current[rank], load(step.b + key_index));
                if constexpr (HasPrevious) {
                    value = multiply_add(value, previous[rank], load(step.prev_b + key_index));
                }
            }
            store(state_row + col, value);
            for (std::size_t rank = 0; rank < Rank; ++rank) {
                sums[rank] = multiply_add(sums[rank], value, load(step.c + rank * step.states + col));
            }
        }

        float readout[Rank];
        for (std::size_t rank = 0; rank < Rank; ++rank) {
            readout[rank] = reduce(sums[rank]);
        }
        for (std::size_t offset = 0; offset < tail; ++offset) {
            const std::size_t col = vector_end + offset;
            float value = step.alpha * state_row[col];
            for (std::size_t rank = 0; rank < Rank; ++rank) {
                const std::size_t input_index = rank * step.width + row;
                const std::size_t key_index = rank * step.states + col;
                value += step.current[input_index] * step.b[key_index];
                if constexpr (HasPrevious) {
                    value += step.previous[input_index] * step.prev_b[key_index];
                }
            }
            state_row[col] = value;
            for (std::size_t rank = 0; rank < Rank; ++rank) {
                readout[rank] += value * step.c[rank * step.states + col];
            }
        }
        for (std::size_t rank = 0; rank < Rank; ++rank) {
            step.readout[rank * step.width + row] = readout[rank];
        }
    }
}
#endif

} // namespace

void step_simd(const HeadStep &step) {
#if defined(MAMBA_SIMD_NEON) || defined(MAMBA_SIMD_AVX2)
    if (step.rank == 1) {
        if (step.previous == nullptr) {
            update<1, false>(step);
        } else {
            update<1, true>(step);
        }
    } else if (step.rank == 4 && step.previous != nullptr) {
        update<4, true>(step);
    } else {
        throw std::invalid_argument("Unsupported decode rank or previous input");
    }
#else
    (void)step;
    throw std::runtime_error("This build has no supported Mamba SIMD backend");
#endif
}

const char *simd_name() { return backend; }

} // namespace mamba
