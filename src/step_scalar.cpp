#include "kernels.h"

#include <stdexcept>

namespace mamba {
namespace {

template <std::size_t Rank, bool HasPrevious>
void update(const HeadStep &step) {
    for (std::size_t row = 0; row < step.width; ++row) {
        float *state_row = step.state + row * step.states;
        float sums[Rank] = {};
        for (std::size_t col = 0; col < step.states; ++col) {
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
                sums[rank] += value * step.c[rank * step.states + col];
            }
        }
        for (std::size_t rank = 0; rank < Rank; ++rank) {
            step.readout[rank * step.width + row] = sums[rank];
        }
    }
}

} // namespace

void step_scalar(const HeadStep &step) {
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
}

} // namespace mamba
