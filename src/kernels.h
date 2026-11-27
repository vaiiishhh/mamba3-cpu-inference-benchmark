#pragma once

#include <cstddef>

namespace mamba {

// All views describe one head and use row-major storage. State and readout
// are writable and must not alias any other views.
struct HeadStep {
    std::size_t width;
    std::size_t states;
    std::size_t rank;
    float alpha;
    float *state;            // [width, states]
    const float *b;          // [rank, states], rotated current key
    const float *prev_b;     // [rank, states], rotated previous key
    const float *c;          // [rank, states], rotated readout
    const float *current;    // [rank, width], current input with gamma applied
    const float *previous;   // [rank, width], previous input with beta applied
    float *readout;          // [rank, width], before skip and gating
};

using HeadKernel = void (*)(const HeadStep &);

// Mamba-3 supports rank 1 or 4. Mamba-2 uses rank 1 and passes nullptr for
// both previous-input views, with dt applied to the current input.
void step_scalar(const HeadStep &step);
void step_simd(const HeadStep &step);
const char *simd_name();

} // namespace mamba
