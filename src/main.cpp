#include "decode.h"
#include "platform.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

using namespace mamba;
using Clock = std::chrono::steady_clock;
constexpr double tolerance = 1e-4;
constexpr double tensor_tolerance = 2e-4;
constexpr std::size_t ring_size = 32;

std::string name(const Config &c) {
    if (c.model == 2) return "mamba2";
    return c.rank == 1 ? "siso" : "mimo4";
}

Config default_config(const std::string &variant) {
    if (variant == "siso") return {3, 24, 64, 128, 1, 1, 32, 768};
    if (variant == "mimo4") return {3, 24, 64, 128, 4, 1, 32, 768};
    if (variant == "mamba2") return {2, 24, 64, 128, 1, 1, 0, 0};
    throw std::invalid_argument("unknown variant");
}

struct Backend {
    HeadKernel run;
    const char *label;
};

Backend backend(const std::string &choice) {
    if (choice == "scalar") return {step_scalar, "scalar"};
    if (choice == "simd") {
        if (const char *isa = simd_name()) return {step_simd, isa};
        throw std::runtime_error("this binary has no supported SIMD backend");
    }
    throw std::invalid_argument("backend must be scalar or simd");
}

std::uint32_t read_u32(std::istream &in) {
    std::array<unsigned char, 4> data{};
    if (!in.read(reinterpret_cast<char *>(data.data()), data.size())) {
        throw std::runtime_error("truncated fixture header");
    }
    return std::uint32_t(data[0]) | (std::uint32_t(data[1]) << 8) |
           (std::uint32_t(data[2]) << 16) | (std::uint32_t(data[3]) << 24);
}

void read(std::istream &in, std::vector<float> &values) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    const std::uint32_t endian = 1;
    if (*reinterpret_cast<const unsigned char *>(&endian) != 1) {
        throw std::runtime_error("fixtures require a little-endian host");
    }
    if (!values.empty() && !in.read(reinterpret_cast<char *>(values.data()),
                                    values.size() * sizeof(float))) {
        throw std::runtime_error("truncated fixture tensor");
    }
}

void read(std::istream &in, State &s) {
    for (auto *tensor : {&s.matrix, &s.phase, &s.prev_b, &s.prev_x}) read(in, *tensor);
}

void read(std::istream &in, Weights &w) {
    for (auto *tensor : {&w.in, &w.out, &w.dt_bias, &w.b_norm, &w.c_norm, &w.b_bias, &w.c_bias,
                         &w.skip, &w.value_scale, &w.gate_scale, &w.output_scale}) {
        read(in, *tensor);
    }
}

struct Fixture {
    explicit Fixture(const std::string &file) : stream(file, std::ios::binary), config(header()) {
        steps = read_u32(stream);
        if (steps > 100000) throw std::runtime_error("too many fixture steps");
    }

    Config header() {
        char magic[8];
        if (!stream.read(magic, sizeof(magic)) || std::string(magic, sizeof(magic)) != "MAMBA3V1") {
            throw std::runtime_error("cannot open a valid MAMBA3V1 fixture");
        }
        Config cfg{read_u32(stream), read_u32(stream), read_u32(stream), read_u32(stream),
                   read_u32(stream), read_u32(stream), read_u32(stream), read_u32(stream)};
        cfg.validate();
        return cfg;
    }

    std::size_t fields() const { return config.model_width ? config.model_width : config.packed(); }
    std::size_t outputs() const { return config.model_width ? config.model_width : config.inner(); }

    std::ifstream stream;
    Config config;
    std::uint32_t steps = 0;
};

struct Error {
    double max_abs = 0.0;
    double max_scaled = 0.0;
    double max_elementwise_scaled = 0.0;
    std::size_t worst_index = 0;
    std::size_t worst_step = 0;
    std::size_t first_bad_step = 0;
    std::size_t first_bad_index = 0;
};

void configure_threads(int threads) {
#if defined(_OPENMP)
    omp_set_dynamic(0);
    if (threads > 1) {
        int actual = 0;
#pragma omp parallel num_threads(threads)
        {
#pragma omp single
            actual = omp_get_num_threads();
        }
        if (actual != threads) throw std::runtime_error("OpenMP could not start the requested worker count");
    }
#else
    if (threads > 1) throw std::invalid_argument("the binary was built without OpenMP");
#endif
}

void compare(const char *tensor, const std::vector<float> &actual, const std::vector<float> &ref,
             std::size_t step, Error &error, bool use_tensor_scale) {
    // Rotation and readout can cancel large terms. Allow a small forward
    // error relative to this step's tensor scale at those cancellation points.
    double peak = 0.0;
    if (use_tensor_scale) {
        for (float value : ref) peak = std::max(peak, std::abs(double(value)));
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double absolute = std::abs(double(actual[i]) - double(ref[i]));
        const double bound = tolerance + tolerance * std::abs(double(ref[i]));
        const double scaled = absolute / (bound + tensor_tolerance * peak);
        if (!std::isfinite(scaled)) {
            throw std::runtime_error(std::string(tensor) + " mismatch at step " +
                                     std::to_string(step) + " index " + std::to_string(i) +
                                     ", actual " + std::to_string(actual[i]) +
                                     ", expected " + std::to_string(ref[i]));
        }
        if (scaled > 1.0 && !error.first_bad_step) {
            error.first_bad_step = step;
            error.first_bad_index = i;
        }
        error.max_abs = std::max(absolute, error.max_abs);
        error.max_elementwise_scaled = std::max(absolute / bound, error.max_elementwise_scaled);
        if (scaled > error.max_scaled) {
            error.max_scaled = scaled;
            error.worst_index = i;
            error.worst_step = step;
        }
    }
}

void check(const std::string &file, Backend impl, int threads) {
    Fixture f(file);
    const Config &cfg = f.config;
    if (!f.steps) throw std::runtime_error("fixture contains no validation steps");
    if (threads < 1 || std::size_t(threads) > cfg.heads) {
        throw std::invalid_argument("invalid validation worker count");
    }
    configure_threads(threads);
    Weights weights(cfg);
    State state(cfg), expected_state(cfg);
    Scratch scratch(cfg);
    std::vector<float> input(f.fields()), output(f.outputs()), expected_output(f.outputs());
    read(f.stream, weights);
    read(f.stream, state);
    std::array<Error, 5> errors{};
    constexpr std::array<const char *, 5> tensors = {"output", "matrix", "phase", "prev_b", "prev_x"};
    const auto step = cfg.model_width ? mixer_step : operator_step;
    for (std::size_t i = 1; i <= f.steps; ++i) {
        read(f.stream, input);
        read(f.stream, expected_output);
        read(f.stream, expected_state);
        step(cfg, weights, input.data(), state, scratch, output.data(), impl.run, threads, nullptr);
        compare(tensors[0], output, expected_output, i, errors[0], true);
        compare(tensors[1], state.matrix, expected_state.matrix, i, errors[1], true);
        compare(tensors[2], state.phase, expected_state.phase, i, errors[2], false);
        compare(tensors[3], state.prev_b, expected_state.prev_b, i, errors[3], true);
        compare(tensors[4], state.prev_x, expected_state.prev_x, i, errors[4], false);
    }
    if (f.stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("fixture contains unexpected trailing data");
    }
    bool passed = true;
    std::cout << "{\"variant\":\"" << name(cfg) << "\",\"backend\":\"" << impl.label
              << "\",\"scope\":\"" << (cfg.model_width ? "mixer" : "operator")
              << "\",\"threads\":" << threads << ",\"steps\":" << f.steps
              << ",\"atol\":" << tolerance << ",\"rtol\":"
              << tolerance << ",\"tensor_rtol\":" << tensor_tolerance << ",\"errors\":{";
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto &e = errors[i];
        if (i) std::cout << ',';
        std::cout << '"' << tensors[i] << "\":{\"max_abs\":" << e.max_abs
                  << ",\"max_scaled\":" << e.max_scaled
                  << ",\"max_elementwise_scaled\":" << e.max_elementwise_scaled
                  << ",\"step\":" << e.worst_step
                  << ",\"index\":" << e.worst_index << '}';
        if (e.first_bad_step) {
            passed = false;
            std::cerr << tensors[i] << " mismatch at step " << e.first_bad_step
                      << " index " << e.first_bad_index << ", worst absolute error "
                      << e.max_abs << " and scaled error " << e.max_scaled << '\n';
        }
    }
    std::cout << "}}\n";
    if (!passed) throw std::runtime_error("fixture exceeded tolerance");
}

struct BenchOptions {
    std::string variant = "siso", backend = "simd", scope = "operator", fixture;
    std::size_t working_states = 1;
    int threads = 1;
    std::uint32_t seed = 1234;
    double seconds = 1.0;
    bool profile = false;
};

BenchOptions parse_bench(int argc, char **argv) {
    BenchOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--profile") {
            options.profile = true;
            continue;
        }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--variant") options.variant = value;
        else if (key == "--backend") options.backend = value;
        else if (key == "--scope") options.scope = value;
        else if (key == "--fixture") options.fixture = value;
        else if (key == "--threads") options.threads = std::stoi(value);
        else if (key == "--working-states") options.working_states = std::stoul(value);
        else if (key == "--seed") options.seed = std::stoul(value);
        else if (key == "--seconds") options.seconds = std::stod(value);
        else throw std::invalid_argument("unknown option " + key);
    }
    if (options.scope != "operator" && options.scope != "mixer") {
        throw std::invalid_argument("scope must be operator or mixer");
    }
    if (options.threads < 1 || (options.profile && options.threads != 1) ||
        options.working_states < 1 || options.working_states > 512 ||
        !std::isfinite(options.seconds) || options.seconds <= 0.0 || options.seconds > 300.0) {
        throw std::invalid_argument("invalid benchmark configuration");
    }
    return options;
}

void fill_synthetic(const Config &cfg, Weights &weights, std::vector<std::vector<float>> &ring,
                    bool mixer, std::uint32_t seed) {
    std::mt19937 rng(seed);
    auto uniform = [&]() { return static_cast<float>(rng() >> 8) / 16777216.0f - 0.5f; };
    for (auto *tensor : {&weights.in, &weights.out}) {
        for (float &v : *tensor) v = 0.04f * uniform();
    }
    std::fill(weights.dt_bias.begin(), weights.dt_bias.end(), -3.0f);
    std::fill(weights.b_bias.begin(), weights.b_bias.end(), 1.0f);
    std::fill(weights.c_bias.begin(), weights.c_bias.end(), 1.0f);
    if (cfg.rank > 1) {
        for (float &v : weights.value_scale) v = (1.0f + uniform()) / float(cfg.rank);
        for (float &v : weights.gate_scale) v = 1.0f + uniform();
        for (float &v : weights.output_scale) v = (1.0f + uniform()) / float(cfg.rank);
    }
    for (auto &token : ring) {
        for (float &v : token) v = uniform();
        if (cfg.model == 2 && !mixer) {
            for (std::size_t h = 0; h < cfg.heads; ++h) {
                token[cfg.packed() - 2 * cfg.heads + h] = 0.05f + 0.04f * uniform();
                token[cfg.packed() - cfg.heads + h] = -1.0f + uniform();
            }
        }
    }
}

std::optional<long> major_faults() {
#if defined(_WIN32)
    return std::nullopt;
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("cannot read process page faults");
    return usage.ru_majflt;
#endif
}

void bench(const BenchOptions &options) {
    const Backend impl = backend(options.backend);
    const bool mixer = options.scope == "mixer";
    Config cfg = default_config(options.variant);
    if (!mixer && options.fixture.empty()) cfg.model_width = 0;
    std::unique_ptr<Fixture> fixture;
    if (!options.fixture.empty()) {
        fixture = std::make_unique<Fixture>(options.fixture);
        cfg = fixture->config;
    }
    if (options.variant != name(cfg) || (mixer && !cfg.model_width) ||
        std::size_t(options.threads) > cfg.heads) {
        throw std::invalid_argument("fixture, scope, or thread count does not match the variant");
    }
    const std::size_t state_floats = cfg.inner() * cfg.states + cfg.heads * cfg.angles +
        (cfg.model == 3 ? cfg.heads * cfg.rank * cfg.states + cfg.inner() : 0);
    if (state_floats * options.working_states * sizeof(float) > 400000000ULL) {
        throw std::invalid_argument("working-state ring exceeds the recurrent memory budget");
    }
    pin_performance_cores(true, options.threads);
    configure_threads(options.threads);
    Weights weights(cfg);
    State initial(cfg);
    Scratch scratch(cfg);
    std::vector<std::vector<float>> ring(ring_size,
        std::vector<float>(mixer ? cfg.model_width : cfg.packed()));
    if (fixture) {
        if (!fixture->steps) throw std::runtime_error("no benchmark inputs in fixture");
        read(fixture->stream, weights);
        read(fixture->stream, initial);
        ring.resize(std::min<std::size_t>(ring.size(), fixture->steps));
        std::vector<float> source(fixture->fields());
        const auto reference_bytes = static_cast<std::streamoff>(
            (fixture->outputs() + state_floats) * sizeof(float));
        for (auto &token : ring) {
            read(fixture->stream, source);
            if (!fixture->stream.seekg(reference_bytes, std::ios::cur)) {
                throw std::runtime_error("cannot skip fixture reference data");
            }
            if (!mixer && cfg.model_width) project_input(cfg, weights, source.data(), token.data());
            else token = source;
        }
        fixture.reset();
    } else {
        fill_synthetic(cfg, weights, ring, mixer, options.seed);
    }
    std::vector<State> states(options.working_states, initial);
    std::vector<float> output(mixer ? cfg.model_width : cfg.inner());
    PhaseTimes phases{};
    const auto step = mixer ? mixer_step : operator_step;
    auto advance = [&](std::size_t iteration, PhaseTimes *profile) {
        const auto slot = iteration % states.size();
        const auto token = (iteration / states.size() + slot) % ring.size();
        step(cfg, weights, ring[token].data(), states[slot],
             scratch, output.data(), impl.run, options.threads, profile);
    };
    for (std::size_t i = 0; i < std::max(ring.size(), states.size()); ++i) advance(i, nullptr);
    for (auto &state : states) state = initial;
    std::size_t steps = 0;
    double elapsed;
    const auto faults_before = major_faults();
    const auto start = Clock::now();
    do {
        for (int i = 0; i < 8; ++i) advance(steps++, options.profile ? &phases : nullptr);
        elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    } while (elapsed < options.seconds);
    const auto faults_after = major_faults();
    const double checksum = std::accumulate(output.begin(), output.end(), 0.0) +
                            states[(steps - 1) % states.size()].matrix.front();
    if (!std::isfinite(checksum)) throw std::runtime_error("non-finite benchmark checksum");
    std::cout << "{\"variant\":\"" << name(cfg) << "\",\"backend\":\"" << impl.label
              << "\",\"scope\":\"" << options.scope << "\",\"steps\":" << steps
              << ",\"elapsed_s\":" << elapsed << ",\"peak_rss_bytes\":" << peak_rss()
              << ",\"checksum\":" << checksum << ",\"threads\":" << options.threads
              << ",\"heads\":" << cfg.heads << ",\"head_width\":" << cfg.width
              << ",\"state_width\":" << cfg.states << ",\"rank\":" << cfg.rank
              << ",\"groups\":" << cfg.groups << ",\"rotary_pairs\":" << cfg.angles
              << ",\"model_width\":" << cfg.model_width << ",\"working_states\":"
              << options.working_states << ",\"timed_major_faults\":";
    if (faults_before && faults_after) std::cout << *faults_after - *faults_before;
    else std::cout << "null";
    if (options.profile) {
        constexpr std::array<const char *, 5> labels = {"input_projection", "prepare", "state_readout",
                                                       "gate_cache", "output_projection"};
        std::cout << ",\"phase_us\":{";
        for (std::size_t i = 0; i < labels.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << '"' << labels[i] << "\":" << phases[i] * 1e6 / steps;
        }
        std::cout << '}';
    }
    std::cout << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        std::cout << std::setprecision(10);
        if (argc == 2 && std::string(argv[1]) == "capabilities") {
            const auto *simd = simd_name();
            std::cout << "{\"simd\":";
            if (simd) std::cout << '"' << simd << '"';
            else std::cout << "null";
#if defined(_OPENMP)
            std::cout << ",\"openmp\":true}\n";
#else
            std::cout << ",\"openmp\":false}\n";
#endif
        } else if ((argc == 5 || argc == 7) && std::string(argv[1]) == "check" &&
            std::string(argv[3]) == "--backend") {
            if (argc == 7 && std::string(argv[5]) != "--threads") {
                throw std::invalid_argument("expected --threads N");
            }
            check(argv[2], backend(argv[4]), argc == 7 ? std::stoi(argv[6]) : 1);
        } else if (argc >= 2 && std::string(argv[1]) == "bench") {
            bench(parse_bench(argc, argv));
        } else {
            throw std::invalid_argument("usage: mamba_decode check FIXTURE --backend scalar|simd [--threads N] "
                                        "or mamba_decode bench [options]");
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
