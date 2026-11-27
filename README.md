# Mamba-3 CPU decode bonus

Run the commands below from this directory. From the parent repository, enter it with `cd mamba3`. This project contains its own CMake entry point, hardware and STREAM tools, reference scripts, tests, and benchmark runner. To build a copy in a new location, copy the source tree and configure a fresh `build/` directory there.

`mamba_decode` advances one FP32 SSM mixer by one token. The scalar and SIMD paths implement exponential-trapezoidal integration, data-dependent rotary phase, and rank-1 or rank-4 input and output. Each head updates one contiguous state matrix. MIMO-4 reuses that matrix for all four ranks and retains four rotated keys from the previous token. [Mamba-3 paper, §§3.1–3.4](https://arxiv.org/html/2603.15569v1) · [Official implementation at the pinned revision](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa)

At batch one with 24 heads, head width 64, and state width 128, all three comparison operators use a **768 KiB** recurrent matrix. Mamba-3 additionally caches the phase, previous rotated key and previous x before rank expansion. Total persistent state is **789 KiB for SISO** and **825 KiB for MIMO-4**. The shape comes from the released 187M models. [SISO checkpoint](https://huggingface.co/state-spaces/mamba3-siso-187m) · [MIMO checkpoint](https://huggingface.co/state-spaces/mamba3-mimo-187m)

## Build

The bonus needs CMake 3.20+ and a C++17 compiler. Its standalone configuration builds the decode executable, hardware probe and STREAM. Use an OpenMP-capable C/C++ toolchain for the worker sweep. The binary reports its detected SIMD backend and OpenMP availability.

```sh
cmake -S . -B build/mamba
cmake --build build/mamba -j
ctest --test-dir build/mamba --output-on-failure
build/mamba/bin/mamba_decode capabilities
```

On Apple Silicon pass `-DCMAKE_OSX_ARCHITECTURES=arm64`, particularly if CMake itself runs under Rosetta. To build with Apple Clang and an installed Homebrew `libomp`, use the following configure command in place of the first command above.

```sh
OMP_PREFIX="$(brew --prefix libomp)"
cmake -S . -B build/mamba -DCMAKE_OSX_ARCHITECTURES=arm64 \
  '-DOpenMP_C_FLAGS=-Xpreprocessor -fopenmp' '-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp' \
  -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY="$OMP_PREFIX/lib/libomp.dylib" \
  -DOpenMP_C_INCLUDE_DIR="$OMP_PREFIX/include" -DOpenMP_CXX_INCLUDE_DIR="$OMP_PREFIX/include"
```

## Validate against CPU PyTorch

Use Python 3.11+ and this project's Python dependencies. Where PyTorch supplies a separate CPU wheel index, install the CPU wheel first. The native executable reads a streamed little-endian FP32 trace. Each replay rolls forward its own state and checks the output, state matrix, phase, previous key and previous x after every token.

```sh
python3 -m pip install -r scripts/requirements-mamba.txt
python3 scripts/make_mamba_fixtures.py --output build/mamba-small --small-only
python3 tests/test_mamba_decode.py --binary build/mamba/bin/mamba_decode \
  --fixtures build/mamba-small --output build/mamba-small/correctness.json --small-only
```

The full suite needs about 5 GiB of disk space for the streamed state snapshots and checkpoint files. Fetch the pinned upstream code and model weights. The generator checks the checkpoint and upstream source SHA-256 digests and maps only the selected mixer into CPU FP32 tensors.

```sh
mkdir -p build/mamba-checkpoints
git clone https://github.com/state-spaces/mamba.git build/mamba-upstream
git -C build/mamba-upstream checkout --detach e9594ce1c732d97440f0332fdc43170a2294dbfa
curl -fL https://huggingface.co/state-spaces/mamba3-siso-187m/resolve/6792c27c00f3bb41506db1066dcd1c51bb0f4b02/pytorch_model.bin \
  -o build/mamba-checkpoints/siso.bin
curl -fL https://huggingface.co/state-spaces/mamba3-mimo-187m/resolve/8fd6e9eb7b795f2e15d7f6353251d0137980c43e/pytorch_model.bin \
  -o build/mamba-checkpoints/mimo.bin
python3 scripts/make_mamba_fixtures.py --output build/mamba-fixtures \
  --checkpoints build/mamba-checkpoints --upstream-source build/mamba-upstream
python3 tests/test_mamba_decode.py --binary build/mamba/bin/mamba_decode \
  --fixtures build/mamba-fixtures --output build/mamba/correctness.json --threads 4
```

Use `--threads 1` for a serial build. The suite checks each SISO and MIMO kernel with 1,000 consecutive projected inputs and with 1,000 inputs through the checkpoint's full mixer. The eager reference imports the exact upstream rotary and recurrent functions from the pinned GPU source files and executes them as ordinary CPU tensor operations. The suite checks Mamba-2’s exponential-Euler recurrence on the same matrix shape. Smaller tests cover nonzero state, full and partial rotation, disabled rank channels, saturated trapezoidal gates, a two-token impulse with a closed-form answer, SIMD tails and damaged fixtures. [Eager rotary function](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py) · [Eager recurrent function](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py)

For phase, previous x and every synthetic tensor the elementwise FP32 bound is `1e-4 + 1e-4 × abs(reference)`. For checkpoint output, matrix and rotated key, add `2e-4 × max(abs(reference tensor))` for that step. Hundreds of FP32 angle increments and the resulting rotations can cancel at individual coordinates even when the tensor error remains small. The JSON records worst absolute error, step, index, error as a fraction of the acceptance bound, and error against the strict elementwise bound for inspection. Any non-finite element fails the replay.

## Measure the three operators

```sh
python3 scripts/run_mamba_bonus.py --build-dir build/mamba --output build/mamba-run \
  --correctness build/mamba/correctness.json --mixer \
  --fixture siso=build/mamba-fixtures/siso-official-mixer.bin \
  --fixture mimo4=build/mamba-fixtures/mimo-official-mixer.bin \
  --fixture mamba2=build/mamba-fixtures/mamba2-full.bin
```

The runner matches validation to the executable by SHA-256. It interleaves three independent repeats for each operator, backend and worker count. A warm-up, fixture loading and any dense input projection for operator timings happen before the timer. On macOS and Linux the runner records each process's major page faults inside the measured loop. The primary comparison uses one hot recurrent matrix at batch one. The optional mixer comparison includes both dense projections. Mamba-2 consumes prepared post-convolution inputs for its state update and gated readout. The Mamba-3 operator includes coefficient preparation, B/C normalization, rotation, recurrence, readout and gating.

The work model counts one fused multiply-add as two FLOPs. It divides modeled arithmetic by logical bytes for tensor reads, writes and parameters at the tested boundary. Library transcendental calls are tracked separately. STREAM estimates a DRAM-resident bandwidth limit. The hot state can be served from cache, so the logical byte model and the STREAM prediction cannot determine actual DRAM traffic.

`build/mamba-run/results/` contains correctness evidence, run metadata, raw timings, phase timings, STREAM data and the statistical summary. `build/mamba-run/output/` contains the conclusion and the comparison plot. Regenerate the analysis from the published quiet-machine run with the following command.

```sh
python3 scripts/analyze_mamba_bonus.py --results results/quiet-apple-m2 --output build/mamba-replot
```

## Apple M2 results

The primary quiet-machine run started on November 27, 2026 at 21:46 IST. With one NEON worker, Mamba-2 averaged 41.9 ± 0.9 µs, Mamba-3 SISO 75.9 ± 0.1 µs, and Mamba-3 MIMO-4 178.8 ± 3.3 µs per step over three repeats (mean ± sample SD). Rank-4 MIMO took 2.36× as long as SISO with the same 768 KiB main recurrent matrix. Across the sweep, 120 timed repeats and five profiles recorded zero major page faults. macOS logged 76 system-wide swap-ins during the native sweep and four more during STREAM, with zero swap-outs. [Conclusion](results/quiet-apple-m2/conclusion.md) · [Raw timings](results/quiet-apple-m2/raw.csv) · [STREAM](results/quiet-apple-m2/stream.csv) · [Correctness](results/quiet-apple-m2/correctness.json) · [Comparison plot](results/quiet-apple-m2/comparison.png)
