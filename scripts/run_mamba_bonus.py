"""Run validated Mamba CPU benchmarks and retain each independent repeat."""

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import sys

from analyze_mamba_bonus import analyze, work_model


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("mamba2", "siso", "mimo4")
PHASES = ("input_projection", "prepare", "state_readout", "gate_cache", "output_projection")
SEED = 1234
ORDER_SEED = 20261127
MAX_RSS_BYTES = 500_000_000
BACKENDS = {"scalar", "neon", "avx2", "avx2-fma"}
MEASURE_STREAM_RSS = """
import resource
import subprocess
import sys

result = subprocess.run(sys.argv[1:], check=False)
rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
scale = 1 if sys.platform == "darwin" else 1024
print(f"STREAM_RSS_BYTES={rss * scale}", file=sys.stderr)
sys.exit(result.returncode)
"""


def run(command, *, stderr=False, timeout=300, **options):
    env = {**os.environ, "LC_ALL": "C", **options}
    result = subprocess.run(list(map(str, command)), env=env, cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{' '.join(map(str, command))}: {result.stderr.strip()} {result.stdout.strip()}")
    return (result.stdout, result.stderr) if stderr else result.stdout


def optional(command):
    try:
        return run(command).strip()
    except (OSError, RuntimeError):
        return None


def swap_counters():
    if platform.system() == "Darwin":
        output = optional(["vm_stat"]) or ""
        return {m[1].lower(): int(m[2]) for m in re.finditer(r"^(Swapins|Swapouts):\s+(\d+)", output, re.M)}
    if platform.system() == "Linux":
        return {name: int(value) for name, value in
                re.findall(r"^(pswpin|pswpout) (\d+)", Path("/proc/vmstat").read_text(), re.M)}
    return {}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def backend_name(value):
    return value.lower().replace("/", "-")


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def validate_correctness(path, binary_hash, modes, include_mixer):
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("passed") is not True or evidence.get("binary_sha256") != binary_hash:
        raise ValueError("Correctness evidence must pass and match the benchmark executable SHA-256")
    accepted = {(case["variant"], case["scope"], "scalar" if backend_name(case["backend"]) == "scalar" else "simd")
                for case in evidence["cases"]
                if backend_name(case["backend"]) in BACKENDS and case["steps"] >= 1000 and case.get("passed", True)}
    required = {(variant, "operator", mode) for variant in VARIANTS for mode in modes}
    if include_mixer:
        required |= {(variant, "mixer", mode) for variant in VARIANTS[1:] for mode in modes}
    if missing := required - accepted:
        raise ValueError(f"Missing 1,000-step correctness cases: {sorted(missing)}")


def benchmark(binary, config, seconds, working_states, fixture=None, profile=False):
    variant, scope, mode, threads = config
    command = [binary, "bench", "--variant", variant, "--scope", scope, "--backend", mode,
               "--threads", threads, "--seconds", seconds, "--seed", SEED, "--working-states", working_states]
    if fixture:
        command.extend(["--fixture", fixture])
    if profile:
        command.append("--profile")
    payload = json.loads(run(command, timeout=seconds + 60))
    keys = ("steps", "elapsed_s", "peak_rss_bytes", "checksum", "backend", "threads", "heads",
            "head_width", "state_width", "rank", "groups", "rotary_pairs", "model_width", "working_states")
    row = {key: payload[key] for key in keys}
    faults = payload.get("timed_major_faults")
    if faults is not None and (not isinstance(faults, int) or faults < 0):
        raise RuntimeError(f"Invalid major-page-fault count: {faults}")
    row["timed_major_faults"] = faults
    row["backend"] = backend_name(row["backend"])
    expected_rank = 4 if variant == "mimo4" else 1
    if (row["steps"] <= 0 or not math.isfinite(row["elapsed_s"]) or row["elapsed_s"] < seconds
            or not 0 < row["peak_rss_bytes"] < MAX_RSS_BYTES or not math.isfinite(row["checksum"])
            or row["threads"] != threads or row["working_states"] != working_states
            or payload["variant"] != variant or payload["scope"] != scope
            or row["rank"] != expected_rank or row["backend"] not in BACKENDS
            or (row["backend"] == "scalar") != (mode == "scalar")):
        raise RuntimeError(f"Benchmark returned invalid metrics or a different configuration: {payload}")
    row.update(variant=variant, scope=scope, mode=mode, us_per_step=row["elapsed_s"] * 1e6 / row["steps"])
    row.update(work_model(row))
    if profile:
        phases = {name: payload["phase_us"][name] for name in PHASES}
        if any(not math.isfinite(value) or value < 0 for value in phases.values()):
            raise RuntimeError(f"Invalid phase timings: {phases}")
        return {**row, **phases}
    return row


def stream(binary, results, threads, repeats, cpu_indices):
    system = platform.system()
    monitor = [sys.executable, "-c", MEASURE_STREAM_RSS] if system in {"Darwin", "Linux"} else []
    with (results / "stream.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("threads", "repeat", "kernel", "best_MB_s", "average_s", "min_s", "max_s", "peak_rss_bytes"))
        for workers in threads:
            command = [binary]
            affinity = None
            if cpu_indices and system == "Linux" and shutil.which("taskset"):
                affinity = ["taskset", "-c", ",".join(map(str, cpu_indices[:workers]))]
            elif cpu_indices and system == "Windows" and max(cpu_indices) < 64:
                mask = sum(1 << index for index in cpu_indices[:workers])
                affinity = ["cmd.exe", "/d", "/c", "start", '""', "/b", "/wait", "/affinity", f"{mask:X}"]
            if affinity:
                command = affinity + command
            for repeat in range(1, repeats + 1):
                output, stderr = run(monitor + command, stderr=True,
                                     OMP_NUM_THREADS=str(workers), OMP_DYNAMIC="FALSE")
                (results / f"stream_t{workers}_r{repeat}.txt").write_text(output, encoding="utf-8")
                if stderr:
                    (results / f"stream_t{workers}_r{repeat}.stderr.txt").write_text(stderr, encoding="utf-8")
                peak = re.search(r"^STREAM_RSS_BYTES=(\d+)", stderr, re.M)
                rss = int(peak[1]) if peak else None
                if rss and rss >= MAX_RSS_BYTES:
                    raise RuntimeError(f"STREAM exceeded the benchmark memory budget: {rss} bytes")
                counted = re.search(r"Number of Threads counted = (\d+)", output)
                active = int(counted[1]) if counted else 1
                if workers != active:
                    print(f"STREAM has {active} workers for a {workers}-worker request", file=sys.stderr, flush=True)
                    break
                if "Solution Validates" not in output:
                    raise RuntimeError("STREAM failed its array checksum")
                kernels = re.findall(r"^(Copy|Scale|Add|Triad):\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                                     output, re.M)
                if {entry[0] for entry in kernels} != {"Copy", "Scale", "Add", "Triad"}:
                    raise RuntimeError("Incomplete STREAM output")
                writer.writerows((workers, repeat, name, rate, avg, low, high, rss)
                                 for name, rate, avg, low, high in kernels)
                file.flush()
                print(f"STREAM workers={workers} repeat={repeat}", flush=True)


def write_rows(file, rows):
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def fixture_paths(entries):
    fixtures = {}
    for entry in entries:
        variant, divider, path = entry.partition("=")
        if not divider or variant not in VARIANTS or variant in fixtures or not Path(path).is_file():
            raise ValueError("Use --fixture VARIANT=PATH once per variant, with an existing fixture")
        fixtures[variant] = Path(path).resolve()
    return fixtures


def fixture_metadata(path):
    sidecar = path.with_suffix(path.suffix + ".json")
    reference = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else None
    return {"name": path.name, "sha256": sha256(path), "reference": reference}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/mamba")
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--correctness", type=Path, required=True, help="Passing test_mamba_decode.py JSON")
    parser.add_argument("--threads", type=int, nargs="+", help="Worker counts, default one through detected cores")
    parser.add_argument("--seconds", type=float, default=1.0, help="Minimum duration of each timed native repeat")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backend", choices=("both", "scalar", "simd"), default="both")
    parser.add_argument("--working-states", type=int, default=1)
    parser.add_argument("--fixture", action="append", default=[], metavar="VARIANT=PATH")
    parser.add_argument("--mixer", action="store_true", help="Also time the full Mamba-3 mixers")
    parser.add_argument("--skip-stream", action="store_true")
    parser.add_argument("--allow-battery", action="store_true", help="Permit a run on battery and record its power source")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 300 or args.repeats < 3 or not 1 <= args.working_states <= 512:
        parser.error("Use 0 < seconds <= 300, at least three repeats, and 1 to 512 working states")
    bin_dir = args.build_dir.resolve() / "bin"
    cache = (args.build_dir / "CMakeCache.txt").read_text(encoding="utf-8").splitlines()
    suffix = ".exe" if os.name == "nt" else ""
    binary = bin_dir / f"mamba_decode{suffix}"
    capabilities = json.loads(run([binary, "capabilities"]))
    machine = run([bin_dir / f"machine{suffix}"])
    hardware = dict(line.split("=", 1) for line in machine.splitlines() if "=" in line)
    detected = int(hardware.get("worker_count", 0))
    max_default = max(1, detected) if capabilities["openmp"] else 1
    threads = sorted(set(args.threads or range(1, max_default + 1)))
    if min(threads) < 1 or (detected and max(threads) > detected):
        parser.error("Thread counts must be positive and within detected performance worker count")
    if max(threads) > 1 and not capabilities["openmp"]:
        parser.error("This build supports one worker. Rebuild with OpenMP to use more.")
    cpu_indices = [int(value) for value in hardware.get("cpu_indices", "").split(",") if value]
    cache_detail = optional(["sysctl", "hw.perflevel0.name", "hw.perflevel0.l1dcachesize",
                             "hw.perflevel0.l2cachesize", "hw.perflevel1.name", "hw.perflevel1.l1dcachesize",
                             "hw.perflevel1.l2cachesize"]) if platform.system() == "Darwin" else None
    modes = ("scalar", "simd") if args.backend == "both" else (args.backend,)
    if "simd" in modes and capabilities["simd"] is None:
        parser.error("This build has no SIMD. Use --backend scalar for a scalar-only run.")
    fixtures = fixture_paths(args.fixture)
    binary_hash = sha256(binary)
    validate_correctness(args.correctness, binary_hash, modes, args.mixer)
    power = optional(["pmset", "-g", "batt"]) if platform.system() == "Darwin" else None
    if power and "Now drawing from 'Battery Power'" in power and not args.allow_battery:
        parser.error("Connect AC power or use --allow-battery to measure on battery")
    output = args.output.resolve()
    if output in {ROOT, ROOT / "results", ROOT / "output"}:
        parser.error("Choose a separate run folder such as build/mamba-run")
    results = output / "results"
    results.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.correctness, results / "correctness.json")
    (results / "hardware.txt").write_text(machine + (cache_detail + "\n" if cache_detail else ""), encoding="utf-8")
    compiler_lines = [line for line in cache if line.startswith(("CMAKE_CXX_COMPILER:", "CMAKE_CXX_FLAGS",
                                                               "CMAKE_BUILD_TYPE", "OpenMP_CXX_FLAGS:"))]
    compiler = next((line.split("=", 1)[1] for line in compiler_lines if line.startswith("CMAKE_CXX_COMPILER:")), None)
    metadata = {"start_utc": datetime.now(timezone.utc).isoformat(), "os": platform.platform(),
                "architecture": platform.machine(), "cpu": hardware.get("cpu"), "threads": threads,
                "detected_workers": detected, "capabilities": capabilities, "modes": modes,
                "include_mixer": args.mixer,
                "affinity": hardware.get("affinity"), "cpu_indices": cpu_indices, "power": power,
                "compiler": optional([compiler, "--version"]) if compiler else None, "build_flags": compiler_lines,
                "git_commit": optional(["git", "rev-parse", "HEAD"]) if (ROOT / ".git").exists() else None,
                "binary_sha256": binary_hash,
                "repeats": args.repeats, "minimum_seconds": args.seconds, "working_states": args.working_states,
                "seed": SEED, "order_seed": ORDER_SEED,
                "fixtures": {variant: fixture_metadata(path) for variant, path in fixtures.items()},
                "swap_before": swap_counters()}
    write_json(results / "environment.json", metadata)
    scopes = [(variant, "operator") for variant in VARIANTS]
    if args.mixer:
        scopes.extend((variant, "mixer") for variant in VARIANTS[1:])
    configs = [(variant, scope, mode, workers) for workers in threads for mode in modes for variant, scope in scopes]
    order = random.Random(ORDER_SEED)
    with (results / "raw.csv").open("w", newline="", encoding="utf-8") as file:
        writer = None
        for repeat in range(1, args.repeats + 1):
            order.shuffle(configs)
            for variant, scope, mode, workers in configs:
                row = benchmark(binary, (variant, scope, mode, workers), args.seconds, args.working_states,
                                fixtures.get(variant))
                row["repeat"] = repeat
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                file.flush()
                print(f"{variant:6} {scope:8} {row['backend']:8} workers={workers} repeat={repeat}: "
                      f"{row['us_per_step']:.2f} µs/step", flush=True)
    with (results / "profiles.csv").open("w", newline="", encoding="utf-8") as file:
        write_rows(file, [benchmark(binary, (variant, scope, modes[-1], 1), args.seconds, args.working_states,
                                    fixtures.get(variant), profile=True) for variant, scope in scopes])
    metadata["swap_after_native"] = swap_counters()
    write_json(results / "environment.json", metadata)
    if not args.skip_stream:
        stream(bin_dir / f"stream{suffix}", results, threads, args.repeats, cpu_indices)
    metadata["end_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["swap_after_stream"] = swap_counters()
    write_json(results / "environment.json", metadata)
    print(analyze(results, output / "output"))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        raise SystemExit(str(error)) from None
