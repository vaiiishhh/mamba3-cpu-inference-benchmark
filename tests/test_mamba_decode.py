"""Replay every token and cache tensor from the CPU PyTorch fixtures."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


CASES = {
    "siso-small": ("siso", "operator", 1000),
    "mimo-small": ("mimo4", "operator", 1000),
    "siso-impulse": ("siso", "operator", 2),
    "mimo-impulse": ("mimo4", "operator", 2),
    "siso-full-rotary": ("siso", "operator", 64),
    "mimo-no-rotary": ("mimo4", "operator", 64),
    "mimo-trap-extremes": ("mimo4", "operator", 64),
    "siso-full": ("siso", "operator", 1000),
    "mimo-full": ("mimo4", "operator", 1000),
    "mamba2-full": ("mamba2", "operator", 1000),
    "siso-official-mixer": ("siso", "mixer", 1000),
    "mimo-official-mixer": ("mimo4", "mixer", 1000),
}


def call(binary, path, backend, workers):
    return subprocess.run(
        [binary, "check", path, "--backend", backend, "--threads", str(workers)],
        capture_output=True, text=True, check=False,
    )


def replay(binary, fixture_dir, label, contract, backend, workers):
    path = fixture_dir / f"{label}.bin"
    metadata = json.loads(path.with_suffix(".bin.json").read_text(encoding="utf-8"))
    result = call(binary, path, backend, workers)
    if result.returncode:
        raise RuntimeError(f"{label} {backend} with {workers} workers\n{result.stderr.strip()}")
    case = json.loads(result.stdout)
    variant, scope, steps = contract
    if (case["variant"] != variant or case["scope"] != scope or case["steps"] < steps
            or case["threads"] != workers or case["steps"] != metadata["steps"]
            or (case["backend"] == "scalar") != (backend == "scalar")):
        raise RuntimeError(f"{label} reported unexpected case parameters")
    if any(not 0 <= err["max_scaled"] <= 1 for err in case["errors"].values()):
        raise RuntimeError(f"{label} reported error above tolerance")
    if "revision" not in metadata and any(
            not 0 <= err["max_elementwise_scaled"] <= 1 for err in case["errors"].values()):
        raise RuntimeError(f"{label} exceeded the strict elementwise bound for synthetic inputs")
    max_error = max(err["max_scaled"] for err in case["errors"].values())
    print(f"{label:22s} {case['backend']:9s} workers={workers} steps={case['steps']:4d} "
          f"max error/tolerance={max_error:.4f}", flush=True)
    return {**case, "fixture": label, "reference": metadata, "passed": True}


def check_invalid(binary, fixture_dir):
    data = (fixture_dir / "siso-impulse.bin").read_bytes()
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory) / "invalid.bin"
        for payload, error in ((data[:-1], "truncated fixture tensor"),
                               (data + b"\0", "unexpected trailing data"),
                               (b"BADMAGIC" + data[8:], "valid MAMBA3V1 fixture"),
                               (data[:-4] + b"\x00\x00\xc0\x7f", "prev_x mismatch")):
            fixture.write_bytes(payload)
            result = call(binary, fixture, "scalar", 1)
            if result.returncode != 1 or error not in result.stderr:
                raise RuntimeError(f"invalid fixture was not rejected with {error!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("both", "scalar", "simd"), default="both")
    parser.add_argument("--threads", type=int, default=1, help="Also replay full-shape cases at this worker count")
    parser.add_argument("--small-only", action="store_true", help="Quick test using the seven small cases")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    binary = args.binary.resolve()
    backends = ("scalar", "simd") if args.backend == "both" else (args.backend,)
    labels = [label for label, (_, scope, _) in CASES.items()
              if scope == "operator" and not label.endswith("-full")]
    if not args.small_only:
        labels = list(CASES)
    cases = []
    for label in labels:
        workers = (1, args.threads) if args.threads > 1 and label.endswith("-full") else (1,)
        for count in workers:
            for backend in backends:
                cases.append(replay(binary, args.fixtures.resolve(), label, CASES[label], backend, count))
    check_invalid(binary, args.fixtures)
    with binary.open("rb") as file:
        binary_hash = hashlib.file_digest(file, "sha256").hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps({"passed": True, "binary_sha256": binary_hash,
                                     "invalid_fixture_cases": 4, "cases": cases}, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
