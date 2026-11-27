"""Generate the named PyTorch fixtures used by tests/test_mamba_decode.py."""

import argparse
from pathlib import Path
import subprocess
import sys


REFERENCE = Path(__file__).with_name("mamba_reference.py")
SMALL = (
    ("siso-small", "siso", ("--steps", "1000")),
    ("mimo-small", "mimo", ("--steps", "1000", "--initial", "nonzero")),
    ("siso-impulse", "siso", ("--steps", "2", "--scenario", "impulse")),
    ("mimo-impulse", "mimo", ("--steps", "2", "--scenario", "impulse")),
    ("siso-full-rotary", "siso", ("--steps", "100", "--initial", "nonzero", "--rotary", "full")),
    ("mimo-no-rotary", "mimo", ("--steps", "100", "--rotary", "none", "--scenario", "one-active-rank")),
    ("mimo-trap-extremes", "mimo", ("--steps", "100", "--scenario", "trap-extremes")),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, help="Directory containing pinned siso.bin and mimo.bin")
    parser.add_argument("--upstream-source", type=Path, help="Path to the pinned state-spaces/mamba source")
    parser.add_argument("--small-only", action="store_true", help="Generate only the seven small test fixtures")
    args = parser.parse_args()
    recipes = list(SMALL)
    if not args.small_only:
        if args.checkpoints is None or args.upstream_source is None:
            parser.error("The full fixture suite requires --checkpoints and --upstream-source")
        for variant in ("siso", "mimo"):
            weights = args.checkpoints / f"{variant}.bin"
            if not weights.is_file():
                parser.error(f"Missing checkpoint {weights}")
            common = ("--shape", "checkpoint", "--steps", "1000", "--checkpoint", str(weights))
            recipes.extend(((f"{variant}-full", variant, common),
                            (f"{variant}-official-mixer", variant,
                             common + ("--mode", "mixer", "--upstream-source", str(args.upstream_source)))))
        if not args.upstream_source.is_dir():
            parser.error(f"Missing upstream source directory {args.upstream_source}")
        recipes.append(("mamba2-full", "mamba2", ("--shape", "checkpoint", "--steps", "1000")))
    for name, variant, options in recipes:
        destination = args.output / f"{name}.bin"
        print(f"Generating {destination}", flush=True)
        subprocess.run([sys.executable, str(REFERENCE), "--model", variant, "--output", str(destination),
                        *options], check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Fixture generation stopped after a reference command exited with {error.returncode}") from None
