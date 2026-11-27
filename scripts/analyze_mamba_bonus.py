"""Rebuild the Mamba CPU comparison from native and STREAM CSVs."""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


VARIANT_LABELS = {"mamba2": "Mamba-2", "siso": "Mamba-3 SISO", "mimo4": "Mamba-3 MIMO-4"}
PHASE_LABELS = {"input_projection": "input projection", "prepare": "preparation",
                "state_readout": "state update and readout", "gate_cache": "gating and cache writes",
                "output_projection": "output projection"}


def work_model(row):
    """Model arithmetic and minimum FP32 tensor traffic within the timed boundary."""
    h, p, n, r, g, k, d = (int(row[name]) for name in
                         ("heads", "head_width", "state_width", "rank", "groups", "rotary_pairs", "model_width"))
    state = h * p * n
    mamba2 = row["variant"] == "mamba2"
    # One decay multiply, one FMA per outer product, and one FMA per readout rank.
    contractions = (5 if mamba2 else 1 + 6 * r) * state
    if mamba2:
        projected = 2 * h * p + 2 * g * n + 2 * h
        params = h
        auxiliary = 0
        pointwise = h + 7 * h * p  # A*dt, dt*x, skip and SiLU
        math_calls = h * (1 + p)  # decay and SiLU exp
    else:
        projected = 2 * h * p + 2 * r * g * n + 3 * h + k
        params = 2 * h + 2 * n + 2 * h * r * n + (3 * h * r * p if r > 1 else 0)
        auxiliary = h * (k + r * n + p)
        normalizing = 2 * r * g * (4 * n + 3)
        coefficients = 9 * h  # dt bias, decay parameter and trapezoidal coefficients
        rotating = h * (2 * r * n + k * (3 + 12 * r))
        scaling_and_gate = h * p * (8 if r == 1 else 14 * r)
        pointwise = normalizing + coefficients + rotating + scaling_and_gate
        # Counts sqrt, softplus exp/log1p, decay exp, mix exp, tanh/sin/cos, and SiLU exp.
        math_calls = 2 * r * g + h * (4 + 3 * k + r * p)
    if row["scope"] == "mixer":
        # The projected fields and recurrent readout are internal to this boundary.
        dense_params = d * (projected + h * p)
        contractions += 2 * dense_params
        logical_bytes = 4 * (2 * (state + auxiliary) + params + dense_params + 2 * d)
    else:
        logical_bytes = 4 * (2 * (state + auxiliary) + params + projected + h * p)
    return {"persistent_state_bytes": 4 * (state + auxiliary), "main_state_bytes": 4 * state,
            "contraction_flops": contractions, "pointwise_flops": pointwise,
            "modeled_library_math_calls": math_calls, "logical_bytes": logical_bytes,
            "arithmetic_intensity": (contractions + pointwise) / logical_bytes}


def read(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def spread(values):
    return {"mean": mean(values), "sd": stdev(values) if len(values) >= 2 else None,
            "min": min(values), "max": max(values), "n": len(values)}


def summarize(rows, repeats):
    groups = defaultdict(list)
    for row in rows:
        expected = work_model(row)
        for key, value in expected.items():
            if abs(float(row[key]) - value) > 1e-9:
                raise ValueError(f"Inconsistent recorded work model: {key}")
        elapsed = float(row["elapsed_s"])
        latency = elapsed * 1e6 / int(row["steps"])
        if abs(latency - float(row["us_per_step"])) > 1e-8:
            raise ValueError("Recorded latency disagrees with raw steps and elapsed time")
        groups[(row["scope"], row["backend"], int(row["threads"]), row["variant"])].append(row)
    summary = []
    for (scope, backend, threads, variant), samples in sorted(groups.items()):
        if sorted(int(sample["repeat"]) for sample in samples) != list(range(1, repeats + 1)):
            raise ValueError(f"Incomplete or duplicated repeats for {scope}, {backend}, {threads}, {variant}")
        work = work_model(samples[0])
        if any(work_model(sample) != work for sample in samples):
            raise ValueError("Benchmark workload changed between repeats")
        summary.append({"scope": scope, "backend": backend, "threads": threads, "variant": variant,
                        "latency_us": spread([float(s["us_per_step"]) for s in samples]),
                        **{key: int(samples[0][key]) for key in ("heads", "head_width", "state_width", "groups")},
                        "peak_rss_bytes": max(int(s["peak_rss_bytes"]) for s in samples), **work})
    return summary, groups


def plot(output, summary):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    try:
        import matplotlib
    except ModuleNotFoundError:
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    operators = [row for row in summary if row["scope"] == "operator"]
    native = next((row["backend"] for row in operators if row["backend"] != "scalar"), "scalar")
    counts = sorted({row["threads"] for row in operators})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    colors = {"mamba2": "#527a91", "siso": "#087d70", "mimo4": "#aa4e2a"}
    for variant, label in VARIANT_LABELS.items():
        group = sorted((row for row in operators if row["variant"] == variant and row["backend"] == native),
                       key=lambda row: row["threads"])
        axes[0].errorbar([row["threads"] for row in group], [row["latency_us"]["mean"] for row in group],
                         yerr=[row["latency_us"]["sd"] for row in group], marker="o", capsize=3,
                         label=label, color=colors[variant])
        axes[1].bar(label, group[0]["arithmetic_intensity"], color=colors[variant])
    axes[0].set(title=f"Decode latency · {native}", xlabel="CPU workers", ylabel="µs per step", xticks=counts)
    axes[0].legend(frameon=False)
    axes[1].set(title="Modeled arithmetic intensity", ylabel="FLOPs per logical byte")
    axes[1].tick_params(axis="x", labelrotation=15)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "comparison.png", dpi=160)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)


def analyze(results, output):
    metadata = json.loads((results / "environment.json").read_text(encoding="utf-8"))
    rows = read(results / "raw.csv")
    summary, groups = summarize(rows, metadata["repeats"])
    scopes = {(variant, "operator") for variant in VARIANT_LABELS}
    if metadata["include_mixer"]:
        scopes.update({("siso", "mixer"), ("mimo4", "mixer")})
    expected = {(variant, scope, mode, workers) for variant, scope in scopes
                for mode in metadata["modes"] for workers in metadata["threads"]}
    observed = {(row["variant"], row["scope"], row["mode"], int(row["threads"])) for row in rows}
    if observed != expected:
        raise ValueError(f"Benchmark configurations differ from the run metadata. Missing {sorted(expected - observed)}")
    operators = [row for row in summary if row["scope"] == "operator"]
    profile = read(results / "profiles.csv")
    stream_path = results / "stream.csv"
    triad = defaultdict(list)
    if stream_path.is_file():
        for row in read(stream_path):
            if row["kernel"] == "Triad":
                triad[int(row["threads"])].append(float(row["best_MB_s"]) / 1000)
    ratios = []
    for backend, threads in sorted({(row["backend"], row["threads"]) for row in operators}):
        sample = [row for row in operators if row["backend"] == backend and row["threads"] == threads]
        shapes = {tuple(row[key] for key in ("heads", "head_width", "state_width", "groups")) for row in sample}
        if {row["variant"] for row in sample} != VARIANT_LABELS.keys() or len(shapes) != 1:
            raise ValueError(f"Unmatched Mamba-2, SISO and MIMO-4 benchmark for {backend} and {threads} workers")
        siso, mimo = ([float(row["us_per_step"]) for row in sorted(
            groups[("operator", backend, threads, variant)], key=lambda row: int(row["repeat"]))]
            for variant in ("siso", "mimo4"))
        ratios.append({"backend": backend, "threads": threads,
                       "change_pct": 100 * (mean(mimo) / mean(siso) - 1),
                       "paired_change_pct": spread([100 * (m / s - 1) for m, s in zip(mimo, siso)])})
    for row in summary:
        # STREAM models a DRAM-resident logical working set. Hot-state latency can be below this estimate.
        rates = triad[row["threads"]]
        row["stream_triad_gb_s"] = spread(rates) if rates else None
        row["stream_projected_us"] = row["logical_bytes"] / (mean(rates) * 1000) if rates else None
    native = next((row["backend"] for row in operators if row["backend"] != "scalar"), "scalar")
    first = min(row["threads"] for row in operators if row["backend"] == native)
    entries = {row["variant"]: row for row in operators if row["threads"] == first and row["backend"] == native}
    ratio = next(row for row in ratios if row["backend"] == native and row["threads"] == first)
    observations = ", ".join(f"{VARIANT_LABELS[key]} {entries[key]['latency_us']['mean']:.1f} ± "
                             f"{entries[key]['latency_us']['sd']:.1f} µs "
                             f"({entries[key]['arithmetic_intensity']:.2f} FLOPs/byte)"
                             for key in ("siso", "mimo4", "mamba2"))
    paired = ratio["paired_change_pct"]
    timing = f"{abs(ratio['change_pct']):.1f}% {'slower' if ratio['change_pct'] >= 0 else 'faster'}"
    profiles = {(row["variant"], row["scope"]): row for row in profile}
    diagnostic = profiles[("mimo4", "operator")]
    dominant = max(PHASE_LABELS, key=lambda name: float(diagnostic[name]))
    checkpoint_variants = {variant for variant, provenance in metadata["fixtures"].items()
                           if (provenance.get("reference") or {}).get("revision")}
    if {"siso", "mimo4"} <= checkpoint_variants:
        source = "Mamba-3 checkpoint-layer fixtures"
    elif metadata["fixtures"]:
        source = "a mix of supplied fixtures and seeded synthetic inputs"
    else:
        source = "seeded synthetic inputs and weights"
    paragraph = (f"On {metadata['cpu'] or metadata['architecture']} using {native} and {first} CPU worker"
                 f"{'s' if first != 1 else ''}, the FP32 operator timings with {source} were {observations} "
                 f"per step (mean ± sample SD). Rank-4 MIMO was {timing} than SISO by mean latency, "
                 f"with paired changes from {paired['min']:+.1f}% to {paired['max']:+.1f}% across "
                 f"{paired['n']} repeats. All three used a {entries['siso']['main_state_bytes'] // 1024} KiB "
                 f"main recurrent matrix. In the separate single-worker profile, {PHASE_LABELS[dominant]} "
                 f"was the largest timed MIMO-4 phase at {float(diagnostic[dominant]):.1f} µs per step. "
                 "The intensity uses modeled arithmetic FLOPs and minimum logical FP32 traffic. "
                 "Actual DRAM traffic depends on cache residency.")
    fault_counts = [int(row["timed_major_faults"]) for row in rows if row.get("timed_major_faults")]
    swaps_before = metadata["swap_before"]
    swaps_after = metadata["swap_after_native"]
    swaps_after_stream = metadata.get("swap_after_stream", swaps_after)
    system_swapped = swaps_before and swaps_after != swaps_before
    system_swapped_out = any(swaps_after.get(key) != swaps_before.get(key)
                             for key in ("swapouts", "pswpout"))
    if any(fault_counts) or system_swapped_out or (system_swapped and len(fault_counts) < len(rows)):
        paragraph += " Paging evidence makes these timings provisional."
    elif system_swapped:
        paragraph += " The system recorded swap-ins, with zero major page faults during all native timed loops."
    if stream_path.is_file() and any(swaps_after_stream.get(key) != swaps_after.get(key)
                                     for key in ("swapouts", "pswpout")):
        paragraph += " The STREAM estimate is provisional because system swap-outs rose during that sweep."
    output.mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_text(json.dumps({
        "work_model": {"flops": "FMA counts as two. Pointwise arithmetic counts the ordinary scalar operations "
                                "alongside the contractions. SIMD horizontal reductions, branching, and "
                                "the internal FLOPs of library math functions are excluded.",
                       "library_math_calls": "Modeled calls assume the softplus evaluates exp and log1p.",
                       "logical_bytes": "One read per visible input and weight, one output write, and one read "
                                        "and one write per persistent state element. This is a minimum traffic model."},
        "paging": {"timed_loops_with_fault_counts": len(fault_counts),
                   "total_timed_major_faults": sum(fault_counts) if fault_counts else None,
                   "system_swap_before": swaps_before, "system_swap_after_native": swaps_after,
                   "system_swap_after_stream": swaps_after_stream},
        "measurements": summary, "mimo_vs_siso": ratios},
                                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "conclusion.md").write_text(paragraph + "\n", encoding="utf-8")
    plot(output, summary)
    return paragraph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(analyze(args.results, args.output))
