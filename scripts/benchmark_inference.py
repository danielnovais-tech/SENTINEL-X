#!/usr/bin/env python3
"""
scripts/benchmark_inference.py – TFLite Inference Benchmarker
==============================================================

Measures inference latency and memory usage for any SENTINEL-X TFLite model
(dynamic-range or int8).  Designed to run on:

* Development workstation (x86-64, arm64)
* Raspberry Pi 4 / Zero 2 W (arm64 / ARMv6)
* Any device with ``tflite-runtime`` or full ``tensorflow`` installed

The output is a human-readable table **plus** an optional JSON report file
suitable for automated CI comparisons or hardware certification records.

Usage
-----
::

    # Basic benchmark (100 warm-up + 1000 timed inferences)
    python scripts/benchmark_inference.py

    # Benchmark a specific model with custom parameters
    python scripts/benchmark_inference.py \\
        --model sentinel_x_model_int8.tflite \\
        --warmup 50 \\
        --runs 500 \\
        --output latency_report.json

    # Benchmark both dynamic-range and int8 models side-by-side
    python scripts/benchmark_inference.py \\
        --model sentinel_x_model.tflite sentinel_x_model_int8.tflite

    # Show only the summary table (no per-run output)
    python scripts/benchmark_inference.py --quiet

Output
------
Printed to stdout::

    ══════════════════════════════════════════════════════════════
      SENTINEL-X TFLite Inference Benchmarker
    ══════════════════════════════════════════════════════════════
      Model : sentinel_x_model_int8.tflite
      Runs  : 1000 (+ 100 warm-up)
      Device: Linux-5.15.0 x86_64  Python 3.12.3
    ──────────────────────────────────────────────────────────────
      Latency (ms)   min=0.028  p50=0.033  p95=0.041  max=0.127
      Throughput     30 248 inferences / second
      Peak RSS       38.2 MB
    ══════════════════════════════════════════════════════════════

JSON report format
------------------
::

    {
      "model": "sentinel_x_model_int8.tflite",
      "warmup_runs": 100,
      "timed_runs": 1000,
      "device": { "platform": "Linux-5.15.0", "arch": "x86_64", "python": "3.12.3" },
      "latency_ms": {
        "min":  0.028, "p5":  0.030, "p25": 0.031,
        "p50":  0.033, "p75": 0.035, "p95": 0.041,
        "p99":  0.062, "max": 0.127, "mean": 0.034, "std": 0.005
      },
      "throughput_ips": 30248,
      "peak_rss_mb": 38.2,
      "model_size_bytes": 12432,
      "action_distribution": {"DO_NOTHING": 412, "RESTART": 193, ...}
    }

Raspberry Pi tips
-----------------
Install the lightweight ``tflite-runtime`` wheel (much smaller than full TF)::

    pip install tflite-runtime   # ARM wheel provided by TF team

Expected latency on Raspberry Pi 4 with the int8 model: **~1–2 ms**.
Expected latency on STM32H7 (Cortex-M7 @ 480 MHz) via TFLite Micro: **~0.5–1 ms**.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STATE_DIM    = 11
_DEFAULT_MODEL = "sentinel_x_model_int8.tflite"
_ACTION_LABELS = {
    0: "DO_NOTHING",
    1: "RESTART",
    2: "SWITCH_REDUNDANT",
    3: "SAFE_MODE",
}


# ---------------------------------------------------------------------------
# TFLite loader (same pattern as replay_tflite.py)
# ---------------------------------------------------------------------------

def _load_interpreter(model_path: str):
    """Load a TFLite model; try tflite-runtime first, then full TF."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"TFLite model not found: {model_path}\n"
            "Run 'python sentinel_x_advanced.py' or 'python run_experiment.py' "
            "to generate it."
        )
    try:
        import tflite_runtime.interpreter as tflite  # type: ignore[import]
        interp = tflite.Interpreter(model_path=str(path))
    except ImportError:
        try:
            import tensorflow as tf
            interp = tf.lite.Interpreter(model_path=str(path))
        except ImportError as exc:
            raise ImportError(
                "Neither 'tflite-runtime' nor 'tensorflow' is installed.\n"
                "Install: pip install tflite-runtime   # lightweight\n"
                "     or: pip install tensorflow        # full"
            ) from exc
    interp.allocate_tensors()
    return interp


# ---------------------------------------------------------------------------
# Single-inference helper
# ---------------------------------------------------------------------------

def _run_one(interp, state: np.ndarray) -> tuple[int, float]:
    """Run one forward pass; return (action, latency_ms)."""
    in_d  = interp.get_input_details()
    out_d = interp.get_output_details()

    inp_dtype = in_d[0]["dtype"]
    if inp_dtype == np.int8:
        scale, zp = in_d[0]["quantization"]
        if scale == 0.0:
            scale = 1.0
        x = np.round(state.astype(np.float32) / scale + zp).astype(np.int8)
    else:
        x = state.astype(np.float32)
    x = x.reshape(1, -1)

    t0 = time.perf_counter()
    interp.set_tensor(in_d[0]["index"], x)
    interp.invoke()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    out = interp.get_tensor(out_d[0]["index"])[0]
    if out_d[0]["dtype"] == np.int8:
        scale, zp = out_d[0]["quantization"]
        if scale == 0.0:
            scale = 1.0
        out = (out.astype(np.float32) - zp) * scale

    return int(np.argmax(out)), latency_ms


# ---------------------------------------------------------------------------
# RSS memory helper
# ---------------------------------------------------------------------------

def _peak_rss_mb() -> Optional[float]:
    """Return current process RSS in MB, or None if unavailable."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        pass
    try:
        with open(f"/proc/{os.getpid()}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Benchmark one model
# ---------------------------------------------------------------------------

def _benchmark(
    model_path: str,
    warmup: int,
    runs: int,
    rng: np.random.Generator,
    quiet: bool,
) -> dict:
    """Benchmark a single TFLite model; return a report dict."""
    interp      = _load_interpreter(model_path)
    model_bytes = Path(model_path).stat().st_size

    # Pre-generate all states so the random-number gen doesn't pollute timing
    states = [rng.random(_STATE_DIM).astype(np.float32) for _ in range(warmup + runs)]

    # ── Warm-up ────────────────────────────────────────────────────────────
    for i in range(warmup):
        _run_one(interp, states[i])

    # ── Timed runs ─────────────────────────────────────────────────────────
    latencies: list[float] = []
    action_counts: dict[str, int] = {}

    for i in range(runs):
        action, lat = _run_one(interp, states[warmup + i])
        latencies.append(lat)
        label = _ACTION_LABELS.get(action, str(action))
        action_counts[label] = action_counts.get(label, 0) + 1

    rss = _peak_rss_mb()

    if not latencies:
        raise ValueError("No timed runs were executed; runs must be > 0.")

    lats = np.array(latencies)
    report = {
        "model":          model_path,
        "warmup_runs":    warmup,
        "timed_runs":     runs,
        "device": {
            "platform": platform.system() + "-" + platform.release(),
            "arch":     platform.machine(),
            "python":   platform.python_version(),
        },
        "latency_ms": {
            "min":  round(float(lats.min()),  4),
            "p5":   round(float(np.percentile(lats,  5)), 4),
            "p25":  round(float(np.percentile(lats, 25)), 4),
            "p50":  round(float(np.percentile(lats, 50)), 4),
            "p75":  round(float(np.percentile(lats, 75)), 4),
            "p95":  round(float(np.percentile(lats, 95)), 4),
            "p99":  round(float(np.percentile(lats, 99)), 4),
            "max":  round(float(lats.max()),  4),
            "mean": round(float(lats.mean()), 4),
            "std":  round(float(lats.std()),  4),
        },
        "throughput_ips":    int(1000.0 / lats.mean()),
        "peak_rss_mb":       round(rss, 1) if rss is not None else None,
        "model_size_bytes":  model_bytes,
        "action_distribution": action_counts,
    }

    if not quiet:
        _print_report(report)

    return report


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_report(r: dict) -> None:
    lat = r["latency_ms"]
    dev = r["device"]
    print("=" * 65)
    print("  SENTINEL-X TFLite Inference Benchmarker")
    print("=" * 65)
    print(f"  Model   : {r['model']}")
    print(f"  Size    : {r['model_size_bytes']:,} bytes")
    print(f"  Runs    : {r['timed_runs']} (+ {r['warmup_runs']} warm-up)")
    print(f"  Device  : {dev['platform']} {dev['arch']}  Python {dev['python']}")
    print("-" * 65)
    print(
        f"  Latency (ms)  min={lat['min']}  p50={lat['p50']}  "
        f"p95={lat['p95']}  max={lat['max']}"
    )
    print(f"  Throughput    {r['throughput_ips']:,} inferences / second")
    rss_str = f"{r['peak_rss_mb']:.1f} MB" if r["peak_rss_mb"] is not None else "N/A"
    print(f"  Peak RSS      {rss_str}")
    print(
        "  Actions       " +
        "  ".join(f"{a}={c}" for a, c in r["action_distribution"].items())
    )
    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SENTINEL-X TFLite inference benchmarker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", nargs="+", default=[_DEFAULT_MODEL], metavar="FILE",
        help="TFLite model file(s) to benchmark (default: sentinel_x_model_int8.tflite)",
    )
    parser.add_argument(
        "--warmup", type=int, default=100,
        help="Number of warm-up inferences before timing (default: 100)",
    )
    parser.add_argument(
        "--runs", type=int, default=1000,
        help="Number of timed inferences (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducible state generation (default: 0)",
    )
    parser.add_argument(
        "--output", metavar="FILE", default=None,
        help="Write JSON report to this file (optional)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-model stdout table; only write JSON output",
    )
    return parser.parse_args()


def main() -> None:
    args    = _parse_args()
    rng     = np.random.default_rng(args.seed)
    reports = []

    for model_path in args.model:
        try:
            report = _benchmark(model_path, args.warmup, args.runs, rng, args.quiet)
            reports.append(report)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}\n", file=sys.stderr)

    if args.output and reports:
        payload = reports[0] if len(reports) == 1 else reports
        Path(args.output).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
