#!/usr/bin/env python3
"""
scripts/hardware_timing_validator.py – Hardware Deployment Timing Validator
============================================================================

Validates that the SENTINEL-X inference pipeline meets real-time timing
requirements on the target hardware.  Runs four test suites:

1. **Inference latency** – measures TFLite int8 inference time and verifies
   it is below the configurable budget (default 10 ms for 100 Hz control).

2. **State-vector normalisation** – measures the time to normalise a
   SensorReading to the 11-dim state vector on the target platform.

3. **SafetyMonitor veto latency** – measures the time for the
   SafetyMonitor rule check.

4. **UART round-trip** (optional) – if a serial port is available, measures
   the round-trip time for a 22-byte request/response with the MCU.  Skipped
   automatically if no port is specified.

All tests print PASS / FAIL against configurable budgets and write a JSON
report to ``--output`` (default ``timing_report.json``).

Usage
-----
::

    # Simulation mode (no hardware needed)
    python scripts/hardware_timing_validator.py

    # With a real serial MCU (STM32 / RPi)
    python scripts/hardware_timing_validator.py --port /dev/ttyUSB0

    # Adjust timing budgets
    python scripts/hardware_timing_validator.py \\
        --inference-budget-ms 2.0 \\
        --uart-budget-ms 5.0 \\
        --runs 500

    # Quiet JSON output only
    python scripts/hardware_timing_validator.py --quiet --output report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _median_p95(times_ns: list) -> dict:
    arr = np.array(times_ns, dtype=np.float64) / 1e6   # → ms
    return {
        "min_ms":  float(np.min(arr)),
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "p99_ms":  float(np.percentile(arr, 99)),
        "max_ms":  float(np.max(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms":  float(np.std(arr)),
    }


def _print_result(name: str, stats: dict, budget_ms: float, quiet: bool) -> bool:
    passed = stats["p99_ms"] <= budget_ms
    if not quiet:
        mark = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {mark}  {name}")
        print(f"         median={stats['p50_ms']:.3f} ms  "
              f"p95={stats['p95_ms']:.3f} ms  "
              f"p99={stats['p99_ms']:.3f} ms  "
              f"(budget={budget_ms:.1f} ms)")
    return passed


# ---------------------------------------------------------------------------
# Test 1 – Inference latency
# ---------------------------------------------------------------------------

def _test_inference(model_path: str, runs: int, quiet: bool) -> tuple:
    """Run TFLite inference timing.  Returns (stats_dict, passed)."""
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        inp  = interp.get_input_details()[0]
        out  = interp.get_output_details()[0]
        state_dim = inp["shape"][1]
        rng  = np.random.default_rng(0)
    except Exception as exc:
        return {"error": str(exc)}, False

    times = []
    for _ in range(runs):
        state = rng.random(state_dim).astype(np.float32)
        if inp["dtype"] == np.int8:
            scale = float(inp["quantization"][0]) or 1e-6
            zp    = int(inp["quantization"][1])
            state_q = np.clip(
                np.round(state / scale + zp), -128, 127
            ).astype(np.int8)
            interp.set_tensor(inp["index"], state_q[None])
        else:
            interp.set_tensor(inp["index"], state[None])

        t0 = time.perf_counter_ns()
        interp.invoke()
        times.append(time.perf_counter_ns() - t0)

    return _median_p95(times), True


# ---------------------------------------------------------------------------
# Test 2 – State-vector normalisation
# ---------------------------------------------------------------------------

def _test_state_normalisation(runs: int, quiet: bool) -> tuple:
    """Measure SensorReading.to_state_vector() latency."""
    try:
        from hardware.hal import SensorReading
    except ImportError:
        return {"error": "hardware package not found"}, False

    rng = np.random.default_rng(1)
    times = []
    for _ in range(runs):
        reading = SensorReading(
            timestamp_s           = time.time(),
            memory_error_count    = int(rng.integers(0, 10)),
            parity_error_flag     = bool(rng.integers(0, 2)),
            sensor_deviation      = float(rng.normal(0, 1)),
            sensor_stuck          = bool(rng.integers(0, 2)),
            time_since_recovery_s = float(rng.uniform(0, 3600)),
            health_flag           = bool(rng.integers(0, 2)),
            thermal_fault         = bool(rng.integers(0, 2)),
            power_level_v         = float(rng.uniform(0, 5)),
            attitude_rate_dps     = float(rng.normal(0, 10)),
            comm_quality_db       = float(rng.uniform(-120, 0)),
        )
        t0 = time.perf_counter_ns()
        _ = reading.to_state_vector()
        times.append(time.perf_counter_ns() - t0)
    return _median_p95(times), True


# ---------------------------------------------------------------------------
# Test 3 – SafetyMonitor veto latency
# ---------------------------------------------------------------------------

def _test_safety_monitor(runs: int, quiet: bool) -> tuple:
    """Measure SafetyMonitor.veto() latency."""
    try:
        import sentinel_x_advanced as sx
        monitor = sx.SafetyMonitor()
    except Exception as exc:
        return {"error": str(exc)}, False

    rng    = np.random.default_rng(2)
    times  = []
    for _ in range(runs):
        state  = rng.random(11).tolist()
        action = int(rng.integers(0, 4))
        t0 = time.perf_counter_ns()
        _ = monitor.veto(action, state, action_dim=4)
        times.append(time.perf_counter_ns() - t0)
    return _median_p95(times), True


# ---------------------------------------------------------------------------
# Test 4 – UART round-trip (optional)
# ---------------------------------------------------------------------------

def _test_uart_roundtrip(port: str, runs: int, baud: int, quiet: bool) -> tuple:
    """Measure UART request/response round-trip time."""
    try:
        import serial
    except ImportError:
        return {"error": "pyserial not installed"}, False

    try:
        ser = serial.Serial(port, baudrate=baud, timeout=0.05)
    except Exception as exc:
        return {"error": f"Cannot open {port}: {exc}"}, False

    times = []
    try:
        for _ in range(runs):
            t0 = time.perf_counter_ns()
            ser.write(b"\x52")        # request frame
            _ = ser.read(22)          # wait for 22-byte response
            times.append(time.perf_counter_ns() - t0)
    finally:
        ser.close()

    if not times:
        return {"error": "no responses received"}, False
    return _median_p95(times), True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SENTINEL-X hardware deployment timing validator"
    )
    parser.add_argument("--port",                default=None,
                        help="Serial port for UART round-trip test (optional)")
    parser.add_argument("--baud",                type=int,   default=115_200)
    parser.add_argument("--runs",                type=int,   default=200,
                        help="Number of timed iterations per test (default 200)")
    parser.add_argument("--inference-budget-ms", type=float, default=10.0,
                        help="Pass/fail threshold for inference latency (default 10 ms)")
    parser.add_argument("--norm-budget-ms",      type=float, default=0.5,
                        help="Pass/fail threshold for normalisation (default 0.5 ms)")
    parser.add_argument("--safety-budget-ms",    type=float, default=0.1,
                        help="Pass/fail threshold for SafetyMonitor (default 0.1 ms)")
    parser.add_argument("--uart-budget-ms",      type=float, default=5.0,
                        help="Pass/fail threshold for UART round-trip (default 5 ms)")
    parser.add_argument("--output",              default="timing_report.json",
                        help="Output JSON report path")
    parser.add_argument("--quiet",               action="store_true",
                        help="Suppress console output; write JSON only")
    args = parser.parse_args(argv)

    if not args.quiet:
        print("SENTINEL-X Hardware Timing Validator")
        print("=" * 50)

    # Build / locate TFLite model
    import sentinel_x_advanced as sx
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "val_int8.tflite")
        swarm = sx.FederatedSwarm(
            num_spacecraft=2, action_dim=4,
            mission_profile=sx.MissionProfile(sx.MissionProfile.BALANCED),
        )
        sx.export_tflite_int8(swarm.agents[0], output_path=model_path, n_calib_samples=8)

        results = {}
        all_passed = True

        # ── Test 1: Inference latency ─────────────────────────────────────
        if not args.quiet:
            print("\n[1] TFLite int8 inference latency")
        stats, ok = _test_inference(model_path, args.runs, args.quiet)
        passed1 = ok and _print_result(
            "Inference", stats, args.inference_budget_ms, args.quiet
        )
        results["inference"] = {"stats": stats, "budget_ms": args.inference_budget_ms,
                                 "passed": passed1}
        all_passed &= passed1

        # ── Test 2: State normalisation ───────────────────────────────────
        if not args.quiet:
            print("\n[2] State-vector normalisation latency")
        stats, ok = _test_state_normalisation(args.runs, args.quiet)
        passed2 = ok and _print_result(
            "Normalisation", stats, args.norm_budget_ms, args.quiet
        )
        results["normalisation"] = {"stats": stats, "budget_ms": args.norm_budget_ms,
                                     "passed": passed2}
        all_passed &= passed2

        # ── Test 3: SafetyMonitor veto ────────────────────────────────────
        if not args.quiet:
            print("\n[3] SafetyMonitor veto latency")
        stats, ok = _test_safety_monitor(args.runs, args.quiet)
        passed3 = ok and _print_result(
            "SafetyMonitor", stats, args.safety_budget_ms, args.quiet
        )
        results["safety_monitor"] = {"stats": stats, "budget_ms": args.safety_budget_ms,
                                      "passed": passed3}
        all_passed &= passed3

        # ── Test 4: UART round-trip (optional) ────────────────────────────
        if args.port:
            if not args.quiet:
                print(f"\n[4] UART round-trip ({args.port} @ {args.baud} baud)")
            stats, ok = _test_uart_roundtrip(
                args.port, args.runs, args.baud, args.quiet
            )
            passed4 = ok and _print_result(
                "UART round-trip", stats, args.uart_budget_ms, args.quiet
            )
            results["uart_roundtrip"] = {"stats": stats, "budget_ms": args.uart_budget_ms,
                                          "passed": passed4, "port": args.port}
            all_passed &= passed4
        else:
            results["uart_roundtrip"] = {"skipped": True,
                                          "reason": "No --port specified"}

        # ── Summary ───────────────────────────────────────────────────────
        results["summary"] = {
            "all_passed": all_passed,
            "runs":       args.runs,
            "platform":   sys.platform,
        }
        Path(args.output).write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

        if not args.quiet:
            print("\n" + "=" * 50)
            status = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
            print(f"Result: {status}")
            print(f"Report: {args.output}")
            print("=" * 50)

        return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
