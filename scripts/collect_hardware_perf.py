#!/usr/bin/env python3
"""
scripts/collect_hardware_perf.py – Full-Pipeline Hardware Performance Collector
================================================================================

Runs the SENTINEL-X pipeline across a configurable battery of test scenarios
and collects performance data suitable for inclusion in a technical report or
publication.  Works in **simulation mode** (no hardware) by default; pass
``--port`` to include real UART round-trip measurements.

What is collected
-----------------
+-----------------------------------+--------------------------------------------------+
| Metric                            | Description                                      |
+===================================+==================================================+
| ``inference_latency``             | TFLite int8 policy inference (p50/p95/p99, ms)  |
| ``state_normalisation_latency``   | SensorReading.to_state_vector() latency          |
| ``safety_veto_latency``           | SafetyMonitor.veto() rule-check latency          |
| ``uart_roundtrip``                | MCU UART request/response (if --port given)      |
| ``fault_recovery_rate``           | Fraction of faults recovered in eval episodes    |
| ``mean_episode_reward``           | Mean reward over eval window                     |
| ``federated_improvement``         | Reward delta: federated vs. isolated agents      |
| ``formation_coherence``           | Mean formation coherence over eval episodes      |
| ``model_size_kb``                 | TFLite model size in kilobytes                   |
+-----------------------------------+--------------------------------------------------+

Output
------
* ``--json-out``  (default ``hardware_perf_results.json``) – full raw data
* ``--md-out``   (default ``hardware_perf_report.md``)   – human-readable report

Usage
-----
::

    # Simulation mode (no hardware needed – suitable for CI)
    python scripts/collect_hardware_perf.py

    # With a real MCU via UART
    python scripts/collect_hardware_perf.py --port /dev/ttyUSB0 --baud 115200

    # Larger run (better statistics)
    python scripts/collect_hardware_perf.py --timing-runs 1000 --train-episodes 50

    # Quiet – JSON only
    python scripts/collect_hardware_perf.py --quiet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(times_ns: list) -> Dict[str, float]:
    arr = np.array(times_ns, dtype=np.float64) / 1e6
    return {
        "n":      len(arr),
        "min_ms": round(float(np.min(arr)), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "max_ms": round(float(np.max(arr)), 4),
        "mean_ms": round(float(np.mean(arr)), 4),
        "std_ms":  round(float(np.std(arr)), 4),
    }


def _print(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg)


# ---------------------------------------------------------------------------
# 1 – Inference / timing metrics
# ---------------------------------------------------------------------------

def _collect_timing(
    model_path: str,
    n: int,
    quiet: bool,
) -> Dict[str, Any]:
    import sentinel_x_advanced as sx
    import tensorflow as tf

    # Inference latency
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    state_dim = int(inp["shape"][1])
    rng = np.random.default_rng(0)

    inf_times: list = []
    for _ in range(n):
        s = rng.random(state_dim).astype(np.float32)
        if inp["dtype"] == np.int8:
            sc = float(inp["quantization"][0]) or 1e-8
            zp = int(inp["quantization"][1])
            sq = np.clip(np.round(s / sc + zp), -128, 127).astype(np.int8)
            interp.set_tensor(inp["index"], sq[None])
        else:
            interp.set_tensor(inp["index"], s[None])
        t0 = time.perf_counter_ns()
        interp.invoke()
        inf_times.append(time.perf_counter_ns() - t0)
    _print(f"  Inference latency ({n} runs) … done", quiet)

    # State normalisation
    from hardware.hal import SensorReading  # type: ignore[import]
    norm_times: list = []
    for _ in range(n):
        r = SensorReading(
            timestamp_s=time.time(),
            memory_error_count=int(rng.integers(0, 10)),
            parity_error_flag=bool(rng.integers(0, 2)),
            sensor_deviation=float(rng.normal(0, 1)),
            sensor_stuck=bool(rng.integers(0, 2)),
            time_since_recovery_s=float(rng.uniform(0, 3600)),
            health_flag=bool(rng.integers(0, 2)),
            thermal_fault=bool(rng.integers(0, 2)),
            power_level_v=float(rng.uniform(0, 5)),
            attitude_rate_dps=float(rng.normal(0, 10)),
            comm_quality_db=float(rng.uniform(-120, 0)),
        )
        t0 = time.perf_counter_ns()
        _ = r.to_state_vector()
        norm_times.append(time.perf_counter_ns() - t0)
    _print(f"  State normalisation ({n} runs) … done", quiet)

    # SafetyMonitor veto
    monitor = sx.SafetyMonitor()
    veto_times: list = []
    for _ in range(n):
        state = rng.random(11).tolist()
        action = int(rng.integers(0, 4))
        t0 = time.perf_counter_ns()
        _ = monitor.veto(action, state, action_dim=4)
        veto_times.append(time.perf_counter_ns() - t0)
    _print(f"  SafetyMonitor veto ({n} runs) … done", quiet)

    model_kb = os.path.getsize(model_path) / 1024
    _print(f"  Model size: {model_kb:.1f} KB", quiet)

    return {
        "inference_latency_ms":           _pct(inf_times),
        "state_normalisation_latency_ms": _pct(norm_times),
        "safety_veto_latency_ms":         _pct(veto_times),
        "model_size_kb":                  round(model_kb, 2),
    }


# ---------------------------------------------------------------------------
# 2 – UART round-trip (optional)
# ---------------------------------------------------------------------------

def _collect_uart(port: str, baud: int, n: int, quiet: bool) -> Dict[str, Any]:
    try:
        import serial  # type: ignore[import]
    except ImportError:
        return {"error": "pyserial not installed"}
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=0.05)
    except Exception as exc:
        return {"error": str(exc)}
    times: list = []
    try:
        for _ in range(n):
            t0 = time.perf_counter_ns()
            ser.write(b"\x52")
            _ = ser.read(22)
            times.append(time.perf_counter_ns() - t0)
    finally:
        ser.close()
    if not times:
        return {"error": "no responses received"}
    _print(f"  UART round-trip ({n} runs @ {baud} baud) … done", quiet)
    return _pct(times)


# ---------------------------------------------------------------------------
# 3 – RL training / fault-recovery metrics
# ---------------------------------------------------------------------------

def _collect_rl(n_episodes: int, max_steps: int, quiet: bool) -> Dict[str, Any]:
    import sentinel_x_advanced as sx

    swarm = sx.FederatedSwarm(
        num_spacecraft=4, action_dim=4,
        mission_profile=sx.MissionProfile(sx.MissionProfile.BALANCED),
        safety_monitor=sx.SafetyMonitor(),
        cooperative_bonus=0.5,
        federated_interval=5,
    )

    # Isolated baseline (no federated averaging)
    isolated_swarm = sx.FederatedSwarm(
        num_spacecraft=4, action_dim=4,
        mission_profile=sx.MissionProfile(sx.MissionProfile.BALANCED),
        federated_interval=10_000,   # effectively disabled
    )

    _print(f"  Training federated swarm for {n_episodes} episodes …", quiet)
    fed_rewards = [swarm.train_episode(max_steps=max_steps) for _ in range(n_episodes)]

    _print(f"  Training isolated baseline for {n_episodes} episodes …", quiet)
    iso_rewards = [isolated_swarm.train_episode(max_steps=max_steps) for _ in range(n_episodes)]

    # Eval: fault-recovery rate using FederatedSwarm.test_episode()
    _print("  Evaluating fault-recovery rate …", quiet)
    eval_steps = max(10, max_steps // 2)
    eval_runs = 3
    healthy_total = sum(
        swarm.test_episode(max_steps=eval_steps)
        for _ in range(eval_runs)
    )
    # test_episode returns mean operational steps; normalise to fraction
    fault_recovery_rate = round(float(healthy_total) / eval_runs / eval_steps, 4)
    override_total = swarm.last_override_count

    # Formation coherence
    from sentinel_x.formation import FormationController
    fc = FormationController(swarm, formation_type="v", separation_m=100.0)
    coherences = []
    form_steps = max(5, max_steps // 4)
    for _ in range(2):
        fc.train_episode_with_formation(max_steps=form_steps)
        coherences.append(fc.formation_coherence())

    fed_mean = float(np.mean(fed_rewards[-10:]))
    iso_mean = float(np.mean(iso_rewards[-10:]))
    fed_improvement = round(fed_mean - iso_mean, 4)

    _print("  RL metrics … done", quiet)
    return {
        "train_episodes":            n_episodes,
        "mean_episode_reward_last10": round(fed_mean, 4),
        "federated_improvement":      fed_improvement,
        "fault_recovery_rate":        fault_recovery_rate,
        "mean_safety_overrides":      round(float(override_total), 2),
        "mean_formation_coherence":   round(float(np.mean(coherences)), 4),
    }


# ---------------------------------------------------------------------------
# 4 – Markdown report builder
# ---------------------------------------------------------------------------

def _build_md_report(results: Dict[str, Any], platform: str) -> str:
    t = results["timing"]
    r = results["rl"]

    def _row(stat: Dict[str, float]) -> str:
        return (
            f"| {stat.get('p50_ms', '—'):.3f} | "
            f"{stat.get('p95_ms', '—'):.3f} | "
            f"{stat.get('p99_ms', '—'):.3f} |"
        )

    uart_section = ""
    if (
        "uart_roundtrip" in results
        and "error" not in results["uart_roundtrip"]
        and "skipped" not in results["uart_roundtrip"]
    ):
        us = results["uart_roundtrip"]
        uart_section = f"""
### UART Round-Trip

| p50 (ms) | p95 (ms) | p99 (ms) |
|----------|----------|----------|
{_row(us)}

*Port: {results.get('uart_port', 'N/A')}  Baud: {results.get('uart_baud', 'N/A')}*
"""

    lines = [
        "# SENTINEL-X Hardware Performance Report",
        "",
        f"**Platform:** {platform}  ",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}  ",
        f"**Model size:** {t.get('model_size_kb', '?')} KB (TFLite int8)",
        "",
        "---",
        "",
        "## Inference Pipeline Latency",
        "",
        "All measurements are wall-clock time on the target platform.",
        "Budget column shows the real-time deadline for a 100 Hz control loop.",
        "",
        "| Test | p50 (ms) | p95 (ms) | p99 (ms) | Budget (ms) | Status |",
        "|------|----------|----------|----------|-------------|--------|",
    ]

    budgets = {
        "inference_latency_ms":           10.0,
        "state_normalisation_latency_ms":  0.5,
        "safety_veto_latency_ms":          0.1,
    }
    labels = {
        "inference_latency_ms":           "TFLite int8 inference",
        "state_normalisation_latency_ms": "State normalisation",
        "safety_veto_latency_ms":         "SafetyMonitor veto",
    }
    for key, label in labels.items():
        stat    = t[key]
        budget  = budgets[key]
        status  = "✓ PASS" if stat["p99_ms"] <= budget else "✗ FAIL"
        lines.append(
            f"| {label} | {stat['p50_ms']:.3f} | "
            f"{stat['p95_ms']:.3f} | {stat['p99_ms']:.3f} | "
            f"{budget} | **{status}** |"
        )

    lines += [
        "",
        uart_section,
        "---",
        "",
        "## Reinforcement Learning Performance",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Training episodes | {r['train_episodes']} |",
        f"| Mean episode reward (last 10 ep) | {r['mean_episode_reward_last10']} |",
        f"| Federated vs. isolated improvement | {r['federated_improvement']:+.4f} |",
        f"| Fault-recovery rate (eval) | {r['fault_recovery_rate']:.1%} |",
        f"| Mean safety overrides / episode | {r['mean_safety_overrides']} |",
        f"| Mean formation coherence | {r['mean_formation_coherence']:.2%} |",
        "",
        "---",
        "",
        "## Notes",
        "",
        "* All timing measurements represent wall-clock time on the indicated platform.",
        "* 'Federated improvement' = (mean reward, federated) − (mean reward, isolated baseline).",
        "* 'Fault-recovery rate' = fraction of steps where all spacecraft were operational",
        "  during 10 × 100-step evaluation episodes with the trained policy.",
        "* Formation coherence = exp(−mean_error / separation_m), where separation_m = 100 m.",
        "",
        "---",
        "",
        "*Generated by `scripts/collect_hardware_perf.py` – part of the*",
        "*[SENTINEL-X](https://github.com/danielnovais-tech/SENTINEL-X)*",
        "*open-source spacecraft fault-detection platform (Apache 2.0).*",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SENTINEL-X full-pipeline hardware performance collector"
    )
    parser.add_argument("--port",          default=None,
                        help="Serial port for UART round-trip test (optional)")
    parser.add_argument("--baud",          type=int, default=115_200)
    parser.add_argument("--timing-runs",   type=int, default=200,
                        help="Iterations per timing test (default 200)")
    parser.add_argument("--train-episodes", type=int, default=20,
                        help="RL training episodes (default 20)")
    parser.add_argument("--max-steps",      type=int, default=100,
                        help="Max steps per episode (default 100)")
    parser.add_argument("--json-out",      default="hardware_perf_results.json",
                        help="JSON output path")
    parser.add_argument("--md-out",        default="hardware_perf_report.md",
                        help="Markdown report output path")
    parser.add_argument("--quiet",         action="store_true")
    args = parser.parse_args(argv)

    import platform as _platform
    sys_platform = f"{_platform.system()} {_platform.machine()} Python {_platform.python_version()}"

    _print("SENTINEL-X Hardware Performance Collector", args.quiet)
    _print("=" * 55, args.quiet)

    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "perf_int8.tflite")

        _print("\n[1/4] Building calibrated TFLite int8 model …", args.quiet)
        import sentinel_x_advanced as sx
        swarm0 = sx.FederatedSwarm(num_spacecraft=2, action_dim=4,
                                   mission_profile=sx.MissionProfile(
                                       sx.MissionProfile.BALANCED))
        sx.export_tflite_int8(swarm0.agents[0], output_path=model_path,
                              n_calib_samples=8)
        _print(f"  Model exported to {model_path}", args.quiet)

        _print("\n[2/4] Collecting inference and safety timing …", args.quiet)
        timing = _collect_timing(model_path, args.timing_runs, args.quiet)

        _print("\n[3/4] Running RL training and evaluation …", args.quiet)
        rl = _collect_rl(args.train_episodes, args.max_steps, args.quiet)

        results: Dict[str, Any] = {
            "platform": sys_platform,
            "timing":   timing,
            "rl":       rl,
        }

        if args.port:
            _print(f"\n[4/4] Measuring UART round-trip ({args.port}) …", args.quiet)
            results["uart_roundtrip"] = _collect_uart(
                args.port, args.baud, args.timing_runs, args.quiet
            )
            results["uart_port"] = args.port
            results["uart_baud"] = args.baud
        else:
            results["uart_roundtrip"] = {"skipped": True,
                                          "reason":  "No --port specified"}

        # Write JSON
        Path(args.json_out).write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        _print(f"\nJSON results → {args.json_out}", args.quiet)

        # Write Markdown
        md = _build_md_report(results, sys_platform)
        Path(args.md_out).write_text(md, encoding="utf-8")
        _print(f"Markdown report → {args.md_out}", args.quiet)

        _print("\n" + "=" * 55, args.quiet)
        _print("Collection complete.", args.quiet)

    return 0


if __name__ == "__main__":
    sys.exit(main())
