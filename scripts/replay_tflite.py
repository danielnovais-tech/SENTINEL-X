#!/usr/bin/env python3
"""
scripts/replay_tflite.py – Embedded deployment demo for SENTINEL-X
====================================================================

Loads a TensorFlow Lite int8 model (produced by ``export_tflite_int8()``)
and replays a fault scenario, measuring per-step inference latency and memory
footprint.  Designed to run identically on:

* Development workstation (x86-64)
* Raspberry Pi 4 / Zero 2 W (ARM64 / ARMv6)
* STM32 H-series + TFLite Micro (with minor C-port adaptations)

Usage
-----
::

    # 1. Train a model and export the int8 TFLite file:
    python sentinel_x_advanced.py    # writes sentinel_x_model_int8.tflite

    # 2. Replay the default built-in scenario:
    python scripts/replay_tflite.py

    # 3. Use a custom model path and scenario file:
    python scripts/replay_tflite.py \\
        --model sentinel_x_model_int8.tflite \\
        --scenario path/to/scenario.json \\
        --steps 100

    # 4. Print a latency / memory report at the end:
    python scripts/replay_tflite.py --report

Scenario file format (JSON)
----------------------------
A JSON array of state vectors (each an array of 11 floats, normalised to
[0, 1]).  Example::

    [
      [0.0, 0.0, 0.1, 0.0, 0.5, 0.0, 0.0, 0.8, 0.1, 0.9, 0.0],
      [0.1, 1.0, 0.2, 0.0, 0.4, 1.0, 0.0, 0.7, 0.1, 0.8, 0.0],
      ...
    ]

If no scenario file is supplied the script generates 50 synthetic states
covering common fault patterns (healthy, memory fault, thermal fault, power
brownout, all-fault cascade).

Output
------
For each step the script prints the chosen action, Q-values (or probabilities
for PPO), and the inference time in milliseconds.  At the end a summary
table is printed with min / mean / max latency and peak RSS memory.

Notes
-----
* The TFLite runtime is included in the standard ``tensorflow`` wheel.
  On bare-metal MCUs use ``tflite-micro`` instead.
* On Raspberry Pi the ``tflite-runtime`` wheel (much smaller than the full TF)
  is recommended: ``pip install tflite-runtime``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Action labels (must match the training environment)
# ---------------------------------------------------------------------------
ACTION_LABELS = {
    0: "DO_NOTHING",
    1: "RESTART",
    2: "SWITCH_REDUNDANT",
    3: "SAFE_MODE",
}
STATE_DIM = 11  # FederatedSwarm.COORD_STATE_DIM


# ---------------------------------------------------------------------------
# TFLite interpreter loader (graceful fallback)
# ---------------------------------------------------------------------------

def _load_interpreter(model_path: str):
    """Load a TFLite model; try tflite-runtime first, then full TF."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"TFLite model not found: {model_path}\n"
            "Run 'python sentinel_x_advanced.py' first to generate it."
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
                "Install one with: pip install tflite-runtime   # (lightweight)\n"
                "           or:    pip install tensorflow        # (full)"
            ) from exc

    interp.allocate_tensors()
    return interp


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _run_inference(interp, state: np.ndarray) -> tuple[int, np.ndarray, float]:
    """Run one forward pass; return (action, output_vector, latency_ms)."""
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # Quantise if int8 model
    inp_dtype = input_details[0]["dtype"]
    if inp_dtype == np.int8:
        scale, zero_point = input_details[0]["quantization"]
        if scale == 0.0:
            scale = 1.0
        state_q = (state / scale + zero_point).clip(-128, 127).astype(np.int8)
    else:
        state_q = state.astype(np.float32)

    interp.set_tensor(input_details[0]["index"], state_q[np.newaxis, :])

    t0 = time.perf_counter()
    interp.invoke()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    raw = interp.get_tensor(output_details[0]["index"])[0]

    # Dequantise if int8
    if output_details[0]["dtype"] == np.int8:
        out_scale, out_zp = output_details[0]["quantization"]
        if out_scale == 0.0:
            out_scale = 1.0
        raw = (raw.astype(np.float32) - out_zp) * out_scale

    action = int(np.argmax(raw))
    return action, raw, latency_ms


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def _builtin_scenario(n_steps: int = 50) -> list[np.ndarray]:
    """Generate a synthetic fault scenario covering key state patterns."""
    rng = np.random.default_rng(42)
    states = []
    patterns = [
        # (description, state template override)
        np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.9, 0.05, 0.95, 0.0]),  # healthy
        np.array([0.3, 1.0, 0.1,  0.0, 0.2, 1.0, 0.0, 0.8, 0.05, 0.9,  0.0]),  # memory fault
        np.array([0.0, 0.0, 0.05, 0.0, 0.3, 1.0, 1.0, 0.75,0.05, 0.9,  0.0]),  # thermal fault
        np.array([0.0, 0.0, 0.05, 0.0, 0.5, 1.0, 0.0, 0.10,0.05, 0.85, 0.0]),  # power brownout
        np.array([0.2, 1.0, 0.15, 1.0, 0.6, 1.0, 1.0, 0.12,0.2,  0.3,  0.1]),  # cascade
    ]
    for i in range(n_steps):
        base = patterns[i % len(patterns)].copy()
        noise = rng.normal(0, 0.02, size=STATE_DIM).astype(np.float32)
        state = np.clip(base + noise, 0.0, 1.0).astype(np.float32)
        states.append(state)
    return states


def _load_scenario(path: str) -> list[np.ndarray]:
    """Load a scenario from a JSON file (array of state vectors)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Scenario JSON must be a list of state vectors.")
    states = []
    for i, vec in enumerate(data):
        arr = np.array(vec, dtype=np.float32)
        if arr.shape != (STATE_DIM,):
            raise ValueError(
                f"State {i} has shape {arr.shape}; expected ({STATE_DIM},)."
            )
        states.append(arr)
    return states


# ---------------------------------------------------------------------------
# Memory footprint helper
# ---------------------------------------------------------------------------

def _peak_rss_mb() -> Optional[float]:
    """Return the current process RSS memory in MiB (Linux / macOS only)."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux: ru_maxrss in kB; macOS: ru_maxrss in bytes
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 ** 2)
        return usage.ru_maxrss / 1024
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    model_path: str = "sentinel_x_model_int8.tflite",
    scenario_path: Optional[str] = None,
    n_steps: int = 50,
    report: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Replay a fault scenario through a TFLite int8 model and measure latency.

    Parameters
    ----------
    model_path : str
        Path to the TFLite model file.
    scenario_path : str or None
        Path to a JSON scenario file.  If None, uses the built-in scenario.
    n_steps : int
        Number of steps to run (ignored if scenario_path is given).
    report : bool
        Print a latency / memory summary at the end.
    verbose : bool
        Print per-step output.

    Returns
    -------
    dict
        ``{latency_ms: list, actions: list, peak_rss_mb: float|None}``
    """
    interp = _load_interpreter(model_path)

    if scenario_path is not None:
        states = _load_scenario(scenario_path)
    else:
        states = _builtin_scenario(n_steps)

    latencies: list[float] = []
    actions: list[int] = []

    if verbose:
        print(f"\nSENTINEL-X TFLite Replay  —  {model_path}")
        print(f"Scenario: {'custom' if scenario_path else 'built-in'}  |  "
              f"{len(states)} steps")
        print("-" * 60)
        print(f"{'Step':>5}  {'Action':<18}  {'Latency (ms)':>12}  Q-values / logits")
        print("-" * 60)

    for step, state in enumerate(states):
        action, output, lat = _run_inference(interp, state)
        latencies.append(lat)
        actions.append(action)
        if verbose:
            label = ACTION_LABELS.get(action, str(action))
            q_str = "  ".join(f"{v:+.3f}" for v in output)
            print(f"{step:>5}  {label:<18}  {lat:>12.3f}  [{q_str}]")

    if report:
        arr = np.array(latencies)
        rss = _peak_rss_mb()
        print("\n" + "=" * 60)
        print("  SENTINEL-X TFLite Replay Summary")
        print("=" * 60)
        print(f"  Steps              : {len(latencies)}")
        print(f"  Latency  min       : {arr.min():.3f} ms")
        print(f"  Latency  mean      : {arr.mean():.3f} ms")
        print(f"  Latency  max       : {arr.max():.3f} ms")
        print(f"  Latency  p95       : {np.percentile(arr, 95):.3f} ms")
        if rss is not None:
            print(f"  Peak RSS           : {rss:.1f} MiB")
        # Action distribution
        unique, counts = np.unique(actions, return_counts=True)
        print("  Action distribution:")
        for a, c in zip(unique, counts):
            print(f"    {ACTION_LABELS.get(int(a), str(a)):<20} {c:>4}× "
                  f"({100*c/len(actions):.0f}%)")
        print("=" * 60)

    return {
        "latency_ms": latencies,
        "actions": actions,
        "peak_rss_mb": _peak_rss_mb(),
    }


def _parse_args():
    p = argparse.ArgumentParser(
        description="Replay a SENTINEL-X TFLite int8 model on a fault scenario.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="sentinel_x_model_int8.tflite",
        help="Path to the TFLite int8 model (default: sentinel_x_model_int8.tflite)",
    )
    p.add_argument(
        "--scenario",
        default=None,
        help="JSON scenario file (optional; default: built-in synthetic scenario)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of synthetic steps when no --scenario file is given (default: 50)",
    )
    p.add_argument(
        "--report",
        action="store_true",
        default=True,
        help="Print latency / memory summary at the end (default: on)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step output (summary still printed if --report is set)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        model_path=args.model,
        scenario_path=args.scenario,
        n_steps=args.steps,
        report=args.report,
        verbose=not args.quiet,
    )
