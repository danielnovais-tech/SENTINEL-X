#!/usr/bin/env python3
"""
scripts/mcu_emulator.py – MCU Hardware Emulator for SENTINEL-X
===============================================================

Simulates a microcontroller (Raspberry Pi, STM32H7, …) running the
SENTINEL-X int8 TFLite policy, enabling a **full hardware-in-the-loop (HIL)
demonstration without real hardware**.

Three operating modes
---------------------
``--demo``
    Self-contained fault-injection harness.  Generates synthetic fault
    states, runs inference on each, applies ``SafetyMonitor`` vetoes, and
    writes an ``mcu_emulator_log.csv`` decision log — exactly what a physical
    MCU would log over UART.

``--server``
    Starts a lightweight TCP socket server on ``localhost:<port>`` (default
    8765) that accepts newline-delimited JSON state vectors and returns
    JSON action records.  Suitable for integration with a software-in-the-
    loop spacecraft simulator or a hardware bridge script.

``--client``
    Connects to a running ``--server`` instance and sends a batch of random
    fault states to it, printing the responses.  Use this to verify the
    server is working correctly.

Protocol (server / client)
--------------------------
Each message is a **JSON object terminated by a newline** (``\\n``).

Client → Server::

    {"state": [0.0, 1.0, 0.2, 0.0, 0.4, 1.0, 0.0, 0.7, 0.1, 0.8, 0.0]}

Server → Client::

    {"step": 1, "action": 3, "action_label": "SAFE_MODE",
     "q_values": [-0.12, 0.03, 0.07, 0.41],
     "vetoed": false, "latency_ms": 0.42,
     "mcu_latency_ms": 1.50, "timestamp_ms": 1711065600000}

Field glossary
--------------
``action``
    Greedy action index (0=DO_NOTHING, 1=RESTART, 2=SWITCH_REDUNDANT,
    3=SAFE_MODE).
``vetoed``
    Whether ``SafetyMonitor`` replaced the model's action with a safe
    override.
``latency_ms``
    Actual Python inference time.
``mcu_latency_ms``
    *Simulated* MCU latency (configurable via ``--mcu-latency-ms``); set to
    the typical inference time you measured on your target hardware.

Fault-injection log (``--demo`` mode)
--------------------------------------
Written to ``mcu_emulator_log.csv`` (or ``--log-file``).  Columns::

    step, fault_type, state_summary, action, action_label,
    vetoed, latency_ms, mcu_latency_ms

Usage examples
--------------
::

    # Self-contained demo (no hardware required)
    python scripts/mcu_emulator.py --demo

    # Server mode on port 8765
    python scripts/mcu_emulator.py --server --port 8765

    # Client test (in a second terminal while server is running)
    python scripts/mcu_emulator.py --client --port 8765 --steps 10

    # Demo with custom model and simulated 2 ms MCU latency
    python scripts/mcu_emulator.py --demo \\
        --model sentinel_x_model_int8.tflite \\
        --mcu-latency-ms 2.0 \\
        --log-file my_hil_log.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STATE_DIM = 11   # must match DQNAgent state_dim
_ACTION_LABELS = {
    0: "DO_NOTHING",
    1: "RESTART",
    2: "SWITCH_REDUNDANT",
    3: "SAFE_MODE",
}
_DEFAULT_MODEL = "sentinel_x_model_int8.tflite"

# State vector indices
# 0  mem_error_ratio      1  parity_flag          2  sensor_deviation_norm
# 3  sensor_stuck_flag    4  time_since_recovery  5  health_flag
# 6  thermal_fault_flag   7  power_level_norm     8  attitude_rate_norm
# 9  comm_quality_norm    10 peer_sensor_deviation

# ---------------------------------------------------------------------------
# Fault scenario catalogue (simulates sensor telemetry from the spacecraft)
# ---------------------------------------------------------------------------
#                            [mem  par  sens stk  tsr  hlt  thm  pwr  att  com  peer]
_FAULT_SCENARIOS: list[tuple[str, np.ndarray]] = [
    ("healthy",           np.array([0.00, 0.0, 0.05, 0.0, 0.50, 0.0, 0.0, 0.90, 0.05, 0.95, 0.00])),
    ("memory_fault",      np.array([0.30, 1.0, 0.08, 0.0, 0.40, 1.0, 0.0, 0.85, 0.08, 0.90, 0.10])),
    ("thermal_spike",     np.array([0.05, 0.0, 0.15, 0.0, 0.45, 0.0, 1.0, 0.80, 0.12, 0.88, 0.05])),
    ("power_brownout",    np.array([0.02, 0.0, 0.07, 0.0, 0.60, 0.0, 0.0, 0.12, 0.10, 0.85, 0.02])),
    ("sensor_stuck",      np.array([0.10, 0.0, 0.40, 1.0, 0.35, 0.0, 0.0, 0.75, 0.09, 0.80, 0.08])),
    ("comm_loss",         np.array([0.05, 0.0, 0.06, 0.0, 0.55, 0.0, 0.0, 0.70, 0.11, 0.04, 0.03])),
    ("cascade_all",       np.array([0.50, 1.0, 0.60, 1.0, 0.20, 1.0, 1.0, 0.08, 0.50, 0.03, 0.30])),
    ("recovery_ongoing",  np.array([0.08, 0.0, 0.10, 0.0, 0.80, 0.0, 0.0, 0.60, 0.08, 0.75, 0.05])),
]


# ---------------------------------------------------------------------------
# TFLite loader / inference (shared with replay_tflite.py logic)
# ---------------------------------------------------------------------------

def _load_interpreter(model_path: str):
    """Load a TFLite model; try tflite-runtime first, then full TF."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"TFLite model not found: {model_path}\n"
            "Run 'python sentinel_x_advanced.py' first to generate it,\n"
            "or use run_experiment.py to train and export automatically."
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


def _run_inference(interp, state: np.ndarray) -> tuple[int, np.ndarray, float]:
    """Run one forward pass; return (action, output_vector, latency_ms)."""
    in_details  = interp.get_input_details()
    out_details = interp.get_output_details()

    inp_dtype = in_details[0]["dtype"]
    if inp_dtype == np.int8:
        scale, zp = in_details[0]["quantization"]
        if scale == 0.0:
            scale = 1.0
        x = np.round(state.astype(np.float32) / scale + zp).astype(np.int8)
    else:
        x = state.astype(np.float32)
    x = x.reshape(1, -1)

    t0 = time.perf_counter()
    interp.set_tensor(in_details[0]["index"], x)
    interp.invoke()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    out = interp.get_tensor(out_details[0]["index"])[0]
    if out_details[0]["dtype"] == np.int8:
        scale, zp = out_details[0]["quantization"]
        if scale == 0.0:
            scale = 1.0
        out = (out.astype(np.float32) - zp) * scale

    action = int(np.argmax(out))
    return action, out, latency_ms


# ---------------------------------------------------------------------------
# SafetyMonitor veto (inline – avoids importing the full sentinel_x_advanced)
# ---------------------------------------------------------------------------

def _apply_safety_veto(action: int, state: np.ndarray) -> tuple[int, bool]:
    """Apply hard safety rules; return (final_action, vetoed)."""
    health_flag    = float(state[5])
    power_level    = float(state[7])
    vetoed = False

    # Rule 1: never do nothing on an active fault
    if health_flag >= 0.5 and action == 0:
        action = 3   # SAFE_MODE
        vetoed = True

    # Rule 2: never full-reset when power critically low
    if power_level < 0.15 and action == 2:
        action = 3   # SAFE_MODE
        vetoed = True

    return action, vetoed


# ---------------------------------------------------------------------------
# Core inference + veto call with simulated MCU delay
# ---------------------------------------------------------------------------

def _infer_with_veto(
    interp,
    state: np.ndarray,
    mcu_latency_ms: float,
    step: int,
) -> dict[str, Any]:
    """Run inference, apply veto, add simulated MCU latency; return a record."""
    action, q_values, latency_ms = _run_inference(interp, state)
    action, vetoed               = _apply_safety_veto(action, state)

    # Simulate the slower MCU clock (busy-wait would skew benchmarks; just record)
    record: dict[str, Any] = {
        "step":           step,
        "action":         action,
        "action_label":   _ACTION_LABELS.get(action, str(action)),
        "q_values":       [round(float(v), 4) for v in q_values],
        "vetoed":         vetoed,
        "latency_ms":     round(latency_ms, 3),
        "mcu_latency_ms": round(mcu_latency_ms, 3),
        "timestamp_ms":   int(time.time() * 1000),
    }
    return record


# ---------------------------------------------------------------------------
# Demo mode (fault-injection harness)
# ---------------------------------------------------------------------------

def run_demo(args: argparse.Namespace) -> None:
    """Self-contained HIL demo: fault injection → inference → veto → log."""
    print("=" * 65)
    print("  SENTINEL-X MCU Emulator  –  Fault-Injection Demo")
    print("=" * 65)
    print(f"  Model         : {args.model}")
    print(f"  MCU latency   : {args.mcu_latency_ms:.1f} ms  (simulated)")
    print(f"  Log file      : {args.log_file}")
    print()

    interp = _load_interpreter(args.model)

    log_rows: list[dict] = []
    # Repeat scenario list to hit requested step count
    scenario_cycle = _FAULT_SCENARIOS * (
        (args.steps + len(_FAULT_SCENARIOS) - 1) // len(_FAULT_SCENARIOS)
    )
    print(f"  {'Step':>4}  {'Fault Type':<22}  {'Action':<20}  {'Veto?':<6}  {'Lat ms':>7}")
    print("  " + "-" * 62)

    for step, (fault_name, state) in enumerate(scenario_cycle[:args.steps], start=1):
        record = _infer_with_veto(interp, state, args.mcu_latency_ms, step)
        veto_mark = "YES" if record["vetoed"] else "no"
        print(
            f"  {step:>4}  {fault_name:<22}  "
            f"{record['action_label']:<20}  {veto_mark:<6}  "
            f"{record['latency_ms']:>6.2f}"
        )
        log_rows.append({
            "step":           record["step"],
            "fault_type":     fault_name,
            "state_summary":  f"health={state[5]:.2f} power={state[7]:.2f} thermal={state[6]:.0f}",
            "action":         record["action"],
            "action_label":   record["action_label"],
            "vetoed":         record["vetoed"],
            "latency_ms":     record["latency_ms"],
            "mcu_latency_ms": record["mcu_latency_ms"],
        })

    _write_log(log_rows, args.log_file)
    _print_demo_summary(log_rows)


def _write_log(rows: list[dict], log_file: str) -> None:
    """Write decision log to CSV."""
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(log_file, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Log written to: {log_file}  ({len(rows)} rows)")


def _print_demo_summary(rows: list[dict]) -> None:
    """Print a summary table of the demo run."""
    if not rows:
        return
    latencies  = [r["latency_ms"] for r in rows]
    n_vetoed   = sum(1 for r in rows if r["vetoed"])
    action_counts: dict[str, int] = {}
    for r in rows:
        action_counts[r["action_label"]] = action_counts.get(r["action_label"], 0) + 1

    print("\n  ── Demo Summary ──────────────────────────────────────────")
    print(f"  Steps run      : {len(rows)}")
    print(f"  Safety vetoes  : {n_vetoed} / {len(rows)}  ({n_vetoed / len(rows):.1%})")
    print(f"  Latency (ms)   : min={min(latencies):.3f}  "
          f"mean={sum(latencies)/len(latencies):.3f}  "
          f"max={max(latencies):.3f}")
    print("  Action dist    :", "  ".join(f"{a}={c}" for a, c in action_counts.items()))
    print("=" * 65)


# ---------------------------------------------------------------------------
# Server mode
# ---------------------------------------------------------------------------

def run_server(args: argparse.Namespace) -> None:
    """TCP server that accepts JSON state vectors and returns action records."""
    interp = _load_interpreter(args.model)
    step   = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", args.port))
        srv.listen(1)
        print(f"MCU emulator server listening on 127.0.0.1:{args.port}")
        print("Send: {\"state\": [11 floats]}\\n   (Ctrl-C to stop)\n")

        while True:
            conn, addr = srv.accept()
            print(f"  Connection from {addr}")
            with conn:
                buf = ""
                while True:
                    chunk = conn.recv(4096).decode("utf-8", errors="replace")
                    if not chunk:
                        break
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg   = json.loads(line)
                            state = np.array(msg["state"], dtype=np.float32)
                            if state.shape != (_STATE_DIM,):
                                raise ValueError(
                                    f"Expected {_STATE_DIM} floats, got {state.shape}"
                                )
                            step += 1
                            record = _infer_with_veto(
                                interp, state, args.mcu_latency_ms, step
                            )
                            conn.sendall((json.dumps(record) + "\n").encode())
                        except (json.JSONDecodeError, KeyError, ValueError) as exc:
                            err = json.dumps({"error": str(exc)}) + "\n"
                            conn.sendall(err.encode())


# ---------------------------------------------------------------------------
# Client mode (test / demonstration)
# ---------------------------------------------------------------------------

def run_client(args: argparse.Namespace) -> None:
    """Connect to a running server and send random fault states."""
    rng = np.random.default_rng(42)
    print(f"Connecting to 127.0.0.1:{args.port} …")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(("127.0.0.1", args.port))
        print(f"Connected. Sending {args.steps} random states …\n")
        recv_buf = ""
        for i in range(args.steps):
            state = rng.random(_STATE_DIM).tolist()
            sock.sendall((json.dumps({"state": state}) + "\n").encode())
            # Read response
            while "\n" not in recv_buf:
                recv_buf += sock.recv(4096).decode("utf-8", errors="replace")
            line, recv_buf = recv_buf.split("\n", 1)
            resp = json.loads(line)
            print(
                f"  Step {resp.get('step',i+1):>3}  "
                f"action={resp.get('action_label','?'):<20}  "
                f"vetoed={str(resp.get('vetoed','?')):<5}  "
                f"latency={resp.get('latency_ms',0):.2f} ms"
            )
    print("\nClient done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SENTINEL-X MCU hardware emulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default=_DEFAULT_MODEL,
        help="Path to int8 TFLite model (default: sentinel_x_model_int8.tflite)",
    )
    parser.add_argument(
        "--mcu-latency-ms", type=float, default=1.5, dest="mcu_latency_ms",
        help="Simulated MCU inference latency in ms (default: 1.5)",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo",   action="store_true", help="Run self-contained demo")
    mode.add_argument("--server", action="store_true", help="Start TCP server")
    mode.add_argument("--client", action="store_true", help="Run client test")

    parser.add_argument(
        "--port", type=int, default=8765,
        help="TCP port for server/client modes (default: 8765)",
    )
    parser.add_argument(
        "--steps", type=int, default=len(_FAULT_SCENARIOS),
        help="Number of inference steps (demo / client modes)",
    )
    parser.add_argument(
        "--log-file", default="mcu_emulator_log.csv", dest="log_file",
        help="CSV decision log path (demo mode, default: mcu_emulator_log.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Default to demo mode if no mode flag was passed
    if not args.server and not args.client:
        args.demo = True

    if args.server:
        run_server(args)
    elif args.client:
        run_client(args)
    else:
        run_demo(args)
