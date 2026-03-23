"""
sentinel_x.hardware.stm32_driver
=================================

Re-exports :class:`hardware.stm32_driver.STM32Driver` under the
``sentinel_x.hardware`` namespace and provides a command-line entry point
for quick connectivity tests::

    # Ping the MCU and print one sensor reading
    python -m sentinel_x.hardware.stm32_driver --port /dev/ttyUSB0 --baud 115200 --ping

    # Run N timing iterations and report UART round-trip latency
    python -m sentinel_x.hardware.stm32_driver --port /dev/ttyUSB0 --timing-runs 100
"""

from __future__ import annotations

import argparse
import sys
import time

# Re-export so ``from sentinel_x.hardware.stm32_driver import STM32Driver``
# works identically to ``from hardware.stm32_driver import STM32Driver``.
from hardware.stm32_driver import STM32Driver  # noqa: F401
from hardware.rpi_driver import SimulatedDriver  # noqa: F401

__all__ = ["STM32Driver", "SimulatedDriver"]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sentinel_x.hardware.stm32_driver",
        description=(
            "SENTINEL-X STM32 driver CLI – test UART connectivity and "
            "measure round-trip latency."
        ),
    )
    parser.add_argument(
        "--port", default=None,
        help=(
            "Serial port (e.g. /dev/ttyUSB0, COM3). "
            "Omit to use the simulated driver (no hardware needed)."
        ),
    )
    parser.add_argument(
        "--baud", type=int, default=115_200,
        help="UART baud rate (default 115200).",
    )
    parser.add_argument(
        "--ping", action="store_true",
        help=(
            "Send one request frame, print the parsed SensorReading, and exit. "
            "Use this to verify the MCU is connected and responding."
        ),
    )
    parser.add_argument(
        "--timing-runs", type=int, default=0,
        help=(
            "Number of round-trip timing iterations to run. "
            "Prints p50/p95/p99 latency summary when > 0."
        ),
    )
    args = parser.parse_args(argv)

    # ── Choose driver ──────────────────────────────────────────────────────
    if args.port:
        try:
            driver = STM32Driver(port=args.port, baud_rate=args.baud)
        except ImportError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Using STM32Driver on {args.port} @ {args.baud} baud")
    else:
        driver = SimulatedDriver()
        print("No --port specified – using SimulatedDriver (no hardware needed)")

    # ── Ping ───────────────────────────────────────────────────────────────
    if args.ping or args.timing_runs == 0:
        with driver:
            if not driver.is_connected() and args.port:
                print("ERROR: driver not connected after open()", file=sys.stderr)
                return 1
            reading = driver.read_sensors()
            vec = reading.to_state_vector()
            print("\nSensorReading:")
            print(f"  timestamp_s          = {reading.timestamp_s:.3f}")
            print(f"  memory_error_count   = {reading.memory_error_count}")
            print(f"  parity_error_flag    = {reading.parity_error_flag}")
            print(f"  sensor_deviation     = {reading.sensor_deviation:.3f}")
            print(f"  health_flag          = {reading.health_flag}")
            print(f"  thermal_fault        = {reading.thermal_fault}")
            print(f"  power_level_v        = {reading.power_level_v:.3f} V")
            print(f"  attitude_rate_dps    = {reading.attitude_rate_dps:.3f} dps")
            print(f"  comm_quality_db      = {reading.comm_quality_db:.1f} dB")
            print(f"\nNormalised state vector: {[round(v, 4) for v in vec]}")
            if not args.timing_runs:
                return 0

    # ── Timing runs ────────────────────────────────────────────────────────
    if args.timing_runs > 0:
        import numpy as np

        print(f"\nRunning {args.timing_runs} round-trip timing iterations …")
        times_ns: list = []

        with driver:
            for _ in range(args.timing_runs):
                t0 = time.perf_counter_ns()
                driver.read_sensors()
                times_ns.append(time.perf_counter_ns() - t0)

        arr = (
            __import__("numpy").array(times_ns, dtype=float) / 1e6  # → ms
        )
        print(
            f"\nUART / sensor-read round-trip latency ({args.timing_runs} runs):"
        )
        print(f"  min   = {arr.min():.3f} ms")
        print(f"  p50   = {float(__import__('numpy').percentile(arr, 50)):.3f} ms")
        print(f"  p95   = {float(__import__('numpy').percentile(arr, 95)):.3f} ms")
        print(f"  p99   = {float(__import__('numpy').percentile(arr, 99)):.3f} ms")
        print(f"  max   = {arr.max():.3f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(_main())
