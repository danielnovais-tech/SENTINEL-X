"""
hardware/sensor_interfaces.py – Sensor Interface Layer
=======================================================

Maps raw :class:`~hardware.hal.SensorReading` objects (produced by a
concrete driver) to the normalised 11-dimensional state vector consumed by
every SENTINEL-X RL agent.

This module also provides :class:`SpacecraftSensorInterface`, a thin
convenience wrapper around a :class:`~hardware.hal.HardwareInterface` that
adds:

* **Moving-average filter** – smooths noisy sensor readings.
* **Stuck-sensor detection** – flags a sensor as stuck if consecutive
  readings deviate by less than ``stuck_threshold`` for more than
  ``stuck_window`` samples.
* **Anomaly logging** – records telemetry to an in-memory ring buffer for
  post-flight analysis.

Typical usage
-------------
::

    from hardware import create_hardware_interface, SpacecraftSensorInterface

    hw     = create_hardware_interface("rpi")
    sensor = SpacecraftSensorInterface(hw, filter_len=5, stuck_window=10)

    with hw:                          # open / close serial port automatically
        state = sensor.read_state()   # returns normalised 11-dim list
        print(state)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from hardware.hal import HardwareInterface, SensorReading


class SpacecraftSensorInterface:
    """
    Sensor pre-processing layer for SENTINEL-X.

    Parameters
    ----------
    driver : HardwareInterface
        Concrete hardware driver (RPi, STM32, or simulation).
    filter_len : int
        Length of the moving-average window for each state feature.
        Set to 1 to disable filtering.
    stuck_window : int
        Number of consecutive identical readings before a sensor is flagged
        as stuck.  Set to 0 to disable stuck detection.
    stuck_threshold : float
        Absolute change below which a reading is considered "unchanged".
    log_capacity : int
        Maximum number of telemetry records to keep in the ring buffer.
    """

    def __init__(
        self,
        driver: HardwareInterface,
        filter_len: int = 3,
        stuck_window: int = 5,
        stuck_threshold: float = 1e-4,
        log_capacity: int = 1000,
    ) -> None:
        self._driver          = driver
        self._filter_len      = max(1, filter_len)
        self._stuck_window    = max(0, stuck_window)
        self._stuck_threshold = float(stuck_threshold)
        self._log: Deque[dict] = deque(maxlen=log_capacity)

        # Circular buffers – one per state dimension (11)
        self._history: list[Deque[float]] = [
            deque(maxlen=self._filter_len) for _ in range(11)
        ]
        # Stuck-sensor detection buffers
        self._stuck_history: list[Deque[float]] = [
            deque(maxlen=max(1, self._stuck_window)) for _ in range(11)
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def read_state(self) -> List[float]:
        """
        Sample hardware, apply filters, and return a normalised 11-dim state.

        Returns
        -------
        list of float
            Values in [0, 1], ordered as:
            ``[mem_err, parity, sensor_dev, stuck, recovery,
               health, thermal, power, att_rate, comm, peer_dev]``
        """
        reading = self._driver.read_sensors()
        raw     = reading.to_state_vector()

        # Update filter buffers
        for i, v in enumerate(raw):
            self._history[i].append(v)
            self._stuck_history[i].append(v)

        # Moving-average filter
        filtered = [
            float(np.mean(list(buf))) for buf in self._history
        ]

        # Override stuck-sensor feature (index 3) with our own detection
        filtered[3] = 1.0 if self._is_stuck(0) else 0.0  # monitor sensor_dev

        # Log
        self._log.append({
            "ts": reading.timestamp_s,
            "raw": raw,
            "filtered": filtered,
        })

        return filtered

    def drain_log(self) -> list:
        """
        Return and clear the telemetry ring buffer.

        Returns a list of dicts with keys ``ts``, ``raw``, ``filtered``.
        """
        records = list(self._log)
        self._log.clear()
        return records

    def reset_filters(self) -> None:
        """Clear all filter and stuck-detection history."""
        for buf in self._history:
            buf.clear()
        for buf in self._stuck_history:
            buf.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_stuck(self, feature_index: int) -> bool:
        """Return True if feature *feature_index* has been stuck."""
        if self._stuck_window < 2:
            return False
        buf = list(self._stuck_history[feature_index])
        if len(buf) < self._stuck_window:
            return False
        return float(np.ptp(buf)) < self._stuck_threshold
