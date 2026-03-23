"""
hardware/rpi_driver.py – Raspberry Pi Hardware Driver
=====================================================

Concrete :class:`~hardware.hal.HardwareInterface` implementation for
Raspberry Pi (any model with a 40-pin GPIO header).

Two classes are provided:

* :class:`RaspberryPiDriver` – uses real ``RPi.GPIO`` and ``pyserial``.
  Requires ``pip install RPi.GPIO pyserial`` and root / GPIO group membership.

* :class:`SimulatedDriver` – pure-Python simulation.  Generates synthetic
  fault states without any hardware dependency.  Use this on development
  machines, in CI, and for unit tests.

Wiring
------
See ``docs/hardware_integration.md`` for GPIO pin assignments and UART wiring.

Quick reference::

    Raspberry Pi 40-pin header → Spacecraft MCU board
    ─────────────────────────────────────────────────
    Pin 8  (GPIO 14, UART TX) → MCU RX
    Pin 10 (GPIO 15, UART RX) → MCU TX
    Pin 6  (GND)              → MCU GND
    Pin 4  (5V)               → MCU VIN  (if MCU is 5V tolerant)

Default UART settings: 115200 baud, 8N1, no hardware flow control.
"""

from __future__ import annotations

import random
import time
import warnings
from typing import Optional

from hardware.hal import ActuatorCommand, HardwareInterface, SensorReading

# ---------------------------------------------------------------------------
# Optional hardware imports (fail gracefully on non-Pi systems)
# ---------------------------------------------------------------------------
try:
    import RPi.GPIO as GPIO          # type: ignore[import]
    _HAS_GPIO = True
except ImportError:
    _HAS_GPIO = False

try:
    import serial                     # type: ignore[import]
    _HAS_SERIAL = True
except ImportError:
    _HAS_SERIAL = False


# ---------------------------------------------------------------------------
# RaspberryPiDriver
# ---------------------------------------------------------------------------

class RaspberryPiDriver(HardwareInterface):
    """
    SENTINEL-X hardware driver for Raspberry Pi (RPi.GPIO + pyserial).

    Parameters
    ----------
    uart_port : str
        Serial device path (default ``/dev/serial0`` – the Pi hardware UART).
    baud_rate : int
        UART baud rate (default 115 200).
    fault_gpio_pin : int
        BCM pin number used to trigger an external fault-injection signal
        (output, active-high).  Set to ``None`` to disable GPIO output.
    watchdog_gpio_pin : int
        BCM pin number used to toggle the hardware watchdog every
        ``watchdog_interval_s`` seconds.  Set to ``None`` to disable.
    watchdog_interval_s : float
        Watchdog pulse interval in seconds (default 0.5).

    Raises
    ------
    ImportError
        If ``RPi.GPIO`` or ``pyserial`` is not installed.
    """

    def __init__(
        self,
        uart_port: str = "/dev/serial0",
        baud_rate: int = 115_200,
        fault_gpio_pin: Optional[int] = None,
        watchdog_gpio_pin: Optional[int] = None,
        watchdog_interval_s: float = 0.5,
    ) -> None:
        if not _HAS_GPIO:
            raise ImportError(
                "RPi.GPIO is not installed. Run: pip install RPi.GPIO\n"
                "On non-Pi hardware use SimulatedDriver instead."
            )
        if not _HAS_SERIAL:
            raise ImportError(
                "pyserial is not installed. Run: pip install pyserial"
            )

        self._uart_port          = uart_port
        self._baud_rate          = baud_rate
        self._fault_pin          = fault_gpio_pin
        self._watchdog_pin       = watchdog_gpio_pin
        self._watchdog_interval  = watchdog_interval_s
        self._last_watchdog_ping = 0.0
        self._port: Optional["serial.Serial"] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        if self._fault_pin is not None:
            GPIO.setup(self._fault_pin, GPIO.OUT, initial=GPIO.LOW)
        if self._watchdog_pin is not None:
            GPIO.setup(self._watchdog_pin, GPIO.OUT, initial=GPIO.LOW)

        self._port = serial.Serial(
            port=self._uart_port,
            baudrate=self._baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.02,    # 20 ms read timeout
        )

    def close(self) -> None:
        if self._port and self._port.is_open:
            self._port.close()
        GPIO.cleanup()

    # ── Core interface ────────────────────────────────────────────────────────

    def read_sensors(self) -> SensorReading:
        """
        Request a sensor frame from the MCU via UART and parse the response.

        Protocol: host sends ``0x52`` ("R"), MCU responds with a 22-byte
        packed frame (see docs/hardware_integration.md §3 for encoding).

        If the MCU does not respond within the timeout a warning is logged
        and the last known good reading is returned.
        """
        if self._port is None or not self._port.is_open:
            raise RuntimeError("UART port is not open. Call open() first.")

        # Request a sensor frame
        self._port.write(b"\x52")        # ASCII 'R'
        raw = self._port.read(22)

        if len(raw) < 22:
            warnings.warn(
                f"Short UART frame ({len(raw)}/22 bytes). "
                "Check wiring and MCU firmware.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._fallback_reading()

        return _parse_uart_frame(raw)

    def write_action(self, cmd: ActuatorCommand) -> None:
        """Transmit a 2-byte action packet to the MCU."""
        if self._port is None or not self._port.is_open:
            raise RuntimeError("UART port is not open. Call open() first.")
        self._port.write(cmd.to_bytes())

    def is_connected(self) -> bool:
        return (
            self._port is not None
            and self._port.is_open
            and _HAS_GPIO
        )

    def watchdog_ping(self) -> None:
        """Toggle the watchdog GPIO pin if configured."""
        if self._watchdog_pin is None:
            return
        now = time.monotonic()
        if now - self._last_watchdog_ping >= self._watchdog_interval:
            GPIO.output(self._watchdog_pin, GPIO.HIGH)
            time.sleep(0.001)
            GPIO.output(self._watchdog_pin, GPIO.LOW)
            self._last_watchdog_ping = now

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _fallback_reading() -> SensorReading:
        return SensorReading(
            timestamp_s=time.time(),
            health_flag=True,          # flag a health event for missing frame
        )


def _parse_uart_frame(data: bytes) -> SensorReading:
    """
    Parse the 22-byte MCU telemetry frame into a :class:`SensorReading`.

    Frame layout (all little-endian):

    ======  ====  =========================================================
    Offset  Size  Field
    ======  ====  =========================================================
    0       1     Start byte (0xAB)
    1       2     memory_error_count (uint16)
    3       1     flags: bit0=parity, bit1=stuck, bit2=health, bit3=thermal
    4       4     sensor_deviation (float32)
    8       4     time_since_recovery_s (float32)
    12      4     power_level_v (float32)
    16      4     attitude_rate_dps (float32)
    20      1     comm_quality (uint8, 0–255 → –120…0 dBm)
    21      1     End byte (0xCD)
    ======  ====  =========================================================
    """
    import struct
    if data[0] != 0xAB or data[21] != 0xCD:
        raise ValueError(
            f"Invalid frame framing bytes: start=0x{data[0]:02X} end=0x{data[21]:02X}"
        )
    mem_err   = struct.unpack_from("<H", data, 1)[0]
    flags     = data[3]
    sensor_dev, recovery, power, att_rate = struct.unpack_from("<ffff", data, 4)
    comm_raw  = data[20]
    comm_db   = -120.0 + (comm_raw / 255.0) * 120.0   # map [0,255] → [−120, 0]

    return SensorReading(
        timestamp_s          = time.time(),
        memory_error_count   = mem_err,
        parity_error_flag    = bool(flags & 0x01),
        sensor_deviation     = sensor_dev,
        sensor_stuck         = bool(flags & 0x02),
        time_since_recovery_s = recovery,
        health_flag          = bool(flags & 0x04),
        thermal_fault        = bool(flags & 0x08),
        power_level_v        = power,
        attitude_rate_dps    = att_rate,
        comm_quality_db      = comm_db,
    )


# ---------------------------------------------------------------------------
# SimulatedDriver (no hardware required)
# ---------------------------------------------------------------------------

class SimulatedDriver(HardwareInterface):
    """
    Pure-Python simulation driver.

    Generates synthetic spacecraft telemetry with configurable fault
    injection.  No hardware dependency — works on any platform.

    Parameters
    ----------
    fault_prob : float
        Per-step probability of injecting a random fault (default 0.05).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, fault_prob: float = 0.05, seed: Optional[int] = None) -> None:
        self._fault_prob = float(fault_prob)
        self._rng        = random.Random(seed)
        self._step       = 0
        self._connected  = True
        self._power_v    = 3.8    # start with healthy power

    def open(self) -> None:
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def read_sensors(self) -> SensorReading:
        self._step += 1
        inject = self._rng.random() < self._fault_prob

        # Slowly drain power
        self._power_v = max(0.0, self._power_v - self._rng.uniform(0.0, 0.02))

        return SensorReading(
            timestamp_s           = time.time(),
            memory_error_count    = self._rng.randint(0, 3) if inject else 0,
            parity_error_flag     = inject and self._rng.random() < 0.4,
            sensor_deviation      = self._rng.gauss(0, 0.5 if inject else 0.05),
            sensor_stuck          = inject and self._rng.random() < 0.2,
            time_since_recovery_s = float(self._step * 5),
            health_flag           = inject and self._rng.random() < 0.6,
            thermal_fault         = inject and self._rng.random() < 0.3,
            power_level_v         = self._power_v,
            attitude_rate_dps     = self._rng.gauss(0, 2.0 if inject else 0.2),
            comm_quality_db       = self._rng.uniform(-90, -50) if inject else -60.0,
        )

    def write_action(self, cmd: ActuatorCommand) -> None:
        if cmd.action_index in (1, 2, 3):
            # Simulate recovery: restore power partially
            self._power_v = min(3.8, self._power_v + 0.5)

    def is_connected(self) -> bool:
        return self._connected
