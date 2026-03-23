"""
hardware/hal.py – Hardware Abstraction Layer (HAL)
==================================================

Platform-independent abstract base classes for the SENTINEL-X hardware
integration layer.  Concrete driver modules (``rpi_driver.py``,
``stm32_driver.py``) implement these interfaces.

Design principles
-----------------
* **Single responsibility** – each class covers exactly one hardware concern.
* **No hardware dependency in this file** – importing ``hal`` on any platform
  (CI server, Windows, macOS) always succeeds.
* **Simulation-friendly** – all abstract methods have sensible docstring
  contracts that a pure-Python simulation can satisfy.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------

@dataclass
class SensorReading:
    """
    A raw sensor sample from one spacecraft subsystem.

    All physical values are in SI units before normalisation.

    Attributes
    ----------
    timestamp_s : float
        Unix timestamp of the reading (seconds since epoch).
    memory_error_count : int
        Number of single-event upsets (SEUs) detected since last reset.
    parity_error_flag : bool
        True if a parity / ECC error is currently active.
    sensor_deviation : float
        Attitude sensor deviation from expected value (degrees or normalised).
    sensor_stuck : bool
        True if the sensor output has not changed in the last N samples.
    time_since_recovery_s : float
        Seconds since the last successful subsystem recovery.
    health_flag : bool
        True if any watchdog / FDIR flag is currently set.
    thermal_fault : bool
        True if temperature is outside safe operating range.
    power_level_v : float
        Bus voltage (V).  Typical range 3.3–5.0 V for 3.3V / 5V buses.
    attitude_rate_dps : float
        Angular rate in degrees per second.
    comm_quality_db : float
        Communication link quality / SNR in dBm.
    peer_deviation : float
        Deviation reported by peer spacecraft (federated context).
    """
    timestamp_s:          float
    memory_error_count:   int   = 0
    parity_error_flag:    bool  = False
    sensor_deviation:     float = 0.0
    sensor_stuck:         bool  = False
    time_since_recovery_s: float = 0.0
    health_flag:          bool  = False
    thermal_fault:        bool  = False
    power_level_v:        float = 3.3
    attitude_rate_dps:    float = 0.0
    comm_quality_db:      float = -60.0
    peer_deviation:       float = 0.0

    # -----------------------------------------------------------------
    # Normalisation constants (tune per mission hardware)
    # -----------------------------------------------------------------
    _MAX_MEMORY_ERRORS:   int   = 10
    _MAX_SENSOR_DEV:      float = 10.0    # degrees
    _MAX_RECOVERY_S:      float = 3600.0  # 1 hour
    _MIN_POWER_V:         float = 0.0
    _MAX_POWER_V:         float = 5.0
    _MAX_RATE_DPS:        float = 180.0
    _MIN_COMM_DB:         float = -120.0
    _MAX_COMM_DB:         float = 0.0
    _MAX_PEER_DEV:        float = 10.0

    def to_state_vector(self) -> List[float]:
        """
        Convert raw readings to the normalised 11-dim state vector used by
        every SENTINEL-X RL agent.

        All values are clipped to [0, 1].

        Returns
        -------
        list of float
            ``[mem_err, parity, sensor_dev, stuck, recovery,
               health, thermal, power, att_rate, comm, peer_dev]``
        """
        def _clip(x: float) -> float:
            return float(max(0.0, min(1.0, x)))

        return [
            _clip(self.memory_error_count / self._MAX_MEMORY_ERRORS),
            1.0 if self.parity_error_flag  else 0.0,
            _clip(abs(self.sensor_deviation) / self._MAX_SENSOR_DEV),
            1.0 if self.sensor_stuck        else 0.0,
            _clip(self.time_since_recovery_s / self._MAX_RECOVERY_S),
            1.0 if self.health_flag         else 0.0,
            1.0 if self.thermal_fault       else 0.0,
            _clip((self.power_level_v - self._MIN_POWER_V)
                  / (self._MAX_POWER_V - self._MIN_POWER_V)),
            _clip(abs(self.attitude_rate_dps) / self._MAX_RATE_DPS),
            _clip((self.comm_quality_db - self._MIN_COMM_DB)
                  / (self._MAX_COMM_DB - self._MIN_COMM_DB)),
            _clip(abs(self.peer_deviation) / self._MAX_PEER_DEV),
        ]


@dataclass
class ActuatorCommand:
    """
    An action command sent to the spacecraft actuators.

    Attributes
    ----------
    action_index : int
        Discrete action from the RL policy (0=DO_NOTHING, 1=RESTART,
        2=SWITCH_REDUNDANT, 3=SAFE_MODE).
    action_label : str
        Human-readable action name.
    confidence : float
        Max Q-value (or probability) associated with the chosen action.
    timestamp_s : float
        Unix timestamp when the command was issued.
    """
    action_index: int
    action_label: str         = "DO_NOTHING"
    confidence:   float       = 1.0
    timestamp_s:  float       = field(default_factory=time.time)

    _LABELS = {0: "DO_NOTHING", 1: "RESTART", 2: "SWITCH_REDUNDANT", 3: "SAFE_MODE"}

    def __post_init__(self) -> None:
        self.action_label = self._LABELS.get(self.action_index, str(self.action_index))

    def to_bytes(self) -> bytes:
        """Encode command as a 2-byte UART packet ``[0xAA, action_index]``."""
        return bytes([0xAA, self.action_index & 0xFF])

    @classmethod
    def from_bytes(cls, data: bytes) -> "ActuatorCommand":
        """Decode a 2-byte UART packet produced by ``to_bytes()``."""
        if len(data) < 2 or data[0] != 0xAA:
            raise ValueError(f"Invalid packet: {data!r}")
        return cls(action_index=data[1])


# ---------------------------------------------------------------------------
# Abstract hardware interface
# ---------------------------------------------------------------------------

class HardwareInterface(abc.ABC):
    """
    Abstract base class for all SENTINEL-X hardware drivers.

    Concrete implementations must override the three abstract methods.
    The optional lifecycle methods (``open`` / ``close``) may be left as
    no-ops for drivers that manage resources differently.

    Example
    -------
    ::

        class MyDriver(HardwareInterface):
            def read_sensors(self) -> SensorReading:
                ...
            def write_action(self, cmd: ActuatorCommand) -> None:
                ...
            def is_connected(self) -> bool:
                ...
    """

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Initialise hardware resources (GPIO, serial port, I²C bus, …)."""

    def close(self) -> None:
        """Release hardware resources gracefully."""

    def __enter__(self) -> "HardwareInterface":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Core interface (must be overridden) ──────────────────────────────────

    @abc.abstractmethod
    def read_sensors(self) -> SensorReading:
        """
        Sample all spacecraft sensors and return a :class:`SensorReading`.

        This method must be non-blocking and complete within the control-loop
        deadline (default: 10 ms).
        """

    @abc.abstractmethod
    def write_action(self, cmd: ActuatorCommand) -> None:
        """
        Send an action command to the spacecraft actuators.

        Parameters
        ----------
        cmd : ActuatorCommand
            Encoded command to transmit (UART, GPIO, I²C, …).
        """

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Return True if the hardware link is currently healthy."""

    # ── Optional helpers ─────────────────────────────────────────────────────

    def read_state_vector(self) -> List[float]:
        """
        Convenience wrapper: read sensors and return normalised state vector.

        Equivalent to ``read_sensors().to_state_vector()``.
        """
        return self.read_sensors().to_state_vector()

    def watchdog_ping(self) -> None:
        """
        Send a watchdog keep-alive signal.

        Override in drivers that require periodic watchdog pings to prevent
        hardware reset.  Default implementation is a no-op.
        """
