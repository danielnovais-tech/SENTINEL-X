"""
hardware – SENTINEL-X Hardware Abstraction Layer (HAL)
=======================================================

This package provides the bridge between the SENTINEL-X software pipeline
and real embedded hardware.  It is structured in three layers:

1. **Abstract HAL** (``hal.py``) – platform-independent base classes that every
   concrete driver must implement.  Write your application code against these
   interfaces; swap the underlying driver without changing application logic.

2. **Concrete drivers** – platform-specific implementations:

   * ``rpi_driver.py`` – Raspberry Pi 4 / Zero 2 W via ``RPi.GPIO`` and
     ``pyserial``.  Works on any Pi with the standard 40-pin GPIO header.
   * ``stm32_driver.py`` – STM32 (or any serial-connected MCU) via
     ``pyserial``.  Suitable for STM32H7, Nucleo boards, or ESP32.

3. **Sensor interfaces** (``sensor_interfaces.py``) – maps raw ADC / UART
   telemetry to the normalised 11-dimensional state vector expected by
   every SENTINEL-X RL agent.

Typical usage
-------------
::

    from hardware import create_hardware_interface, SpacecraftSensorInterface
    from sentinel_x_advanced import DQNAgent

    # Auto-detect target platform
    hw = create_hardware_interface("rpi")          # or "stm32", "simulation"
    sensor = SpacecraftSensorInterface(hw)

    while True:
        state = sensor.read_state()                # normalised 11-dim vector
        action = agent.act(state, explore=False)
        hw.write_action(action)

See ``docs/hardware_integration.md`` for wiring diagrams and step-by-step
integration instructions.
"""

from hardware.hal import (  # noqa: F401
    HardwareInterface,
    SensorReading,
    ActuatorCommand,
)
from hardware.sensor_interfaces import SpacecraftSensorInterface  # noqa: F401
from hardware.rpi_driver import RaspberryPiDriver  # noqa: F401
from hardware.stm32_driver import STM32Driver  # noqa: F401


def create_hardware_interface(target: str = "simulation") -> HardwareInterface:
    """
    Factory function – return a ``HardwareInterface`` for the given target.

    Parameters
    ----------
    target : str
        One of ``"rpi"`` (Raspberry Pi), ``"stm32"`` (STM32 / any serial
        MCU), or ``"simulation"`` (pure-Python, no hardware required).

    Returns
    -------
    HardwareInterface
        A fully-initialised concrete driver ready to call ``read_sensors()``
        and ``write_action()``.

    Raises
    ------
    ValueError
        If *target* is not one of the supported platforms.
    ImportError
        If the required driver library (``RPi.GPIO``, ``pyserial``) is not
        installed for the requested target.
    """
    target = target.lower().strip()
    if target == "rpi":
        return RaspberryPiDriver()
    if target in ("stm32", "serial"):
        return STM32Driver()
    if target == "simulation":
        from hardware.rpi_driver import SimulatedDriver
        return SimulatedDriver()
    raise ValueError(
        f"Unknown target '{target}'. "
        "Choose from: 'rpi', 'stm32', 'simulation'."
    )


__all__ = [
    "HardwareInterface",
    "SensorReading",
    "ActuatorCommand",
    "SpacecraftSensorInterface",
    "RaspberryPiDriver",
    "STM32Driver",
    "create_hardware_interface",
]
