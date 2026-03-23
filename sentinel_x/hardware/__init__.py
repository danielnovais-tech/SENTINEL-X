"""
sentinel_x.hardware – Hardware Abstraction Layer (subpackage mirror)
====================================================================

Re-exports the full ``hardware`` package so that both import paths work:

    from hardware import STM32Driver          # legacy / direct
    from sentinel_x.hardware import STM32Driver  # package form

The concrete driver modules are the single source of truth and live in
``hardware/``.  This subpackage simply re-exports them under the
``sentinel_x`` namespace.
"""

from hardware import (  # noqa: F401
    HardwareInterface,
    SensorReading,
    ActuatorCommand,
    SpacecraftSensorInterface,
    RaspberryPiDriver,
    STM32Driver,
    create_hardware_interface,
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
