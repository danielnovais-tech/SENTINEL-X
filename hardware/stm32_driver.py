"""
hardware/stm32_driver.py – STM32 / Generic Serial MCU Driver
============================================================

Concrete :class:`~hardware.hal.HardwareInterface` implementation for any
MCU connected via a USB-to-UART adapter (FTDI, CP2102, CH340, …) or a
native UART port.

Tested with:

* STM32H743 Nucleo board (via ST-Link V3 VCP)
* STM32L432 Nucleo-32 (via onboard USB CDC)
* ESP32-S3 DevKit (via CP2102N)

Protocol
--------
Same 22-byte framing as the Raspberry Pi driver (see
:func:`hardware.rpi_driver._parse_uart_frame`).  The MCU firmware must
implement the same frame format.  A reference FreeRTOS task is provided in
``hardware/freertos_task.c``.

Usage
-----
::

    from hardware import create_hardware_interface

    hw = create_hardware_interface("stm32")   # auto-detects first USB serial
    # or:
    from hardware.stm32_driver import STM32Driver
    hw = STM32Driver(port="COM3")             # Windows
    hw = STM32Driver(port="/dev/ttyUSB0")     # Linux

    with hw:
        reading = hw.read_sensors()
        print(reading.to_state_vector())
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

from hardware.hal import ActuatorCommand, HardwareInterface, SensorReading
from hardware.rpi_driver import SimulatedDriver, _parse_uart_frame

try:
    import serial                          # type: ignore[import]
    import serial.tools.list_ports         # type: ignore[import]
    _HAS_SERIAL = True
except ImportError:
    _HAS_SERIAL = False


def _auto_detect_port() -> Optional[str]:
    """
    Return the first serial port that looks like an STM32 virtual COM port.

    Heuristic: matches CP210x, FTDI, CH340, or ST-Link VCP vendor strings.
    """
    if not _HAS_SERIAL:
        return None
    known = {"CP210", "FTDI", "CH340", "ST-Link", "STM32", "USB Serial"}
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "") + (p.manufacturer or "")
        if any(k.lower() in desc.lower() for k in known):
            return p.device
    # Fall back to first available port
    ports = list(serial.tools.list_ports.comports())
    return ports[0].device if ports else None


class STM32Driver(HardwareInterface):
    """
    SENTINEL-X hardware driver for STM32 and other serial-connected MCUs.

    Parameters
    ----------
    port : str or None
        Serial port device (e.g. ``/dev/ttyUSB0``, ``COM3``).
        Pass ``None`` to auto-detect the first USB serial port.
    baud_rate : int
        UART baud rate (default 115 200; must match MCU firmware).
    read_timeout_s : float
        Serial read timeout in seconds (default 0.02 = 20 ms).

    Raises
    ------
    ImportError
        If ``pyserial`` is not installed.
    RuntimeError
        If *port* is ``None`` and auto-detection fails.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baud_rate: int = 115_200,
        read_timeout_s: float = 0.02,
    ) -> None:
        if not _HAS_SERIAL:
            raise ImportError(
                "pyserial is not installed. Run: pip install pyserial"
            )
        if port is None:
            port = _auto_detect_port()
        if port is None:
            raise RuntimeError(
                "No USB serial port detected. Connect your MCU and retry, "
                "or specify the port explicitly: STM32Driver(port='/dev/ttyUSB0')."
            )

        self._port_name      = port
        self._baud_rate      = baud_rate
        self._read_timeout   = read_timeout_s
        self._serial: Optional["serial.Serial"] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        import serial as _serial
        self._serial = _serial.Serial(
            port=self._port_name,
            baudrate=self._baud_rate,
            bytesize=_serial.EIGHTBITS,
            parity=_serial.PARITY_NONE,
            stopbits=_serial.STOPBITS_ONE,
            timeout=self._read_timeout,
        )

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()

    # ── Core interface ────────────────────────────────────────────────────────

    def read_sensors(self) -> SensorReading:
        """Request and parse a 22-byte sensor frame from the MCU."""
        if self._serial is None or not self._serial.is_open:
            raise RuntimeError("Serial port not open. Call open() first.")

        self._serial.write(b"\x52")    # request frame ('R')
        data = self._serial.read(22)

        if len(data) < 22:
            warnings.warn(
                f"STM32Driver: short frame ({len(data)}/22 bytes). "
                "Check MCU firmware and baud rate.",
                RuntimeWarning,
                stacklevel=2,
            )
            return SensorReading(timestamp_s=time.time(), health_flag=True)

        return _parse_uart_frame(data)

    def write_action(self, cmd: ActuatorCommand) -> None:
        """Transmit a 2-byte action packet to the MCU."""
        if self._serial is None or not self._serial.is_open:
            raise RuntimeError("Serial port not open. Call open() first.")
        self._serial.write(cmd.to_bytes())

    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    @property
    def port(self) -> str:
        """The serial port device name."""
        return self._port_name
