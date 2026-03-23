"""
sentinel_x/mission_control.py – Mission Control System Integration
==================================================================

Optional adapters that bridge the SENTINEL-X software pipeline with
industry-standard flight-software stacks:

* **NASA F´ (F Prime)** – Component-based flight software framework used on
  multiple JPL and NASA CubeSat missions.
  https://nasa.github.io/fprime/

* **ESA TASTE** – The ASSERT Set of Tools for Engineering flight software,
  based on ASN.1 / ACN data modelling and AADLv2 architecture.
  https://taste.tools/

Both adapters are **optional dependencies**.  If the underlying library is
not installed, the constructor raises :class:`ImportError` with clear
installation instructions.  The base class :class:`MissionControlBridge`
provides a platform-independent interface that works without either tool.

Architecture
------------
::

    SENTINEL-X FederatedSwarm
           │  telemetry events (dict)
           ▼
    MissionControlBridge.publish_telemetry(event)
           │
           ├── FPrimeAdapter ──► F´ GDS (fprime-gds)  [optional]
           ├── TASTEAdapter  ──► TASTE telemetry bus    [optional]
           └── (base class)  ──► JSON log file / stdout

Typical usage
-------------
::

    from sentinel_x.mission_control import FPrimeAdapter

    bridge = FPrimeAdapter(
        gds_host="127.0.0.1",
        gds_port=50050,
        component_name="SentinelX.FaultDetector",
    )
    bridge.connect()

    # During training loop:
    event = {
        "spacecraft_id": 0,
        "step": 42,
        "action": 2,
        "action_label": "SWITCH_REDUNDANT",
        "reward": 3.5,
        "health": True,
    }
    bridge.publish_telemetry(event)
    bridge.send_command("RESET_OVERRIDE_COUNT", {})

    bridge.disconnect()
"""

from __future__ import annotations

import abc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------
try:
    import fprime_gds                              # type: ignore[import]
    _HAS_FPRIME = True
except ImportError:
    _HAS_FPRIME = False

try:
    import asn1tools                               # type: ignore[import]
    _HAS_ASN1 = True
except ImportError:
    _HAS_ASN1 = False


# ---------------------------------------------------------------------------
# Base bridge
# ---------------------------------------------------------------------------

class MissionControlBridge(abc.ABC):
    """
    Abstract base class for mission-control adapters.

    Concrete implementations override :meth:`publish_telemetry` and
    :meth:`send_command`.  The base implementation logs to a JSON-lines
    file so that events are always persisted even when no GDS is connected.

    Parameters
    ----------
    log_path : str or None
        Path to a JSON-lines log file.  Pass ``None`` to disable local
        logging.  Default: ``"sentinel_x_telemetry.jsonl"``.
    """

    def __init__(self, log_path: Optional[str] = "sentinel_x_telemetry.jsonl") -> None:
        self._log_path   = log_path
        self._connected  = False
        self._tx_count   = 0
        self._rx_count   = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the connection to the mission-control system."""
        self._connected = True
        self._on_connect()

    def disconnect(self) -> None:
        """Close the connection gracefully."""
        self._on_disconnect()
        self._connected = False

    def __enter__(self) -> "MissionControlBridge":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def tx_count(self) -> int:
        """Number of telemetry frames published since connect."""
        return self._tx_count

    @property
    def rx_count(self) -> int:
        """Number of commands received since connect."""
        return self._rx_count

    # ── Core interface ────────────────────────────────────────────────────────

    def publish_telemetry(self, event: Dict[str, Any]) -> None:
        """
        Publish a telemetry event to the mission-control system.

        The base implementation writes to the JSON-lines log.  Override in
        subclasses to also forward to F´ GDS / TASTE bus.

        Parameters
        ----------
        event : dict
            Arbitrary key-value telemetry record.  Recommended keys:
            ``spacecraft_id``, ``step``, ``action``, ``reward``, ``health``.
        """
        frame = {"ts": time.time(), **event}
        if self._log_path:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(frame) + "\n")
        self._tx_count += 1
        self._publish_impl(frame)

    def send_command(self, command: str, args: Dict[str, Any]) -> None:
        """
        Send a command to the spacecraft / FDIR subsystem.

        The base implementation logs the command.  Override in subclasses to
        forward to the actual flight software command uplink.

        Parameters
        ----------
        command : str
            Command mnemonic (e.g. ``"RESET_OVERRIDE_COUNT"``).
        args : dict
            Command arguments.
        """
        record = {"ts": time.time(), "command": command, "args": args}
        if self._log_path:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        self._rx_count += 1
        self._command_impl(command, args)

    # ── Hooks for subclasses ──────────────────────────────────────────────────

    def _on_connect(self) -> None:
        """Called when connect() is invoked.  No-op by default."""

    def _on_disconnect(self) -> None:
        """Called when disconnect() is invoked.  No-op by default."""

    def _publish_impl(self, frame: Dict[str, Any]) -> None:
        """Transport-specific publish hook.  No-op in the base class."""

    def _command_impl(self, command: str, args: Dict[str, Any]) -> None:
        """Transport-specific command hook.  No-op in the base class."""

    # ── Convenience helpers ───────────────────────────────────────────────────

    def publish_episode_summary(self, episode: int, reward: float, overrides: int) -> None:
        """Publish a standardised episode-summary telemetry record."""
        self.publish_telemetry({
            "type":      "episode_summary",
            "episode":   episode,
            "reward":    round(float(reward), 4),
            "overrides": overrides,
        })


def available_adapters() -> Dict[str, bool]:
    """
    Return a dict indicating which mission-control adapters are available.

    Returns
    -------
    dict
        ``{"fprime": bool, "taste": bool}``
    """
    return {"fprime": _HAS_FPRIME, "taste": _HAS_ASN1}


# ---------------------------------------------------------------------------
# NASA F´ adapter
# ---------------------------------------------------------------------------

class FPrimeAdapter(MissionControlBridge):
    """
    Adapter for NASA F´ (F Prime) Ground Data System (GDS).

    Connects to the F´ GDS dictionary server via the ``fprime-gds`` Python
    package and forwards SENTINEL-X telemetry as F´ telemetry channels.
    Commands received from the GDS are dispatched back to the adapter's
    :meth:`send_command` callback.

    Parameters
    ----------
    gds_host : str
        Hostname of the F´ GDS server (default ``"127.0.0.1"``).
    gds_port : int
        Port of the F´ GDS server (default ``50050``).
    component_name : str
        F´ component path prefix used for channel naming
        (default ``"SentinelX.FaultDetector"``).
    log_path : str or None
        Path for the JSON-lines fallback log.

    Raises
    ------
    ImportError
        If ``fprime-gds`` is not installed.
    """

    def __init__(
        self,
        gds_host: str = "127.0.0.1",
        gds_port: int = 50050,
        component_name: str = "SentinelX.FaultDetector",
        log_path: Optional[str] = "sentinel_x_telemetry.jsonl",
    ) -> None:
        if not _HAS_FPRIME:
            raise ImportError(
                "fprime-gds is not installed.\n"
                "Install with: pip install fprime-gds\n"
                "See docs/mission_control_integration.md for full setup."
            )
        super().__init__(log_path=log_path)
        self._host      = gds_host
        self._port      = gds_port
        self._component = component_name
        self._client    = None

    def _on_connect(self) -> None:
        """Open the F´ GDS TCP connection."""
        try:
            from fprime_gds.common.client_socket.client_socket import (  # type: ignore[import]
                ClientSocket,
            )
            self._client = ClientSocket(self._host, self._port)
            self._client.connect()
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to F´ GDS at {self._host}:{self._port}: {exc}"
            ) from exc

    def _on_disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None

    def _publish_impl(self, frame: Dict[str, Any]) -> None:
        """
        Forward telemetry to F´ GDS as a channel update.

        Maps SENTINEL-X event keys to F´ telemetry channel mnemonics:

        ==============================  ==========================================
        SENTINEL-X key                  F´ channel
        ==============================  ==========================================
        ``spacecraft_id``               ``<component>.SpacecraftId``
        ``action``                      ``<component>.LastAction``
        ``reward``                      ``<component>.EpisodeReward``
        ``health``                      ``<component>.HealthStatus``
        ``overrides``                   ``<component>.SafetyOverrides``
        ==============================  ==========================================
        """
        if self._client is None:
            return
        channel_map = {
            "spacecraft_id": "SpacecraftId",
            "action":        "LastAction",
            "reward":        "EpisodeReward",
            "health":        "HealthStatus",
            "overrides":     "SafetyOverrides",
        }
        for sx_key, ch_suffix in channel_map.items():
            if sx_key in frame:
                channel = f"{self._component}.{ch_suffix}"
                try:
                    self._client.send_telemetry(channel, frame[sx_key])
                except Exception:
                    pass   # best-effort; local log already written

    def _command_impl(self, command: str, args: Dict[str, Any]) -> None:
        """Uplink a command to the F´ GDS command dictionary."""
        if self._client is None:
            return
        cmd_name = f"{self._component}.{command}"
        try:
            self._client.send_command(cmd_name, args)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ESA TASTE adapter
# ---------------------------------------------------------------------------

class TASTEAdapter(MissionControlBridge):
    """
    Adapter for ESA TASTE (The ASSERT Set of Tools for Engineering).

    TASTE uses ASN.1 / ACN data modelling.  This adapter encodes SENTINEL-X
    telemetry into ASN.1 DER-encoded records via the ``asn1tools`` library
    and exposes a simple file-based transport (compatible with TASTE's
    virtual function bus, HLT bus, or TCP gateway).

    Parameters
    ----------
    schema_path : str or None
        Path to an ASN.1 schema file defining the ``SentinelXTelemetry``
        sequence type.  If ``None``, a built-in minimal schema is used.
    output_dir : str
        Directory where encoded telemetry frames are written
        (default: ``"taste_telemetry/"``)
    log_path : str or None
        Path for the JSON-lines fallback log.

    Raises
    ------
    ImportError
        If ``asn1tools`` is not installed.
    """

    # Minimal built-in schema for SENTINEL-X telemetry
    _BUILTIN_SCHEMA = """\
SENTINEL-X-TELEMETRY DEFINITIONS AUTOMATIC TAGS ::= BEGIN

SentinelXTelemetry ::= SEQUENCE {
    timestamp    INTEGER (0..MAX),
    spacecraftId INTEGER (0..63),
    action       INTEGER (0..3),
    reward       INTEGER (-1000..1000),  -- scaled by 1000
    healthFlag   BOOLEAN,
    overrides    INTEGER (0..MAX)
}

END
"""

    def __init__(
        self,
        schema_path: Optional[str] = None,
        output_dir: str = "taste_telemetry",
        log_path: Optional[str] = "sentinel_x_telemetry.jsonl",
    ) -> None:
        if not _HAS_ASN1:
            raise ImportError(
                "asn1tools is not installed.\n"
                "Install with: pip install asn1tools\n"
                "See docs/mission_control_integration.md for TASTE setup."
            )
        super().__init__(log_path=log_path)
        self._output_dir = Path(output_dir)
        self._schema_path = schema_path
        self._codec = None
        self._frame_idx = 0

    def _on_connect(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if self._schema_path:
            self._codec = asn1tools.compile_files(  # type: ignore[name-defined]
                self._schema_path, codec="ber"
            )
        else:
            self._codec = asn1tools.compile_string(  # type: ignore[name-defined]
                self._BUILTIN_SCHEMA, codec="ber"
            )

    def _on_disconnect(self) -> None:
        self._codec = None

    def _publish_impl(self, frame: Dict[str, Any]) -> None:
        """
        Encode a telemetry frame to ASN.1 DER and write it to a file.

        The output directory acts as a virtual telemetry bus; a TASTE
        adapter process can watch this directory and forward frames to the
        actual bus (TCP, SpaceWire, …).
        """
        if self._codec is None:
            return
        record = {
            "timestamp":    int(frame.get("ts", time.time()) * 1000),
            "spacecraftId": int(frame.get("spacecraft_id", 0)),
            "action":       int(frame.get("action", 0)),
            "reward":       int(round(float(frame.get("reward", 0.0)) * 1000)),
            "healthFlag":   bool(frame.get("health", True)),
            "overrides":    int(frame.get("overrides", 0)),
        }
        try:
            encoded = self._codec.encode("SentinelXTelemetry", record)
            out_file = self._output_dir / f"frame_{self._frame_idx:06d}.ber"
            out_file.write_bytes(encoded)
            self._frame_idx += 1
        except Exception:
            pass   # best-effort; JSON log already written
