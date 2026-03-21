"""
sentinel_x.config – YAML/JSON configuration loader
====================================================

Provides :func:`load_config` and :func:`save_default_config` so that all
hyperparameters (agent, federation, fault model, mission profile, training)
can be driven from a single YAML file instead of hard-coded values.

Typical usage
-------------
::

    from sentinel_x.config import load_config
    cfg = load_config("sentinel_x_config.yaml")

    from sentinel_x import DQNAgent, FederatedSwarm, MissionProfile

    agent = DQNAgent(
        state_dim=cfg["agent"]["state_dim"],
        action_dim=cfg["agent"]["action_dim"],
        learning_rate=cfg["agent"]["learning_rate"],
        gamma=cfg["agent"]["gamma"],
        epsilon=cfg["agent"]["epsilon"],
        epsilon_min=cfg["agent"]["epsilon_min"],
        epsilon_decay=cfg["agent"]["epsilon_decay"],
        memory_size=cfg["agent"]["memory_size"],
        batch_size=cfg["agent"]["batch_size"],
    )

    swarm = FederatedSwarm(
        num_spacecraft=cfg["swarm"]["num_spacecraft"],
        action_dim=cfg["agent"]["action_dim"],
        mission_profile=MissionProfile(cfg["mission"]["profile"]),
        federated_interval=cfg["federation"]["federated_interval"],
        comm_delay_steps=cfg["federation"]["comm_delay_steps"],
        link_dropout_prob=cfg["federation"]["link_dropout_prob"],
        cooperative_bonus=cfg["swarm"].get("cooperative_bonus", 0.0),
    )

YAML format
-----------
See :func:`save_default_config` for the canonical schema, or the bundled
``sentinel_x_config.yaml`` file.

Dependencies
------------
PyYAML is only required when loading or saving YAML files.  JSON files
(``*.json``) are handled by the standard library without any extra dependency.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------
# Default configuration (single source of truth)
# -----------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    "agent": {
        "state_dim": 11,
        "action_dim": 4,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "memory_size": 2000,
        "batch_size": 32,
    },
    "swarm": {
        "num_spacecraft": 5,
        "cooperative_bonus": 0.0,
    },
    "federation": {
        "federated_interval": 10,
        "comm_delay_steps": 0,
        "link_dropout_prob": 0.0,
    },
    "mission": {
        # One of: "balanced", "data_return", "lifespan", "power_constrained"
        "profile": "balanced",
        "recovery_bonus_scale": 1.0,
        "fault_penalty_scale": 1.0,
    },
    "faults": {
        "flip_rate_per_bit": 1e-4,
        "sensor_stuck_prob": 0.01,
        "thermal_drift_std": 0.5,
        "thermal_spike_prob": 0.02,
        "power_drain_rate": 0.01,
    },
    "training": {
        "episodes": 200,
        "max_steps": 200,
        "target_update_interval": 10,
    },
    "curiosity": {
        "enabled": False,
        "bins": 5,
        "bonus_scale": 0.1,
        "clip": 1.0,
    },
    "ppo": {
        "lr": 3e-4,
        "gamma": 0.99,
        "clip_ratio": 0.2,
        "epochs": 4,
    },
    "hierarchical": {
        "n_phases": 3,
        "phase_duration": 10,
    },
    "deployment": {
        # Path written by export_tflite_int8()
        "tflite_int8_path": "sentinel_x_model_int8.tflite",
    },
}


# -----------------------------------------------------------------
# Public API
# -----------------------------------------------------------------

def save_default_config(path: str | os.PathLike = "sentinel_x_config.yaml") -> None:
    """Write the default configuration to *path*.

    Supports ``.yaml`` / ``.yml`` and ``.json`` extensions.  The written
    file serves as a template that users can edit to override any setting.

    Parameters
    ----------
    path : str or PathLike
        Destination file path.  The parent directory must already exist.
    """
    path = Path(path)
    cfg = copy.deepcopy(_DEFAULT_CONFIG)
    if path.suffix.lower() in (".yaml", ".yml"):
        _write_yaml(cfg, path)
    elif path.suffix.lower() == ".json":
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported config extension '{path.suffix}'. "
            "Use '.yaml', '.yml', or '.json'."
        )


def load_config(
    path: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Load configuration from a YAML or JSON file, merged with defaults.

    Parameters
    ----------
    path : str, PathLike, or None
        Path to a YAML (``.yaml`` / ``.yml``) or JSON (``.json``) file.
        When *None* the default configuration is returned unchanged.

    Returns
    -------
    dict
        A fully-populated configuration dictionary.  Missing keys are filled
        from the built-in defaults so partial override files are allowed.
    """
    cfg = copy.deepcopy(_DEFAULT_CONFIG)
    if path is None:
        return cfg

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() in (".yaml", ".yml"):
        user = _read_yaml(path)
    elif path.suffix.lower() == ".json":
        user = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(
            f"Unsupported config extension '{path.suffix}'. "
            "Use '.yaml', '.yml', or '.json'."
        )

    _deep_merge(cfg, user)
    return cfg


def get_default_config() -> dict[str, Any]:
    """Return a deep copy of the built-in default configuration."""
    return copy.deepcopy(_DEFAULT_CONFIG)


# -----------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _write_yaml(data: dict, path: Path) -> None:
    """Write *data* as YAML to *path*."""
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to write YAML config files. "
            "Install it with: pip install pyyaml"
        ) from exc
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False),
                    encoding="utf-8")


def _read_yaml(path: Path) -> dict:
    """Read and parse a YAML file, returning a dict."""
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read YAML config files. "
            "Install it with: pip install pyyaml"
        ) from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
