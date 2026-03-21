"""
sentinel_x – SENTINEL-X Python package
=======================================

Provides a clean, importable interface to the full SENTINEL-X feature set.
All symbols are re-exported from the canonical ``sentinel_x_advanced`` module
so that both

    import sentinel_x_advanced as sx           # legacy / direct
    from sentinel_x import DQNAgent, PPOAgent  # package import

work identically.  Callers should prefer the ``sentinel_x`` package form for
new code.

Public API
----------

Fault generators
    MemoryArray, Sensor, ThermalSubsystem, PowerSubsystem,
    AttitudeControlSubsystem, CommSubsystem

Spacecraft & swarms
    Spacecraft, Swarm, FederatedSwarm, FederatedServer, GossipServer

Agents
    DQNAgent, PPOAgent, HierarchicalAgent

Mission configuration
    MissionProfile, MissionScenario, build_swarm_for_scenario
    load_config, save_default_config

Training utilities
    CuriosityBonus, LTLConstraintChecker

Safety & verification
    SafetyMonitor, PolicyVerifier, AdversarialTester

Deployment
    export_tflite, export_tflite_int8, extract_decision_tree
    benchmark_federation, run_do_nothing_baseline
"""

# Re-export the full public API from the canonical module so that
# ``from sentinel_x import X`` works for every public symbol.
from sentinel_x_advanced import (  # noqa: F401
    # Fault generators
    MemoryArray,
    Sensor,
    ThermalSubsystem,
    PowerSubsystem,
    AttitudeControlSubsystem,
    CommSubsystem,
    # Spacecraft & swarms
    Spacecraft,
    Swarm,
    FederatedSwarm,
    FederatedServer,
    GossipServer,
    # Agents
    DQNAgent,
    PPOAgent,
    HierarchicalAgent,
    # Mission configuration
    MissionProfile,
    MissionScenario,
    build_swarm_for_scenario,
    # Training utilities
    CuriosityBonus,
    LTLConstraintChecker,
    # Safety & verification
    SafetyMonitor,
    PolicyVerifier,
    AdversarialTester,
    # Deployment
    export_tflite,
    export_tflite_int8,
    extract_decision_tree,
    benchmark_federation,
    run_do_nothing_baseline,
)

# Configuration helpers live in the config sub-module but are surfaced at the
# top level for convenience.
from sentinel_x.config import load_config, save_default_config  # noqa: F401

__all__ = [
    # Fault generators
    "MemoryArray",
    "Sensor",
    "ThermalSubsystem",
    "PowerSubsystem",
    "AttitudeControlSubsystem",
    "CommSubsystem",
    # Spacecraft & swarms
    "Spacecraft",
    "Swarm",
    "FederatedSwarm",
    "FederatedServer",
    "GossipServer",
    # Agents
    "DQNAgent",
    "PPOAgent",
    "HierarchicalAgent",
    # Mission configuration
    "MissionProfile",
    "MissionScenario",
    "build_swarm_for_scenario",
    # Training utilities
    "CuriosityBonus",
    "LTLConstraintChecker",
    # Safety & verification
    "SafetyMonitor",
    "PolicyVerifier",
    "AdversarialTester",
    # Deployment
    "export_tflite",
    "export_tflite_int8",
    "extract_decision_tree",
    "benchmark_federation",
    "run_do_nothing_baseline",
    # Configuration
    "load_config",
    "save_default_config",
]
