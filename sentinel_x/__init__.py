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

Formal verification (external tools – optional)
    MarabouVerifier, ERANVerifier, export_to_onnx, available_verifiers
    (requires ``maraboupy``/``eran`` + ``onnx tf2onnx``; graceful fallback)

Formation-flying coordination
    ConsensusProtocol, FormationController, FormationGeometry, FormationMetrics

Mission control integration (optional)
    MissionControlBridge, FPrimeAdapter, TASTEAdapter, available_adapters
    (requires ``fprime-gds`` / ``asn1tools``; graceful fallback)

Deployment
    export_tflite, export_tflite_int8, extract_decision_tree
    export_decision_tree_rules, dt_fidelity_report
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
    export_decision_tree_rules,
    dt_fidelity_report,
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
    "export_decision_tree_rules",
    "dt_fidelity_report",
    "benchmark_federation",
    "run_do_nothing_baseline",
    # Configuration
    "load_config",
    "save_default_config",
    # Formal verification (external tools – optional)
    "MarabouVerifier",
    "ERANVerifier",
    "export_to_onnx",
    "available_verifiers",
    # Formation-flying coordination
    "ConsensusProtocol",
    "FormationController",
    "FormationGeometry",
    "FormationMetrics",
    # Mission control integration (optional)
    "MissionControlBridge",
    "FPrimeAdapter",
    "TASTEAdapter",
    "available_adapters",
]

# ---------------------------------------------------------------------------
# Optional: formal verification module (lazy import – never fails on import)
# ---------------------------------------------------------------------------
from sentinel_x.formal_verification import (  # noqa: F401, E402
    MarabouVerifier,
    ERANVerifier,
    export_to_onnx,
    available_verifiers,
)

# ---------------------------------------------------------------------------
# Formation-flying coordination (always available – no optional deps)
# ---------------------------------------------------------------------------
from sentinel_x.formation import (  # noqa: F401, E402
    ConsensusProtocol,
    FormationController,
    FormationGeometry,
    FormationMetrics,
)

# ---------------------------------------------------------------------------
# Mission control integration (lazy import – never fails on import)
# ---------------------------------------------------------------------------
from sentinel_x.mission_control import (  # noqa: F401, E402
    MissionControlBridge,
    FPrimeAdapter,
    TASTEAdapter,
    available_adapters,
)
