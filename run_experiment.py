#!/usr/bin/env python3
"""
run_experiment.py – Config-Driven SENTINEL-X Experiment Runner
===============================================================

Orchestrates a full SENTINEL-X experiment from a single YAML or JSON
configuration file:

    1. **Build** – create FederatedSwarm from config (with optional LTL
       shaping, SafetyMonitor, cooperative bonus)
    2. **Train** – run *N* episodes and log per-episode metrics
    3. **Evaluate** – test episode returning avg operational steps
    4. **Export** – write dynamic-range and int8 TFLite models
    5. **Verify** – run PolicyVerifier safety checks
    6. **Certify** – adversarial robustness certification via FGSM
    7. **Surrogate** – extract decision-tree surrogate, print fidelity
       report, and export if-then-else rules

Usage
-----
::

    # Run with built-in defaults (no config file needed)
    python run_experiment.py

    # Run with a custom config file
    python run_experiment.py --config sentinel_x_config.yaml

    # Use a scenario preset and override episodes
    python run_experiment.py --scenario lunar_gateway --episodes 50

    # Disable steps you don't need
    python run_experiment.py --no-verify --no-certify --no-surrogate

    # Save a template config and exit
    python run_experiment.py --save-config my_experiment.yaml

Output
------
Results are printed to stdout.  Exported artefacts are written to the
current working directory (override with ``deployment.tflite_int8_path``
in the config file).

Config keys honoured
--------------------
All top-level sections from ``sentinel_x/config.py`` are used:

* ``agent``     – DQN hyperparameters (used to configure each spacecraft's agent)
* ``swarm``     – swarm size, cooperative bonus
* ``federation``– interval, comm delay, link dropout
* ``mission``   – profile name, recovery/fault scales
* ``faults``    – flip rate, sensor stuck, thermal, power drain
* ``training``  – episodes, max_steps
* ``ltl``       – enabled flag and penalty magnitude
* ``deployment``– output path for the int8 TFLite model

Scenario presets
----------------
Pass ``--scenario {lunar_gateway|mars_orbiter|cubesat_swarm}`` to load a
pre-built ``MissionScenario``.  Scenario parameters take precedence over
the ``swarm``, ``federation``, ``faults``, and ``mission`` config sections.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import sentinel_x_advanced as sx
from sentinel_x.config import load_config, save_default_config

# ---------------------------------------------------------------------------
# Scenario presets lookup
# ---------------------------------------------------------------------------
_SCENARIO_PRESETS = {
    "lunar_gateway": sx.MissionScenario.lunar_gateway,
    "mars_orbiter":  sx.MissionScenario.mars_orbiter,
    "cubesat_swarm": sx.MissionScenario.cubesat_swarm,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title: str = "", width: int = 65) -> None:
    bar = "=" * width
    print(f"\n{bar}")
    if title:
        print(f"  {title}")
        print(bar)


def _build_swarm_from_config(cfg: dict, scenario: sx.MissionScenario | None) -> sx.FederatedSwarm:
    """Construct a FederatedSwarm from *cfg* (and optional *scenario* preset)."""
    ltl_checker = None
    if cfg["ltl"].get("enabled", False):
        ltl_checker = sx.LTLConstraintChecker(penalty=cfg["ltl"]["penalty"])
        print(f"  LTL shaping ENABLED  penalty={cfg['ltl']['penalty']}  "
              f"constraints={ltl_checker.active_names()}")
    else:
        print("  LTL shaping DISABLED  (set ltl.enabled=true to activate)")

    safety_monitor = sx.SafetyMonitor()

    if scenario is not None:
        print(scenario.summary())
        swarm = sx.build_swarm_for_scenario(
            scenario,
            federated_interval=cfg["federation"]["federated_interval"],
            safety_monitor=safety_monitor,
            cooperative_bonus=cfg["swarm"].get("cooperative_bonus", 0.0),
            ltl_checker=ltl_checker,
        )
    else:
        swarm = sx.FederatedSwarm(
            num_spacecraft=cfg["swarm"]["num_spacecraft"],
            action_dim=cfg["agent"]["action_dim"],
            mission_profile=sx.MissionProfile(cfg["mission"]["profile"]),
            federated_interval=cfg["federation"]["federated_interval"],
            comm_delay_steps=cfg["federation"]["comm_delay_steps"],
            link_dropout_prob=cfg["federation"]["link_dropout_prob"],
            safety_monitor=safety_monitor,
            cooperative_bonus=cfg["swarm"].get("cooperative_bonus", 0.0),
            ltl_checker=ltl_checker,
        )
        # Apply fault parameters from config to all spacecraft using the
        # private attributes so the values persist across reset() calls,
        # matching the pattern used in build_swarm_for_scenario.
        for sc in swarm.spacecraft:
            sc._flip_rate_per_bit  = cfg["faults"]["flip_rate_per_bit"]
            sc._sensor_stuck_prob  = cfg["faults"]["sensor_stuck_prob"]
            sc._thermal_drift_std  = cfg["faults"]["thermal_drift_std"]
            sc._thermal_spike_prob = cfg["faults"]["thermal_spike_prob"]
            sc._power_drain_rate   = cfg["faults"]["power_drain_rate"]

    return swarm


# ---------------------------------------------------------------------------
# Experiment steps
# ---------------------------------------------------------------------------

def step_train(swarm: sx.FederatedSwarm, cfg: dict) -> list[float]:
    """Train for *cfg.training.episodes* episodes; return per-episode rewards."""
    episodes   = cfg["training"]["episodes"]
    max_steps  = cfg["training"]["max_steps"]
    rewards: list[float] = []
    t0 = time.time()

    for ep in range(episodes):
        r = swarm.train_episode(max_steps=max_steps)
        rewards.append(r)
        if ep == 0 or (ep + 1) % max(1, episodes // 5) == 0:
            elapsed = time.time() - t0
            mean_r  = float(np.mean(rewards[-max(1, episodes // 5):]))
            print(
                f"  Ep {ep + 1:>4}/{episodes}  "
                f"avg_reward={mean_r:+.2f}  "
                f"veto={swarm.last_override_count}  "
                f"ltl_pen={swarm.last_ltl_penalty:.2f}  "
                f"({elapsed:.0f}s)"
            )

    return rewards


def step_evaluate(swarm: sx.FederatedSwarm, cfg: dict) -> float:
    """Run one evaluation episode; return avg operational steps."""
    ops = swarm.test_episode(max_steps=cfg["training"]["max_steps"])
    print(f"  Avg operational steps : {ops:.1f} / {cfg['training']['max_steps']}")
    return ops


def step_export(swarm: sx.FederatedSwarm, cfg: dict) -> dict[str, str]:
    """Export TFLite models for every agent; return {agent_id: path}."""
    agent   = swarm.agents[0]
    base    = cfg["deployment"].get("tflite_int8_path", "sentinel_x_model_int8.tflite")
    stem    = base.replace("_int8.tflite", "")

    dyn_path  = f"{stem}.tflite"
    int8_path = f"{stem}_int8.tflite"

    dyn_bytes  = sx.export_tflite(agent, output_path=dyn_path)
    int8_bytes = sx.export_tflite_int8(agent, output_path=int8_path, n_calib_samples=32)

    print(f"  Dynamic-range model : {dyn_path}  ({len(dyn_bytes):,} bytes)")
    print(f"  Int8 model          : {int8_path}  ({len(int8_bytes):,} bytes)")
    return {"dynamic": dyn_path, "int8": int8_path}


def step_verify(swarm: sx.FederatedSwarm) -> dict:
    """Run PolicyVerifier on the primary agent; return report dict."""
    agent    = swarm.agents[0]
    verifier = sx.PolicyVerifier(agent, n_samples=500, margin_threshold=0.05)
    report   = verifier.verify()

    sc       = report["safety_constraint"]
    margin   = report["qvalue_margin"]
    coverage = report["action_coverage"]

    print(f"  Overall passed       : {report['overall_passed']}")
    print(f"  Safety violations    : {sc['violations']}/{verifier.n_samples}  "
          f"({sc['rate']:.1%})")
    print(f"  Q-value margin       : mean={margin['mean_margin']:.3f}  "
          f"min={margin['min_margin']:.3f}  "
          f"passed={margin['passed']}")
    covered   = [str(a) for a in coverage.get("covered_actions", [])]
    uncovered = [str(a) for a in coverage.get("missing_actions", [])]
    print(f"  Action coverage      : covered={covered}  missing={uncovered}")
    return report


def step_certify(swarm: sx.FederatedSwarm, n_adv: int = 200) -> dict:
    """Run adversarial robustness certification; return cert dict."""
    agent  = swarm.agents[0]
    tester = sx.AdversarialTester(agent, epsilon=0.05, rng_seed=42)

    adv    = tester.find_adversarial_examples(n_examples=n_adv)
    flipped = sum(1 for r in adv if r["flipped_action"] != r["original_action"])
    flip_rate = flipped / max(len(adv), 1)
    print(f"  FGSM flip rate @ε=0.05 : {flip_rate * 100:.1f}%  "
          f"({len(adv)} states)")

    cert = tester.certify_robustness(n_samples=100, eps_hi=0.3, n_bisect=10)
    print(f"  Certified mean radius  : {cert['mean_radius']:.4f}  "
          f"robust_frac={cert['robust_frac']:.1%}")
    return cert


def step_surrogate(swarm: sx.FederatedSwarm, rules_path: str = "") -> dict:
    """Extract DT surrogate, print fidelity report, export rules."""
    try:
        agent = swarm.agents[0]
        dt    = sx.extract_decision_tree(agent, n_samples=1500, max_depth=6)

        report = sx.dt_fidelity_report(agent, dt, n_eval=1000)
        print(f"\n  DT overall fidelity  : {report['fidelity'] * 100:.1f}%")

        if not rules_path:
            rules_path = "sentinel_x_policy_rules.txt"
        rules = sx.export_decision_tree_rules(dt)
        with open(rules_path, "w", encoding="utf-8") as fh:
            fh.write(f"# SENTINEL-X DQN surrogate (depth={dt.get_depth()})\n")
            fh.write(f"# Fidelity: {report['fidelity'] * 100:.1f}%\n\n")
            fh.write(rules)
            fh.write("\n")
        print(f"  Rules written to     : {rules_path}")
        return report
    except ImportError:
        print("  scikit-learn not installed; skipping surrogate step.")
        return {}


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """Execute the full experiment pipeline."""

    # ── 0. Save template and exit? ─────────────────────────────────────────
    if args.save_config:
        save_default_config(args.save_config)
        print(f"Default config written to: {args.save_config}")
        return

    # ── 1. Load config ──────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.episodes:
        cfg["training"]["episodes"] = args.episodes
    if args.max_steps:
        cfg["training"]["max_steps"] = args.max_steps

    _sep("SENTINEL-X Experiment Runner")
    print(f"  Config file    : {args.config or '(built-in defaults)'}")
    print(f"  Scenario preset: {args.scenario or '(none – using config)'}")
    print(f"  Episodes       : {cfg['training']['episodes']}")
    print(f"  Max steps/ep   : {cfg['training']['max_steps']}")

    # ── 2. Resolve scenario preset ──────────────────────────────────────────
    scenario: sx.MissionScenario | None = None
    if args.scenario:
        if args.scenario not in _SCENARIO_PRESETS:
            print(f"ERROR: unknown scenario '{args.scenario}'. "
                  f"Choose from: {list(_SCENARIO_PRESETS)}")
            sys.exit(1)
        scenario = _SCENARIO_PRESETS[args.scenario]()

    # ── 3. Build swarm ──────────────────────────────────────────────────────
    _sep("Step 1 – Build Swarm")
    swarm = _build_swarm_from_config(cfg, scenario)
    print(f"  Spacecraft     : {len(swarm.spacecraft)}")
    print(f"  Mission profile: {swarm.mission_profile.profile}")

    # ── 4. Train ─────────────────────────────────────────────────────────────
    _sep("Step 2 – Training")
    t_train_start = time.time()
    rewards = step_train(swarm, cfg)
    train_time = time.time() - t_train_start
    print(f"\n  Training complete in {train_time:.1f}s  "
          f"| mean_reward={float(np.mean(rewards)):.2f}  "
          f"| final_20ep={float(np.mean(rewards[-20:])):.2f}")

    # ── 5. Evaluate ──────────────────────────────────────────────────────────
    _sep("Step 3 – Evaluation")
    ops = step_evaluate(swarm, cfg)

    # ── 6. Export TFLite ─────────────────────────────────────────────────────
    if not args.no_export:
        _sep("Step 4 – TFLite Export")
        step_export(swarm, cfg)

    # ── 7. Verify ────────────────────────────────────────────────────────────
    if not args.no_verify:
        _sep("Step 5 – Policy Verification")
        step_verify(swarm)

    # ── 8. Certify robustness ────────────────────────────────────────────────
    if not args.no_certify:
        _sep("Step 6 – Adversarial Robustness Certification")
        step_certify(swarm)

    # ── 9. Decision-tree surrogate ───────────────────────────────────────────
    if not args.no_surrogate:
        _sep("Step 7 – Decision-Tree Surrogate")
        step_surrogate(swarm)

    # ── 10. Summary ──────────────────────────────────────────────────────────
    _sep("Experiment Complete")
    print(f"  Episodes       : {cfg['training']['episodes']}")
    print(f"  Training time  : {train_time:.1f}s")
    print(f"  Avg ops/ep     : {ops:.1f} / {cfg['training']['max_steps']}")
    print("=" * 65)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SENTINEL-X config-driven experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", metavar="FILE", default=None,
        help="Path to YAML or JSON config file (default: built-in defaults)",
    )
    parser.add_argument(
        "--scenario",
        choices=list(_SCENARIO_PRESETS),
        metavar="PRESET",
        default=None,
        help="Mission scenario preset: lunar_gateway | mars_orbiter | cubesat_swarm",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override training episodes from config",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, dest="max_steps",
        help="Override max steps per episode from config",
    )
    parser.add_argument(
        "--save-config", metavar="FILE", default=None, dest="save_config",
        help="Write default config to FILE and exit",
    )
    parser.add_argument("--no-export",    action="store_true", help="Skip TFLite export")
    parser.add_argument("--no-verify",    action="store_true", help="Skip PolicyVerifier")
    parser.add_argument("--no-certify",   action="store_true", help="Skip robustness cert")
    parser.add_argument("--no-surrogate", action="store_true", help="Skip DT surrogate")
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
