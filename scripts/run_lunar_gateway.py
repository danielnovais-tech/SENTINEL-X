#!/usr/bin/env python3
"""
scripts/run_lunar_gateway.py – Full Lunar Gateway Experiment Cookbook
======================================================================

End-to-end example for the **Lunar Gateway (HALO node)** mission scenario.
Demonstrates:

1. Scenario construction – ``MissionScenario.lunar_gateway()``
2. Training with:
   - Power-constrained mission reward profile
   - Safety monitor (FDIR hard constraints)
   - LTL constrained-RL penalty shaping
3. Policy export – dynamic-range and int8 TFLite models
4. Policy verification – ``PolicyVerifier`` safety checks
5. Adversarial robustness certification – ``AdversarialTester``
6. Decision-tree surrogate – ``extract_decision_tree``,
   ``dt_fidelity_report``, ``export_decision_tree_rules``

Usage
-----
::

    python scripts/run_lunar_gateway.py

Output files written to the current directory:
- ``lunar_gateway_model.tflite``      – dynamic-range quantised model
- ``lunar_gateway_model_int8.tflite`` – full int8 model for embedded MCU
- ``lunar_gateway_policy_rules.txt``  – if-then-else rule export
"""

from __future__ import annotations

import os
import sys

# Keep TF/XLA log noise at warning level
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

# Allow running from repo root or scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sentinel_x_advanced as sx

# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────
EPISODES        = 80          # training episodes (increase for better convergence)
MAX_STEPS       = 150         # steps per episode
FEDERATED_INTERVAL = 10       # episodes between FedAvg aggregations
LTL_PENALTY     = -0.5        # reward deduction per violated LTL constraint
DT_N_SAMPLES    = 1500        # imitation-learning samples for decision tree
DT_MAX_DEPTH    = 6           # maximum decision-tree depth
ADV_N_EXAMPLES  = 200         # adversarial examples for robustness check
ADV_EPSILON     = 0.05        # FGSM perturbation magnitude
CERT_N_SAMPLES  = 100         # states for robustness certification
TFLITE_PATH     = "lunar_gateway_model.tflite"
TFLITE_INT8_PATH = "lunar_gateway_model_int8.tflite"
RULES_PATH      = "lunar_gateway_policy_rules.txt"

SEP = "=" * 65

# ───────────────────────────────────────────────────────────────────────────
# Step 1 – Scenario construction
# ───────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  Step 1 – Scenario Construction")
print(SEP)

scenario = sx.MissionScenario.lunar_gateway()
print(scenario.summary())

# ───────────────────────────────────────────────────────────────────────────
# Step 2 – Build swarm with safety monitor and LTL shaping
# ───────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  Step 2 – Build Swarm (SafetyMonitor + LTL shaping)")
print(SEP)

monitor = sx.SafetyMonitor()
ltl = sx.LTLConstraintChecker(penalty=LTL_PENALTY)

print(f"  Active LTL constraints : {ltl.active_names()}")
print(f"  LTL penalty per step   : {LTL_PENALTY}")

swarm = sx.build_swarm_for_scenario(
    scenario,
    federated_interval=FEDERATED_INTERVAL,
    safety_monitor=monitor,
    ltl_checker=ltl,
)

print(f"  Swarm size             : {len(swarm.spacecraft)}")
print(f"  Mission profile        : {swarm.mission_profile.profile}")

# ───────────────────────────────────────────────────────────────────────────
# Step 3 – Training loop
# ───────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  Step 3 – Training ({EPISODES} episodes × {MAX_STEPS} steps)")
print(SEP)

rewards: list[float] = []
for ep in range(EPISODES):
    avg_r = swarm.train_episode(max_steps=MAX_STEPS)
    rewards.append(avg_r)
    if ep == 0 or (ep + 1) % 20 == 0:
        mean_r = float(np.mean(rewards[-20:]))
        overrides = swarm.last_override_count
        ltl_pen = swarm.last_ltl_penalty
        print(
            f"  Episode {ep + 1:>3}/{EPISODES}  "
            f"avg_reward={mean_r:+.2f}  "
            f"veto_overrides={overrides}  "
            f"ltl_penalty={ltl_pen:.2f}"
        )

ops = swarm.test_episode(max_steps=200)
print(f"\n  Evaluation – avg operational steps: {ops:.1f} / 200")

# ───────────────────────────────────────────────────────────────────────────
# Step 4 – Export TFLite models
# ───────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  Step 4 – TFLite Export")
print(SEP)

agent0 = swarm.agents[0]

dyn_bytes = sx.export_tflite(agent0, output_path=TFLITE_PATH)
print(f"  Dynamic-range model : {TFLITE_PATH}  ({len(dyn_bytes):,} bytes)")

int8_bytes = sx.export_tflite_int8(agent0, output_path=TFLITE_INT8_PATH, n_calib_samples=32)
print(f"  Int8 model          : {TFLITE_INT8_PATH}  ({len(int8_bytes):,} bytes)")

# ───────────────────────────────────────────────────────────────────────────
# Step 5 – Policy verification
# ───────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  Step 5 – Policy Verification")
print(SEP)

verifier = sx.PolicyVerifier(agent0, n_samples=500, margin_threshold=0.05)
report = verifier.verify()

print(f"\n  Overall passed : {report['overall_passed']}")
sc_check = report["safety_constraint"]
print(
    f"  Safety constraint violations : "
    f"{sc_check['violations']}/{verifier.n_samples} "
    f"({sc_check['rate']:.1%})"
)

# ───────────────────────────────────────────────────────────────────────────
# Step 6 – Adversarial robustness certification
# ───────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  Step 6 – Adversarial Robustness")
print(SEP)

tester = sx.AdversarialTester(agent0, epsilon=ADV_EPSILON, rng_seed=42)

adv_results = tester.find_adversarial_examples(n_examples=ADV_N_EXAMPLES)
flip_rate = float(
    sum(1 for r in adv_results if r["flipped_action"] != r["original_action"])
) / max(len(adv_results), 1)
print(f"  FGSM flip rate @ ε={ADV_EPSILON}: {flip_rate * 100:.1f}%  "
      f"({len(adv_results)} states tested)")

cert = tester.certify_robustness(n_samples=CERT_N_SAMPLES, eps_hi=0.3, n_bisect=10)
print(
    f"  Certified mean radius  : {cert['mean_radius']:.4f}  "
    f"(robust_frac={cert['robust_frac']:.1%})"
)

# ───────────────────────────────────────────────────────────────────────────
# Step 7 – Decision-tree surrogate and rule export
# ───────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  Step 7 – Decision-Tree Surrogate")
print(SEP)

try:
    dt = sx.extract_decision_tree(
        agent0, n_samples=DT_N_SAMPLES, max_depth=DT_MAX_DEPTH
    )

    # Held-out fidelity comparison
    print()
    fid_report = sx.dt_fidelity_report(agent0, dt, n_eval=1000)
    print(f"\n  DT overall fidelity    : {fid_report['fidelity'] * 100:.1f}%")

    # Export if-then-else rules
    feature_names = [
        "mem_error_ratio",
        "parity_flag",
        "sensor_deviation_norm",
        "sensor_stuck_flag",
        "time_since_recovery_norm",
        "health_flag",
        "thermal_fault_flag",
        "power_level_norm",
        "attitude_rate_norm",
        "comm_quality_norm",
        "peer_sensor_deviation_norm",
    ]
    rules = sx.export_decision_tree_rules(
        dt, feature_names=feature_names
    )
    with open(RULES_PATH, "w", encoding="utf-8") as fh:
        fh.write(f"# Lunar Gateway – DQN surrogate policy (depth={dt.get_depth()})\n")
        fh.write(f"# Fidelity: {fid_report['fidelity'] * 100:.1f}%\n\n")
        fh.write(rules)
        fh.write("\n")
    print(f"\n  Rules written to       : {RULES_PATH}")
    print(f"  Rule file size         : {os.path.getsize(RULES_PATH):,} bytes")

    # Show first 20 lines of rules as preview
    preview_lines = rules.split("\n")[:20]
    print("\n  --- Rules preview (first 20 lines) ---")
    for line in preview_lines:
        print(f"  {line}")
    if len(rules.split("\n")) > 20:
        print("  ...")

except ImportError:
    print("  scikit-learn not installed; skipping decision-tree step.")

# ───────────────────────────────────────────────────────────────────────────
# Final summary
# ───────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  SENTINEL-X  –  Lunar Gateway Experiment Complete")
print(SEP)
print(f"  [✓] Scenario : {scenario.name}")
print(f"  [✓] Trained  : {EPISODES} episodes  |  avg ops = {ops:.1f}/200")
print(f"  [✓] TFLite (dynamic)   : {TFLITE_PATH}")
print(f"  [✓] TFLite (int8)      : {TFLITE_INT8_PATH}")
if os.path.exists(RULES_PATH):
    print(f"  [✓] DT rules file      : {RULES_PATH}")
print(f"  [✓] Policy verification: passed={report['overall_passed']}")
print(
    f"  [✓] Robustness cert    : mean_radius={cert['mean_radius']:.4f}  "
    f"robust@ε=0.3: {cert['robust_frac']:.1%}"
)
print(SEP)
