#!/usr/bin/env python3
"""
examples/lunar_gateway.py – Lunar Gateway Quick Demo (< 1 minute)
==================================================================

Minimal end-to-end example for the **Lunar Gateway (HALO node)** mission
scenario.  Designed to run in under a minute on a modern laptop (CPU only):

* Train a 3-spacecraft swarm for 15 episodes using the power-constrained
  reward profile with LTL safety shaping and SafetyMonitor integration.
* Evaluate operational reliability.
* Export an int8 TFLite model ready for embedded MCU deployment.
* Run formal policy verification.

Run::

    python examples/lunar_gateway.py
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sentinel_x_advanced as sx

# ── 1. Mission scenario ────────────────────────────────────────────────────
scenario = sx.MissionScenario.lunar_gateway()
print(scenario.summary())

# ── 2. Build swarm (SafetyMonitor + LTL shaping) ──────────────────────────
swarm = sx.build_swarm_for_scenario(
    scenario,
    safety_monitor=sx.SafetyMonitor(),
    ltl_checker=sx.LTLConstraintChecker(penalty=-0.5),
)
print(f"Swarm ready: {len(swarm.spacecraft)} spacecraft, "
      f"profile={swarm.mission_profile.profile}")

# ── 3. Quick training (15 episodes) ───────────────────────────────────────
print("\nTraining (15 episodes) …")
for ep in range(15):
    r = swarm.train_episode(max_steps=60)
    if (ep + 1) % 5 == 0:
        print(f"  Episode {ep + 1:>2}/15  reward={r:+.2f}  "
              f"veto={swarm.last_override_count}  "
              f"ltl_pen={swarm.last_ltl_penalty:.2f}")

# ── 4. Evaluate ────────────────────────────────────────────────────────────
ops = swarm.test_episode(max_steps=100)
print(f"\nEvaluation: {ops:.1f}/100 avg operational steps")

# ── 5. Export int8 TFLite ──────────────────────────────────────────────────
tflite_path = os.path.join(tempfile.gettempdir(), "lunar_gateway_int8.tflite")
payload = sx.export_tflite_int8(swarm.agents[0], output_path=tflite_path, n_calib_samples=16)
print(f"\nInt8 TFLite model: {tflite_path}  ({len(payload):,} bytes)")

# ── 6. Policy verification ─────────────────────────────────────────────────
verifier = sx.PolicyVerifier(swarm.agents[0], n_samples=200, margin_threshold=0.05)
report   = verifier.verify()
sc       = report["safety_constraint"]
print(f"\nVerification: passed={report['overall_passed']}  "
      f"safety_violations={sc['violations']}/200  ({sc['rate']:.1%})")

print("\n✓  Lunar Gateway quick demo complete.")
