#!/usr/bin/env python3
"""
examples/earth_observation.py – Earth Observation Constellation Demo
=====================================================================

End-to-end demonstration of the SENTINEL-X advanced coordination features:

* A six-spacecraft MEO Earth-observation constellation with dynamic task
  allocation (imaging + relay tasks auctioned each episode).
* Automatic swarm reconfiguration when fault levels rise above configured
  thresholds (formation reshape, leader promotion, degraded mode).
* Formation-keeping via :class:`~sentinel_x.formation.FormationController`
  using a circular geometry.

Run::

    python examples/earth_observation.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sentinel_x_advanced as sx
from sentinel_x.coordination import TaskAllocator, SwarmReconfigurationManager
from sentinel_x.formation import FormationController

# ── 1. Mission scenario ────────────────────────────────────────────────────
scenario = sx.MissionScenario.earth_observation_constellation()
print(scenario.summary())

# ── 2. Build swarm ────────────────────────────────────────────────────────
swarm = sx.build_swarm_for_scenario(
    scenario,
    safety_monitor=sx.SafetyMonitor(),
    ltl_checker=sx.LTLConstraintChecker(penalty=-0.3),
    cooperative_bonus=0.5,
)
print(f"\nSwarm ready: {len(swarm.spacecraft)} spacecraft  "
      f"profile={swarm.mission_profile.profile}")

# ── 3. Attach coordination layers ─────────────────────────────────────────
# Formation controller – circular geometry, 150 m separation
fc = FormationController(
    swarm,
    formation_type="circular",
    separation_m=150.0,
    formation_weight=0.08,
)

# Task allocator – spacecraft evenly distributed on a 400 m circle
import math
n = len(swarm.spacecraft)
positions = [
    (400.0 * math.cos(2 * math.pi * i / n),
     400.0 * math.sin(2 * math.pi * i / n))
    for i in range(n)
]
allocator = TaskAllocator(swarm, agent_positions=positions, exclusive=True)

# Reconfiguration manager
reconfig = SwarmReconfigurationManager(
    swarm,
    reshape_threshold=0.65,
    degraded_threshold=0.40,
    recovery_threshold=0.75,
    on_event=lambda ev: print(f"  ⚡ Reconfiguration: [{ev.action}] {ev.detail}"),
)

# ── 4. Define observation tasks ───────────────────────────────────────────
tasks = [
    {"id": "img_0", "type": "imaging",   "priority": 0.95,
     "position": (200.0,   0.0)},
    {"id": "img_1", "type": "imaging",   "priority": 0.85,
     "position": (-200.0,  0.0)},
    {"id": "img_2", "type": "imaging",   "priority": 0.80,
     "position": (0.0,    200.0)},
    {"id": "rel_0", "type": "relay",     "priority": 0.60,
     "position": (400.0,  400.0)},
    {"id": "sci_0", "type": "science",   "priority": 0.75,
     "position": (0.0,   -200.0)},
    {"id": "sci_1", "type": "science",   "priority": 0.70,
     "position": (-100.0, 100.0)},
]

# ── 5. Training loop ──────────────────────────────────────────────────────
EPISODES = 20
print(f"\nTraining ({EPISODES} episodes) with formation + task allocation …\n")

for ep in range(EPISODES):
    # Train one episode with formation shaping
    reward = fc.train_episode_with_formation(max_steps=80)

    # Allocate tasks for the next episode
    alloc = allocator.allocate(tasks)

    # Check for needed reconfigurations
    ev = reconfig.check_and_reconfigure()

    # Re-auction tasks won by now-failed spacecraft
    realloc = allocator.reallocate_failed(alloc, tasks)

    if (ep + 1) % 5 == 0:
        coherence = fc.formation_coherence()
        consensus = fc.get_consensus_estimate()
        print(
            f"  Episode {ep + 1:>2}/{EPISODES}  "
            f"reward={reward:+.2f}  "
            f"coherence={coherence:.3f}  "
            f"healthy={reconfig.healthy_fraction:.0%}  "
            f"assigned={len(alloc.assignments)}/{len(tasks)}  "
            f"reassigned={len(realloc.assignments)}"
        )

# ── 6. Results ────────────────────────────────────────────────────────────
print("\n── Formation metrics ─────────────────────────────────────────")
fm = fc.metrics.summary()
print(f"  Mean coherence : {fm['mean_coherence']:.3f}")
print(f"  Min  coherence : {fm['min_coherence']:.3f}")
print(f"  Mean error (m) : {fm['mean_error_m']:.1f}")

print("\n── Task allocation summary ───────────────────────────────────")
ta = allocator.summary()
print(f"  Rounds         : {ta['rounds']}")
print(f"  Assigned       : {ta['total_assigned']}")
print(f"  Unassigned     : {ta['total_unassigned']}")
print(f"  Assignment rate: {ta['assignment_rate']:.1%}")

print("\n── Reconfiguration log ───────────────────────────────────────")
rc = reconfig.summary()
print(f"  Total events   : {rc['total_events']}")
for action, count in rc['event_counts'].items():
    print(f"  {action:<22}: {count}")
print(f"  Current leader : spacecraft {rc['current_leader']}")
print(f"  Formation      : {rc['suggested_formation']}")
print(f"  Degraded mode  : {rc['is_degraded']}")

print("\n✓  Earth Observation Constellation demo complete.")
