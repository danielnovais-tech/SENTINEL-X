# Advanced Swarm Coordination

`sentinel_x/coordination.py` provides two complementary coordination layers
that extend the core federated-RL pipeline with higher-level mission
management:

| Class | Purpose |
|---|---|
| `TaskAllocator` | Distributed greedy-auction task assignment |
| `SwarmReconfigurationManager` | Automated fault-triggered reconfiguration |

Both classes are designed to be **decoupled** from the RL training loop.
They can be attached to any `FederatedSwarm` instance and called each
episode.

---

## TaskAllocator

### Problem

In a multi-spacecraft mission, tasks such as imaging a target, relaying
data, or running a science instrument must be assigned to the spacecraft
best positioned and most able to execute them.  Centralised assignment is
brittle (single-point-of-failure); purely random assignment ignores
health and geometry.

### Algorithm — Greedy Sequential Auction

For each task *t* (sorted by descending priority):

```
bid(i, t) = health_score(i) × proximity_weight(t.position, pos_i) × priority(t)
```

where:

* `health_score(i)` ∈ [0, 1] combines memory, power, and thermal health.
* `proximity_weight` is a Gaussian kernel:
  `exp(−dist² / (2 σ²))`, σ = `proximity_scale` (default 500 m).

The spacecraft with the highest bid wins.  In `exclusive=True` mode (default)
each spacecraft wins at most one task per round.

### Usage

```python
from sentinel_x_advanced import FederatedSwarm, MissionProfile
from sentinel_x.coordination import TaskAllocator

swarm = FederatedSwarm(num_spacecraft=6, action_dim=4,
                       mission_profile=MissionProfile("maximize_data_return"))

# 2-D positions for each spacecraft (metres)
positions = [(i * 100.0, 0.0) for i in range(6)]
allocator = TaskAllocator(swarm, agent_positions=positions)

tasks = [
    {"id": "img_0", "type": "imaging", "priority": 0.9, "position": (300.0, 0.0)},
    {"id": "rel_0", "type": "relay",   "priority": 0.6, "position": (0.0, 400.0)},
]
result = allocator.allocate(tasks)
print(result.assignments)   # {"img_0": 3, "rel_0": 1}
```

### Re-auctioning failed assignments

If a winning spacecraft fails between allocation rounds, `reallocate_failed`
re-auctions only the orphaned tasks:

```python
result2 = allocator.reallocate_failed(result, tasks)
```

### API reference

| Method | Description |
|---|---|
| `allocate(tasks)` | Run one allocation round; return `AllocationResult` |
| `reallocate_failed(prev, tasks)` | Re-auction tasks whose winner has failed |
| `update_position(idx, pos)` | Update a spacecraft's 2-D position |
| `summary()` | Aggregate statistics over all rounds |

---

## SwarmReconfigurationManager

### Problem

When spacecraft fail during a mission the swarm must adapt: a new leader
must be elected, the formation may need to be simplified, and federation
frequency should be reduced to conserve bandwidth.

### Reconfiguration triggers

Three independent triggers are evaluated in `check_and_reconfigure()`
(called each episode):

| Priority | Condition | Action |
|---|---|---|
| 1 (highest) | healthy fraction < `degraded_threshold` (default 0.40) | Emit `degraded_mode`; increase `suggested_fed_interval` to 20 |
| 2 | Was degraded, healthy fraction ≥ `recovery_threshold` (0.70) | Emit `recovered`; restore interval to 10 |
| 3 | healthy fraction < `reshape_threshold` (0.60) | Emit `formation_reshaped`; set `suggested_formation = "line"` |
| 4 | Leader spacecraft (index `current_leader`) is not operational | Emit `leader_promoted`; update `current_leader` |

### Usage

```python
from sentinel_x.coordination import SwarmReconfigurationManager

reconfig = SwarmReconfigurationManager(
    swarm,
    reshape_threshold=0.65,
    degraded_threshold=0.40,
    on_event=lambda ev: print(f"[{ev.action}] {ev.detail}"),
)

# Each episode:
event = reconfig.check_and_reconfigure()
if event:
    if event.action == "formation_reshaped":
        fc.reshape(event.new_formation)     # apply to FormationController
    elif event.action == "degraded_mode":
        swarm.federated_interval = reconfig.suggested_fed_interval
```

### Manual override (ground control)

```python
event = reconfig.force_reconfiguration("diamond")
```

### API reference

| Property / Method | Description |
|---|---|
| `healthy_fraction` | Current fraction of operational spacecraft |
| `current_leader` | Index of logical leader spacecraft |
| `suggested_formation` | Current recommended formation type |
| `is_degraded` | True when in degraded mode |
| `check_and_reconfigure()` | Evaluate triggers; return event or None |
| `force_reconfiguration(formation)` | Manual formation override |
| `summary()` | Event-log statistics |

---

## Integration example – Earth Observation Constellation

`examples/earth_observation.py` demonstrates all three layers together:

```
FederatedSwarm (6 × MEO spacecraft)
    │
    ├── FormationController  (circular, 150 m separation)
    │       └── consensus on health indices
    │
    ├── TaskAllocator        (imaging + relay tasks auctioned each episode)
    │       └── reallocate_failed() after each episode
    │
    └── SwarmReconfigurationManager
            ├── leader_promoted  → new logical leader elected
            ├── formation_reshaped → switch from circular to line
            └── degraded_mode    → reduce federation interval
```

Run the demo:

```bash
python examples/earth_observation.py
```

---

## New mission scenarios

Two additional `MissionScenario` presets were added to `sentinel_x_advanced`:

### `MissionScenario.earth_observation_constellation()`

| Parameter | Value |
|---|---|
| Orbit | Medium-Earth (Walker-Delta) |
| Spacecraft | 6 |
| Profile | `maximize_data_return` |
| Use case | Paired with `TaskAllocator` for multi-target imaging |

### `MissionScenario.deep_space_telescope_array()`

| Parameter | Value |
|---|---|
| Orbit | Sun-Earth L2 |
| Spacecraft | 4 |
| Profile | `power_constrained` |
| Use case | Paired with `FormationController(formation_type="diamond")` |

---

## Design notes

* **No RL coupling** – both classes are pure Python with NumPy.  They
  inspect spacecraft health via `is_operational()` and `state_vector()` but
  never modify RL weights.
* **Thread-safe reads** – `check_and_reconfigure()` reads spacecraft state
  atomically within a single call; no locks required.
* **Extensibility** – subclass `TaskAllocator` to implement capacity
  constraints, temporal windows, or multi-objective auction variants.
