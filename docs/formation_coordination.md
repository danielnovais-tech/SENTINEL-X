# Formation-Flying Coordination

SENTINEL-X supports explicit geometric formation-keeping and distributed
consensus protocols through the `sentinel_x.formation` module.

This is a research-extension layer on top of the federated RL pipeline.
While the RL agents handle fault detection and recovery autonomously, the
formation controller maintains relative spacecraft geometry and drives
swarm-level consensus on health estimates.

---

## Classes

| Class | Description |
|-------|-------------|
| `ConsensusProtocol` | Distributed averaging consensus (DeGroot) |
| `FormationController` | Leader-follower formation + reward shaping |
| `FormationGeometry` | Standard geometry builders (line, V, diamond, circular) |
| `FormationMetrics` | Per-step statistics recorder |

---

## Consensus Protocol

```python
from sentinel_x.formation import ConsensusProtocol

# 4-agent ring topology
cp = ConsensusProtocol(n_agents=4, mixing_weight=0.5, topology="ring")

# Initial health estimates
cp.set_values([1.0, 0.0, 1.0, 0.5])   # agent 1 has a fault

# Run 10 mixing steps
final = cp.run(n_steps=10)
print(final)   # [0.625, 0.625, 0.625, 0.625]  ← converged to mean

print(f"Converged: {cp.converged}")              # True
print(f"Converged at step: {cp.convergence_step}")
```

### Topology options

| Topology | Description | Convergence |
|----------|-------------|-------------|
| `"ring"` | Each agent talks to two neighbours | Moderate |
| `"complete"` | All-to-all communication | Fastest |
| `"star"` | Hub-and-spoke (agent 0 = hub) | Fast but hub-dependent |

---

## Formation Geometries

```python
from sentinel_x.formation import FormationGeometry

# 3 followers in a V-shape behind the leader, 200 m separation
geom = FormationGeometry.build(
    n_followers=3,
    formation_type="v",
    separation_m=200.0,
)
print(geom.offsets)
# [(-200.0, 120.0), (-200.0, -120.0), (-400.0, 240.0)]
```

| Formation | Description | Typical use |
|-----------|-------------|-------------|
| `"line"` | Column behind leader | Earth observation pass |
| `"v"` | Chevron / arrowhead | Attitude sensors |
| `"diamond"` | Rhombus around leader | All-sky survey |
| `"circular"` | Equal spacing on a circle | Interferometry |

---

## Formation Controller

```python
from sentinel_x_advanced import FederatedSwarm, MissionProfile, SafetyMonitor
from sentinel_x.formation import FormationController

swarm = FederatedSwarm(
    num_spacecraft=4,
    action_dim=4,
    mission_profile=MissionProfile(MissionProfile.BALANCED),
    safety_monitor=SafetyMonitor(),
)

fc = FormationController(
    swarm,
    formation_type="v",
    separation_m=100.0,
    formation_weight=0.1,     # formation penalty weight
    consensus_topology="ring",
    consensus_mixing=0.5,
)

# Training loop with formation reward shaping
for episode in range(50):
    reward = fc.train_episode_with_formation(max_steps=200)
    metrics = fc.metrics.summary()
    print(f"Episode {episode}: reward={reward:.2f}  "
          f"coherence={metrics['mean_coherence']:.2%}")

# Formation status
consensus = fc.get_consensus_estimate()
print(f"Consensus health mean: {consensus['mean']:.2f}")
print(f"Formation coherence:   {fc.formation_coherence():.2%}")
```

---

## Formation metrics

```python
summary = fc.metrics.summary()
print(summary)
# {
#   "mean_coherence": 0.87,
#   "min_coherence":  0.61,
#   "mean_error_m":   12.4,
#   "max_error_m":    45.1,
#   "n_steps":        50,
# }

# Full step-by-step log
records = fc.metrics.to_list()
# [{"step": 0, "coherence": ..., "mean_error_m": ..., "consensus": [...]}, ...]
```

---

## Formation error reward shaping

The total reward at each episode is:

```
total_reward = base_RL_reward − formation_weight × mean_formation_error_m
```

The `formation_weight` parameter controls the trade-off between fault
recovery (maximise `base_RL_reward`) and formation-keeping (minimise
`mean_formation_error_m`).  Start with `0.05`–`0.1` for sparse fault
scenarios; increase to `0.5` if formation is the primary mission objective.

---

## Convergence analysis

```python
import matplotlib.pyplot as plt
import numpy as np

cp = ConsensusProtocol(n_agents=6, mixing_weight=0.4, topology="ring")
cp.set_values([1.0, 0.0, 0.5, 0.8, 0.2, 0.6])
cp.run(n_steps=30)

history = np.array(cp.history)    # shape (31, 6)
for i in range(6):
    plt.plot(history[:, i], label=f"Agent {i}")
plt.axhline(y=np.mean(cp.values), linestyle="--", color="k", label="Mean")
plt.xlabel("Mixing step")
plt.ylabel("Value")
plt.legend()
plt.title("Consensus convergence (ring topology)")
plt.savefig("consensus_convergence.png", dpi=100)
```

---

## Integration with MissionControlBridge

```python
from sentinel_x.mission_control import MissionControlBridge

class FormationAwareBridge(MissionControlBridge):
    def __init__(self, fc: FormationController):
        super().__init__(log_path="formation_log.jsonl")
        self.fc = fc

    def _publish_impl(self, frame):
        frame["coherence"] = self.fc.formation_coherence()
        # forward to GDS …
```

---

## Further reading

- [ConsensusProtocol source](../sentinel_x/formation.py)
- [FormationController source](../sentinel_x/formation.py)
- [docs/mission_control_integration.md](mission_control_integration.md)
