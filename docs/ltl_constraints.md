# How LTL Constraints Work in SENTINEL-X

Linear Temporal Logic (LTL) is a formal language for expressing safety and
liveness properties over sequences of states.  SENTINEL-X uses a lightweight
*predicate-based* subset of LTL—sufficient for the fault-detection and
recovery domain—without requiring a full model-checker.

---

## Motivation

A DQN agent trained purely by reward maximisation may occasionally learn
policies that violate engineering safety rules, especially early in training
when the replay buffer is sparse.  Two canonical violations in spacecraft FDIR:

| Violation | Why it is dangerous |
|-----------|---------------------|
| **Inaction on fault** – action=0 when `health_flag`=1 | The fault persists, propagates, and may cause an unrecoverable cascade |
| **Full reset on low power** – action=2 when `power_level` < 15 % | The hardware reset cycle exhausts the remaining energy budget, causing a power-off |

LTL-style penalties *reshape the reward signal* at the moment of violation,
directly penalising the offending (state, action) pair instead of waiting for
the distant downstream consequence.  This is **constrained RL**:

```
reward_total = mission_reward + ltl_penalty
```

---

## Built-in Constraints

`LTLConstraintChecker` ships with four built-in predicates:

| Constraint name | Trigger condition | Example penalty |
|-----------------|-------------------|-----------------|
| `no_inaction_on_fault` | `health_flag ≥ 0.5` **and** `action == 0` | −0.5 |
| `no_full_reset_low_power` | `action == 2` **and** `power_level < 0.15` | −0.5 |
| `no_simultaneous_faults` | ≥ 3 of {thermal, health, parity, stuck} flags are active | −0.5 |
| `comm_link_recovery` | `comm_quality < 0.1` **and** action ∉ {2, 3} | −0.5 |

Each violated constraint subtracts `penalty` (< 0) from the reward.
Multiple simultaneous violations stack additively.

---

## Integration into Training

Enable LTL shaping in the `sentinel_x_config.yaml` file:

```yaml
ltl:
  enabled: true
  penalty: -0.5
```

Or pass an `LTLConstraintChecker` instance directly to `FederatedSwarm`:

```python
from sentinel_x import FederatedSwarm, LTLConstraintChecker, MissionProfile

ltl = LTLConstraintChecker(penalty=-0.5)
swarm = FederatedSwarm(
    num_spacecraft=3,
    action_dim=4,
    mission_profile=MissionProfile("power_constrained"),
    ltl_checker=ltl,
)

for _ in range(100):
    reward = swarm.train_episode()
    print(f"reward={reward:.2f}  ltl_penalty={swarm.last_ltl_penalty:.2f}")
```

`swarm.last_ltl_penalty` accumulates the total LTL deduction over the episode
so you can monitor how often constraints are being violated during training.

---

## Adding Custom Constraints

```python
def no_attitude_thruster_on_comm_loss(state, action):
    """Block thruster firing (action=1) when comm quality < 5 %."""
    comm_q = float(state[9])   # comm_quality_norm index
    return comm_q < 0.05 and action == 1

ltl.add_constraint("no_attitude_on_comm_loss", no_attitude_thruster_on_comm_loss)
```

Remove a constraint by name:

```python
ltl.remove_constraint("comm_link_recovery")
```

---

## Relationship to SafetyMonitor

`LTLConstraintChecker` and `SafetyMonitor` are complementary:

| Component | Role | Applied at |
|-----------|------|-----------|
| `LTLConstraintChecker` | *Soft* penalty that shapes the reward; the agent still learns from the bad experience | **Training** time |
| `SafetyMonitor` | *Hard* veto that replaces the unsafe action before it is executed; constraint is never actually violated in the environment | **Inference + Training** time |

Combining both gives the strongest safety guarantee:
- The safety monitor prevents violations at runtime.
- LTL shaping teaches the agent to *prefer* safe actions autonomously, so
  vetoes become rare as training progresses.

---

## Evaluating Constraint Satisfaction

After training, use `PolicyVerifier` to measure residual constraint violations:

```python
from sentinel_x import PolicyVerifier

verifier = PolicyVerifier(swarm.agents[0], n_samples=1000)
report = verifier.verify()
sc = report["safety_constraint"]
print(f"Safety violations: {sc['violations']}/1000  ({sc['rate']:.1%})")
```

A well-trained agent with LTL shaping should converge to near-zero violations.

---

## Further Reading

- Sadigh *et al.* (2014) – "A Learning Based Approach to Control Synthesis of
  Markov Decision Processes for Linear Temporal Logic Specifications"
- Hasanbeig *et al.* (2019) – "Logically-Constrained Reinforcement Learning"
- ECSS-E-ST-70-11C – Spacecraft FDIR safety requirements
