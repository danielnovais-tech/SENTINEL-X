# Using the Decision Tree for Policy Certification

A DQN neural network is a black box.  Deploying it aboard a spacecraft
requires a certifiable representation that a human safety engineer can review
and that an automated formal-verification tool can exhaustively check.
SENTINEL-X extracts a **decision-tree surrogate** via imitation learning,
providing exactly that.

---

## What Is a Decision-Tree Surrogate?

A surrogate is a second model trained to *imitate* the DQN's greedy policy.
The surrogate is not used for learning—it is used for auditing:

```
DQN policy (black box)  →  label random states with argmax Q  →  fit a
DecisionTreeClassifier (white box)  →  readable rules + formal verification
```

Because the tree partitions the state space into a *finite set of rectangular
regions* (axis-aligned splits), it can be:

1. **Enumerated** – every leaf is a rule; a human can read all of them.
2. **Verified** – solvers like Marabou, α,β-CROWN, or a simple interval
   propagator can check whether any region violates a safety property.
3. **Deployed as a fallback** – if the neural network is unavailable
   (hardware fault, bit-flip in weights), the tree can serve as a
   fail-safe policy.

---

## Extracting the Tree

```python
from sentinel_x import extract_decision_tree, dt_fidelity_report

# dt is a fitted scikit-learn DecisionTreeClassifier
dt = extract_decision_tree(agent, n_samples=2000, max_depth=8)
# Prints: Decision-tree surrogate: depth=7, leaves=62, fidelity=94.3%
```

| Parameter | Effect |
|-----------|--------|
| `n_samples` | More samples → higher fidelity but slower fitting |
| `max_depth` | Shallower tree → fewer rules, easier to verify; deeper → higher fidelity |
| `rng_seed` | Fix for reproducibility |

### Fidelity

*Fidelity* (action-match rate) measures agreement between the DQN's greedy
action and the tree's prediction on held-out random states:

```python
report = dt_fidelity_report(agent, dt, n_eval=1000)
# DT fidelity report  (n=1000, tree depth=7, leaves=62)
#   Overall fidelity   : 94.3%
#   DO_NOTHING            :  411/435  matched  (94.5%)
#   RESTART               :   98/110  matched  (89.1%)
#   SWITCH_REDUNDANT      :  305/319  matched  (95.6%)
#   SAFE_MODE             :  130/136  matched  (95.6%)
```

A fidelity ≥ 90 % is generally considered a reliable surrogate for
certification purposes.

---

## Exporting Rules for Formal Verification

```python
from sentinel_x import export_decision_tree_rules

feature_names = [
    "mem_error_ratio", "parity_flag", "sensor_deviation_norm",
    "sensor_stuck_flag", "time_since_recovery_norm", "health_flag",
    "thermal_fault_flag", "power_level_norm", "attitude_rate_norm",
    "comm_quality_norm", "peer_sensor_deviation_norm",
]

rules = export_decision_tree_rules(dt, feature_names=feature_names)
print(rules[:500])
```

Example output (excerpt):

```
if health_flag <= 0.500000:
  if power_level_norm <= 0.150000:
    return SAFE_MODE  # action=3, coverage=8.2%
  else:  # power_level_norm > 0.150000
    if comm_quality_norm <= 0.095000:
      return SWITCH_REDUNDANT  # action=2, coverage=3.1%
    else:  # comm_quality_norm > 0.095000
      return DO_NOTHING  # action=0, coverage=42.7%
else:  # health_flag > 0.500000
  return SAFE_MODE  # action=3, coverage=46.0%
```

Each rule can be expressed as an SMT constraint and fed to a solver to verify
that unsafe actions are never selected in provably dangerous regions.

### Saving the rule file

```python
with open("policy_rules.txt", "w") as f:
    f.write(rules)
```

---

## Using the Tree as a Fallback in SafetyMonitor

When the DQN weight memory is suspect (e.g., after a radiation-induced
bit-flip), the decision tree—stored as plain integers (split indices and
thresholds)—can serve as a fallback policy:

```python
from sentinel_x import SafetyMonitor

monitor = SafetyMonitor(dt_fallback=dt)
```

Behaviour with `dt_fallback`:

| Scenario | Outcome |
|----------|---------|
| Faulted state + DQN says "do nothing" | Hard veto → `SAFE_MODE` (unchanged) |
| Healthy state + DQN says "do nothing" + DT says "restart" | DT nudge applied → `RESTART` |
| DQN proposes any non-zero action | DT not consulted; DQN action returned |

This proactive nudge helps catch developing faults *before* the health flag
trips, guided by the certified tree's knowledge of the state space.

---

## Comparison with the DQN

| Property | DQN | Decision Tree |
|----------|-----|---------------|
| Action quality (avg reward) | High | Moderate (fidelity ≈ 90–95 %) |
| Human readability | No | Yes |
| Formal verification | Not directly | Yes (finite partition) |
| Memory footprint | ~100 KB (int8 TFLite) | ~10 KB (JSON/text) |
| Bit-flip resilience | Low (weights are floats) | High (integers) |
| Training required | Yes | No (imitation only) |

The recommended deployment architecture uses **both**:
- DQN as the primary policy (high performance)
- Decision tree as the certified fallback via `SafetyMonitor(dt_fallback=dt)`

---

## Running the Full Certification Pipeline

```bash
# 1. Train
python run_experiment.py --scenario lunar_gateway --episodes 100

# 2. Extract surrogate and export rules (done automatically by run_experiment.py)
#    Rules written to: sentinel_x_policy_rules.txt

# 3. Feed rules to your SMT solver of choice, e.g.:
#    Marabou:  python marabou_encode.py sentinel_x_policy_rules.txt
#    Z3:       python z3_encode.py sentinel_x_policy_rules.txt
```

---

## Further Reading

- Bastani *et al.* (2018) – "Verifiable Reinforcement Learning via Policy
  Extraction"
- Tjeng *et al.* (2019) – "Evaluating Robustness of Neural Networks with
  Mixed Integer Programming"
- Katz *et al.* (2019) – "Marabou: A Verification Framework for Deep Neural
  Networks"
