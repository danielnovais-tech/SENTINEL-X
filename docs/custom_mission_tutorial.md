# Custom Mission Scenario Tutorial

This tutorial walks you through creating a brand-new mission scenario in
SENTINEL-X from scratch — from requirements to a fully trained, verified,
and exported policy — in under 30 minutes.

We will build a **Jupiter flyby CubeSat** mission as the running example:
a 6U CubeSat on a gravity-assist trajectory past Jupiter, with high radiation,
limited solar power, and intermittent deep-space communications.

---

## Prerequisites

```bash
pip install -r requirements.txt
# scikit-learn is needed for decision-tree extraction
pip install scikit-learn
```

---

## Step 1 – Define Your Mission Requirements

Before writing any code, answer these four questions:

| Question | Jupiter Flyby CubeSat answer |
|----------|------------------------------|
| **Mission objective** | Collect magnetometer / particle data during Jupiter approach; maximise data returned to Earth |
| **Number of spacecraft** | 1 (single CubeSat; no swarm) |
| **Reward priority** | Data return with aggressive fault response (can't afford inaction) |
| **Environmental stressors** | Very high radiation (SEU), reduced solar power, thermal extremes |

---

## Step 2 – Choose a Reward Profile

SENTINEL-X ships four `MissionProfile` constants:

| Constant | Best for |
|----------|----------|
| `BALANCED` | General-purpose; equal weight on health and data |
| `MAXIMIZE_DATA_RETURN` | Science missions; reward data throughput |
| `EXTEND_LIFESPAN` | Long-duration spacecraft; penalise wear |
| `POWER_CONSTRAINED` | Missions near power margins (Moon, deep space) |

For the Jupiter flyby, `MAXIMIZE_DATA_RETURN` is the best fit:

```python
from sentinel_x_advanced import MissionProfile
profile = MissionProfile(MissionProfile.MAXIMIZE_DATA_RETURN)
```

---

## Step 3 – Set Fault Parameters

Map each physical stressor to its SENTINEL-X parameter:

| Physical stressor | Parameter | Jupiter value | LEO CubeSat (reference) |
|-------------------|-----------|--------------|------------------------|
| Radiation (SEU) | `flip_rate_per_bit` | `8e-4` | `2e-4` |
| Thermal extremes | `thermal_drift_std` | `0.30` | `0.25` |
| Thermal spikes | `thermal_spike_prob` | `0.020` | `0.018` |
| Sensor reliability | `sensor_stuck_prob` | `0.020` | `0.025` |
| Solar power (5.2 AU) | `power_drain_rate` | `0.18` | `0.11` |
| Comm delay (light-time) | `comm_delay_steps` | `30` | `1` |
| Link dropout | `link_dropout_prob` | `0.20` | `0.25` |

**Parameter guidance:**

- `flip_rate_per_bit`: multiply the LEO baseline (1e-4) by the ratio of the
  trapped-proton flux at your mission altitude vs. LEO.  Jupiter's radiation
  belt is ~8× LEO.
- `power_drain_rate`: scales as 1/r² where r is the solar distance in AU.
  Jupiter is 5.2 AU, so power per panel is ~1/27 of Earth's.  Set the drain
  rate proportionally higher.
- `thermal_drift_std`: larger for rapidly eclipsing or Sun-pointing spacecraft.
- `comm_delay_steps`: 1 training step ≈ ~90 min of real mission time.
  Jupiter light-time ≈ 45 min one-way → use 30 steps for safety.

---

## Step 4 – Create a `MissionScenario`

```python
from sentinel_x_advanced import MissionScenario, MissionProfile

jupiter_flyby = MissionScenario(
    name="Jupiter Flyby CubeSat",
    profile=MissionProfile.MAXIMIZE_DATA_RETURN,
    num_spacecraft=1,           # single spacecraft
    comm_delay_steps=30,        # Jupiter light-time
    link_dropout_prob=0.20,     # DSN scheduling gaps
    thermal_drift_std=0.30,     # extreme temperature swing
    thermal_spike_prob=0.020,
    flip_rate_per_bit=8e-4,     # high radiation
    sensor_stuck_prob=0.020,
    power_drain_rate=0.18,      # 5.2 AU from Sun
)
print(jupiter_flyby.summary())
```

Expected output:

```
=======================================================
  Mission Scenario: Jupiter Flyby CubeSat
=======================================================
  Profile            : data_return
  Swarm size         : 1
  Comm delay (eps)   : 30
  Link dropout       : 20%
  Thermal drift std  : 0.3
  Thermal spike prob : 0.02
  SEU flip rate      : 8.0e-04 /bit/step
  Sensor stuck prob  : 0.02
  Power drain rate   : 0.18
=======================================================
```

---

## Step 5 – Build the Swarm with Safety Constraints

```python
from sentinel_x_advanced import (
    build_swarm_for_scenario,
    SafetyMonitor,
    LTLConstraintChecker,
)

# Hard safety rules (veto unsafe actions at inference time)
monitor = SafetyMonitor()

# Soft LTL reward shaping (penalise constraint violations during training)
ltl = LTLConstraintChecker(penalty=-1.0)   # strong penalty for Jupiter's high fault rate

swarm = build_swarm_for_scenario(
    jupiter_flyby,
    safety_monitor=monitor,
    ltl_checker=ltl,
    cooperative_bonus=0.0,   # no swarm → no cooperative bonus needed
)
print(f"Swarm: {len(swarm.spacecraft)} spacecraft, "
      f"profile={swarm.mission_profile.profile}")
```

---

## Step 6 – Tune the Reward Weights (Optional)

The default reward weights may need adjusting for extreme missions.
Use `MissionProfile.update_weights` to do this at runtime (no retraining needed):

```python
# Increase urgency of fault response for the radiation-intense flyby
swarm.mission_profile.update_weights(
    fault_penalty_scale=1.5,     # stronger penalty for letting faults persist
    recovery_bonus_scale=1.2,    # extra reward for successful recoveries
)
```

You can call `update_weights` between training episodes to simulate a
mission phase transition (e.g., switch from cruise to flyby phase):

```python
for ep in range(200):
    if ep == 100:
        # Phase 2: Jupiter approach — increase fault urgency
        swarm.mission_profile.update_weights(fault_penalty_scale=2.0)
    swarm.train_episode(max_steps=200)
```

---

## Step 7 – Train

```python
import numpy as np

rewards = []
for ep in range(150):                    # 150 episodes is usually enough
    r = swarm.train_episode(max_steps=200)
    rewards.append(r)
    if (ep + 1) % 30 == 0:
        print(f"Ep {ep+1}/150  reward={np.mean(rewards[-30:]):+.2f}  "
              f"ltl_pen={swarm.last_ltl_penalty:.2f}  "
              f"veto={swarm.last_override_count}")
```

**Healthy training signs:**

- `reward` increases towards the end of training.
- `ltl_pen` decreases as the agent learns to avoid constraint violations.
- `veto` count decreases as the agent's policy aligns with the safety rules.

---

## Step 8 – Evaluate

```python
ops = swarm.test_episode(max_steps=200)
print(f"Avg operational steps: {ops:.1f} / 200  ({ops/200:.1%})")
```

A well-trained agent for the Jupiter flyby should achieve ≥ 70 % operational
uptime despite the hostile radiation environment.

---

## Step 9 – Export TFLite

```python
from sentinel_x_advanced import export_tflite, export_tflite_int8

# Dynamic-range model (larger, higher precision)
export_tflite(swarm.agents[0], output_path="jupiter_flyby.tflite")

# Int8 model (MCU-ready, ~4× smaller)
export_tflite_int8(
    swarm.agents[0],
    output_path="jupiter_flyby_int8.tflite",
    n_calib_samples=64,
)
print("Models exported.")
```

---

## Step 10 – Verify and Certify

```python
from sentinel_x_advanced import PolicyVerifier, AdversarialTester

# Formal policy verification
verifier = PolicyVerifier(swarm.agents[0], n_samples=1000, margin_threshold=0.05)
report   = verifier.verify()
sc       = report["safety_constraint"]
print(f"Verification: passed={report['overall_passed']}  "
      f"violations={sc['violations']}/1000  ({sc['rate']:.1%})")

# Adversarial robustness certification
tester = AdversarialTester(swarm.agents[0], epsilon=0.05)
cert   = tester.certify_robustness(n_samples=200, eps_hi=0.3, n_bisect=10)
print(f"Robustness: mean_radius={cert['mean_radius']:.4f}  "
      f"robust_frac={cert['robust_frac']:.1%}")
```

---

## Step 11 – Extract a Certifiable Decision Tree

```python
from sentinel_x_advanced import (
    extract_decision_tree,
    dt_fidelity_report,
    export_decision_tree_rules,
)

dt     = extract_decision_tree(swarm.agents[0], n_samples=2000, max_depth=8)
report = dt_fidelity_report(swarm.agents[0], dt, n_eval=1000)
print(f"DT fidelity: {report['fidelity']*100:.1f}%")

# Export human-readable rules for review / SMT verification
rules = export_decision_tree_rules(dt)
with open("jupiter_flyby_policy_rules.txt", "w") as f:
    f.write(rules)
```

---

## Step 12 – Putting It All Together with `run_experiment.py`

Once you have validated your scenario parameters, you can run the full pipeline
from a configuration file without touching Python code.

Create `configs/jupiter_flyby.yaml`:

```yaml
agent:
  state_dim: 11
  action_dim: 4
  learning_rate: 0.001
  gamma: 0.99
  epsilon: 1.0
  epsilon_min: 0.01
  epsilon_decay: 0.995
  memory_size: 2000
  batch_size: 32

swarm:
  num_spacecraft: 1
  cooperative_bonus: 0.0

federation:
  federated_interval: 10
  comm_delay_steps: 30
  link_dropout_prob: 0.20

mission:
  profile: data_return
  recovery_bonus_scale: 1.2
  fault_penalty_scale: 1.5

faults:
  flip_rate_per_bit: 8.0e-4
  sensor_stuck_prob: 0.020
  thermal_drift_std: 0.30
  thermal_spike_prob: 0.020
  power_drain_rate: 0.18

training:
  episodes: 150
  max_steps: 200
  target_update_interval: 10

ltl:
  enabled: true
  penalty: -1.0

deployment:
  tflite_int8_path: jupiter_flyby_int8.tflite
```

Run it:

```bash
python run_experiment.py --config configs/jupiter_flyby.yaml
```

The runner will train, evaluate, export TFLite, verify, certify, and extract
the decision-tree surrogate — all in one step.

---

## Parameter Reference Card

| Parameter | Effect | Typical range |
|-----------|--------|---------------|
| `num_spacecraft` | Swarm size | 1–20 |
| `comm_delay_steps` | Federated round-trip delay | 0–30 |
| `link_dropout_prob` | Fraction of fed. rounds lost | 0.0–0.5 |
| `thermal_drift_std` | Thermal noise amplitude | 0.05–0.50 |
| `thermal_spike_prob` | Per-step spike probability | 0.0–0.05 |
| `flip_rate_per_bit` | SEU bit-flip rate | 1e-5 – 1e-3 |
| `sensor_stuck_prob` | Per-step stuck-sensor probability | 0.0–0.05 |
| `power_drain_rate` | Mean power consumption per step | 0.05–0.25 |
| `ltl.penalty` | LTL constraint violation penalty | −0.1 – −2.0 |
| `fault_penalty_scale` | Urgency multiplier for fault tolerance | 0.5–3.0 |
| `recovery_bonus_scale` | Incentive for successful recovery | 0.5–3.0 |

---

## Troubleshooting

**Agent never recovers (reward stays negative)**
→ Reduce `fault_penalty_scale` and/or increase `recovery_bonus_scale`.  The agent may
  be overwhelmed by simultaneous faults; reduce `flip_rate_per_bit` for initial training.

**Too many LTL violations (ltl_pen very negative)**
→ The fault rate is too high for the agent to avoid violations from the start.
  Increase `epsilon_decay` (slower exploration) or reduce `ltl.penalty` magnitude.

**Training curves don't converge**
→ Increase `episodes` or reduce `max_steps` per episode.  For large `comm_delay_steps`,
  federated knowledge may arrive too late; try reducing `federated_interval`.

**DT fidelity < 80 %**
→ Increase `n_samples` in `extract_decision_tree()` or `max_depth`.  A shallow tree
  (depth ≤ 4) may be too coarse for a complex radiation environment.

---

## Further Reading

- [docs/ltl_constraints.md](ltl_constraints.md) – LTL reward shaping API
- [docs/decision_tree_certification.md](decision_tree_certification.md) – DT surrogate and certification
- [examples/lunar_gateway.py](../examples/lunar_gateway.py) – Full < 1-minute quickstart
- [scripts/run_lunar_gateway.py](../scripts/run_lunar_gateway.py) – 7-step Lunar Gateway cookbook
