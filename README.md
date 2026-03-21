# SENTINEL-X

**Self-Healing Spacecraft Networks via Reinforcement Learning**

Adaptive recovery is the next frontier in space autonomy, and those who invest now will define the future of interplanetary reliability.

Deep-space systems are fragile, and rule-based recovery can't keep pace with unpredictable failures. SENTINEL-X introduces self-healing spacecraft networks that don't just react—they learn. By leveraging reinforcement learning, these systems detect anomalies, adapt recovery strategies, and improve resilience over time. The result: autonomous missions that endure radiation, delays, and uncertainty—without waiting for human intervention.

---

## Features

- **Basic Q-learning simulation** (`sentinel_x.py`) – A tabular Q-learning agent learns to recover a simulated spacecraft from random faults (memory corruption, sensor failure, CPU hang).
- **Advanced DQN simulation** (`sentinel_x_advanced.py`) – Extends the prototype with:
  - **Realistic fault generators**: `MemoryArray` (SEUs/bit flips), `Sensor` (Gaussian noise + stuck-at), `ThermalSubsystem` (overheating/overcooling), `PowerSubsystem` (brownout), `AttitudeControlSubsystem` (gyro-drift/tumble), and `CommSubsystem` (link-quality degradation/dropout).
  - **Deep Q-Network (DQN) agent** using TensorFlow/Keras with experience replay and a separate target network.
  - **Swarm simulation** – multiple spacecraft each managed by an independent DQN agent.
  - **Mission-specific reward profiles** – `MissionProfile` lets you tune the reward function to mission priorities (balanced, data-return, lifespan, power-constrained) with runtime-adjustable weights via `update_weights()`.
  - **Federated learning** – `FederatedServer` averages DQN weights across all agents (FedAvg), with optional deep-space communication latency (`comm_delay_steps`) and stochastic link dropout (`link_dropout_prob`).
  - **Gossip-based federation** – `GossipServer` provides a decentralised alternative where each spacecraft gossips with k random neighbours, scaling to large constellations without a central server.
  - **Multi-agent coordination** – `FederatedSwarm` cross-checks peer sensor readings to detect stuck sensors that individual agents cannot diagnose alone.
  - **TFLite export** – `export_tflite()` (dynamic-range) and `export_tflite_int8()` (full int8, for Cortex-M / MCU) serialise a trained DQN for embedded deployment.
  - **Safety monitor** – `SafetyMonitor` is a rule-based FDIR veto layer: never "do nothing" on a fault, never reset hardware on critically low power.  Pass it to `FederatedSwarm` via `safety_monitor=` to train the agent to avoid veto-triggering actions.
  - **Adversarial testing** – `AdversarialTester` uses FGSM to find minimal state perturbations that flip the greedy action; `augment_replay_buffer()` injects adversarial transitions to harden the policy; `certify_robustness()` binary-searches the minimum per-state perturbation radius.
  - **Formal verification** – `PolicyVerifier` runs lightweight safety proofs (safety constraint, Q-value margin, action coverage) over a trained policy and prints a CI-friendly report.
  - **LTL reward shaping** – `LTLConstraintChecker` encodes safety properties as Linear Temporal Logic (LTL) predicates and penalises violations at training time (constrained RL).
  - **Decision-tree extraction** – `extract_decision_tree()` trains a scikit-learn `DecisionTreeClassifier` surrogate via imitation learning, providing a certifiable, human-readable proxy for the DQN.
  - **Mission scenario presets** – `MissionScenario.lunar_gateway()`, `.mars_orbiter()`, `.cubesat_swarm()` bundle fault-model and reward-profile parameters for three real mission concepts; `build_swarm_for_scenario()` builds the ready-to-train swarm.
  - **Federation benchmark** – `benchmark_federation()` compares FedAvg and gossip under configurable link-dropout rates and prints a summary table.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Basic Q-learning simulation

```bash
python sentinel_x.py
```

Trains a tabular Q-learning agent for 500 episodes and then evaluates it against a rule-based baseline.

### Advanced DQN + swarm simulation

```bash
python sentinel_x_advanced.py
```

Trains a basic swarm and then a `FederatedSwarm` under each of the four mission profiles, exports both dynamic-range and int8 TFLite models, demonstrates deep-space communication latency, gossip-based federation, dynamic reward shaping, safety monitor, and adversarial testing. Training curves are saved to `sentinel_x_training_curve.png`.

---

## Architecture Overview

```
+----------------------+------------------------------------------+
| SENTINEL-X System                                               |
+----------------------+------------------------------------------+
| Fault Generators     | MemoryArray (bit flips)                  |
|                      | Sensor (noise / stuck-at)                |
|                      | ThermalSubsystem (overheating/cooling)   |
|                      | PowerSubsystem (brownout/depletion)      |
|                      | AttitudeControlSubsystem (gyro/tumble)   |
|                      | CommSubsystem (link dropout/degradation) |
+----------------------+------------------------------------------+
| Spacecraft           | Aggregates subsystem health,             |
| Environment          | produces 10-dim normalised state vector  |
+----------------------+------------------------------------------+
| Recovery Agent       | DQN (64->64->actions) with replay        |
|                      | buffer and target network                |
+----------------------+------------------------------------------+
| SafetyMonitor        | FDIR veto layer: hard safety rules       |
|                      | override unsafe DQN recommendations      |
+----------------------+------------------------------------------+
| Swarm                | N independent spacecraft + agents        |
+----------------------+------------------------------------------+
| FederatedSwarm       | Swarm + peer sensor cross-check          |
|                      | (11-dim state) + MissionProfile          |
+----------------------+------------------------------------------+
| FederatedServer      | FedAvg with comm-delay queue &           |
|                      | stochastic link-dropout simulation       |
+----------------------+------------------------------------------+
| GossipServer         | Decentralised k-neighbour gossip         |
|                      | averaging (no central server needed)     |
+----------------------+------------------------------------------+
| PolicyVerifier       | Formal safety checks: constraint,        |
|                      | Q-margin bound, action coverage          |
+----------------------+------------------------------------------+
| AdversarialTester    | FGSM corner-case discovery;              |
|                      | generates adversarial training examples  |
+----------------------+------------------------------------------+
```

### Recovery Actions

| Action | Description                          | Effective against                              |
|--------|--------------------------------------|------------------------------------------------|
| 0      | Do nothing                           | —                                                         |
| 1      | Restart subsystem                    | Memory, sensor, thermal, attitude, comm faults            |
| 2      | Switch to redundant hardware         | All fault types (full reset; high power draw)             |
| 3      | Safe mode (targeted)                 | Stuck sensor, parity, thermal, power, attitude, comm      |

---

## Mission-Specific Reward Profiles

`MissionProfile` shapes the reward signal to align the agent's behaviour with mission priorities.

| Profile                  | Objective                                            |
|--------------------------|------------------------------------------------------|
| `BALANCED` (default)     | General-purpose; equal weight on uptime and cost     |
| `MAXIMIZE_DATA_RETURN`   | Maximise operational time; penalise inaction on faults |
| `EXTEND_LIFESPAN`        | Prefer cheap recovery actions; preserve spares       |
| `POWER_CONSTRAINED`      | Conserve power; penalise high-current redundant switch |

```python
from sentinel_x_advanced import FederatedSwarm, MissionProfile

swarm = FederatedSwarm(
    num_spacecraft=5,
    action_dim=4,
    mission_profile=MissionProfile(MissionProfile.MAXIMIZE_DATA_RETURN),
    federated_interval=10,
    link_dropout_prob=0.1,   # 10% stochastic link outage
)
avg_reward = swarm.train_episode(max_steps=150)

# Adjust reward weights at runtime
swarm.mission_profile.update_weights(fault_penalty_scale=2.0)
```

---

## Federated Learning

`FederatedServer` implements **FedAvg**: every `federated_interval` training episodes it averages the online-network weights from all agents and redistributes the result, then synchronises each agent's target network.

```python
from sentinel_x_advanced import FederatedServer, DQNAgent

# Immediate aggregation (default)
server = FederatedServer()
agents = [DQNAgent(state_dim=11, action_dim=4) for _ in range(5)]
# ... train agents independently for one episode each ...
server.aggregate(agents)   # all agents now share the same averaged weights

# Simulate a 5-episode deep-space link delay
server_delayed = FederatedServer(comm_delay_steps=5)
server_delayed.aggregate(agents)   # weights queued, not yet applied
for _ in range(5):
    server_delayed.tick(agents)    # advance delay counter each episode
# weights are now applied after 5 tick() calls
```

---

## Multi-Agent Coordination

`FederatedSwarm` extends the state vector with a **peer sensor deviation** feature (11th dimension). Before each time step, `_cross_check_sensors()` computes the deviation of each spacecraft's sensor reading from the swarm median. A spacecraft whose reading is an outlier receives a high `peer_sensor_deviation_norm` value, giving its DQN agent an additional signal to suspect a stuck-sensor fault.

```
State vector (FederatedSwarm) - 11 features:
  [mem_error_ratio, parity_flag, sensor_deviation_norm,
   sensor_stuck_flag, time_since_recovery_norm, health_flag,
   thermal_fault_flag, power_level_norm,
   attitude_rate_norm, comm_quality_norm,
   peer_sensor_deviation_norm]   <- coordination feature
```

---

## TFLite Export

Convert any trained `DQNAgent` to TensorFlow Lite for deployment on embedded hardware (e.g., microcontrollers, CubeSat avionics):

```python
from sentinel_x_advanced import export_tflite, export_tflite_int8, DQNAgent

agent = DQNAgent(state_dim=11, action_dim=4)
# ... train agent ...

# Dynamic-range quantisation (float32 inference)
export_tflite(agent, output_path="sentinel_x_model.tflite")

# Full integer-only quantisation for Cortex-M / STM32
export_tflite_int8(agent, output_path="sentinel_x_model_int8.tflite")
```

`export_tflite()` applies dynamic-range weight quantisation (`tf.lite.Optimize.DEFAULT`).
`export_tflite_int8()` performs full integer-only quantisation using a representative calibration dataset, producing integer-only arithmetic suitable for MCUs without FPUs.

---

## Formal Verification

`PolicyVerifier` audits a trained `DQNAgent` against three certifiable safety properties without requiring an external SMT or model-checking solver, making it compatible with standard ground-segment CI pipelines.

| Check | Property | Passes when |
|---|---|---|
| **Safety constraint** | Agent never selects "do nothing" (action 0) on a faulty state | Zero violations across all sampled fault states |
| **Q-value margin** | Policy is not ambiguous between actions | Mean (best − 2nd-best) Q-value gap ≥ `margin_threshold` |
| **Action coverage** | Every recovery action is reachable on some fault state | All actions 1…N appear in the greedy policy |

```python
from sentinel_x_advanced import PolicyVerifier, DQNAgent

agent = DQNAgent(state_dim=11, action_dim=4)
# ... train agent ...

verifier = PolicyVerifier(agent, n_samples=500, margin_threshold=0.1)
report = verifier.verify()
# Prints:
# =======================================================
#   SENTINEL-X Policy Verification Report
# =======================================================
#   [PASS] Overall
#   [PASS] Safety constraint – 0 violation(s) / 500 states  (0.0%)
#   [PASS] Q-value margin  – mean=0.2341, min=0.0031, threshold=0.1
#   [PASS] Action coverage  – covered=[0, 1, 2, 3], missing=[none]
# =======================================================
```

> **Note on certifiability**: The verifier uses probabilistic sampling over the normalised state space.  For flight-critical certification (e.g., DO-178C / ECSS standards) this lightweight check should be complemented by interval-arithmetic bounds or a dedicated neural-network verification tool such as α,β-CROWN or Marabou.

---

## Safety Monitor

`SafetyMonitor` wraps any DQN agent with a rule-based FDIR veto layer enforcing hard safety constraints at inference time:

```python
from sentinel_x_advanced import SafetyMonitor
import numpy as np

monitor = SafetyMonitor()
state = np.zeros(11, dtype=np.float32)
state[5] = 1.0   # health_flag = faulted

safe_action = monitor.veto(proposed_action, state, action_dim=4)
# action 0 on faulted state -> forced to 3 (safe mode)
# action 2 on critical-power state (state[7] < 0.15) -> forced to 3
```

---

## Gossip-Based Federated Learning

`GossipServer` provides a decentralised alternative to `FederatedServer` for large constellations:

```python
from sentinel_x_advanced import GossipServer, FederatedSwarm

swarm = FederatedSwarm(num_spacecraft=10, action_dim=4, federated_interval=999)
gossip = GossipServer(k=2, comm_delay_steps=3)

for episode in range(200):
    swarm.train_episode(max_steps=150)
    if episode % 10 == 0:
        gossip.gossip_round(swarm.agents)
    gossip.tick(swarm.agents)
```

---

## Adversarial Testing

`AdversarialTester` uses FGSM to find minimal perturbations that flip the agent's greedy action, and provides two additional tools:

```python
from sentinel_x_advanced import AdversarialTester

tester = AdversarialTester(agent, epsilon=0.05)

# Find adversarial examples
results = tester.find_adversarial_examples(n_examples=200)
tester.summary(results, n_tested=200)

# Augment replay buffers to harden the policy
n = tester.augment_replay_buffer(agents, n_examples=200, safety_monitor=monitor)

# Certify per-state robustness radius
cert = tester.certify_robustness(n_samples=100, eps_hi=0.3, n_bisect=10)
print(f"Mean certified radius: {cert['mean_radius']:.4f}")
```

---

## LTL Reward Shaping

`LTLConstraintChecker` encodes safety properties as LTL-style predicates and returns a penalty at each step for any active violation, implementing constrained RL:

```python
from sentinel_x_advanced import LTLConstraintChecker

ltl = LTLConstraintChecker(penalty=-0.5)

# Built-in constraints: no_inaction_on_fault, no_full_reset_low_power,
#                       no_simultaneous_faults, comm_link_recovery

# Add a custom constraint
ltl.add_constraint(
    "safe_attitude",
    lambda state, action: float(state[8]) > 0.9 and action == 0,
)

penalty = ltl.evaluate(state, action)   # total penalty this step
```

Integrate into a training loop by adding `penalty` to the mission reward before storing the transition in the replay buffer.

---

## Decision-Tree Policy Extraction

`extract_decision_tree()` trains a scikit-learn `DecisionTreeClassifier` to imitate the DQN policy (behavioural cloning), producing a certifiable and human-auditable surrogate:

```python
from sentinel_x_advanced import extract_decision_tree

# Requires: pip install scikit-learn
dt = extract_decision_tree(agent, n_samples=2000, max_depth=8)
# Decision-tree surrogate: depth=8, leaves=127, fidelity=94.2%

# Use the surrogate for inference
action = int(dt.predict(state[np.newaxis, :])[0])
```

The tree can be exported to DOT format (`sklearn.tree.export_graphviz`) for visual inspection, or fed into interval-arithmetic solvers (e.g., α,β-CROWN, Marabou) for flight-certifiable verification.

---

## Mission Scenario Presets

`MissionScenario` bundles fault-model parameters and reward profile for three real mission concepts:

```python
from sentinel_x_advanced import MissionScenario, build_swarm_for_scenario, SafetyMonitor

# Lunar Gateway – NRHO, power-constrained, Earth-proximity comms
scenario = MissionScenario.lunar_gateway()

# Mars Sample Return orbiter – deep-space delay, high radiation
scenario = MissionScenario.mars_orbiter()

# LEO CubeSat swarm – frequent eclipses, data-return, gossip-friendly
scenario = MissionScenario.cubesat_swarm()

print(scenario.summary())

# Build a ready-to-train FederatedSwarm for this mission
swarm = build_swarm_for_scenario(scenario, safety_monitor=SafetyMonitor())
avg_reward = swarm.train_episode(max_steps=150)
```

---

## Deployment Goals

### For Research

SENTINEL-X provides a reproducible baseline for papers on:

- **Federated reinforcement learning for space autonomy**: compare FedAvg vs gossip convergence under deep-space delays and link outages using `benchmark_federation()`.
- **Adversarial robustness of safety-critical policies**: measure flip rates before/after `augment_replay_buffer()` and report certified radii from `certify_robustness()`.
- **Constrained RL with LTL specifications**: demonstrate that `LTLConstraintChecker` penalties steer agents toward constraint-compliant policies.
- **Formal verification of neural network policies**: use `PolicyVerifier` as a lightweight baseline, then compare to Marabou or α,β-CROWN for full-space certification.
- **Certifiable policy surrogates**: show that `extract_decision_tree()` achieves >90% fidelity at depth ≤ 8 and can be exhaustively verified.

### For Engineering (Hardware-in-the-Loop)

The recommended HIL workflow:

1. **Export** the trained DQN to int8 TFLite: `export_tflite_int8(agent, "model_int8.tflite")`.
2. **Flash** the `.tflite` flatbuffer to a Cortex-M target (STM32H7, Raspberry Pi Pico) using TensorFlow Lite for Microcontrollers.
3. **Measure** inference latency (target < 1 ms), power draw, and bit-exact determinism.
4. **Inject faults** via an FPGA bit-flip board or JTAG-controlled memory corruptor while the MCU runs inference; verify that `SafetyMonitor` vetoes unsafe actions.
5. **Close the loop** by connecting the MCU's UART output to the spacecraft fault simulator; run full-stack HIL episodes with real round-trip latency.
6. **Log overrides** – feed `SafetyMonitor` veto events back as training signal (already supported via `FederatedSwarm.last_override_count`).

### For Mission Planning

Pick a mission concept and tailor the simulation:

```python
from sentinel_x_advanced import (
    MissionScenario, build_swarm_for_scenario,
    GossipServer, SafetyMonitor, LTLConstraintChecker,
)

# 1. Choose a pre-built scenario or define your own
scenario = MissionScenario(
    name="My CubeSat",
    profile="data_return",
    num_spacecraft=6,
    comm_delay_steps=2,
    link_dropout_prob=0.3,
    thermal_spike_prob=0.02,
    flip_rate_per_bit=2e-4,
    sensor_stuck_prob=0.02,
    power_drain_rate=0.12,
)

# 2. Build a swarm with safety constraints and LTL reward shaping
ltl = LTLConstraintChecker(penalty=-0.5)
swarm = build_swarm_for_scenario(scenario, safety_monitor=SafetyMonitor())

# 3. Add gossip-based federation for partial connectivity
gossip = GossipServer(k=2, comm_delay_steps=scenario.comm_delay_steps)

# 4. Train and log results
for ep in range(500):
    reward = swarm.train_episode(max_steps=200)
    if ep % 10 == 0:
        gossip.gossip_round(swarm.agents)
    gossip.tick(swarm.agents)
```

---

## Hardware-in-the-Loop (HIL) Testing Strategies

Moving from simulation to real hardware requires careful validation:

1. **Fault injection at hardware level** – FPGA-based injectors that flip bits via JTAG; voltage-offset signal generators for sensor faults.
2. **Emulated space environment** – Thermal-vacuum chambers cycling temperature while injecting faults; vibration tables for launch loads.
3. **Real-time telemetry & logging** – Agent decisions logged alongside system state; safety supervisor overrides the agent if it deviates from safe bounds.
4. **Incremental deployment** – Hardware emulator → flight-like board (CubeSat avionics) → end-to-end tests with ground segment.
5. **Model-based verification** – Formal verification tools to bound Q-value differences and certify critical action selection under specific fault conditions.
6. **Hybrid architecture** – The RL agent recommends actions to a deterministic FDIR layer that retains final authority, allowing gradual adoption.

---

## Limitations and Real-World Considerations

- **Safety & Verification**: Neural networks are not directly certifiable. A hybrid FDIR + RL architecture is recommended for production.
- **Simulation Fidelity**: Physics-based fault models (radiation dose, sensor drift) are needed before deployment validation.
- **Continuous Learning**: Onboard retraining is compute-limited; periodic ground-simulated policy updates are more practical.
- **Multi-Agent Swarm**: Distributed RL with shared experiences can further improve swarm resilience when inter-spacecraft communication is available.

