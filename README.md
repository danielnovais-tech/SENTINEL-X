# SENTINEL-X

**Self-Healing Spacecraft Networks via Reinforcement Learning**

Adaptive recovery is the next frontier in space autonomy, and those who invest now will define the future of interplanetary reliability.

Deep-space systems are fragile, and rule-based recovery can't keep pace with unpredictable failures. SENTINEL-X introduces self-healing spacecraft networks that don't just react—they learn. By leveraging reinforcement learning, these systems detect anomalies, adapt recovery strategies, and improve resilience over time. The result: autonomous missions that endure radiation, delays, and uncertainty—without waiting for human intervention.

---

## Features

- **Basic Q-learning simulation** (`sentinel_x.py`) – A tabular Q-learning agent learns to recover a simulated spacecraft from random faults (memory corruption, sensor failure, CPU hang).
- **Advanced DQN simulation** (`sentinel_x_advanced.py`) – Extends the prototype with:
  - **Realistic fault generators**: `MemoryArray` (single-event upsets / bit flips) and `Sensor` (Gaussian noise + stuck-at faults).
  - **Deep Q-Network (DQN) agent** using TensorFlow/Keras with experience replay and a separate target network.
  - **Swarm simulation** – multiple spacecraft each managed by an independent DQN agent.
  - **Mission-specific reward profiles** – `MissionProfile` lets you tune the reward function to mission priorities.
  - **Federated learning** – `FederatedServer` averages DQN weights across all agents (FedAvg), aligning the swarm's policy without sharing raw data.
  - **Multi-agent coordination** – `FederatedSwarm` cross-checks peer sensor readings to detect stuck sensors that individual agents cannot diagnose alone.
  - **TFLite export** – `export_tflite()` quantises and serialises a trained DQN for microcontroller deployment.
  - **Formal verification** – `PolicyVerifier` runs lightweight safety proofs (safety constraint, Q-value margin, action coverage) over a trained policy and prints a CI-friendly report.

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

Trains a basic swarm and then a `FederatedSwarm` under each of the three mission profiles, evaluates the trained agents, exports a TFLite model, and saves the training curves to `sentinel_x_training_curve.png`.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      SENTINEL-X System                       │
├────────────────────┬─────────────────────────────────────────┤
│  Fault Generators  │  MemoryArray (bit flips)                 │
│                    │  Sensor (noise / stuck-at)               │
├────────────────────┼─────────────────────────────────────────┤
│  Spacecraft        │  Aggregates subsystem health,            │
│  Environment       │  produces normalised state vector        │
├────────────────────┼─────────────────────────────────────────┤
│  Recovery Agent    │  DQN (64→64→actions) with replay         │
│                    │  buffer and target network               │
├────────────────────┼─────────────────────────────────────────┤
│  Swarm             │  N independent spacecraft + agents       │
├────────────────────┼─────────────────────────────────────────┤
│  FederatedSwarm    │  Swarm + peer sensor cross-check         │
│                    │  (7-dim state) + MissionProfile          │
├────────────────────┼─────────────────────────────────────────┤
│  FederatedServer   │  FedAvg weight aggregation               │
│                    │  across all DQN agents                   │
├────────────────────┼─────────────────────────────────────────┤
│  PolicyVerifier    │  Formal safety checks: constraint,       │
│                    │  Q-margin bound, action coverage         │
└────────────────────┴─────────────────────────────────────────┘
```

### Recovery Actions

| Action | Description                          | Effective against                  |
|--------|--------------------------------------|------------------------------------|
| 0      | Do nothing                           | —                                  |
| 1      | Restart subsystem                    | Memory errors, CPU hang            |
| 2      | Switch to redundant hardware         | All fault types (full recovery)    |
| 3      | Safe mode                            | Stuck sensor, parity error         |

---

## Mission-Specific Reward Profiles

`MissionProfile` shapes the reward signal to align the agent's behaviour with mission priorities.

| Profile                  | Objective                                            |
|--------------------------|------------------------------------------------------|
| `BALANCED` (default)     | General-purpose; equal weight on uptime and cost     |
| `MAXIMIZE_DATA_RETURN`   | Maximise operational time; penalise inaction on faults |
| `EXTEND_LIFESPAN`        | Prefer cheap recovery actions; preserve spares       |

```python
from sentinel_x_advanced import FederatedSwarm, MissionProfile

swarm = FederatedSwarm(
    num_spacecraft=5,
    action_dim=4,
    mission_profile=MissionProfile(MissionProfile.MAXIMIZE_DATA_RETURN),
    federated_interval=10,
)
avg_reward = swarm.train_episode(max_steps=150)
```

---

## Federated Learning

`FederatedServer` implements **FedAvg**: every `federated_interval` training episodes it averages the online-network weights from all agents and redistributes the result, then synchronises each agent's target network.

```python
from sentinel_x_advanced import FederatedServer, DQNAgent

server = FederatedServer()
agents = [DQNAgent(state_dim=7, action_dim=4) for _ in range(5)]
# ... train agents independently for one episode each ...
server.aggregate(agents)   # all agents now share the same averaged weights
```

---

## Multi-Agent Coordination

`FederatedSwarm` extends the state vector with a **peer sensor deviation** feature (7th dimension). Before each time step, `_cross_check_sensors()` computes the deviation of each spacecraft's sensor reading from the swarm median. A spacecraft whose reading is an outlier receives a high `peer_sensor_deviation_norm` value, giving its DQN agent an additional signal to suspect a stuck-sensor fault.

```
State vector (FederatedSwarm):
  [mem_error_ratio, parity_flag, sensor_deviation_norm,
   sensor_stuck_flag, time_since_recovery_norm, health_flag,
   peer_sensor_deviation_norm]   ← coordination feature
```

---

## TFLite Export

Convert any trained `DQNAgent` to TensorFlow Lite for deployment on embedded hardware (e.g., microcontrollers, CubeSat avionics):

```python
from sentinel_x_advanced import export_tflite, DQNAgent

agent = DQNAgent(state_dim=7, action_dim=4)
# ... train agent ...
export_tflite(agent, output_path="sentinel_x_agent.tflite")
# → TFLite model exported to 'sentinel_x_agent.tflite' (8.9 KB)
```

The converter applies dynamic-range weight quantisation (`tf.lite.Optimize.DEFAULT`), reducing the model footprint while preserving inference accuracy.

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

agent = DQNAgent(state_dim=7, action_dim=4)
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

