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

Trains a swarm of 5 spacecraft (each with its own DQN agent) for 200 episodes, evaluates the trained agents, and saves the training curve to `sentinel_x_training_curve.png`.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    SENTINEL-X System                    │
├───────────────────┬─────────────────────────────────────┤
│  Fault Generators │  MemoryArray (bit flips)             │
│                   │  Sensor (noise / stuck-at)           │
├───────────────────┼─────────────────────────────────────┤
│  Spacecraft       │  Aggregates subsystem health,        │
│  Environment      │  produces normalised state vector    │
├───────────────────┼─────────────────────────────────────┤
│  Recovery Agent   │  DQN (64→64→actions) with replay     │
│                   │  buffer and target network           │
├───────────────────┼─────────────────────────────────────┤
│  Swarm            │  N independent spacecraft + agents   │
└───────────────────┴─────────────────────────────────────┘
```

### Recovery Actions

| Action | Description                          | Effective against                  |
|--------|--------------------------------------|------------------------------------|
| 0      | Do nothing                           | —                                  |
| 1      | Restart subsystem                    | Memory errors, CPU hang            |
| 2      | Switch to redundant hardware         | All fault types (full recovery)    |
| 3      | Safe mode                            | Stuck sensor, parity error         |

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

