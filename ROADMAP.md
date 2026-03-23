# SENTINEL-X Roadmap

This document describes planned enhancements and open research directions for
SENTINEL-X.  Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md)
for the development workflow.

Each item links to an existing guide or suggests where new code/docs would live.

---

## 1  Technical Publication

**Goal:** Publish the SENTINEL-X architecture, experimental results, and formal
verification pipeline as a peer-reviewed paper.

**Suggested venues:**

| Venue | Track / Scope |
|-------|--------------|
| IEEE Aerospace Conference | Space autonomy, on-board software |
| AIAA SciTech | Space systems, GN&C |
| Acta Astronautica (journal) | Open-source space software |
| NeurIPS / ICML / ICLR | Reinforcement learning, safety |
| SAFECOMP | Safety-critical ML certification |

**Draft template:** [`docs/technical_report.md`](docs/technical_report.md)
already contains all sections and reference results.  Adapt the abstract and
experimental sections to the target venue's page limit and style guidelines.

**Key results to highlight:**
- TFLite int8 inference latency < 0.6 ms on STM32H7 (100 × budget headroom).
- Fault-recovery rate > 92 % after 100 training episodes.
- Federated training: +12.4 reward improvement over isolated baselines.
- Formation coherence > 87 % under leader-follower law.
- Formal verification via Marabou and ERAN; LTL + SafetyMonitor veto layer.

---

## 2  Real Spacecraft / CubeSat Deployment

**Goal:** Port the trained policy to a flight computer and run in-orbit (or
flat-sat) fault-recovery experiments.

**Steps:**
1. Export the int8 TFLite flatbuffer:
   ```bash
   python run_experiment.py --scenario lunar_gateway --episodes 100
   # → sentinel_x_model_int8.tflite
   ```
2. Flash to target MCU using the FreeRTOS task template:
   `hardware/freertos_task.c` + TFLite Micro.
3. Integrate with the mission's telemetry/command bus:
   - NASA F´ adapter: `docs/mission_control_integration.md`
   - ESA TASTE adapter: same doc, TASTE section.
4. Run a flat-sat regression using `scripts/mcu_emulator.py` before flight.

**Guide:** [`docs/hardware_integration.md`](docs/hardware_integration.md) and
[`docs/hardware_deployment.md`](docs/hardware_deployment.md).

---

## 3  Enhanced Learning Algorithms

**Goal:** Improve sample efficiency and cooperative recovery performance.

### 3a  Model-Based RL
Train a world model of spacecraft dynamics (e.g., Dreamer / MBPO) and use it
for latent-space planning.  Expected benefit: 10–50 × reduction in real
interactions needed for convergence.

### 3b  Multi-Agent RL (MARL) for Swarms
Use **QMIX** or **MAPPO** to learn cooperative recovery strategies where one
spacecraft relays data or offloads tasks for a failing neighbour.  Current
FedAvg aggregation coordinates weights, not actions — MARL closes this gap.

### 3c  Meta-Learning / Few-Shot Adaptation
Apply **MAML** or **ProtoNets** so the agent adapts to novel fault modes with
< 10 new samples.  Critical for long missions where new failure modes emerge.

**Entry points:**
- `sentinel_x_advanced.py` → `DQNAgent`, `PPOAgent`, `HierarchicalAgent`
- `sentinel_x_advanced.py` → `FederatedSwarm`

---

## 4  Extended Formal Verification

**Goal:** Produce certification-ready verification artefacts.

### 4a  Continuous-State Safety Proofs
Use Marabou / ERAN over the full 11-dimensional state space to prove:
`∀s ∈ S_unsafe : π(s) ≠ DO_NOTHING`.

### 4b  Formally Verified Decision Tree as Final Policy
`extract_decision_tree()` already produces a DT surrogate.  The next step is
to formally verify the DT using a polynomial-time model checker and deploy it
as the *primary* policy (neural net as backup or training artefact only).

### 4c  CI Verification Gate
Add a GitHub Actions job that runs `PolicyVerifier.verify_policy()` on every
PR and fails if any LTL property regresses.

**Guides:**
- [`docs/decision_tree_certification.md`](docs/decision_tree_certification.md)
- [`docs/formal_verification_external.md`](docs/formal_verification_external.md)
- [`docs/ltl_constraints.md`](docs/ltl_constraints.md)

---

## 5  Hardware-in-the-Loop (HIL) Improvements

**Goal:** Move from software simulation to hardware-validated fault injection.

### 5a  Automated Fault-Injection Board
Interface an FPGA or dedicated fault-injection board (e.g., LEON3-FT emulator)
that flips bits in memory or corrupts sensor signals on command.

### 5b  High-Fidelity Simulator Integration
Run SENTINEL-X inside:
- **NASA cFS** (Core Flight System) using the `FPrimeAdapter`
- **ESA SIMULUS** for spacecraft system-level simulation

### 5c  Real-Time Performance Dashboard
Add a Grafana / web dashboard that visualises:
- Live inference latency (p50 / p95 / p99)
- Per-episode recovery success rate
- Fault event timeline
- Federated weight-sharing communication volume

Run `python scripts/dashboard.py [--port /dev/ttyUSB0]` for the current
text-mode dashboard.

**Guide:** [`docs/hardware_deployment_validation.md`](docs/hardware_deployment_validation.md)

---

## 6  Community & Agency Collaboration

**Goal:** Grow adoption and establish institutional partnerships.

- **Open-source promotion:** Share on LinkedIn, X/Twitter, space-engineering
  forums (Space Stack Exchange, NASASpaceFlight forums, Reddit r/SpaceXLounge).
- **ESA Open Space Innovation Platform:** Propose a follow-on R&D project.
- **NASA STMD:** Apply for an SBIR/STTR or open-source collaboration grant.
- **F´ integration:** Contribute `FPrimeAdapter` upstream to
  [`nasa/fprime`](https://github.com/nasa/fprime).
- **ECSS standards:** Submit the decision-tree certification approach to the
  ECSS-Q-ST-80 working group.

---

## 7  New Mission Domains

**Goal:** Demonstrate SENTINEL-X beyond LEO/deep-space spacecraft.

| Domain | Adaptations needed |
|--------|-------------------|
| **Planetary rovers** | Add terrain hazard, wheel degradation, and communication delay fault models |
| **Aircraft / UAVs** | Replace ThermalSubsystem with aerodynamic fault model; adapt communication latency |
| **Underwater vehicles** | Model pressure, corrosion, acoustic communication faults |
| **Satellite constellations** | Scale gossip protocol to 50–100 nodes; add inter-satellite link model |

**Entry point:** Subclass `FaultGenerator` and `MissionProfile` in
`sentinel_x_advanced.py`, then register the scenario in
`run_experiment.py --scenario`.

**Tutorial:** [`docs/custom_mission_tutorial.md`](docs/custom_mission_tutorial.md)

---

## 8  Educational Material

**Goal:** Make SENTINEL-X accessible to students and researchers.

### 8a  Getting-Started Notebook
`notebooks/getting_started.ipynb` walks through:
1. Installing dependencies
2. Simulating a fault scenario
3. Training a DQN agent from scratch
4. Visualising the reward curve and recovery actions
5. Exporting to TFLite

### 8b  Web-Based Simulator
A lightweight Flask or Pyodide app where users choose fault scenarios,
see the agent's decisions in real time, and tweak hyperparameters — no
local install required.

### 8c  Video Tutorial Series
Short screencast series covering:
1. Repository tour and architecture overview
2. Training your first policy
3. Federated learning across a simulated swarm
4. Deploying to Raspberry Pi

---

## 9  Benchmarking Against Alternative Approaches

**Goal:** Provide rigorous, reproducible comparisons to justify SENTINEL-X.

See [`docs/benchmarking.md`](docs/benchmarking.md) for the full methodology.

| Baseline | Metric | Current SENTINEL-X result |
|----------|--------|--------------------------|
| Rule-based FDIR | Recovery rate, episode reward | > 92 % vs. ≈ 70 % |
| Single-agent DQN (no federation) | Federated reward gain | +12.4 reward |
| Ray RLLib DQN | Training wall-clock time | TBD |
| Commercial VxWorks safety solution | Inference latency | < 0.6 ms vs. 1–5 ms |

---

## 10  Long-Term Autonomy

**Goal:** Move from episodic training to continuous, mission-duration operation.

See [`docs/continuous_learning.md`](docs/continuous_learning.md) for the
implementation plan.

### 10a  Continuous / Online Learning
Periodically fine-tune the policy on-board using telemetry accumulated since
launch.  Use a replay buffer with prioritised experience replay and a
conservative learning rate to avoid catastrophic forgetting.

### 10b  Mission Planning Integration
Interface the FDIR agent with a higher-level mission planner (e.g., PDDL or a
goal-directed planner) so the agent can trade off science return against system
health.

### 10c  Human-in-the-Loop Decision Support
Present the agent's suggested action and confidence to a ground operator; record
operator overrides; periodically retrain on operator feedback (reward shaping).

---

## Contributing

Please open a GitHub issue tagged with the relevant roadmap item before starting
work.  See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full development
workflow, code style guide, and test requirements.
