# Benchmarking SENTINEL-X Against Alternative Approaches

This guide explains how to reproduce the comparison results presented in
`docs/technical_report.md` Section 3, and how to extend the benchmark
suite for new baselines.

---

## 1  Baselines

### 1a  Rule-Based FDIR (hand-crafted threshold logic)

The simplest baseline uses fixed thresholds and a priority-ordered rule tree:

```python
# scripts/baselines/rule_based_fdir.py  (included in examples/)
from examples.rule_based_fdir import RuleBasedFDIR
from sentinel_x_advanced import SpacecraftEnv

env  = SpacecraftEnv()
agent = RuleBasedFDIR()

total_reward = 0
state, _ = env.reset()
for _ in range(200):
    action = agent.select_action(state)
    state, reward, done, _, _ = env.step(action)
    total_reward += reward
    if done:
        break

print(f"Rule-based FDIR total reward: {total_reward:.1f}")
```

Expected result: **≈ 700–900** total reward over 200 steps (varies by fault
draw), versus **> 1 800** for a trained DQN policy.

### 1b  Single-Agent DQN (no federation)

Train a single DQN on one spacecraft and compare cumulative reward against the
federated result:

```bash
# Single agent (no federation)
python run_experiment.py --scenario lunar_gateway --episodes 50 --no-federation

# Federated (default)
python run_experiment.py --scenario lunar_gateway --episodes 50
```

Expected federated improvement: **+12.4** reward per episode on average.

### 1c  Random Policy

```python
import random
from sentinel_x_advanced import SpacecraftEnv

env = SpacecraftEnv()
state, _ = env.reset()
total_reward = 0
for _ in range(200):
    action = random.randint(0, 3)
    state, reward, done, _, _ = env.step(action)
    total_reward += reward
    if done:
        break
print(f"Random policy reward: {total_reward:.1f}")
```

Expected: **< 400** total reward (random safe-mode churning).

### 1d  Ray RLLib DQN

If you have Ray installed (`pip install ray[rllib]`), you can compare SENTINEL-X
to RLLib's reference DQN implementation on the same environment:

```python
import ray
from ray.rllib.algorithms.dqn import DQNConfig
from sentinel_x_advanced import SpacecraftEnv

# Register the env with Ray
ray.init()
config = (
    DQNConfig()
    .environment(SpacecraftEnv)
    .training(lr=1e-3, train_batch_size=32)
)
algo = config.build()
for i in range(50):
    result = algo.train()
    print(f"Ep {i}: reward={result['episode_reward_mean']:.1f}")
```

Compare wall-clock time and final reward to SENTINEL-X's `DQNAgent`.

---

## 2  Hardware Latency Comparison

The `scripts/collect_hardware_perf.py` script produces timing measurements
for direct comparison:

```bash
# SENTINEL-X int8 inference on STM32H7 (requires hardware)
python scripts/collect_hardware_perf.py \
    --port /dev/ttyACM0 \
    --timing-runs 1000 \
    --output results_stm32.json

# Simulation (no hardware)
python scripts/collect_hardware_perf.py \
    --timing-runs 1000 \
    --output results_sim.json
```

Reference table (see `docs/hardware_performance_results.md` for full data):

| Platform | Inference mode | p50 (ms) | p95 (ms) | p99 (ms) |
|---------|---------------|---------|---------|---------|
| STM32H7 @ 480 MHz | int8 TFLite Micro | 0.48 | 0.55 | 0.58 |
| STM32F4 @ 168 MHz | int8 TFLite Micro | 2.1 | 2.4 | 2.6 |
| Raspberry Pi 4B | tflite-runtime | 0.9 | 1.1 | 1.2 |
| Desktop (i7-12700) | TF eager (float32) | 0.3 | 0.4 | 0.5 |

Budget: **10 ms** (100 Hz control loop).  SENTINEL-X comfortably satisfies
this constraint on all tested platforms.

---

## 3  Fault-Recovery Rate

Run a fixed random seed evaluation across 500 episodes to get stable statistics:

```bash
python run_experiment.py \
    --scenario lunar_gateway \
    --episodes 500 \
    --eval-only \
    --seed 42 \
    --output eval_results.json
```

Metrics reported:
- `recovery_rate` – fraction of injected faults resolved within 10 steps
- `mean_episode_reward` – average total reward per episode
- `safety_violations` – number of SafetyMonitor vetoes that occurred

---

## 4  Formal Verification Comparison

| Method | Property checked | Result | Tool |
|--------|-----------------|--------|------|
| Marabou (DNN) | `power_level < 1.0 → action ≠ DO_NOTHING` | UNSAT (safe) | Marabou 2.0 |
| ERAN (abstract interp.) | L∞ robustness, ε = 0.1 | Certified | ERAN |
| Decision-tree surrogate | Exhaustive state enumeration | PASS | Scikit-learn |
| LTL SafetyMonitor | G(health_flag=True → F[0,5] action≠DO_NOTHING) | Enforced | runtime |

Run: `python -c "from sentinel_x_advanced import PolicyVerifier; ..."`
See `docs/formal_verification_external.md` for full instructions.

---

## 5  Adding a New Baseline

1. Implement `select_action(state: np.ndarray) -> int` in a new class
   under `examples/`.
2. Run the standard evaluation loop (Section 3 above) with your baseline.
3. Add a row to the comparison table in `docs/technical_report.md`
   Section 3 and open a PR.

See `CONTRIBUTING.md` for the pull-request checklist.
