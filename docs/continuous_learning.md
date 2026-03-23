# Continuous / Long-Term Autonomy

SENTINEL-X currently trains episodically before deployment.  This guide
describes how to extend it for **continuous on-board learning**, **mission
planning integration**, and **human-in-the-loop operation** — the three
pillars of long-term spacecraft autonomy.

---

## 1  Continuous / Online Policy Updates

### 1a  Concept

During a multi-year mission, the spacecraft will encounter fault signatures not
seen during ground training (component aging, micro-meteorite impacts, slow
sensor drift).  An online update mechanism allows the agent to adapt without a
full ground-uplinked retrain.

### 1b  Implementation sketch

```python
from sentinel_x_advanced import DQNAgent, SpacecraftEnv, SafetyMonitor

agent  = DQNAgent.load("sentinel_x_model_int8.tflite")   # pre-trained
env    = SpacecraftEnv()
safety = SafetyMonitor()

# --- Mission loop (runs continuously on-board) ---
FINETUNE_INTERVAL = 500   # steps between gradient updates
CONSERVATIVE_LR   = 1e-5  # small LR to avoid catastrophic forgetting

state, _ = env.reset()
step = 0

while True:                             # real mission: replace with cFS loop
    action = agent.act(state, epsilon=0.05)
    action = safety.veto(state, action)

    next_state, reward, done, _, info = env.step(action)
    agent.remember(state, action, reward, next_state, done)
    state = next_state if not done else env.reset()[0]
    step += 1

    # Periodic fine-tuning
    if step % FINETUNE_INTERVAL == 0 and len(agent.memory) >= agent.batch_size:
        agent.replay()                  # uses prioritised experience replay
        # Optionally: re-export TFLite and hot-swap the model flatbuffer
```

**Key safeguards:**
- Keep the pre-trained model frozen as a fallback; only update the online copy.
- Run `PolicyVerifier.verify_policy()` after each fine-tune step and revert
  if any LTL property is violated.
- Use a conservative learning rate (1 × 10⁻⁵) to prevent catastrophic
  forgetting of the ground-trained safety behaviour.

---

## 2  Mission Planning Integration

### 2a  Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Mission Planner                         │
│   (PDDL / goal-directed / timeline-based)                │
│                                                           │
│   Goal: max science return subject to health constraints  │
└────────────────────┬─────────────────────────────────────┘
                     │ high-level health constraint / mode
                     ▼
┌──────────────────────────────────────────────────────────┐
│            SENTINEL-X FDIR Agent (DQN / DT)              │
│   Observes: spacecraft state (11-dim)                     │
│   Acts: DO_NOTHING / RESTART / SWITCH_REDUNDANT / SAFE_MODE│
│                                                           │
│   Negotiation: can request SCIENCE_HOLD from planner     │
│   if subsystem health is below configurable threshold     │
└──────────────────────────────────────────────────────────┘
```

### 2b  Health-aware reward shaping

Pass the planner's current science priority as a context variable to the
reward function:

```python
class MissionAwareEnv(SpacecraftEnv):
    def __init__(self, science_priority: float = 1.0):
        super().__init__()
        self.science_priority = science_priority

    def step(self, action):
        obs, base_reward, done, truncated, info = super().step(action)
        # Scale recovery urgency against science return
        reward = base_reward + self.science_priority * info.get("science_value", 0)
        return obs, reward, done, truncated, info
```

Set `science_priority = 0.0` during high-radiation events when health
preservation dominates; set it to `1.0` during nominal science windows.

---

## 3  Human-in-the-Loop Decision Support

### 3a  Concept

Rather than acting autonomously, the agent proposes the top-2 actions with
confidence scores.  A ground operator can approve, override, or defer.  The
agent learns from operator decisions via reward shaping.

### 3b  Operator interface (CLI prototype)

```python
from sentinel_x_advanced import DQNAgent, SafetyMonitor
import numpy as np

ACTION_NAMES = ["DO_NOTHING", "RESTART", "SWITCH_REDUNDANT", "SAFE_MODE"]

def propose_action(agent: DQNAgent, safety: SafetyMonitor, state: np.ndarray):
    q_values   = agent.model.predict(state[np.newaxis], verbose=0)[0]
    ranked     = np.argsort(q_values)[::-1]
    top1, top2 = ranked[0], ranked[1]
    top1 = safety.veto(state, top1)

    print(f"\nAgent recommends : {ACTION_NAMES[top1]}  (Q={q_values[top1]:.3f})")
    print(f"  Alternative     : {ACTION_NAMES[top2]}  (Q={q_values[top2]:.3f})")

    choice = input("Accept [Enter] / override action index [0-3] / skip [s]: ").strip()
    if choice == "s":
        return None, None          # defer decision
    elif choice in ("0", "1", "2", "3"):
        operator_action = int(choice)
        # Shape reward: +1 if operator agrees, -0.5 if they override
        feedback = 1.0 if operator_action == top1 else -0.5
        return operator_action, feedback
    else:
        return top1, 0.0           # agent's recommendation accepted
```

### 3c  Feedback integration

```python
# After each operator decision, store the transition with shaped reward
feedback_reward = base_reward + feedback
agent.remember(state, operator_action, feedback_reward, next_state, done)
# Periodically retrain on the augmented buffer
if len(agent.memory) % 100 == 0:
    agent.replay()
```

---

## 4  Continuous Deployment Workflow

```
┌──────────────┐    uplink policy      ┌─────────────────────┐
│ Ground Ops   │──────────────────────►│ Spacecraft (on-board)│
│              │◄─────────────────────│                     │
│  Verify new  │    downlink telemetry │  Fine-tune locally  │
│  policy with │    + replay buffer    │  Verify with DT /   │
│  Marabou /   │                       │  LTL before deploy  │
│  ERAN        │                       │                     │
└──────────────┘                       └─────────────────────┘
```

**Uplink cadence:** Once per week (or after significant fault event) the
spacecraft downlinks its compressed replay buffer (≈ 100 KB for 1 000 transitions
at 11 floats each).  Ground verifies, fine-tunes, re-verifies, then uplinks the
updated flatbuffer.

**On-board verification gate:**
```python
from sentinel_x_advanced import PolicyVerifier, DQNAgent

agent    = DQNAgent.load("new_policy.tflite")
verifier = PolicyVerifier(agent)
report   = verifier.verify_policy(n_samples=500)

if report["ltl_pass"] and report["safety_pass"]:
    # Hot-swap the active policy
    os.replace("new_policy.tflite", "active_policy.tflite")
else:
    # Revert to previous policy; alert ground
    raise RuntimeError("Policy verification failed — retaining previous policy")
```

---

## 5  References

- Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural
  networks. *PNAS*, 114(13), 3521–3526.  *(Elastic Weight Consolidation)*
- Rashidinejad, P. et al. (2021). Bridging offline reinforcement learning and
  imitation learning. *NeurIPS 2021*.  *(Offline RL for safe fine-tuning)*
- Katz, G. et al. (2022). The Marabou framework for verification and analysis
  of deep neural networks. *CAV 2022*.
