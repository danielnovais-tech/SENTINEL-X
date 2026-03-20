"""
SENTINEL-X: Self-Healing Spacecraft Networks via Reinforcement Learning
=======================================================================
Basic Q-learning simulation for autonomous fault recovery.

Demonstrates a Spacecraft environment that experiences random faults and a
Q-learning RecoveryAgent that learns to choose recovery actions to maximise
operational time.
"""

import numpy as np
import random
from collections import defaultdict


# ------------------------------
# Simulation Environment
# ------------------------------
class Spacecraft:
    """A simplified spacecraft with internal states and faults."""

    def __init__(self):
        self.healthy = True
        self.fault_type = None   # None, 'memory_flip', 'sensor_stuck', 'cpu_hang'
        self.time_healthy = 0
        self.time_total = 0
        self.recovery_attempts = 0

    def step(self):
        """Advance one time step. Faults may occur naturally."""
        self.time_total += 1
        if self.healthy:
            self.time_healthy += 1
            # Random fault generation (simulate radiation etc.)
            if random.random() < 0.05:   # 5% chance per step
                self.healthy = False
                self.fault_type = random.choice(
                    ['memory_flip', 'sensor_stuck', 'cpu_hang']
                )

    def apply_recovery(self, action):
        """
        Attempt to recover the spacecraft.

        Actions:
            0: do nothing
            1: restart subsystem (clears memory flips, may fix cpu_hang)
            2: switch to redundant hardware (fixes all faults)
            3: safe mode (fixes memory_flip and sensor_stuck)
        """
        self.recovery_attempts += 1
        success = False

        if action == 0:
            pass
        elif action == 1:   # restart
            if self.fault_type in ['memory_flip', 'cpu_hang']:
                success = True
        elif action == 2:   # switch to redundant
            success = True   # redundant hardware covers all faults
        elif action == 3:   # safe mode
            if self.fault_type in ['memory_flip', 'sensor_stuck']:
                success = True

        if success:
            self.healthy = True
            self.fault_type = None
        return success

    def get_state(self):
        """Return a discrete state representation for the RL agent."""
        if self.healthy:
            return 0
        fault_map = {'memory_flip': 1, 'sensor_stuck': 2, 'cpu_hang': 3}
        return fault_map[self.fault_type]

    def is_operational(self):
        return self.healthy


# ------------------------------
# Q-Learning Agent
# ------------------------------
class RecoveryAgent:
    """Q-learning agent that selects recovery actions from discrete states."""

    def __init__(self, n_states=4, n_actions=4, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q = defaultdict(lambda: np.zeros(n_actions))
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        self.n_actions = n_actions

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q[state]))

    def learn(self, state, action, reward, next_state):
        """Update Q-value using the Q-learning update rule."""
        best_next = np.max(self.q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_error


# ------------------------------
# Training Loop
# ------------------------------
def train_agent(episodes=1000, steps_per_episode=200):
    """Train the Q-learning agent over a number of episodes."""
    agent = RecoveryAgent()
    episode_rewards = []

    for ep in range(episodes):
        spacecraft = Spacecraft()
        total_reward = 0
        state = spacecraft.get_state()

        for _ in range(steps_per_episode):
            spacecraft.step()
            new_state = spacecraft.get_state()

            if new_state != 0:   # fault present — agent decides action
                action = agent.choose_action(new_state)
                success = spacecraft.apply_recovery(action)
                reward = 10 if success else -2
            else:
                action = 0
                reward = 1   # small positive reward for staying healthy

            total_reward += reward
            next_state = spacecraft.get_state()
            agent.learn(state, action, reward, next_state)
            state = next_state

        episode_rewards.append(total_reward)
        if ep % 100 == 0:
            print(
                f"Episode {ep}: Avg Reward = "
                f"{np.mean(episode_rewards[-100:]):.2f}"
            )

    return agent


# ------------------------------
# Test the trained agent
# ------------------------------
def test_agent(agent, episodes=10):
    """Evaluate the trained agent on fresh episodes."""
    total_operational_time = 0
    for ep in range(episodes):
        spacecraft = Spacecraft()
        state = spacecraft.get_state()
        operational_steps = 0
        for _ in range(200):
            spacecraft.step()
            if spacecraft.is_operational():
                operational_steps += 1
            new_state = spacecraft.get_state()
            if new_state != 0:
                action = agent.choose_action(state)
                spacecraft.apply_recovery(action)
            state = new_state
        total_operational_time += operational_steps
        print(f"Test Episode {ep}: Operational steps = {operational_steps}")
    print(f"Average operational steps: {total_operational_time / episodes:.1f}")


# ------------------------------
# Rule-based baseline
# ------------------------------
def run_baseline(episodes=10):
    """Baseline policy: always switch to redundant hardware on any fault."""
    baseline_reward = 0
    for _ in range(episodes):
        sc = Spacecraft()
        steps_op = 0
        for _ in range(200):
            sc.step()
            if sc.is_operational():
                steps_op += 1
            if not sc.is_operational():
                sc.apply_recovery(2)   # action 2 = redundant switch
        baseline_reward += steps_op
    avg = baseline_reward / episodes
    print(f"Average operational steps: {avg:.1f}")
    return avg


# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    print("Training SENTINEL-X agent...")
    trained_agent = train_agent(episodes=500, steps_per_episode=150)

    print("\nTesting trained agent...")
    test_agent(trained_agent, episodes=10)

    print("\nRule-based baseline (always switch to redundant)...")
    run_baseline(episodes=10)
