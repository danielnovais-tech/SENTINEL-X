"""
SENTINEL-X Advanced: Realistic Faults, Deep Q-Network, and Swarm Simulation
============================================================================
Extends the basic Q-learning prototype with:

  1. Realistic fault generators:
       - MemoryArray – simulates single-event upsets (bit flips) in memory
       - Sensor      – models Gaussian noise and stuck-at faults

  2. Deep Q-Network (DQN) agent using TensorFlow/Keras with:
       - Experience replay buffer
       - Separate target network for stable training
       - Epsilon-greedy exploration with decay

  3. Swarm simulation – N spacecraft each managed by an independent DQN agent.

Install dependencies:
    pip install numpy tensorflow matplotlib
"""

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for servers/CI)
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------
# 1. Realistic Fault Generators
# -----------------------------------------------------------------------

class MemoryArray:
    """Simulates a memory region subject to single-event upsets (bit flips)."""

    def __init__(self, size_bits=1024, flip_rate_per_bit=1e-4):
        self.size_bits = size_bits
        self.flip_rate = flip_rate_per_bit
        self.data = np.zeros(size_bits, dtype=np.int8)
        self.error_count = 0

    def step(self):
        """Apply random bit flips based on flip_rate; return cumulative error count."""
        flips = np.random.random(self.size_bits) < self.flip_rate
        if np.any(flips):
            self.data[flips] ^= 1
            self.error_count += int(np.sum(flips))
        return self.error_count

    def check_parity(self):
        """Simple parity check over the entire memory word."""
        return int(np.sum(self.data)) % 2

    def reset(self):
        self.data.fill(0)
        self.error_count = 0


class Sensor:
    """Simulates a sensor with Gaussian noise and stuck-at faults."""

    def __init__(self, true_value=25.0, noise_std=0.5, stuck_prob=0.01):
        self.true_value = true_value
        self.noise_std = noise_std
        self.stuck_prob = stuck_prob
        self.stuck_value = None
        self.is_stuck = False

    def step(self):
        """Return the current sensor reading (may be stuck)."""
        if self.is_stuck:
            return self.stuck_value
        reading = self.true_value + np.random.normal(0, self.noise_std)
        if random.random() < self.stuck_prob:
            self.is_stuck = True
            self.stuck_value = reading
        return reading

    def reset(self):
        self.is_stuck = False
        self.stuck_value = None


# -----------------------------------------------------------------------
# 2. DQN Agent
# -----------------------------------------------------------------------

class DQNAgent:
    """
    Deep Q-Network agent with experience replay and a target network.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state vector.
    action_dim : int
        Number of discrete actions.
    learning_rate : float
    gamma : float
        Discount factor.
    epsilon : float
        Initial exploration probability.
    epsilon_min : float
        Minimum exploration probability after decay.
    epsilon_decay : float
        Multiplicative decay applied each replay step.
    memory_size : int
        Maximum number of transitions in the replay buffer.
    batch_size : int
        Number of transitions sampled per training step.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=2000,
        batch_size=32,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.model = self._build_model(learning_rate)
        self.target_model = self._build_model(learning_rate)
        self.update_target()

    def _build_model(self, lr):
        model = models.Sequential(
            [
                layers.Dense(64, activation="relu", input_shape=(self.state_dim,)),
                layers.Dense(64, activation="relu"),
                layers.Dense(self.action_dim, activation="linear"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse"
        )
        return model

    def update_target(self):
        """Copy weights from the online model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Select an action using epsilon-greedy policy."""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        return int(np.argmax(q_values))

    def replay(self):
        """Sample a mini-batch from memory and perform a gradient update."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])

        target_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])

        self.model.fit(states, target_q, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# -----------------------------------------------------------------------
# 3. Spacecraft with Realistic Faults
# -----------------------------------------------------------------------

class Spacecraft:
    """
    Spacecraft combining a MemoryArray and a Sensor subsystem.

    State vector (6 features, all normalised to [0, 1]):
        [mem_error_ratio, parity_flag, sensor_deviation_norm,
         sensor_stuck_flag, time_since_recovery_norm, health_flag]
    """

    def __init__(self, spacecraft_id=0):
        self.id = spacecraft_id
        self.memory = MemoryArray(size_bits=1024, flip_rate_per_bit=1e-4)
        self.sensor = Sensor(true_value=25.0, noise_std=0.5, stuck_prob=0.01)
        self.healthy = True
        self.time_healthy = 0
        self.time_total = 0
        self.recovery_attempts = 0
        self.last_recovery_step = 0
        self.step_count = 0

        # Normalisation constants
        self.max_memory_errors = 100
        self.max_sensor_deviation = 10.0

        # Initialise observable attributes so get_state() works before the
        # first call to step().
        self.mem_errors = 0
        self.sensor_reading = self.sensor.true_value
        self.parity_ok = True
        self.sensor_stuck = False

    def step(self):
        """Advance one time step: inject faults and update health status."""
        self.step_count += 1
        self.time_total += 1
        if self.healthy:
            self.time_healthy += 1

        self.mem_errors = self.memory.step()
        self.sensor_reading = self.sensor.step()

        self.parity_ok = self.memory.check_parity() == 0
        mem_too_many = self.mem_errors > 50
        self.sensor_stuck = self.sensor.is_stuck

        if mem_too_many or not self.parity_ok or self.sensor_stuck:
            self.healthy = False

    def get_state(self):
        """Return a normalised state vector for the DQN."""
        mem_err_norm = min(self.mem_errors / self.max_memory_errors, 1.0)
        parity_flag = 0 if self.parity_ok else 1
        dev = abs(self.sensor_reading - self.sensor.true_value)
        dev_norm = min(dev / self.max_sensor_deviation, 1.0)
        stuck_flag = 1 if self.sensor_stuck else 0
        steps_since = self.step_count - self.last_recovery_step
        time_since_norm = min(steps_since / 500.0, 1.0)
        health_flag = 0 if self.healthy else 1

        return np.array(
            [mem_err_norm, parity_flag, dev_norm, stuck_flag,
             time_since_norm, health_flag],
            dtype=np.float32,
        )

    def apply_recovery(self, action):
        """
        Apply a recovery action.

        Actions:
            0: do nothing
            1: restart subsystem  (resets memory and sensor)
            2: switch to redundant hardware (full reset)
            3: safe mode (targeted: fixes stuck sensor or parity error)
        """
        self.recovery_attempts += 1
        success = False

        if action == 0:
            pass
        elif action == 1:   # restart
            self.memory.reset()
            self.sensor.reset()
            success = True
        elif action == 2:   # switch to redundant
            self.memory.reset()
            self.sensor.reset()
            success = True
        elif action == 3:   # safe mode
            if self.sensor_stuck:
                self.sensor.reset()
                success = True
            if not self.parity_ok:
                self.memory.reset()
                success = True

        if success:
            self.healthy = True
            self.last_recovery_step = self.step_count

        return success

    def is_operational(self):
        return self.healthy

    def reset(self):
        """Re-initialise the spacecraft for a new episode."""
        self.__init__(spacecraft_id=self.id)


# -----------------------------------------------------------------------
# 4. Swarm Simulation
# -----------------------------------------------------------------------

class Swarm:
    """
    A swarm of spacecraft, each managed by an independent DQN agent.

    Parameters
    ----------
    num_spacecraft : int
        Number of spacecraft in the swarm.
    state_dim : int
        Dimensionality of the state vector.
    action_dim : int
        Number of discrete recovery actions.
    """

    def __init__(self, num_spacecraft, state_dim, action_dim):
        self.spacecraft = [Spacecraft(i) for i in range(num_spacecraft)]
        self.agents = [DQNAgent(state_dim, action_dim) for _ in range(num_spacecraft)]

    def train_episode(self, max_steps=200):
        """Run one training episode for the entire swarm."""
        for sc in self.spacecraft:
            sc.reset()

        episode_rewards = [0.0] * len(self.spacecraft)

        for step in range(max_steps):
            for i, sc in enumerate(self.spacecraft):
                state = sc.get_state()
                action = self.agents[i].act(state, training=True)
                sc.step()

                # Apply recovery if a fault is present
                was_healthy = sc.is_operational()
                if not sc.is_operational():
                    sc.apply_recovery(action)
                new_state = sc.get_state()

                # Reward: operational → +1, fault persists → -1;
                # bonus for a recovery action that actually restored health.
                now_operational = sc.is_operational()
                reward = 1.0 if now_operational else -1.0
                if action != 0 and now_operational and not was_healthy:
                    reward += 5.0   # successful recovery bonus
                elif action != 0 and not now_operational:
                    reward -= 2.0   # failed recovery penalty

                episode_rewards[i] += reward
                done = step == max_steps - 1
                self.agents[i].remember(state, action, reward, new_state, done)
                self.agents[i].replay()

            if step % 10 == 0:
                for agent in self.agents:
                    agent.update_target()

        return float(np.mean(episode_rewards))

    def test_episode(self, max_steps=200):
        """Evaluate the swarm without exploration; returns avg operational steps."""
        for sc in self.spacecraft:
            sc.reset()

        total_operational = 0

        for _ in range(max_steps):
            for i, sc in enumerate(self.spacecraft):
                state = sc.get_state()
                action = self.agents[i].act(state, training=False)
                sc.step()
                if sc.is_operational():
                    total_operational += 1

        return total_operational / len(self.spacecraft)


# -----------------------------------------------------------------------
# 5. Main Training Loop
# -----------------------------------------------------------------------

def run_do_nothing_baseline(episodes=10, max_steps=200):
    """Baseline: never attempt recovery — let faults accumulate."""
    total_op = 0
    for _ in range(episodes):
        sc = Spacecraft()
        for _ in range(max_steps):
            sc.step()
            if sc.is_operational():
                total_op += 1
    avg = total_op / episodes
    print(f"Baseline (do nothing) avg operational steps: {avg:.1f}")
    return avg


if __name__ == "__main__":
    STATE_DIM = 6
    ACTION_DIM = 4
    SWARM_SIZE = 5
    EPISODES = 200
    MAX_STEPS = 150

    swarm = Swarm(SWARM_SIZE, STATE_DIM, ACTION_DIM)
    rewards_per_episode = []

    print("Training swarm...")
    for ep in range(EPISODES):
        avg_reward = swarm.train_episode(max_steps=MAX_STEPS)
        rewards_per_episode.append(avg_reward)
        if ep % 20 == 0:
            print(f"Episode {ep}: Avg Reward = {avg_reward:.2f}")

    print("\nTesting trained agents...")
    test_operational = swarm.test_episode(max_steps=200)
    print(f"Avg operational steps per spacecraft: {test_operational:.1f}")

    print("\nDo-nothing baseline...")
    run_do_nothing_baseline(episodes=10, max_steps=200)

    # Save training curve to a file (non-interactive environments safe)
    plt.figure()
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("SENTINEL-X Swarm Training")
    plt.tight_layout()
    plt.savefig("sentinel_x_training_curve.png")
    print("\nTraining curve saved to sentinel_x_training_curve.png")
