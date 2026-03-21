"""
SENTINEL-X Advanced: Realistic Faults, DQN, Swarm, Federated Learning & Coordination
======================================================================================
Extends the basic Q-learning prototype with:

  1. Realistic fault generators:
       - MemoryArray – simulates single-event upsets (bit flips) in memory
       - Sensor      – models Gaussian noise and stuck-at faults

  2. Deep Q-Network (DQN) agent using TensorFlow/Keras with:
       - Experience replay buffer
       - Separate target network for stable training
       - Epsilon-greedy exploration with decay

  3. Swarm simulation – N spacecraft each managed by an independent DQN agent.

  4. Mission-specific reward profiles – tailor optimisation to mission priorities
       (balanced / maximise data return / extend lifespan).

  5. Federated learning – a central FederatedServer averages DQN weights across
       all agents (FedAvg) at configurable intervals.

  6. Multi-agent coordination – spacecraft cross-check peer sensor readings to
       detect stuck sensors that an individual agent cannot diagnose alone.

  7. TFLite export – convert a trained DQN to TensorFlow Lite for deployment on
       microcontrollers and embedded hardware.

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

    Base state vector (6 features, all normalised to [0, 1]):
        [mem_error_ratio, parity_flag, sensor_deviation_norm,
         sensor_stuck_flag, time_since_recovery_norm, health_flag]

    Note: FederatedSwarm extends this to a 7-feature vector by appending a
    peer_sensor_deviation_norm coordination feature via
    ``_get_coordinated_state()``.
    """

    def __init__(self, spacecraft_id=0):
        self.id = spacecraft_id
        self._reset_state()

    def _reset_state(self):
        """Initialise (or re-initialise) all mutable state for a new episode."""
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
        self._reset_state()


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
# 5. Mission-Specific Reward Profiles
# -----------------------------------------------------------------------

class MissionProfile:
    """
    Encapsulates reward shaping based on mission objectives.

    Three built-in profiles are provided:

    * **BALANCED** (default)
        General-purpose recovery. Equal weight on uptime and recovery cost.

    * **MAXIMIZE_DATA_RETURN**
        Prioritise operational time; penalise inaction during faults and
        incentivise fast recovery regardless of resource cost.

    * **EXTEND_LIFESPAN**
        Preserve redundant hardware. Prefer cheap recovery actions (restart,
        safe mode) and accept brief downtime to avoid wearing out spares.
    """

    BALANCED = "balanced"
    MAXIMIZE_DATA_RETURN = "data_return"
    EXTEND_LIFESPAN = "lifespan"

    def __init__(self, profile: str = BALANCED):
        if profile not in (self.BALANCED, self.MAXIMIZE_DATA_RETURN, self.EXTEND_LIFESPAN):
            raise ValueError(
                f"Unknown profile '{profile}'. Choose from: "
                f"{self.BALANCED}, {self.MAXIMIZE_DATA_RETURN}, {self.EXTEND_LIFESPAN}"
            )
        self.profile = profile

    def compute(self, was_healthy: bool, now_operational: bool, action: int) -> float:
        """
        Return a scalar reward.

        Parameters
        ----------
        was_healthy : bool
            Spacecraft health *before* the recovery action was applied.
        now_operational : bool
            Spacecraft health *after* the recovery action.
        action : int
            Recovery action (0 = do nothing, 1 = restart,
            2 = switch to redundant, 3 = safe mode).
        """
        if self.profile == self.MAXIMIZE_DATA_RETURN:
            # High reward for being up; penalise lingering faults and inaction.
            reward = 2.0 if now_operational else -2.0
            if not was_healthy and now_operational:
                reward += 8.0        # fast recovery greatly rewarded
            elif not now_operational and not was_healthy:
                if action == 0:
                    reward -= 1.0    # penalise doing nothing when faulty
                else:
                    reward -= 3.0    # failed recovery attempt
            return reward

        elif self.profile == self.EXTEND_LIFESPAN:
            # Balanced uptime; steer agent toward cheap recovery actions.
            reward = 1.0 if now_operational else -1.0
            if not was_healthy and now_operational:
                reward += 5.0
                if action in (1, 3):   # restart or safe mode – cheap
                    reward += 1.0
                elif action == 2:       # redundant switch – expensive
                    reward -= 1.0
            elif action != 0 and not now_operational:
                reward -= 2.0
            return reward

        else:  # BALANCED (default)
            reward = 1.0 if now_operational else -1.0
            if action != 0 and now_operational and not was_healthy:
                reward += 5.0
            elif action != 0 and not now_operational:
                reward -= 2.0
            return reward


# -----------------------------------------------------------------------
# 6. Federated Learning Server
# -----------------------------------------------------------------------

class FederatedServer:
    """
    Central server that aggregates DQN weights across a swarm (FedAvg).

    After aggregation every agent receives the same globally averaged
    weights, promoting convergence while still allowing each agent to
    continue learning from its own local experience.
    """

    def aggregate(self, agents: list) -> None:
        """
        Average online-network weights across all agents and redistribute.

        The target network of each agent is also synchronised so that the
        next round of local training starts from a consistent baseline.

        Parameters
        ----------
        agents : list[DQNAgent]
            All agents participating in this aggregation round.
        """
        if not agents:
            return
        all_weights = [agent.model.get_weights() for agent in agents]
        n_layers = len(all_weights[0])
        avg_weights = [
            np.mean([all_weights[a][layer] for a in range(len(agents))], axis=0)
            for layer in range(n_layers)
        ]
        for agent in agents:
            agent.model.set_weights(avg_weights)
            agent.update_target()


# -----------------------------------------------------------------------
# 7. Federated Swarm with Multi-Agent Coordination
# -----------------------------------------------------------------------

class FederatedSwarm(Swarm):
    """
    Extends Swarm with federated learning, sensor cross-checking, and
    mission-profile reward shaping.

    The state vector is extended to 7 features:

        [mem_error_ratio, parity_flag, sensor_deviation_norm,
         sensor_stuck_flag, time_since_recovery_norm, health_flag,
         peer_sensor_deviation_norm]   ← coordination feature

    Parameters
    ----------
    num_spacecraft : int
        Number of spacecraft in the swarm.
    action_dim : int
        Number of discrete recovery actions.
    mission_profile : MissionProfile, optional
        Reward-shaping strategy. Defaults to MissionProfile.BALANCED.
    federated_interval : int
        Number of episodes between federated weight aggregations.
    """

    COORD_STATE_DIM = 7   # base 6 features + 1 peer-sensor deviation feature

    def __init__(
        self,
        num_spacecraft: int,
        action_dim: int,
        mission_profile: MissionProfile = None,
        federated_interval: int = 10,
    ):
        super().__init__(num_spacecraft, self.COORD_STATE_DIM, action_dim)
        self.server = FederatedServer()
        self.mission_profile = mission_profile or MissionProfile()
        self.federated_interval = federated_interval
        self._episode_count = 0
        for sc in self.spacecraft:
            sc._peer_sensor_deviation = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cross_check_sensors(self) -> None:
        """
        Compute each spacecraft's sensor deviation from the swarm median.

        A spacecraft whose reading is an outlier among its peers is likely
        experiencing a stuck-sensor fault, even if it cannot detect this
        from its own readings alone.  The per-spacecraft
        ``_peer_sensor_deviation`` attribute becomes the 7th state feature
        fed to the DQN, enabling coordinated fault diagnosis.
        """
        readings = np.array(
            [sc.sensor_reading for sc in self.spacecraft], dtype=np.float32
        )
        if len(readings) < 2:
            for sc in self.spacecraft:
                sc._peer_sensor_deviation = 0.0
            return
        median = float(np.median(readings))
        for sc in self.spacecraft:
            dev = abs(sc.sensor_reading - median)
            sc._peer_sensor_deviation = float(
                min(dev / sc.max_sensor_deviation, 1.0)
            )

    def _get_coordinated_state(self, sc: Spacecraft) -> np.ndarray:
        """Return the 7-feature state vector for *sc*, including peer deviation."""
        base = sc.get_state()                                          # shape (6,)
        peer = np.float32(getattr(sc, "_peer_sensor_deviation", 0.0))
        return np.append(base, peer)

    # ------------------------------------------------------------------
    # Training / testing overrides
    # ------------------------------------------------------------------

    def train_episode(self, max_steps: int = 200) -> float:
        """
        Run one training episode for the federated swarm.

        All spacecraft step simultaneously each tick so that cross-sensor
        readings are always synchronised before actions are selected.
        Federated weight aggregation is performed every
        ``federated_interval`` episodes.
        """
        for sc in self.spacecraft:
            sc.reset()
            sc._peer_sensor_deviation = 0.0

        episode_rewards = [0.0] * len(self.spacecraft)

        for step in range(max_steps):
            # Observe coordinated states (pre-step cross-check)
            self._cross_check_sensors()
            states = [self._get_coordinated_state(sc) for sc in self.spacecraft]
            actions = [
                self.agents[i].act(states[i], training=True)
                for i in range(len(self.spacecraft))
            ]

            # All spacecraft advance one time step together
            for sc in self.spacecraft:
                sc.step()

            # Cross-check after step; apply recovery actions
            self._cross_check_sensors()
            was_healthy = [sc.is_operational() for sc in self.spacecraft]
            for i, sc in enumerate(self.spacecraft):
                if not sc.is_operational():
                    sc.apply_recovery(actions[i])

            # Observe new states and compute mission-shaped rewards
            for i, sc in enumerate(self.spacecraft):
                new_state = self._get_coordinated_state(sc)
                now_operational = sc.is_operational()
                reward = self.mission_profile.compute(
                    was_healthy[i], now_operational, actions[i]
                )
                episode_rewards[i] += reward
                done = step == max_steps - 1
                self.agents[i].remember(states[i], actions[i], reward, new_state, done)
                self.agents[i].replay()

            if step % 10 == 0:
                for agent in self.agents:
                    agent.update_target()

        self._episode_count += 1
        if self._episode_count % self.federated_interval == 0:
            self.server.aggregate(self.agents)

        return float(np.mean(episode_rewards))

    def test_episode(self, max_steps: int = 200) -> float:
        """Evaluate the swarm (no exploration); returns avg operational steps."""
        for sc in self.spacecraft:
            sc.reset()
            sc._peer_sensor_deviation = 0.0

        total_operational = 0
        for _ in range(max_steps):
            self._cross_check_sensors()
            for i, sc in enumerate(self.spacecraft):
                state = self._get_coordinated_state(sc)
                action = self.agents[i].act(state, training=False)
                sc.step()
                if not sc.is_operational():
                    sc.apply_recovery(action)
                if sc.is_operational():
                    total_operational += 1

        return total_operational / len(self.spacecraft)


# -----------------------------------------------------------------------
# 8. TFLite Export for Embedded / Microcontroller Deployment
# -----------------------------------------------------------------------

def export_tflite(agent: DQNAgent, output_path: str = "sentinel_x_agent.tflite") -> bytes:
    """
    Convert a trained DQNAgent to TensorFlow Lite for embedded hardware.

    Applies default optimisations (dynamic-range weight quantisation) so the
    resulting model is suitable for microcontrollers with limited flash and
    RAM budgets.

    Parameters
    ----------
    agent : DQNAgent
        A fully trained agent whose ``model`` attribute will be converted.
    output_path : str
        File path where the ``.tflite`` model will be written.

    Returns
    -------
    bytes
        The serialised TFLite flatbuffer (also written to *output_path*).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(agent.model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"TFLite model exported to '{output_path}' ({size_kb:.1f} KB)")
    return tflite_model


# -----------------------------------------------------------------------
# 9. Main Training Loop
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
    ACTION_DIM = 4
    SWARM_SIZE = 5
    EPISODES = 200
    MAX_STEPS = 150

    # ------------------------------------------------------------------
    # 1. Basic swarm (independent agents, no federation)
    # ------------------------------------------------------------------
    STATE_DIM = 6
    swarm = Swarm(SWARM_SIZE, STATE_DIM, ACTION_DIM)
    rewards_per_episode = []

    print("Training basic swarm...")
    for ep in range(EPISODES):
        avg_reward = swarm.train_episode(max_steps=MAX_STEPS)
        rewards_per_episode.append(avg_reward)
        if ep % 20 == 0:
            print(f"Episode {ep}: Avg Reward = {avg_reward:.2f}")

    print("\nTesting basic swarm...")
    test_operational = swarm.test_episode(max_steps=200)
    print(f"Avg operational steps per spacecraft: {test_operational:.1f}")

    print("\nDo-nothing baseline...")
    run_do_nothing_baseline(episodes=10, max_steps=200)

    # ------------------------------------------------------------------
    # 2. Federated swarm – mission profiles + coordination + TFLite
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Federated swarm with mission-specific reward profiles")
    print("=" * 60)

    all_fed_rewards = {}
    for profile_name in (
        MissionProfile.BALANCED,
        MissionProfile.MAXIMIZE_DATA_RETURN,
        MissionProfile.EXTEND_LIFESPAN,
    ):
        print(f"\nProfile: {profile_name}")
        fed_swarm = FederatedSwarm(
            num_spacecraft=SWARM_SIZE,
            action_dim=ACTION_DIM,
            mission_profile=MissionProfile(profile_name),
            federated_interval=10,
        )
        profile_rewards = []
        for ep in range(EPISODES):
            avg_reward = fed_swarm.train_episode(max_steps=MAX_STEPS)
            profile_rewards.append(avg_reward)
            if ep % 40 == 0:
                print(f"  Episode {ep}: Avg Reward = {avg_reward:.2f}")

        fed_ops = fed_swarm.test_episode(max_steps=200)
        print(f"  Avg operational steps: {fed_ops:.1f}")
        all_fed_rewards[profile_name] = profile_rewards

    # ------------------------------------------------------------------
    # 3. Export one trained agent to TFLite for embedded deployment
    # ------------------------------------------------------------------
    print("\nExporting agent to TFLite for microcontroller deployment...")
    export_tflite(fed_swarm.agents[0], output_path="sentinel_x_agent.tflite")

    # ------------------------------------------------------------------
    # 4. Save training curves (basic swarm + all three fed profiles)
    # ------------------------------------------------------------------
    plt.figure()
    plt.plot(rewards_per_episode, label="Basic swarm")
    for pname, rewards in all_fed_rewards.items():
        plt.plot(rewards, label=f"Federated ({pname})")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("SENTINEL-X Swarm Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sentinel_x_training_curve.png")
    print("\nTraining curve saved to sentinel_x_training_curve.png")
