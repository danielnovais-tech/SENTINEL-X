"""
SENTINEL-X Advanced: Realistic Faults, DQN, Swarm, Federated Learning & Coordination
======================================================================================
Extends the basic Q-learning prototype with:

  1. Realistic fault generators:
       - MemoryArray              – single-event upsets (bit flips) in memory
       - Sensor                   – Gaussian noise and stuck-at faults
       - ThermalSubsystem         – temperature random-walk; overheating /
                                    overcooling fault detection
       - PowerSubsystem           – power-budget drain model; brownout fault
       - AttitudeControlSubsystem – gyro-drift random walk; tumble detection
       - CommSubsystem            – antenna link-quality degradation and
                                    stochastic dropout fault detection

  2. Deep Q-Network (DQN) agent using TensorFlow/Keras with:
       - Experience replay buffer
       - Separate target network for stable training
       - Epsilon-greedy exploration with decay

  3. Swarm simulation – N spacecraft each managed by an independent DQN agent.

  4. Mission-specific reward profiles – tailor optimisation to mission priorities
       (balanced / maximise data return / extend lifespan / power-constrained).
       Dynamic reward shaping lets operators adjust weights at runtime.

  5. Federated learning – a central FederatedServer averages DQN weights across
       all agents (FedAvg) at configurable intervals, with optional deep-space
       communication latency (comm_delay_steps) and stochastic link dropout
       (link_dropout_prob).

  6. Gossip-based federated learning – GossipServer provides a decentralised
       alternative where each spacecraft shares weights with k random neighbours,
       scaling to large constellations without a central aggregation point.

  7. Multi-agent coordination – spacecraft cross-check peer sensor readings to
       detect stuck sensors that an individual agent cannot diagnose alone.

  8. TFLite export – convert a trained DQN to TensorFlow Lite for deployment on
       microcontrollers; includes full integer-only (int8) quantisation mode for
       deterministic inference on Cortex-M / FPGA platforms.

  9. Formal verification – PolicyVerifier runs lightweight safety proofs over
       the trained policy: safety constraints, Q-value margin bounds, and action
       coverage across fault-state samples.

  10. Safety monitor – SafetyMonitor is a rule-based FDIR veto layer that
       overrides DQN actions violating hard spacecraft safety constraints,
       enabling a hybrid RL + deterministic architecture.  Pass a
       ``SafetyMonitor`` instance to ``FederatedSwarm`` via the
       ``safety_monitor`` parameter to integrate veto into the training loop:
       the agent learns from vetoed actions and the per-episode override count
       is available via ``FederatedSwarm.last_override_count``.

  11. Adversarial testing – AdversarialTester finds minimal L∞-bounded state
       perturbations (FGSM) that flip the greedy action, revealing policy
       fragility and generating adversarial training examples.

  12. Adversarial training – AdversarialTester.augment_replay_buffer() injects
       adversarial transitions into agent replay buffers, hardening the policy
       against corner-case perturbations.

  13. Robustness certification – AdversarialTester.certify_robustness() uses
       binary search to estimate the per-state minimum perturbation magnitude
       required to flip the greedy action (certified robustness radius).

  14. Federation benchmark – benchmark_federation() compares centralised FedAvg
       against decentralised gossip under varying link-dropout rates and prints
       a formatted summary table.

  15. LTL reward shaping – LTLConstraintChecker encodes safety properties as
       Linear Temporal Logic (LTL)-style state predicates.  At each training
       step any active LTL violation incurs a configurable penalty, turning
       hard safety rules into a soft constrained-RL signal.

  16. Decision-tree policy extraction – extract_decision_tree() trains a
       scikit-learn DecisionTreeClassifier to imitate the DQN greedy policy
       (imitation learning), producing a certifiable surrogate that can be
       inspected, exhaustively tested, and submitted to formal tools.

  17. Mission scenarios – MissionScenario bundles fault-model parameters and
       reward-profile settings for three real mission concepts:
       Lunar Gateway (HALO node), Mars Sample Return orbiter, and CubeSat
       swarm (low Earth orbit constellation).
       build_swarm_for_scenario() returns a ready-to-train FederatedSwarm.

Install dependencies:
    pip install numpy tensorflow matplotlib scikit-learn
"""
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for servers/CI)
import matplotlib.pyplot as plt

try:
    from sklearn.tree import DecisionTreeClassifier
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

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


class ThermalSubsystem:
    """
    Simulates spacecraft thermal control with overheating and overcooling faults.

    The temperature undergoes a Gaussian random walk each step with occasional
    large spikes representing eclipse transitions or heater failures.  A fault
    is declared when the temperature drifts outside the safe operating range.
    """

    NOMINAL_TEMP = 20.0      # °C
    FAULT_TEMP_HIGH = 75.0   # °C – overheating threshold
    FAULT_TEMP_LOW = -20.0   # °C – overcooling threshold

    def __init__(
        self,
        nominal_temp: float = NOMINAL_TEMP,
        fault_temp_high: float = FAULT_TEMP_HIGH,
        fault_temp_low: float = FAULT_TEMP_LOW,
        drift_std: float = 0.1,
        spike_prob: float = 0.005,
    ):
        self.nominal_temp = nominal_temp
        self.fault_temp_high = fault_temp_high
        self.fault_temp_low = fault_temp_low
        self.drift_std = drift_std
        self.spike_prob = spike_prob
        self.temperature = nominal_temp
        self.is_faulted = False

    def step(self) -> float:
        """Advance thermal state; return current temperature."""
        self.temperature += np.random.normal(0.0, self.drift_std)
        if random.random() < self.spike_prob:
            # Random thermal spike: positive (overheating) or negative (cooling)
            self.temperature += float(
                np.random.choice([-1, 1]) * np.random.uniform(5.0, 20.0)
            )
        if (self.temperature > self.fault_temp_high
                or self.temperature < self.fault_temp_low):
            self.is_faulted = True
        return self.temperature

    @property
    def fault_flag(self) -> int:
        """1 if the thermal system is currently faulted, else 0."""
        return 1 if self.is_faulted else 0

    def reset(self) -> None:
        self.temperature = self.nominal_temp
        self.is_faulted = False


class PowerSubsystem:
    """
    Simulates spacecraft power budget with gradual depletion and brownout faults.

    The charge level follows a near-zero net balance (generation ≈ consumption)
    with occasional sudden drain events representing solar-panel shadowing or
    high-power fault responses.  A brownout fault is declared when the charge
    drops below ``fault_threshold``.
    """

    def __init__(
        self,
        capacity: float = 100.0,
        drain_rate: float = 0.10,
        recharge_rate: float = 0.08,
        event_prob: float = 0.003,
        fault_threshold: float = 10.0,
    ):
        self.capacity = capacity
        self.drain_rate = drain_rate
        self.recharge_rate = recharge_rate
        self.event_prob = event_prob
        self.fault_threshold = fault_threshold
        self.charge = capacity
        self.is_faulted = False

    def step(self) -> float:
        """Advance power state; return current charge level."""
        net = (self.recharge_rate - self.drain_rate
               + np.random.normal(0.0, 0.02))
        self.charge = float(np.clip(self.charge + net, 0.0, self.capacity))
        if random.random() < self.event_prob:
            # Sudden power drain event (e.g., solar panel shadowing)
            self.charge = float(
                np.clip(
                    self.charge - np.random.uniform(5.0, 20.0),
                    0.0, self.capacity,
                )
            )
        if self.charge < self.fault_threshold:
            self.is_faulted = True
        return self.charge

    @property
    def level_norm(self) -> float:
        """Normalised charge in [0, 1]."""
        return self.charge / self.capacity

    def reset(self) -> None:
        self.charge = self.capacity
        self.is_faulted = False


class AttitudeControlSubsystem:
    """
    Simulates spacecraft attitude control with gyro drift and tumble faults.

    The angular rate vector undergoes a Gaussian random walk each step.
    Occasional torque impulses (reaction-wheel saturation or disturbance
    torques) can cause rapid spin-up.  A tumble fault is declared when the
    magnitude of the angular rate exceeds the detumbling threshold.
    """

    DETUMBLE_THRESHOLD = 5.0   # deg/s - rate above which tumble is declared

    def __init__(
        self,
        detumble_threshold: float = DETUMBLE_THRESHOLD,
        drift_std: float = 0.02,
        impulse_prob: float = 0.004,
    ):
        self.detumble_threshold = detumble_threshold
        self.drift_std = drift_std
        self.impulse_prob = impulse_prob
        self.angular_rate = np.zeros(3, dtype=np.float64)   # deg/s per axis
        self.is_faulted = False

    def step(self) -> float:
        """Advance attitude state; return angular rate magnitude."""
        self.angular_rate += np.random.normal(0.0, self.drift_std, size=3)
        if random.random() < self.impulse_prob:
            axis = random.randrange(3)
            self.angular_rate[axis] += float(
                np.random.choice([-1, 1]) * np.random.uniform(2.0, 8.0)
            )
        magnitude = float(np.linalg.norm(self.angular_rate))
        if magnitude > self.detumble_threshold:
            self.is_faulted = True
        return magnitude

    @property
    def rate_norm(self) -> float:
        """Angular rate magnitude normalised to [0, 1] (clipped at 2x threshold)."""
        return float(
            min(np.linalg.norm(self.angular_rate) / (2.0 * self.detumble_threshold), 1.0)
        )

    def reset(self) -> None:
        self.angular_rate = np.zeros(3, dtype=np.float64)
        self.is_faulted = False


class CommSubsystem:
    """
    Simulates spacecraft communication link quality with dropout faults.

    The link quality undergoes a slow random walk representing atmospheric
    conditions, deep-space ranging geometry, and hardware ageing.  Sudden
    link-dropout events (antenna mispointing, relay loss) can drive quality
    to zero.  A fault is declared when quality drops below the threshold.
    """

    def __init__(
        self,
        drift_std: float = 0.005,
        dropout_prob: float = 0.004,
        fault_threshold: float = 0.2,
    ):
        self.drift_std = drift_std
        self.dropout_prob = dropout_prob
        self.fault_threshold = fault_threshold
        self.link_quality = 1.0   # 1.0 = perfect link, 0.0 = no link
        self.is_faulted = False

    def step(self) -> float:
        """Advance comm state; return current link quality."""
        self.link_quality += np.random.normal(0.0, self.drift_std)
        self.link_quality = float(np.clip(self.link_quality, 0.0, 1.0))
        if random.random() < self.dropout_prob:
            # Sudden link degradation event
            self.link_quality = float(
                np.clip(
                    self.link_quality - np.random.uniform(0.3, 0.8),
                    0.0, 1.0,
                )
            )
        if self.link_quality < self.fault_threshold:
            self.is_faulted = True
        return self.link_quality

    @property
    def quality_norm(self) -> float:
        """Link quality already in [0, 1]."""
        return self.link_quality

    def reset(self) -> None:
        self.link_quality = 1.0
        self.is_faulted = False



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
    Spacecraft combining MemoryArray, Sensor, ThermalSubsystem, PowerSubsystem,
    AttitudeControlSubsystem, and CommSubsystem fault subsystems.

    Base state vector (10 features, all normalised to [0, 1]):
        [0] mem_error_ratio
        [1] parity_flag
        [2] sensor_deviation_norm
        [3] sensor_stuck_flag
        [4] time_since_recovery_norm
        [5] health_flag               <- SafetyMonitor / PolicyVerifier key index
        [6] thermal_fault_flag
        [7] power_level_norm
        [8] attitude_rate_norm
        [9] comm_quality_norm

    Note: FederatedSwarm extends this to an 11-feature vector by appending a
    peer_sensor_deviation_norm coordination feature via
    ``_get_coordinated_state()``.
    """

    HEALTH_FLAG_IDX = 5   # position of health_flag in the base state vector

    def __init__(
        self,
        spacecraft_id: int = 0,
        flip_rate_per_bit: float = 1e-4,
        sensor_stuck_prob: float = 0.01,
        thermal_drift_std: float = 0.1,
        thermal_spike_prob: float = 0.005,
        power_drain_rate: float = 0.10,
    ):
        self.id = spacecraft_id
        # Store fault-model parameters so they survive reset() calls.
        # build_swarm_for_scenario sets these at construction time.
        self._flip_rate_per_bit = flip_rate_per_bit
        self._sensor_stuck_prob = sensor_stuck_prob
        self._thermal_drift_std = thermal_drift_std
        self._thermal_spike_prob = thermal_spike_prob
        self._power_drain_rate = power_drain_rate
        self._reset_state()

    def _reset_state(self):
        """Initialise (or re-initialise) all mutable state for a new episode."""
        self.memory = MemoryArray(
            size_bits=1024, flip_rate_per_bit=self._flip_rate_per_bit
        )
        self.sensor = Sensor(
            true_value=25.0,
            noise_std=0.5,
            stuck_prob=self._sensor_stuck_prob,
        )
        self.thermal = ThermalSubsystem(
            drift_std=self._thermal_drift_std,
            spike_prob=self._thermal_spike_prob,
        )
        self.power = PowerSubsystem(drain_rate=self._power_drain_rate)
        self.attitude = AttitudeControlSubsystem()
        self.comm = CommSubsystem()
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
        self.thermal.step()
        self.power.step()
        self.attitude.step()
        self.comm.step()

        self.parity_ok = self.memory.check_parity() == 0
        mem_too_many = self.mem_errors > 50
        self.sensor_stuck = self.sensor.is_stuck

        if (mem_too_many or not self.parity_ok or self.sensor_stuck
                or self.thermal.is_faulted or self.power.is_faulted
                or self.attitude.is_faulted or self.comm.is_faulted):
            self.healthy = False

    def get_state(self):
        """Return a normalised 10-feature state vector for the DQN."""
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
             time_since_norm, health_flag,
             self.thermal.fault_flag, self.power.level_norm,
             self.attitude.rate_norm, self.comm.quality_norm],
            dtype=np.float32,
        )

    def apply_recovery(self, action):
        """
        Apply a recovery action.

        Actions:
            0: do nothing
            1: restart subsystem  (resets memory, sensor, thermal, attitude,
                                   and comm subsystems)
            2: switch to redundant hardware (full reset of all subsystems)
            3: safe mode (targeted: fixes stuck sensor, parity error, thermal,
                          attitude, comm, or low-power condition)
        """
        self.recovery_attempts += 1
        success = False

        if action == 0:
            pass
        elif action == 1:   # restart
            self.memory.reset()
            self.sensor.reset()
            self.thermal.reset()
            self.attitude.reset()
            self.comm.reset()
            success = True
        elif action == 2:   # switch to redundant
            self.memory.reset()
            self.sensor.reset()
            self.thermal.reset()
            self.power.reset()
            self.attitude.reset()
            self.comm.reset()
            success = True
        elif action == 3:   # safe mode
            if self.sensor_stuck:
                self.sensor.reset()
                success = True
            if not self.parity_ok:
                self.memory.reset()
                success = True
            if self.thermal.is_faulted:
                self.thermal.reset()
                success = True
            if self.power.is_faulted:
                self.power.reset()
                success = True
            if self.attitude.is_faulted:
                self.attitude.reset()
                success = True
            if self.comm.is_faulted:
                self.comm.reset()
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

    Four built-in profiles are provided:

    * **BALANCED** (default)
        General-purpose recovery. Equal weight on uptime and recovery cost.

    * **MAXIMIZE_DATA_RETURN**
        Prioritise operational time; penalise inaction during faults and
        incentivise fast recovery regardless of resource cost.

    * **EXTEND_LIFESPAN**
        Preserve redundant hardware. Prefer cheap recovery actions (restart,
        safe mode) and accept brief downtime to avoid wearing out spares.

    * **POWER_CONSTRAINED**
        Optimise for power budgets. Penalise redundant hardware switches
        (high current draw) and reward power-efficient recovery actions.
    """

    BALANCED = "balanced"
    MAXIMIZE_DATA_RETURN = "data_return"
    EXTEND_LIFESPAN = "lifespan"
    POWER_CONSTRAINED = "power_constrained"

    def __init__(self, profile: str = BALANCED):
        valid = (
            self.BALANCED,
            self.MAXIMIZE_DATA_RETURN,
            self.EXTEND_LIFESPAN,
            self.POWER_CONSTRAINED,
        )
        if profile not in valid:
            raise ValueError(
                f"Unknown profile '{profile}'. Choose from: "
                + ", ".join(valid)
            )
        self.profile = profile
        # Dynamic weight multipliers (1.0 = default, operator-adjustable)
        self.recovery_bonus_scale = 1.0
        self.fault_penalty_scale = 1.0

    def update_weights(
        self,
        *,
        recovery_bonus_scale: float = None,
        fault_penalty_scale: float = None,
    ) -> None:
        """
        Adjust reward weight multipliers at runtime.

        Allows a mission operator to tune the agent's priorities based on
        live telemetry without retraining (e.g., increase fault penalty when
        power is critical).

        Parameters
        ----------
        recovery_bonus_scale : float, optional
            Multiplier applied to every successful-recovery bonus term.
            > 1.0 makes recovery more attractive; < 1.0 de-emphasises it.
        fault_penalty_scale : float, optional
            Multiplier applied to every fault-persistence penalty term.
            > 1.0 increases urgency; < 1.0 allows the agent to tolerate faults.
        """
        if recovery_bonus_scale is not None:
            self.recovery_bonus_scale = float(recovery_bonus_scale)
        if fault_penalty_scale is not None:
            self.fault_penalty_scale = float(fault_penalty_scale)

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
        rs = self.recovery_bonus_scale
        fp = self.fault_penalty_scale

        if self.profile == self.MAXIMIZE_DATA_RETURN:
            # High reward for being up; penalise lingering faults and inaction.
            reward = 2.0 if now_operational else -2.0 * fp
            if not was_healthy and now_operational:
                reward += 8.0 * rs   # fast recovery greatly rewarded
            elif not now_operational and not was_healthy:
                if action == 0:
                    reward -= 1.0 * fp   # penalise doing nothing when faulty
                else:
                    reward -= 3.0 * fp   # failed recovery attempt
            return reward

        elif self.profile == self.EXTEND_LIFESPAN:
            # Balanced uptime; steer agent toward cheap recovery actions.
            reward = 1.0 if now_operational else -1.0 * fp
            if not was_healthy and now_operational:
                reward += 5.0 * rs
                if action in (1, 3):   # restart or safe mode – cheap
                    reward += 1.0 * rs
                elif action == 2:       # redundant switch – expensive
                    reward -= 1.0 * fp
            elif action != 0 and not now_operational:
                reward -= 2.0 * fp
            return reward

        elif self.profile == self.POWER_CONSTRAINED:
            # Favour power-efficient actions (restart, safe mode) and
            # discourage redundant hardware switches that draw high current.
            reward = 1.0 if now_operational else -1.0 * fp
            if not was_healthy and now_operational:
                reward += 5.0 * rs
                if action in (1, 3):   # restart / safe mode – low power draw
                    reward += 1.0 * rs
                elif action == 2:       # redundant switch – high power draw
                    reward -= 2.0 * fp
            elif action != 0 and not now_operational:
                reward -= 2.0 * fp
            # Additional deterrent: penalise action 2 on a healthy spacecraft
            if action == 2 and was_healthy:
                reward -= 1.0 * fp
            return reward

        else:  # BALANCED (default)
            reward = 1.0 if now_operational else -1.0 * fp
            if action != 0 and now_operational and not was_healthy:
                reward += 5.0 * rs
            elif action != 0 and not now_operational:
                reward -= 2.0 * fp
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

    Parameters
    ----------
    comm_delay_steps : int
        Number of training episodes the aggregated weights are held in a
        queue before being distributed to the agents.  A non-zero value
        simulates the round-trip light-time delay of deep-space
        communication links (e.g., Earth–Mars ≈ 3–22 minutes one-way,
        translating to tens of training episodes in simulation time).
        Set to 0 (default) for immediate distribution.
    """

    def __init__(self, comm_delay_steps: int = 0, link_dropout_prob: float = 0.0):
        self.comm_delay_steps = comm_delay_steps
        self.link_dropout_prob = link_dropout_prob
        self._queue: list = []   # list of [remaining_episodes, avg_weights]

    def aggregate(self, agents: list) -> None:
        """
        Average online-network weights across all agents and enqueue.

        If ``comm_delay_steps`` is zero the averaged weights are applied
        immediately.  Otherwise they are queued; call ``tick()`` once per
        training episode to advance the delay counter and release weights
        when due.

        Parameters
        ----------
        agents : list[DQNAgent]
            All agents participating in this aggregation round.
        """
        if not agents:
            return
        # Stochastic link availability: filter out disconnected agents
        if self.link_dropout_prob > 0.0:
            agents = [a for a in agents if random.random() > self.link_dropout_prob]
        if not agents:
            return
        all_weights = [agent.model.get_weights() for agent in agents]
        n_layers = len(all_weights[0])
        avg_weights = [
            np.mean([all_weights[a][layer] for a in range(len(agents))], axis=0)
            for layer in range(n_layers)
        ]
        if self.comm_delay_steps > 0:
            self._queue.append([self.comm_delay_steps, avg_weights])
        else:
            self._apply(agents, avg_weights)

    def tick(self, agents: list) -> None:
        """
        Advance the communication delay queue by one episode.

        Any weight update whose remaining delay reaches zero is immediately
        applied to all agents.  Call this once per training episode inside
        ``FederatedSwarm.train_episode()``.
        """
        still_pending = []
        for remaining, avg_weights in self._queue:
            remaining -= 1
            if remaining <= 0:
                self._apply(agents, avg_weights)
            else:
                still_pending.append([remaining, avg_weights])
        self._queue = still_pending

    def _apply(self, agents: list, avg_weights: list) -> None:
        """Distribute *avg_weights* to every agent's online and target networks."""
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

    The state vector is extended to 11 features:

        [mem_error_ratio, parity_flag, sensor_deviation_norm,
         sensor_stuck_flag, time_since_recovery_norm, health_flag,
         thermal_fault_flag, power_level_norm,
         attitude_rate_norm, comm_quality_norm,
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
    comm_delay_steps : int
        Communication latency in episodes before aggregated weights reach
        agents.  Simulates deep-space round-trip light time (default 0 =
        immediate).
    link_dropout_prob : float
        Probability that any agent drops out of a given aggregation round.
    """

    COORD_STATE_DIM = 11  # base 10 features + 1 peer-sensor deviation feature

    def __init__(
        self,
        num_spacecraft: int,
        action_dim: int,
        mission_profile: MissionProfile = None,
        federated_interval: int = 10,
        comm_delay_steps: int = 0,
        link_dropout_prob: float = 0.0,
        safety_monitor: "SafetyMonitor" = None,
    ):
        """
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
        comm_delay_steps : int
            Communication latency in episodes before aggregated weights reach
            agents.
        link_dropout_prob : float
            Probability that any agent drops out of a given aggregation round.
        safety_monitor : SafetyMonitor, optional
            When provided, every DQN-recommended action is passed through
            ``SafetyMonitor.veto()`` before being applied to the environment
            *and* stored in the replay buffer.  Override events are counted in
            ``self.last_override_count`` and the agent still receives the
            (possibly lower) reward for the vetoed action, encouraging it to
            avoid constraint-violating choices during training.
        """
        super().__init__(num_spacecraft, self.COORD_STATE_DIM, action_dim)
        self.server = FederatedServer(
            comm_delay_steps=comm_delay_steps,
            link_dropout_prob=link_dropout_prob,
        )
        self.mission_profile = mission_profile or MissionProfile()
        self.federated_interval = federated_interval
        self.safety_monitor = safety_monitor
        self._episode_count = 0
        self.last_override_count = 0   # overrides triggered in last train_episode
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
        ``_peer_sensor_deviation`` attribute becomes the 9th state feature
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
        """Return the 11-feature state vector for *sc*, including peer deviation."""
        base = sc.get_state()                                          # shape (10,)
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

        If a ``SafetyMonitor`` was supplied at construction time, every
        DQN-recommended action is passed through ``veto()`` before being
        applied to the environment *and* stored in the replay buffer.  The
        number of veto overrides in this episode is recorded in
        ``self.last_override_count``.
        """
        for sc in self.spacecraft:
            sc.reset()
            sc._peer_sensor_deviation = 0.0

        episode_rewards = [0.0] * len(self.spacecraft)
        override_count = 0

        for step in range(max_steps):
            # Observe coordinated states (pre-step cross-check)
            self._cross_check_sensors()
            states = [self._get_coordinated_state(sc) for sc in self.spacecraft]

            # Select actions and optionally apply safety veto
            raw_actions = [
                self.agents[i].act(states[i], training=True)
                for i in range(len(self.spacecraft))
            ]
            if self.safety_monitor is not None:
                actions = []
                for i, sc in enumerate(self.spacecraft):
                    vetoed = self.safety_monitor.veto(
                        raw_actions[i], states[i], self.agents[i].action_dim
                    )
                    if vetoed != raw_actions[i]:
                        override_count += 1
                    actions.append(vetoed)
            else:
                actions = raw_actions

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
                # Store the vetoed (safe) action so the agent learns to
                # prefer constraint-respecting choices autonomously.
                self.agents[i].remember(states[i], actions[i], reward, new_state, done)
                self.agents[i].replay()

            if step % 10 == 0:
                for agent in self.agents:
                    agent.update_target()

        self._episode_count += 1
        self.last_override_count = override_count
        if self._episode_count % self.federated_interval == 0:
            self.server.aggregate(self.agents)

        # Advance the communication delay queue regardless of whether we
        # triggered an aggregation this episode.
        self.server.tick(self.agents)

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

def export_tflite(agent: DQNAgent, output_path: str = "sentinel_x_model.tflite") -> bytes:
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


def export_tflite_int8(
    agent: "DQNAgent",
    output_path: str = "sentinel_x_model_int8.tflite",
    n_calib_samples: int = 256,
) -> bytes:
    """
    Convert a trained DQNAgent to a full integer-only TFLite model (int8).

    Uses a representative calibration dataset of random fault states to
    determine per-layer quantisation ranges.  The resulting model uses
    integer-only arithmetic throughout, making it suitable for
    deterministic inference on MCUs without floating-point units
    (e.g., ARM Cortex-M4, STM32, radiation-hardened FPGAs).

    Parameters
    ----------
    agent : DQNAgent
        A fully trained agent.
    output_path : str
        Destination file for the int8 ``.tflite`` model.
    n_calib_samples : int
        Number of random states used to calibrate quantisation ranges.

    Returns
    -------
    bytes
        The serialised int8 TFLite flatbuffer (also written to *output_path*).
    """
    rng = np.random.default_rng(seed=42)

    def representative_dataset():
        for _ in range(n_calib_samples):
            sample = rng.uniform(0.0, 1.0, size=(1, agent.state_dim)).astype(
                np.float32
            )
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(agent.model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"Int8 TFLite model exported to '{output_path}' ({size_kb:.1f} KB)")
    return tflite_model


# -----------------------------------------------------------------------
# 9. Formal Verification of Trained Policies
# -----------------------------------------------------------------------

class PolicyVerifier:
    """
    Lightweight formal-verification harness for trained DQN policies.

    Runs three certifiable safety checks without relying on any external
    verification solver, making it suitable for ground-segment toolchains:

    **Check 1 – Safety Constraint (no-inaction on fault)**
        On every sampled faulty state the greedy policy must select a
        *recovery* action (action ≠ 0).  An agent that prefers inaction when
        a fault is active violates the minimum-safety requirement.

    **Check 2 – Q-value Margin Bound**
        For each sampled state the margin between the best and second-best
        Q-value is computed.  A narrow margin (< ``margin_threshold``)
        indicates an ambiguous policy that may be sensitive to noise.
        The check passes when the mean margin across all sampled states
        exceeds the threshold.

    **Check 3 – Action Coverage**
        Every non-inaction recovery action (1, 2, 3, …) must be the greedy
        choice on at least one sampled *faulty* state.  A policy that never
        selects a particular action wastes the available recovery repertoire
        and may fail on fault types that require that action.

    Parameters
    ----------
    agent : DQNAgent
        Trained agent to verify.
    n_samples : int
        Number of random states to draw for probabilistic checks.
    margin_threshold : float
        Minimum acceptable mean Q-value margin (Check 2).
    rng_seed : int or None
        Seed for reproducible sampling.
    """

    def __init__(
        self,
        agent: DQNAgent,
        n_samples: int = 500,
        margin_threshold: float = 0.1,
        rng_seed: int = 42,
    ):
        self.agent = agent
        self.n_samples = n_samples
        self.margin_threshold = margin_threshold
        self.rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_fault_states(self) -> np.ndarray:
        """
        Draw ``n_samples`` random states that represent *faulty* spacecraft.

        ``health_flag`` is always at index 5 in both the base state vector
        and any extended (coordinated) state vector.  Setting it to 1
        while randomising remaining features covers a broad slice of the
        fault-state space without requiring access to the live environment.
        """
        dim = self.agent.state_dim
        states = self.rng.uniform(0.0, 1.0, size=(self.n_samples, dim)).astype(
            np.float32
        )
        states[:, Spacecraft.HEALTH_FLAG_IDX] = 1.0   # health_flag = 1 → faulty
        return states

    def _q_values(self, states: np.ndarray) -> np.ndarray:
        """Return Q-value matrix of shape (n_samples, action_dim)."""
        return self.agent.model.predict(states, verbose=0)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_safety_constraint(self) -> dict:
        """
        Verify that the greedy policy never selects "do nothing" (action 0)
        on a faulty state.

        Returns
        -------
        dict with keys:
            ``passed``     – bool, True when no violation found
            ``violations`` – int, number of states where action 0 was chosen
            ``rate``       – float, fraction of sampled states with violations
        """
        states = self._sample_fault_states()
        q_vals = self._q_values(states)
        greedy_actions = np.argmax(q_vals, axis=1)
        violations = int(np.sum(greedy_actions == 0))
        rate = violations / self.n_samples
        return {
            "passed": violations == 0,
            "violations": violations,
            "rate": rate,
        }

    def check_qvalue_margin(self) -> dict:
        """
        Verify that the mean Q-value margin (best minus second-best) across
        sampled fault states exceeds ``margin_threshold``.

        A margin below the threshold indicates that the policy is nearly
        indifferent between actions, which is a sign of under-training or
        instability.

        Returns
        -------
        dict with keys:
            ``passed``       – bool
            ``mean_margin``  – float, mean margin across sampled states
            ``min_margin``   – float, worst-case (smallest) margin observed
            ``threshold``    – float, the configured threshold
        """
        states = self._sample_fault_states()
        q_vals = self._q_values(states)
        sorted_q = np.sort(q_vals, axis=1)[:, ::-1]   # descending per row
        margins = sorted_q[:, 0] - sorted_q[:, 1]
        mean_margin = float(np.mean(margins))
        min_margin = float(np.min(margins))
        return {
            "passed": mean_margin >= self.margin_threshold,
            "mean_margin": mean_margin,
            "min_margin": min_margin,
            "threshold": self.margin_threshold,
        }

    def check_action_coverage(self) -> dict:
        """
        Verify that every recovery action (action ≥ 1) is the greedy choice
        on at least one sampled faulty state.

        Returns
        -------
        dict with keys:
            ``passed``          – bool
            ``covered_actions`` – set of action indices that appear in the
                                  greedy policy
            ``missing_actions`` – set of action indices never chosen
        """
        states = self._sample_fault_states()
        q_vals = self._q_values(states)
        greedy_actions = set(int(a) for a in np.argmax(q_vals, axis=1))
        recovery_actions = set(range(1, self.agent.action_dim))
        missing = recovery_actions - greedy_actions
        return {
            "passed": len(missing) == 0,
            "covered_actions": greedy_actions,
            "missing_actions": missing,
        }

    # ------------------------------------------------------------------
    # Composite report
    # ------------------------------------------------------------------

    def verify(self) -> dict:
        """
        Run all three checks and return a consolidated verification report.

        The overall ``passed`` flag is True only when every individual check
        passes.  The report is also printed to stdout in a human-readable
        format for integration into ground-segment CI pipelines.

        Returns
        -------
        dict with keys:
            ``overall_passed``     – bool
            ``safety_constraint``  – result dict from check_safety_constraint
            ``qvalue_margin``      – result dict from check_qvalue_margin
            ``action_coverage``    – result dict from check_action_coverage
        """
        safety = self.check_safety_constraint()
        margin = self.check_qvalue_margin()
        coverage = self.check_action_coverage()
        overall = safety["passed"] and margin["passed"] and coverage["passed"]

        label = lambda ok: "PASS" if ok else "FAIL"   # noqa: E731
        print("=" * 55)
        print("  SENTINEL-X Policy Verification Report")
        print("=" * 55)
        print(
            f"  [{'PASS' if overall else 'FAIL'}] Overall"
        )
        print(
            f"  [{label(safety['passed'])}] Safety constraint – "
            f"{safety['violations']} violation(s) / {self.n_samples} states"
            f"  ({safety['rate'] * 100:.1f}%)"
        )
        print(
            f"  [{label(margin['passed'])}] Q-value margin  – "
            f"mean={margin['mean_margin']:.4f}, "
            f"min={margin['min_margin']:.4f}, "
            f"threshold={margin['threshold']}"
        )
        covered_str = ", ".join(str(a) for a in sorted(coverage["covered_actions"]))
        missing_str = (
            ", ".join(str(a) for a in sorted(coverage["missing_actions"]))
            if coverage["missing_actions"]
            else "none"
        )
        print(
            f"  [{label(coverage['passed'])}] Action coverage  – "
            f"covered=[{covered_str}], missing=[{missing_str}]"
        )
        print("=" * 55)

        return {
            "overall_passed": overall,
            "safety_constraint": safety,
            "qvalue_margin": margin,
            "action_coverage": coverage,
        }


# -----------------------------------------------------------------------
# 10. Safety Monitor (FDIR Veto Layer)
# -----------------------------------------------------------------------

class SafetyMonitor:
    """
    Rule-based FDIR safety layer that vetoes unsafe DQN-recommended actions.

    Implements hard safety constraints that must never be violated regardless
    of what the learned policy recommends, providing a deterministic safety
    net around the neural network inference.

    Hard constraints enforced:

    * **No inaction on fault** – When ``health_flag`` (state[5]) is 1.0, the
      agent must select a recovery action; action 0 ("do nothing") is replaced
      by safe-mode (action 3).

    * **No full reset on critical power** – When the normalised power level
      (state[7]) is critically low, switching to redundant hardware (action 2)
      is prohibited because the full reset would exhaust the remaining energy
      budget; safe-mode (action 3) is used instead.

    This architecture mirrors the hybrid RL + FDIR layer recommended for
    space-qualified autonomy under ECSS-E-ST-70-11 and NASA autonomy standards.
    """

    HEALTH_FLAG_IDX = Spacecraft.HEALTH_FLAG_IDX   # 5
    POWER_LEVEL_IDX = 7
    CRITICAL_POWER_THRESHOLD = 0.15

    def veto(self, proposed_action: int, state: np.ndarray, action_dim: int) -> int:
        """
        Apply hard safety constraints and return a safe action.

        Returns ``proposed_action`` unchanged if no constraint is violated.

        Parameters
        ----------
        proposed_action : int
            Action recommended by the DQN agent.
        state : np.ndarray
            Current (normalised) state vector.
        action_dim : int
            Total number of actions available (unused; reserved for future use).

        Returns
        -------
        int
            A safe action (may equal ``proposed_action`` if no veto triggered).
        """
        health_flag = float(state[self.HEALTH_FLAG_IDX])
        power_level = (
            float(state[self.POWER_LEVEL_IDX])
            if len(state) > self.POWER_LEVEL_IDX
            else 1.0
        )

        # Constraint 1: must attempt recovery when faulted
        if health_flag >= 0.5 and proposed_action == 0:
            return 3   # fall back to safe mode

        # Constraint 2: no full hardware reset on critically low power
        if proposed_action == 2 and power_level < self.CRITICAL_POWER_THRESHOLD:
            return 3   # safe mode instead

        return proposed_action


# -----------------------------------------------------------------------
# 11. Gossip-Based Federated Learning (Decentralised)
# -----------------------------------------------------------------------

class GossipServer:
    """
    Decentralised gossip-based federated learning for large constellations.

    Instead of routing all updates through a central server, each spacecraft
    randomly selects ``k`` neighbours and averages its DQN weights with theirs.
    This approach:

    * Scales to arbitrarily large fleets without a single aggregation point.
    * Tolerates partial link outages (unreachable agents are skipped without
      stalling the round).
    * Converges to the same global average as FedAvg when run repeatedly,
      but distributes the communication load across the constellation.

    Parameters
    ----------
    k : int
        Number of random neighbours each agent gossips with per round.
    comm_delay_steps : int
        Episodes of delay before gossiped weights take effect (same
        semantics as ``FederatedServer.comm_delay_steps``).
    """

    def __init__(self, k: int = 2, comm_delay_steps: int = 0):
        self.k = k
        self.comm_delay_steps = comm_delay_steps
        self._queue: list = []

    def gossip_round(self, agents: list) -> None:
        """
        Run one round of gossip averaging across the agent population.

        Each agent independently samples k peers (without replacement) and
        computes the mean of its weights and their weights.  All snapshots
        are taken *before* any weights are updated to avoid order-dependent
        results.
        """
        n = len(agents)
        if n < 2:
            return

        # Snapshot current weights before any update
        snapshots = [agent.model.get_weights() for agent in agents]
        n_layers = len(snapshots[0])

        updates = []
        for i in range(n):
            peer_indices = random.sample(
                [j for j in range(n) if j != i], min(self.k, n - 1)
            )
            all_w = [snapshots[i]] + [snapshots[p] for p in peer_indices]
            avg = [
                np.mean([w[layer] for w in all_w], axis=0)
                for layer in range(n_layers)
            ]
            updates.append((i, avg))

        if self.comm_delay_steps > 0:
            self._queue.append([self.comm_delay_steps, updates])
        else:
            self._apply(agents, updates)

    def tick(self, agents: list) -> None:
        """Advance the gossip delay queue by one episode."""
        still_pending = []
        for remaining, updates in self._queue:
            remaining -= 1
            if remaining <= 0:
                self._apply(agents, updates)
            else:
                still_pending.append([remaining, updates])
        self._queue = still_pending

    def _apply(self, agents: list, updates: list) -> None:
        """Distribute gossip-averaged weights to each agent."""
        for idx, weights in updates:
            if idx < len(agents):
                agents[idx].model.set_weights(weights)
                agents[idx].update_target()


# -----------------------------------------------------------------------
# 12. Adversarial Testing (FGSM)
# -----------------------------------------------------------------------

class AdversarialTester:
    """
    Gradient-based adversarial testing (FGSM) for trained DQN policies.

    Finds minimal L∞-bounded state perturbations that change the agent's
    greedy action.  These corner cases expose policy fragility near decision
    boundaries and can be fed back into training as adversarial examples to
    improve robustness.

    Parameters
    ----------
    agent : DQNAgent
        Trained agent to probe.
    epsilon : float
        Maximum L∞ perturbation magnitude (in normalised state space).
        A value of 0.05 corresponds to a 5% shift in any single feature.
    rng_seed : int
        Seed for reproducible state sampling.
    """

    def __init__(self, agent: "DQNAgent", epsilon: float = 0.05, rng_seed: int = 0):
        self.agent = agent
        self.epsilon = epsilon
        self.rng = np.random.default_rng(rng_seed)

    def find_adversarial_examples(self, n_examples: int = 100) -> list:
        """
        Attempt to flip the greedy action on *n_examples* random fault states.

        Uses FGSM to perturb states in the direction that reduces the Q-value
        margin between the best and second-best action.

        Returns
        -------
        list of dict, each with keys:
            ``original_state``, ``perturbed_state``,
            ``original_action``, ``flipped_action``
        """
        dim = self.agent.state_dim
        states = self.rng.uniform(0.0, 1.0, size=(n_examples, dim)).astype(np.float32)
        states[:, Spacecraft.HEALTH_FLAG_IDX] = 1.0   # force fault condition

        results = []
        for state in states:
            original_action = self._greedy_action(state)
            perturbed = self._fgsm_perturb(state)
            perturbed_action = self._greedy_action(perturbed)
            if perturbed_action != original_action:
                results.append(
                    {
                        "original_state": state.copy(),
                        "perturbed_state": perturbed.copy(),
                        "original_action": original_action,
                        "flipped_action": perturbed_action,
                    }
                )
        return results

    def augment_replay_buffer(
        self,
        agents: list,
        n_examples: int = 100,
        safety_monitor: "SafetyMonitor" = None,
    ) -> int:
        """
        Harden agents by adding adversarial transitions to their replay buffers.

        For every adversarial example found, a synthetic transition is created:
        the *original* (clean) state is paired with the *safe* recovery action
        (i.e., the non-flipped action, optionally passed through a
        ``SafetyMonitor`` veto) and a reward of ``+1.0``.  The next_state is
        the perturbed state, and ``done`` is False.

        This teaches the agent that, near a decision boundary, the original
        action should be maintained even under small state perturbations.

        Parameters
        ----------
        agents : list[DQNAgent]
            Agents whose replay buffers will be augmented.
        n_examples : int
            Number of candidate adversarial states to probe.
        safety_monitor : SafetyMonitor, optional
            When provided the synthetic action is also passed through veto
            so that only constraint-respecting actions are injected.

        Returns
        -------
        int
            Number of adversarial transitions successfully injected.
        """
        examples = self.find_adversarial_examples(n_examples=n_examples)
        injected = 0
        for ex in examples:
            clean_state = ex["original_state"]
            perturbed_state = ex["perturbed_state"]
            safe_action = ex["original_action"]   # original was stable; reinforce it
            if safety_monitor is not None:
                safe_action = safety_monitor.veto(
                    safe_action, clean_state, self.agent.action_dim
                )
            for agent in agents:
                agent.remember(
                    clean_state,
                    safe_action,
                    1.0,           # positive reward: this is the correct behaviour
                    perturbed_state,
                    False,
                )
            injected += 1
        return injected

    def certify_robustness(
        self,
        n_samples: int = 200,
        eps_lo: float = 0.0,
        eps_hi: float = 0.5,
        n_bisect: int = 12,
    ) -> dict:
        """
        Estimate the per-state robustness radius via binary search.

        For each sampled fault state, binary-search the smallest ε in
        ``[eps_lo, eps_hi]`` at which an FGSM perturbation flips the greedy
        action.  States that are robust at ``eps_hi`` are assigned a radius
        of ``eps_hi`` (lower-bound).

        Parameters
        ----------
        n_samples : int
            Number of random fault states to certify.
        eps_lo : float
            Lower bound of the search range.
        eps_hi : float
            Upper bound of the search range (and maximum reported radius).
        n_bisect : int
            Number of bisection iterations (accuracy ≈ ``eps_hi / 2**n_bisect``).

        Returns
        -------
        dict with keys:
            ``mean_radius``  – float, mean minimum flip-ε across samples
            ``min_radius``   – float, worst-case (most fragile) certified radius
            ``max_radius``   – float, best-case certified radius (≤ eps_hi)
            ``robust_frac``  – float, fraction of states robust at eps_hi
            ``eps_hi``       – float, the configured upper bound
            ``n_samples``    – int, number of states evaluated
        """
        dim = self.agent.state_dim
        states = self.rng.uniform(0.0, 1.0, size=(n_samples, dim)).astype(np.float32)
        states[:, Spacecraft.HEALTH_FLAG_IDX] = 1.0

        original_eps = self.epsilon
        radii = []
        robust_count = 0
        for state in states:
            orig_action = self._greedy_action(state)
            lo, hi = eps_lo, eps_hi
            flipped_at_hi = self._action_flips_at_eps(state, orig_action, hi)
            if not flipped_at_hi:
                radii.append(eps_hi)
                robust_count += 1
                continue
            for _ in range(n_bisect):
                mid = (lo + hi) / 2.0
                if self._action_flips_at_eps(state, orig_action, mid):
                    hi = mid
                else:
                    lo = mid
            radii.append((lo + hi) / 2.0)
        self.epsilon = original_eps

        radii_arr = np.array(radii, dtype=np.float64)
        return {
            "mean_radius": float(np.mean(radii_arr)),
            "min_radius": float(np.min(radii_arr)),
            "max_radius": float(np.max(radii_arr)),
            "robust_frac": robust_count / max(n_samples, 1),
            "eps_hi": eps_hi,
            "n_samples": n_samples,
        }

    def _action_flips_at_eps(self, state: np.ndarray, orig_action: int, eps: float) -> bool:
        """Return True if an FGSM perturbation of size *eps* flips the action."""
        saved_eps = self.epsilon
        self.epsilon = eps
        perturbed = self._fgsm_perturb(state)
        self.epsilon = saved_eps
        return self._greedy_action(perturbed) != orig_action

    def summary(self, results: list, n_tested: int) -> None:
        """
        Print a summary of adversarial findings.

        Parameters
        ----------
        results : list
            Output of ``find_adversarial_examples()``.
        n_tested : int
            Total number of states that were tested.
        """
        from collections import Counter

        n_flipped = len(results)
        rate = n_flipped / max(n_tested, 1) * 100
        flip_pairs = Counter(
            (r["original_action"], r["flipped_action"]) for r in results
        )
        print("=" * 55)
        print("  SENTINEL-X Adversarial Testing Report")
        print("=" * 55)
        if n_flipped == 0:
            print("  No adversarial examples found (robust within epsilon).")
        else:
            print(f"  Adversarial flip rate: {rate:.1f}% ({n_flipped}/{n_tested})")
            print("  Action flip distribution:")
            for (orig, flipped), count in sorted(flip_pairs.items()):
                print(f"    action {orig} -> {flipped}: {count} example(s)")
        print("=" * 55)

    def _greedy_action(self, state: np.ndarray) -> int:
        q = self.agent.model.predict(state[np.newaxis, :], verbose=0)[0]
        return int(np.argmax(q))

    def _fgsm_perturb(self, state: np.ndarray) -> np.ndarray:
        """
        Fast Gradient Sign Method perturbation.

        Perturbs ``state`` in the direction that reduces the Q-value margin
        between the best and second-best action, pushing toward a decision
        boundary where the greedy action may flip.
        """
        state_var = tf.Variable(state[np.newaxis, :])
        with tf.GradientTape() as tape:
            q_vals = self.agent.model(state_var, training=False)
            sorted_q = tf.sort(q_vals[0])[::-1]
            loss = sorted_q[0] - sorted_q[1]   # minimise margin
        grad = tape.gradient(loss, state_var)
        if grad is None:
            return state
        # Step in negative gradient direction to shrink the margin
        perturbation = -self.epsilon * tf.sign(grad).numpy()[0]
        perturbed = np.clip(state + perturbation, 0.0, 1.0)
        return perturbed.astype(np.float32)


# -----------------------------------------------------------------------
# 13. LTL Reward Shaping (Constrained RL)
# -----------------------------------------------------------------------

class LTLConstraintChecker:
    """
    Lightweight Linear Temporal Logic (LTL)-style safety-constraint checker.

    Each constraint is a named predicate over the current state vector.  At
    every training step the checker evaluates all active constraints; any
    violation returns a negative penalty that can be added to the mission
    reward to implement *constrained RL*.

    Supported built-in constraints (all evaluated on the normalised state):

    * **no_inaction_on_fault** – penalise action=0 when ``health_flag``=1
    * **no_full_reset_low_power** – penalise action=2 when power ≤ threshold
    * **no_simultaneous_faults** – penalise states where ≥ 3 fault flags
      are simultaneously active (indicates cascading failures)
    * **comm_link_recovery** – penalise any action other than safe-mode (3)
      when ``comm_quality_norm`` drops below a critical level

    Operators may add custom constraints via ``add_constraint()``.

    Parameters
    ----------
    penalty : float
        Reward deduction applied per violated constraint per step (< 0).
    """

    # Indices in the 10/11-dim normalised state vector
    _HEALTH_IDX = Spacecraft.HEALTH_FLAG_IDX           # 5
    _POWER_IDX = 7
    _ATTITUDE_IDX = 8
    _COMM_IDX = 9
    _THERMAL_IDX = 6
    _PARITY_IDX = 1
    _STUCK_IDX = 3

    def __init__(self, penalty: float = -0.5):
        if penalty > 0:
            raise ValueError("penalty must be ≤ 0 (a negative or zero value).")
        self.penalty = float(penalty)
        # dict name -> callable(state, action) -> bool (True = violated)
        self._constraints: dict = {}
        self._add_builtin_constraints()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_constraint(self, name: str, predicate) -> None:
        """
        Register a custom constraint.

        Parameters
        ----------
        name : str
            Unique identifier for the constraint.
        predicate : callable(state: np.ndarray, action: int) -> bool
            Returns True when the constraint is **violated**.
        """
        self._constraints[name] = predicate

    def remove_constraint(self, name: str) -> None:
        """Remove a constraint by name (no-op if not found)."""
        self._constraints.pop(name, None)

    def evaluate(self, state: np.ndarray, action: int) -> float:
        """
        Evaluate all active constraints against (*state*, *action*).

        Returns
        -------
        float
            Total penalty: ``n_violated × self.penalty``.  Zero if no
            constraint is violated.
        """
        n_violated = sum(
            1 for pred in self._constraints.values() if pred(state, action)
        )
        return n_violated * self.penalty

    def active_names(self) -> list:
        """Return a list of currently registered constraint names."""
        return list(self._constraints.keys())

    # ------------------------------------------------------------------
    # Built-in constraints
    # ------------------------------------------------------------------

    def _add_builtin_constraints(self) -> None:
        idx_h = self._HEALTH_IDX
        idx_p = self._POWER_IDX
        idx_c = self._COMM_IDX

        def _no_inaction_on_fault(state, action):
            return float(state[idx_h]) >= 0.5 and action == 0

        def _no_full_reset_low_power(state, action):
            power = float(state[idx_p]) if len(state) > idx_p else 1.0
            return action == 2 and power < 0.15

        def _no_simultaneous_faults(state, _action):
            fault_flags = [
                float(state[self._THERMAL_IDX]),
                float(state[self._HEALTH_IDX]),
                float(state[self._PARITY_IDX]),
                float(state[self._STUCK_IDX]),
            ]
            return sum(f >= 0.5 for f in fault_flags) >= 3

        def _comm_link_recovery(state, action):
            comm_q = float(state[idx_c]) if len(state) > idx_c else 1.0
            return comm_q < 0.1 and action not in (2, 3)

        self._constraints = {
            "no_inaction_on_fault": _no_inaction_on_fault,
            "no_full_reset_low_power": _no_full_reset_low_power,
            "no_simultaneous_faults": _no_simultaneous_faults,
            "comm_link_recovery": _comm_link_recovery,
        }


# -----------------------------------------------------------------------
# 14. Decision-Tree Policy Extraction (Imitation Learning)
# -----------------------------------------------------------------------

def extract_decision_tree(
    agent: "DQNAgent",
    n_samples: int = 2000,
    max_depth: int = 8,
    rng_seed: int = 42,
) -> "DecisionTreeClassifier":
    """
    Fit a decision-tree surrogate that imitates the DQN's greedy policy.

    The surrogate is trained via *imitation learning* (behavioural cloning):
    random states are labelled by the DQN's ``argmax Q``-value and a
    ``DecisionTreeClassifier`` is fitted on those (state, greedy_action)
    pairs.  The resulting tree:

    * Can be exhaustively verified over its finite input partition.
    * Is human-readable and auditable.
    * Can be exported to SMT or interval-arithmetic solvers (e.g., Marabou,
      α,β-CROWN) for flight-level certifiability.

    Parameters
    ----------
    agent : DQNAgent
        Trained agent to imitate.
    n_samples : int
        Number of random states used for imitation.
    max_depth : int
        Maximum depth of the decision tree (controls model complexity).
    rng_seed : int
        Reproducibility seed.

    Returns
    -------
    DecisionTreeClassifier
        Fitted scikit-learn classifier.  Accuracy relative to the DQN is
        printed to stdout.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for decision-tree extraction. "
            "Install it with: pip install scikit-learn"
        )
    rng = np.random.default_rng(rng_seed)
    states = rng.uniform(0.0, 1.0, size=(n_samples, agent.state_dim)).astype(
        np.float32
    )
    # Label with the DQN greedy policy
    q_vals = agent.model.predict(states, verbose=0)
    labels = np.argmax(q_vals, axis=1)

    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=rng_seed)
    dt.fit(states, labels)

    # Compute fidelity (agreement with DQN labels)
    preds = dt.predict(states)
    fidelity = float(np.mean(preds == labels))
    n_leaves = dt.get_n_leaves()
    print(
        f"Decision-tree surrogate: depth={dt.get_depth()}, "
        f"leaves={n_leaves}, fidelity={fidelity * 100:.1f}%"
    )
    return dt


# -----------------------------------------------------------------------
# 15. Mission Scenarios
# -----------------------------------------------------------------------

class MissionScenario:
    """
    Bundles fault-model parameters and reward profile for a concrete mission.

    Three pre-built scenarios are available as class methods:

    * :meth:`lunar_gateway` – Near-rectilinear halo orbit (NRHO).  Moderate
      thermal and attitude requirements; high communication quality with
      Earth/Artemis crew; mission-critical power budget.

    * :meth:`mars_orbiter` – Mars Sample Return relay orbiter.  Deep-space
      comm delays; high radiation flux (elevated bit-flip rate); power
      constrained (solar distance); thermal extremes.

    * :meth:`cubesat_swarm` – Low-Earth orbit 3U/6U swarm.  Frequent eclipse
      cycles (thermal spikes); limited onboard compute; high stuck-sensor
      probability due to space weather; data-return-oriented mission.

    Each scenario is created with ``MissionScenario(**params)`` or via the
    convenience class methods (recommended).  Pass it to
    :func:`build_swarm_for_scenario` to obtain a configured ``FederatedSwarm``.

    Attributes
    ----------
    name : str
        Human-readable mission name.
    profile : str
        ``MissionProfile`` constant ('balanced', 'data_return', etc.)
    num_spacecraft : int
        Suggested swarm size for simulation.
    comm_delay_steps : int
        Federation communication delay (deep-space latency in episodes).
    link_dropout_prob : float
        Fraction of federated rounds that miss due to link outages.
    thermal_drift_std : float
        Standard deviation of the thermal random walk (higher = more dynamic).
    thermal_spike_prob : float
        Probability of a sudden large thermal spike per step.
    flip_rate_per_bit : float
        SEU (bit-flip) probability per bit per step (higher near radiation belts).
    sensor_stuck_prob : float
        Probability that a sensor becomes stuck per step.
    power_drain_rate : float
        Mean power drain per step (higher away from the sun).
    """

    def __init__(
        self,
        name: str,
        profile: str,
        num_spacecraft: int = 4,
        comm_delay_steps: int = 0,
        link_dropout_prob: float = 0.0,
        thermal_drift_std: float = 0.1,
        thermal_spike_prob: float = 0.005,
        flip_rate_per_bit: float = 1e-4,
        sensor_stuck_prob: float = 0.01,
        power_drain_rate: float = 0.10,
    ):
        self.name = name
        self.profile = profile
        self.num_spacecraft = num_spacecraft
        self.comm_delay_steps = comm_delay_steps
        self.link_dropout_prob = link_dropout_prob
        self.thermal_drift_std = thermal_drift_std
        self.thermal_spike_prob = thermal_spike_prob
        self.flip_rate_per_bit = flip_rate_per_bit
        self.sensor_stuck_prob = sensor_stuck_prob
        self.power_drain_rate = power_drain_rate

    # ------------------------------------------------------------------
    # Pre-built scenarios
    # ------------------------------------------------------------------

    @classmethod
    def lunar_gateway(cls) -> "MissionScenario":
        """
        Lunar Gateway (HALO node) in near-rectilinear halo orbit.

        Key characteristics:
        - Moderate thermal environment; weekly eclipse transitions.
        - Excellent Earth comms with < 1.3 s round-trip delay.
        - Crew safety is paramount → power-constrained profile.
        - Low radiation flux compared to deep space.
        """
        return cls(
            name="Lunar Gateway (HALO)",
            profile=MissionProfile.POWER_CONSTRAINED,
            num_spacecraft=3,
            comm_delay_steps=0,          # < 1.3 s round-trip → no sim delay
            link_dropout_prob=0.05,      # occasional Artemis crew comm priority
            thermal_drift_std=0.15,      # eclipse cycles every ~7 days
            thermal_spike_prob=0.008,
            flip_rate_per_bit=5e-5,      # lower than deep space
            sensor_stuck_prob=0.008,
            power_drain_rate=0.09,       # solar panels at 1 AU
        )

    @classmethod
    def mars_orbiter(cls) -> "MissionScenario":
        """
        Mars Sample Return relay orbiter.

        Key characteristics:
        - Deep-space comms: 3–22 min one-way delay → large federation delay.
        - Higher radiation flux in interplanetary space.
        - Significant power reduction (solar panels at 1.5 AU).
        - Thermal extremes due to Mars solar distance and dust storm risk.
        - Link dropout from planetary occultation or dust storms.
        """
        return cls(
            name="Mars Sample Return Orbiter",
            profile=MissionProfile.MAXIMIZE_DATA_RETURN,
            num_spacecraft=2,
            comm_delay_steps=15,          # ~22-min max delay ≈ 15 training eps
            link_dropout_prob=0.15,       # occultation + dust storm dropouts
            thermal_drift_std=0.20,       # larger thermal swings
            thermal_spike_prob=0.012,
            flip_rate_per_bit=3e-4,       # higher SEU rate in interplanetary space
            sensor_stuck_prob=0.015,
            power_drain_rate=0.14,        # reduced solar flux at Mars
        )

    @classmethod
    def cubesat_swarm(cls) -> "MissionScenario":
        """
        Low-Earth orbit 3U/6U CubeSat swarm (e.g., Planet Labs style).

        Key characteristics:
        - Frequent eclipse cycles (every ~90 min) → thermal spikes.
        - Data-return mission; maximise downlink during ground contacts.
        - Large swarm; gossip-style federation scales naturally.
        - High stuck-sensor rate due to space weather and component quality.
        - Limited onboard compute; simple actions preferred.
        """
        return cls(
            name="LEO CubeSat Swarm",
            profile=MissionProfile.MAXIMIZE_DATA_RETURN,
            num_spacecraft=8,
            comm_delay_steps=1,           # downlink contacts every ~90 min
            link_dropout_prob=0.25,       # limited contact windows
            thermal_drift_std=0.25,       # rapid eclipse cycling
            thermal_spike_prob=0.018,
            flip_rate_per_bit=2e-4,       # moderate LEO radiation
            sensor_stuck_prob=0.025,      # COTS component quality
            power_drain_rate=0.11,
        )

    def summary(self) -> str:
        """Return a formatted one-page summary of the scenario parameters."""
        lines = [
            "=" * 55,
            f"  Mission Scenario: {self.name}",
            "=" * 55,
            f"  Profile            : {self.profile}",
            f"  Swarm size         : {self.num_spacecraft}",
            f"  Comm delay (eps)   : {self.comm_delay_steps}",
            f"  Link dropout       : {self.link_dropout_prob:.0%}",
            f"  Thermal drift std  : {self.thermal_drift_std}",
            f"  Thermal spike prob : {self.thermal_spike_prob}",
            f"  SEU flip rate      : {self.flip_rate_per_bit:.1e} /bit/step",
            f"  Sensor stuck prob  : {self.sensor_stuck_prob}",
            f"  Power drain rate   : {self.power_drain_rate}",
            "=" * 55,
        ]
        return "\n".join(lines)


def build_swarm_for_scenario(
    scenario: MissionScenario,
    action_dim: int = 4,
    federated_interval: int = 10,
    safety_monitor: "SafetyMonitor" = None,
) -> "FederatedSwarm":
    """
    Construct a ``FederatedSwarm`` configured for *scenario*.

    Each spacecraft in the swarm is initialised with fault-model parameters
    drawn from the scenario definition.  The mission reward profile and
    federation settings are applied automatically.  Fault-model parameters
    are stored on each ``Spacecraft`` instance so they persist across
    ``reset()`` calls between training episodes.

    Parameters
    ----------
    scenario : MissionScenario
        The mission scenario configuration.
    action_dim : int
        Number of discrete recovery actions (default 4).
    federated_interval : int
        Episodes between federated weight aggregations.
    safety_monitor : SafetyMonitor, optional
        Safety veto layer to integrate into training.

    Returns
    -------
    FederatedSwarm
        A ready-to-train swarm pre-configured for the scenario.
    """
    swarm = FederatedSwarm(
        num_spacecraft=scenario.num_spacecraft,
        action_dim=action_dim,
        mission_profile=MissionProfile(scenario.profile),
        federated_interval=federated_interval,
        comm_delay_steps=scenario.comm_delay_steps,
        link_dropout_prob=scenario.link_dropout_prob,
        safety_monitor=safety_monitor,
    )
    # Replace each spacecraft with a scenario-configured instance so that
    # fault parameters persist across reset() calls between episodes.
    swarm.spacecraft = [
        Spacecraft(
            spacecraft_id=i,
            flip_rate_per_bit=scenario.flip_rate_per_bit,
            sensor_stuck_prob=scenario.sensor_stuck_prob,
            thermal_drift_std=scenario.thermal_drift_std,
            thermal_spike_prob=scenario.thermal_spike_prob,
            power_drain_rate=scenario.power_drain_rate,
        )
        for i in range(scenario.num_spacecraft)
    ]
    for sc in swarm.spacecraft:
        sc._peer_sensor_deviation = 0.0
    return swarm


# -----------------------------------------------------------------------
# 16. Federation Benchmark (FedAvg vs Gossip)
# -----------------------------------------------------------------------

def benchmark_federation(
    num_spacecraft: int = 4,
    episodes: int = 50,
    max_steps: int = 100,
    dropout_rates: tuple = (0.0, 0.2, 0.5),
    comm_delay: int = 0,
    federated_interval: int = 10,
    gossip_k: int = 2,
    rng_seed: int = 0,
) -> dict:
    """
    Compare centralised FedAvg against decentralised gossip federation.

    Trains one ``FederatedSwarm`` per (method × dropout_rate) combination and
    records the mean reward per episode and the final test operational steps.
    Results are printed as a formatted summary table.

    Parameters
    ----------
    num_spacecraft : int
        Swarm size used for every condition.
    episodes : int
        Training episodes per condition.
    max_steps : int
        Steps per training episode.
    dropout_rates : tuple of float
        Link-dropout probabilities to sweep across.
    comm_delay : int
        Communication delay steps applied to *both* methods equally.
    federated_interval : int
        Episodes between aggregation events.
    gossip_k : int
        Number of gossip neighbours per agent per round.
    rng_seed : int
        Seed for NumPy random state (unused directly; sets Python random seed
        so training conditions are repeatable).

    Returns
    -------
    dict
        Nested dict ``results[method][dropout_rate]`` where each value is a
        dict with keys ``rewards`` (list of per-episode mean rewards) and
        ``test_ops`` (float, final avg operational steps in test episode).
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    profile = MissionProfile(MissionProfile.BALANCED)
    results: dict = {"fedavg": {}, "gossip": {}}

    col_w = 10
    header = (
        f"{'Method':<10} {'Dropout':>{col_w}} {'FinalReward':>{col_w+2}}"
        f" {'TestOps':>{col_w}}"
    )
    print("\n" + "=" * len(header))
    print("  SENTINEL-X  Federation Benchmark")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for dropout in dropout_rates:
        # ---- FedAvg ----
        fed_swarm = FederatedSwarm(
            num_spacecraft=num_spacecraft,
            action_dim=4,
            mission_profile=MissionProfile(MissionProfile.BALANCED),
            federated_interval=federated_interval,
            comm_delay_steps=comm_delay,
            link_dropout_prob=dropout,
        )
        fed_rewards = []
        for _ in range(episodes):
            fed_rewards.append(fed_swarm.train_episode(max_steps=max_steps))
        fed_ops = fed_swarm.test_episode(max_steps=max_steps)
        results["fedavg"][dropout] = {"rewards": fed_rewards, "test_ops": fed_ops}
        print(
            f"  {'FedAvg':<8} {dropout:>{col_w}.1%}"
            f" {fed_rewards[-1]:>{col_w+2}.2f}"
            f" {fed_ops:>{col_w}.1f}"
        )

        # ---- Gossip ----
        gossip_swarm = FederatedSwarm(
            num_spacecraft=num_spacecraft,
            action_dim=4,
            mission_profile=MissionProfile(MissionProfile.BALANCED),
            federated_interval=999,   # disable central FedAvg
            comm_delay_steps=comm_delay,
            link_dropout_prob=0.0,
        )
        gossip_server = GossipServer(k=gossip_k, comm_delay_steps=comm_delay)
        gos_rewards = []
        for ep in range(episodes):
            gos_rewards.append(gossip_swarm.train_episode(max_steps=max_steps))
            if ep % federated_interval == 0:
                gossip_server.gossip_round(gossip_swarm.agents)
                gossip_server.tick(gossip_swarm.agents)
        gos_ops = gossip_swarm.test_episode(max_steps=max_steps)
        results["gossip"][dropout] = {"rewards": gos_rewards, "test_ops": gos_ops}
        print(
            f"  {'Gossip':<8} {dropout:>{col_w}.1%}"
            f" {gos_rewards[-1]:>{col_w+2}.2f}"
            f" {gos_ops:>{col_w}.1f}"
        )

    print("=" * len(header))
    return results


# -----------------------------------------------------------------------
# 17. Main Training Loop
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
    STATE_DIM = 10  # 10-feature state: mem/sensor/thermal/power/attitude/comm
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
        MissionProfile.POWER_CONSTRAINED,
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
    export_tflite(fed_swarm.agents[0], output_path="sentinel_x_model.tflite")

    print("\nExporting int8-quantised TFLite model for MCU deployment...")
    export_tflite_int8(fed_swarm.agents[0], output_path="sentinel_x_model_int8.tflite")

    # ------------------------------------------------------------------
    # 4. Deep-space communication delay demonstration
    #    Train a small swarm with a 5-episode link delay to demonstrate
    #    the effect of intermittent deep-space communication on federated
    #    learning convergence.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Deep-space comm-delay federated learning demo (delay=5 eps)")
    print("=" * 60)
    delayed_swarm = FederatedSwarm(
        num_spacecraft=SWARM_SIZE,
        action_dim=ACTION_DIM,
        mission_profile=MissionProfile(MissionProfile.BALANCED),
        federated_interval=10,
        comm_delay_steps=5,
    )
    delayed_rewards = []
    for ep in range(EPISODES):
        avg_reward = delayed_swarm.train_episode(max_steps=MAX_STEPS)
        delayed_rewards.append(avg_reward)
        if ep % 40 == 0:
            print(f"  Episode {ep}: Avg Reward = {avg_reward:.2f}")
    delayed_ops = delayed_swarm.test_episode(max_steps=200)
    print(f"  Avg operational steps (delayed comms): {delayed_ops:.1f}")
    all_fed_rewards["balanced+delay5"] = delayed_rewards

    # ------------------------------------------------------------------
    # 5. Formal verification of the best-trained agent's policy
    # ------------------------------------------------------------------
    print("\nRunning formal policy verification...")
    verifier = PolicyVerifier(fed_swarm.agents[0], n_samples=500)
    verifier.verify()

    # ------------------------------------------------------------------
    # 6. Safety monitor demo
    # ------------------------------------------------------------------
    print("\nSafety monitor demo...")
    monitor = SafetyMonitor()
    faulted_state = np.zeros(FederatedSwarm.COORD_STATE_DIM, dtype=np.float32)
    faulted_state[Spacecraft.HEALTH_FLAG_IDX] = 1.0
    vetoed = monitor.veto(0, faulted_state, ACTION_DIM)
    print(f"  Proposed action=0 on faulted state -> vetoed to {vetoed}")
    low_power_state = np.zeros(FederatedSwarm.COORD_STATE_DIM, dtype=np.float32)
    low_power_state[SafetyMonitor.POWER_LEVEL_IDX] = 0.05
    vetoed2 = monitor.veto(2, low_power_state, ACTION_DIM)
    print(f"  Proposed action=2 on critical-power state -> vetoed to {vetoed2}")

    # ------------------------------------------------------------------
    # 7. Gossip-based federated learning demo
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Gossip-based federated learning demo (k=2 neighbours)")
    print("=" * 60)
    gossip_swarm = FederatedSwarm(
        num_spacecraft=SWARM_SIZE,
        action_dim=ACTION_DIM,
        mission_profile=MissionProfile(MissionProfile.BALANCED),
        federated_interval=999,   # disable central FedAvg
    )
    gossip_server = GossipServer(k=2, comm_delay_steps=0)
    gossip_rewards = []
    for ep in range(EPISODES):
        avg_reward = gossip_swarm.train_episode(max_steps=MAX_STEPS)
        gossip_rewards.append(avg_reward)
        if ep % 10 == 0:
            gossip_server.gossip_round(gossip_swarm.agents)
        if ep % 40 == 0:
            print(f"  Episode {ep}: Avg Reward = {avg_reward:.2f}")
    gossip_ops = gossip_swarm.test_episode(max_steps=200)
    print(f"  Avg operational steps (gossip k=2): {gossip_ops:.1f}")
    all_fed_rewards["gossip_k2"] = gossip_rewards

    # ------------------------------------------------------------------
    # 8. Dynamic reward shaping demo
    # ------------------------------------------------------------------
    print("\nDynamic reward shaping demo...")
    dyn_profile = MissionProfile(MissionProfile.POWER_CONSTRAINED)
    r_before = dyn_profile.compute(False, True, 2)
    dyn_profile.update_weights(recovery_bonus_scale=0.5, fault_penalty_scale=2.0)
    r_after = dyn_profile.compute(False, True, 2)
    print(f"  Recovery reward before weight adjustment: {r_before:.2f}")
    print(f"  Recovery reward after  weight adjustment: {r_after:.2f}")

    # ------------------------------------------------------------------
    # 9. Adversarial testing demo
    # ------------------------------------------------------------------
    print("\nRunning adversarial testing on trained agent...")
    adv_tester = AdversarialTester(fed_swarm.agents[0], epsilon=0.05)
    N_ADV = 200
    adv_results = adv_tester.find_adversarial_examples(n_examples=N_ADV)
    adv_tester.summary(adv_results, n_tested=N_ADV)

    # ------------------------------------------------------------------
    # 10. Adversarial training – harden the agent with augmented replay
    # ------------------------------------------------------------------
    print("\nAugmenting replay buffers with adversarial examples...")
    monitor_for_adv = SafetyMonitor()
    n_injected = adv_tester.augment_replay_buffer(
        fed_swarm.agents, n_examples=N_ADV, safety_monitor=monitor_for_adv
    )
    print(f"  Injected {n_injected} adversarial transitions into each agent's buffer.")
    # Run a few more training episodes so the injected examples take effect
    print("  Running 20 adversarial fine-tuning episodes...")
    for ep in range(20):
        for agent in fed_swarm.agents:
            agent.replay()

    # Re-run adversarial test to measure improvement
    adv_results_post = adv_tester.find_adversarial_examples(n_examples=N_ADV)
    print(
        f"  Flip rate after augmentation: "
        f"{len(adv_results_post)}/{N_ADV} "
        f"({len(adv_results_post)/N_ADV*100:.1f}%)"
    )

    # ------------------------------------------------------------------
    # 11. Robustness certification
    # ------------------------------------------------------------------
    print("\nRunning robustness certification (binary-search ε)...")
    cert = adv_tester.certify_robustness(n_samples=100, eps_hi=0.3, n_bisect=10)
    print(f"  Mean certified radius : {cert['mean_radius']:.4f}")
    print(f"  Min  certified radius : {cert['min_radius']:.4f}  (worst-case)")
    print(f"  Max  certified radius : {cert['max_radius']:.4f}  (best-case)")
    print(f"  Robust fraction at ε={cert['eps_hi']}: {cert['robust_frac']*100:.1f}%")

    # ------------------------------------------------------------------
    # 12. Safety-aware training demo
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Safety-aware training with integrated SafetyMonitor")
    print("=" * 60)
    safety_aware_swarm = FederatedSwarm(
        num_spacecraft=SWARM_SIZE,
        action_dim=ACTION_DIM,
        mission_profile=MissionProfile(MissionProfile.BALANCED),
        federated_interval=10,
        safety_monitor=SafetyMonitor(),
    )
    total_overrides = 0
    sa_rewards = []
    for ep in range(EPISODES):
        avg_reward = safety_aware_swarm.train_episode(max_steps=MAX_STEPS)
        sa_rewards.append(avg_reward)
        total_overrides += safety_aware_swarm.last_override_count
        if ep % 40 == 0:
            print(
                f"  Episode {ep}: Avg Reward = {avg_reward:.2f}"
                f"  Overrides = {safety_aware_swarm.last_override_count}"
            )
    sa_ops = safety_aware_swarm.test_episode(max_steps=200)
    print(f"  Avg operational steps (safety-aware): {sa_ops:.1f}")
    print(f"  Total safety veto overrides during training: {total_overrides}")
    all_fed_rewards["safety_aware"] = sa_rewards

    # ------------------------------------------------------------------
    # 13. Federation benchmark (FedAvg vs Gossip across dropout rates)
    # ------------------------------------------------------------------
    print()
    benchmark_federation(
        num_spacecraft=3,
        episodes=30,
        max_steps=80,
        dropout_rates=(0.0, 0.3),
    )

    # ------------------------------------------------------------------
    # 14. LTL reward-shaping demo
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LTL constraint checker demo")
    print("=" * 60)
    ltl = LTLConstraintChecker(penalty=-0.5)
    print(f"  Active LTL constraints: {ltl.active_names()}")
    # Violated: action=0 on faulted state (no_inaction_on_fault)
    fault_st = np.zeros(FederatedSwarm.COORD_STATE_DIM, dtype=np.float32)
    fault_st[Spacecraft.HEALTH_FLAG_IDX] = 1.0
    pen = ltl.evaluate(fault_st, action=0)
    print(f"  Penalty for action=0 on faulted state   : {pen}")
    # Safe action – no violation expected
    pen_safe = ltl.evaluate(fault_st, action=3)
    print(f"  Penalty for action=3 on faulted state   : {pen_safe}")
    # Low-power full-reset violation
    lp_st = np.zeros(FederatedSwarm.COORD_STATE_DIM, dtype=np.float32)
    lp_st[SafetyMonitor.POWER_LEVEL_IDX] = 0.05
    pen_lp = ltl.evaluate(lp_st, action=2)
    print(f"  Penalty for action=2 on low-power state : {pen_lp}")

    # ------------------------------------------------------------------
    # 15. Decision-tree policy extraction demo
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Decision-tree policy extraction (imitation learning)")
    print("=" * 60)
    if _SKLEARN_AVAILABLE:
        dt = extract_decision_tree(fed_swarm.agents[0], n_samples=1000, max_depth=6)
        # Smoke-test the surrogate on a few states
        test_state = np.random.rand(1, fed_swarm.agents[0].state_dim).astype(np.float32)
        dt_action = int(dt.predict(test_state)[0])
        dqn_action = int(
            np.argmax(fed_swarm.agents[0].model.predict(test_state, verbose=0)[0])
        )
        print(f"  DQN action: {dqn_action}  |  Decision-tree action: {dt_action}")
    else:
        print("  scikit-learn not installed; skipping decision-tree demo.")

    # ------------------------------------------------------------------
    # 16. Mission scenario demo
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Mission scenario presets demo")
    print("=" * 60)
    for scenario in (
        MissionScenario.lunar_gateway(),
        MissionScenario.mars_orbiter(),
        MissionScenario.cubesat_swarm(),
    ):
        print(scenario.summary())
        sc_swarm = build_swarm_for_scenario(scenario, safety_monitor=SafetyMonitor())
        sc_reward = sc_swarm.train_episode(max_steps=50)
        print(f"  One-episode reward: {sc_reward:.2f}\n")

    # ------------------------------------------------------------------
    # 17. Save training curves (basic swarm + all profiles + gossip)
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
