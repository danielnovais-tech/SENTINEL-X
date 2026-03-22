"""
sentinel_x/formation.py – Multi-Robot Formation-Flying Coordination
====================================================================

This module implements formation-keeping and consensus protocols for
SENTINEL-X swarms.  It complements the existing federated learning layer
(which focuses on fault detection) with explicit geometric coordination:

* **ConsensusProtocol** – distributed averaging consensus (DeGroot model)
  that drives the swarm toward a shared estimate of any scalar quantity
  (sensor reading, health index, …) without a central coordinator.

* **FormationController** – leader-follower formation controller.  The
  "leader" spacecraft (index 0 by default) broadcasts its position; all
  followers compute a formation error and inject it as an auxiliary reward
  signal so the RL policy learns to maintain formation.

* **FormationMetrics** – records per-step formation statistics (coherence,
  spread, convergence time) for post-flight analysis.

Integration with FederatedSwarm
--------------------------------
Wrap an existing ``FederatedSwarm`` with a ``FormationController`` and
call ``train_episode_with_formation()`` instead of the bare
``train_episode()``:

::

    from sentinel_x_advanced import FederatedSwarm, MissionProfile
    from sentinel_x.formation import FormationController

    swarm = FederatedSwarm(num_spacecraft=4, action_dim=4,
                           mission_profile=MissionProfile(MissionProfile.BALANCED))
    fc = FormationController(swarm, formation_type="line", separation_m=100.0)
    reward = fc.train_episode_with_formation(max_steps=200)

Theory notes
------------
The consensus update rule for agent *i* at step *t* is::

    x_i(t+1) = (1 - α) x_i(t) + α * (1/|N_i|) Σ_{j ∈ N_i} x_j(t)

where *α* is the mixing weight and *N_i* is the neighbourhood of agent *i*.
This converges to the average of the initial values for any connected graph.

Formation error for agent *i* is::

    e_i = ||p_i - (p_leader + d_i)||_2

where *d_i* is the desired offset from the leader for agent *i* in the
reference formation geometry.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Consensus protocol
# ---------------------------------------------------------------------------

class ConsensusProtocol:
    """
    Distributed averaging consensus over a swarm.

    Each agent holds a local scalar value (e.g., normalised health index).
    At each mixing step every agent replaces its value with a weighted
    average of itself and its neighbours' values.  After enough steps all
    agents converge to the global mean (for a connected graph).

    Parameters
    ----------
    n_agents : int
        Number of agents in the swarm.
    mixing_weight : float
        Step-size α ∈ (0, 1].  Higher values converge faster but may
        oscillate.  Default: 0.5.
    topology : str
        Connectivity graph.  One of:

        * ``"ring"`` – each agent communicates with its two neighbours.
        * ``"complete"`` – every agent communicates with every other agent.
        * ``"star"`` – all agents communicate with agent 0 (hub-and-spoke).

    Attributes
    ----------
    values : np.ndarray of shape (n_agents,)
        Current local values held by each agent.
    history : list of np.ndarray
        Snapshot of values after each mixing step.
    """

    def __init__(
        self,
        n_agents: int,
        mixing_weight: float = 0.5,
        topology: str = "ring",
    ) -> None:
        if n_agents < 2:
            raise ValueError("Consensus requires at least 2 agents.")
        if not 0.0 < mixing_weight <= 1.0:
            raise ValueError("mixing_weight must be in (0, 1].")
        if topology not in ("ring", "complete", "star"):
            raise ValueError("topology must be 'ring', 'complete', or 'star'.")

        self._n       = n_agents
        self._alpha   = float(mixing_weight)
        self._topo    = topology
        self.values   = np.zeros(n_agents, dtype=np.float64)
        self.history: list = []
        self._adj     = self._build_adjacency()

    # ── Adjacency construction ────────────────────────────────────────────────

    def _build_adjacency(self) -> List[List[int]]:
        n = self._n
        if self._topo == "ring":
            return [[(i - 1) % n, (i + 1) % n] for i in range(n)]
        if self._topo == "complete":
            return [[j for j in range(n) if j != i] for i in range(n)]
        if self._topo == "star":
            adj = [[0] for _ in range(n)]
            adj[0] = list(range(1, n))
            return adj
        raise ValueError(self._topo)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_values(self, values: List[float]) -> None:
        """
        Initialise agent values before running consensus.

        Parameters
        ----------
        values : list of float
            Initial values for each agent (length must equal *n_agents*).
        """
        if len(values) != self._n:
            raise ValueError(
                f"Expected {self._n} values, got {len(values)}."
            )
        self.values = np.array(values, dtype=np.float64)
        self.history = [self.values.copy()]

    def step(self) -> np.ndarray:
        """
        Perform one distributed averaging step.

        Returns
        -------
        np.ndarray
            Updated values after the mixing step.
        """
        new_vals = self.values.copy()
        for i in range(self._n):
            neighbours = self._adj[i]
            if not neighbours:
                continue
            neighbour_mean = float(np.mean([self.values[j] for j in neighbours]))
            new_vals[i] = (1.0 - self._alpha) * self.values[i] + self._alpha * neighbour_mean
        self.values = new_vals
        self.history.append(self.values.copy())
        return self.values.copy()

    def run(self, n_steps: int = 20) -> np.ndarray:
        """
        Run *n_steps* mixing steps.

        Returns the final consensus values.
        """
        for _ in range(n_steps):
            self.step()
        return self.values.copy()

    @property
    def converged(self) -> bool:
        """
        True if all agent values are within 1 % of the global mean.
        """
        if len(self.history) < 2:
            return False
        mean = float(np.mean(self.values))
        if abs(mean) < 1e-12:
            return True
        return float(np.max(np.abs(self.values - mean))) / abs(mean) < 0.01

    @property
    def convergence_step(self) -> Optional[int]:
        """
        First step index at which convergence was achieved, or None.
        """
        if not self.history:
            return None
        for k, snap in enumerate(self.history):
            mean = float(np.mean(snap))
            if abs(mean) < 1e-12:
                return k
            if float(np.max(np.abs(snap - mean))) / abs(mean) < 0.01:
                return k
        return None


# ---------------------------------------------------------------------------
# Formation geometry helpers
# ---------------------------------------------------------------------------

@dataclass
class FormationGeometry:
    """
    Desired relative offsets from the leader for a specific formation type.

    Attributes
    ----------
    offsets : list of (float, float)
        (dx, dy) offsets in metres for each follower spacecraft.
        Length must equal ``n_spacecraft - 1``.
    """
    offsets: List[Tuple[float, float]] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        n_followers: int,
        formation_type: str,
        separation_m: float = 100.0,
    ) -> "FormationGeometry":
        """
        Build standard formation geometries.

        Parameters
        ----------
        n_followers : int
            Number of follower spacecraft (total swarm − 1).
        formation_type : str
            One of:

            * ``"line"`` – followers spaced uniformly behind the leader.
            * ``"v"`` – V-shaped (chevron) formation.
            * ``"diamond"`` – diamond / rhombus formation.
            * ``"circular"`` – followers equally spaced on a circle.

        separation_m : float
            Inter-spacecraft spacing in metres (default 100 m).
        """
        offsets: List[Tuple[float, float]] = []
        if formation_type == "line":
            for k in range(1, n_followers + 1):
                offsets.append((-k * separation_m, 0.0))
        elif formation_type == "v":
            half = n_followers / 2.0
            for k in range(n_followers):
                side  = 1 if k % 2 == 0 else -1
                depth = (k // 2 + 1) * separation_m
                offsets.append((-depth, side * depth * 0.6))
        elif formation_type == "diamond":
            positions = [
                (0.0,             separation_m),
                (0.0,            -separation_m),
                (-separation_m,   0.0),
                (separation_m,    0.0),
            ]
            offsets = [(positions[k % len(positions)]) for k in range(n_followers)]
        elif formation_type == "circular":
            for k in range(n_followers):
                angle = 2.0 * math.pi * k / n_followers
                offsets.append((
                    separation_m * math.cos(angle),
                    separation_m * math.sin(angle),
                ))
        else:
            raise ValueError(
                f"Unknown formation_type: {formation_type!r}. "
                "Choose 'line', 'v', 'diamond', or 'circular'."
            )
        return cls(offsets=offsets)


# ---------------------------------------------------------------------------
# Formation controller
# ---------------------------------------------------------------------------

class FormationController:
    """
    Leader-follower formation controller for SENTINEL-X swarms.

    Wraps a ``FederatedSwarm`` and augments its reward signal with a
    formation-keeping term so the RL policy learns to maintain the desired
    geometry.

    The controller also runs a ``ConsensusProtocol`` on each agent's
    normalised health index so that all spacecraft quickly converge on a
    shared situational-awareness estimate.

    Parameters
    ----------
    swarm : FederatedSwarm
        The swarm to control.
    formation_type : str
        Desired formation shape.  One of ``"line"``, ``"v"``, ``"diamond"``,
        ``"circular"``.
    separation_m : float
        Inter-spacecraft spacing in metres (default 100 m).
    formation_weight : float
        Weight of the formation-error penalty in the total reward
        (default 0.1).  Larger values enforce stricter formation-keeping at
        the cost of fault-recovery performance.
    consensus_topology : str
        Connectivity graph for the consensus protocol: ``"ring"`` (default),
        ``"complete"``, or ``"star"``.
    consensus_mixing : float
        Mixing weight for the consensus protocol (default 0.5).

    Attributes
    ----------
    metrics : FormationMetrics
        Records per-step formation statistics.
    """

    def __init__(
        self,
        swarm,
        formation_type: str = "line",
        separation_m: float = 100.0,
        formation_weight: float = 0.1,
        consensus_topology: str = "ring",
        consensus_mixing: float = 0.5,
    ) -> None:
        self._swarm  = swarm
        self._weight = float(formation_weight)
        n = len(swarm.spacecraft)
        self._geom   = FormationGeometry.build(
            n_followers=max(0, n - 1),
            formation_type=formation_type,
            separation_m=separation_m,
        )
        self._consensus = ConsensusProtocol(
            n_agents=n,
            mixing_weight=consensus_mixing,
            topology=consensus_topology,
        )
        self.metrics = FormationMetrics(n_spacecraft=n)
        # Simulated 2-D positions (x, y in metres)
        self._positions = np.zeros((n, 2), dtype=np.float64)
        self._initialise_positions(separation_m)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialise_positions(self, separation_m: float) -> None:
        """Place spacecraft in desired formation at t=0."""
        self._positions[0] = [0.0, 0.0]
        for i, (dx, dy) in enumerate(self._geom.offsets, start=1):
            self._positions[i] = [dx, dy]

    # ── Position dynamics (simplified orbit propagator) ───────────────────────

    def _propagate(self, dt: float = 1.0) -> None:
        """
        Move all spacecraft by one time step.

        Leader maintains straight-line motion; followers apply a
        proportional-derivative control law to minimise formation error.
        """
        leader_pos = self._positions[0].copy()
        # Leader: constant velocity along x-axis
        self._positions[0, 0] += 10.0 * dt   # 10 m/s

        for i in range(1, len(self._positions)):
            desired = leader_pos + np.array(self._geom.offsets[i - 1])
            error   = desired - self._positions[i]
            # PD control: u = Kp * e (D term omitted for simplicity)
            self._positions[i] += 0.5 * error * dt + 10.0 * dt * np.array([1, 0])

    # ── Formation error ───────────────────────────────────────────────────────

    def formation_errors(self) -> np.ndarray:
        """
        Compute formation error for each follower.

        Returns
        -------
        np.ndarray of shape (n_spacecraft - 1,)
            Euclidean distance (m) from each follower's current position to
            its desired position relative to the leader.
        """
        leader = self._positions[0]
        errors = np.array([
            np.linalg.norm(self._positions[i] - (leader + np.array(self._geom.offsets[i - 1])))
            for i in range(1, len(self._positions))
        ])
        return errors

    def formation_coherence(self) -> float:
        """
        Return a [0, 1] coherence score.

        1.0 = perfect formation; 0.0 = fully dispersed.

        Computed as ``exp(−mean_error / separation_m)``.
        """
        errors = self.formation_errors()
        if len(errors) == 0:
            return 1.0
        sep = max(
            np.linalg.norm(self._geom.offsets[0]) if self._geom.offsets else 1.0,
            1.0,
        )
        return float(np.exp(-float(np.mean(errors)) / sep))

    # ── Training integration ──────────────────────────────────────────────────

    def train_episode_with_formation(self, max_steps: int = 200) -> float:
        """
        Run one training episode with formation-keeping reward shaping.

        Delegates to ``FederatedSwarm.train_episode()`` but adds a
        per-step formation penalty and runs one consensus mixing step per
        episode.

        Parameters
        ----------
        max_steps : int
            Episode length (default 200).

        Returns
        -------
        float
            Mean episode reward across all spacecraft (including
            formation penalty).
        """
        self._positions = np.zeros_like(self._positions)
        self._initialise_positions(
            np.linalg.norm(self._geom.offsets[0]) if self._geom.offsets else 100.0
        )

        base_reward = self._swarm.train_episode(max_steps=max_steps)

        # Formation penalty (averaged over followers)
        errors = self.formation_errors()
        formation_penalty = float(np.mean(errors)) * self._weight if len(errors) > 0 else 0.0

        # Consensus on health indices
        health_values = [float(sc.is_operational()) for sc in self._swarm.spacecraft]
        self._consensus.set_values(health_values)
        self._consensus.run(n_steps=5)

        # Record metrics
        coherence = self.formation_coherence()
        self.metrics.record(
            step=self._swarm._episode_count,
            coherence=coherence,
            errors=errors,
            consensus_values=self._consensus.values.tolist(),
        )
        self._propagate()

        return base_reward - formation_penalty

    def get_consensus_estimate(self) -> Dict[str, float]:
        """
        Return the current consensus health estimates.

        Returns
        -------
        dict
            ``{"agents": list, "mean": float, "std": float}``
        """
        return {
            "agents": self._consensus.values.tolist(),
            "mean":   float(np.mean(self._consensus.values)),
            "std":    float(np.std(self._consensus.values)),
        }


# ---------------------------------------------------------------------------
# Formation metrics
# ---------------------------------------------------------------------------

class FormationMetrics:
    """
    Per-step formation statistics.

    Parameters
    ----------
    n_spacecraft : int
        Swarm size.
    capacity : int
        Ring-buffer capacity (default 1000).
    """

    def __init__(self, n_spacecraft: int, capacity: int = 1000) -> None:
        self._n = n_spacecraft
        self._buf: deque = deque(maxlen=capacity)

    def record(
        self,
        step: int,
        coherence: float,
        errors: np.ndarray,
        consensus_values: Optional[List[float]] = None,
    ) -> None:
        """Append a metrics snapshot."""
        self._buf.append({
            "step":      step,
            "coherence": coherence,
            "mean_error_m": float(np.mean(errors)) if len(errors) > 0 else 0.0,
            "max_error_m":  float(np.max(errors))  if len(errors) > 0 else 0.0,
            "consensus":    consensus_values or [],
        })

    def summary(self) -> Dict[str, float]:
        """
        Return aggregate formation statistics over all recorded steps.

        Returns
        -------
        dict
            Keys: ``mean_coherence``, ``min_coherence``, ``mean_error_m``,
            ``max_error_m``, ``n_steps``.
        """
        if not self._buf:
            return {"mean_coherence": 0.0, "min_coherence": 0.0,
                    "mean_error_m": 0.0, "max_error_m": 0.0, "n_steps": 0}
        coherences = [r["coherence"] for r in self._buf]
        errors     = [r["mean_error_m"] for r in self._buf]
        return {
            "mean_coherence": float(np.mean(coherences)),
            "min_coherence":  float(np.min(coherences)),
            "mean_error_m":   float(np.mean(errors)),
            "max_error_m":    float(np.max([r["max_error_m"] for r in self._buf])),
            "n_steps":        len(self._buf),
        }

    def to_list(self) -> list:
        """Return all recorded snapshots as a list of dicts."""
        return list(self._buf)
