"""
sentinel_x/coordination.py – Advanced Swarm Coordination
=========================================================

Two complementary coordination layers that extend the core federated RL
with higher-level mission management:

* **TaskAllocator** – distributed greedy-auction task allocation.  Each
  spacecraft bids on available tasks based on its current health and
  proximity; the highest bidder wins each task.  Automatically re-auctions
  tasks when the winning agent fails.

* **SwarmReconfigurationManager** – monitors swarm health and triggers
  pre-defined reconfiguration actions (leader promotion, formation reshape,
  degraded-mode transition) when fault levels exceed configurable thresholds.

Design principles
-----------------
Both classes are *decoupled* from the federated RL layer:

* They operate on any object that exposes a ``spacecraft`` list and an
  ``is_operational()`` method on each spacecraft.
* They emit structured event logs so the operator dashboard or a
  mission-control bridge can ingest them.

Quickstart
----------
::

    from sentinel_x_advanced import FederatedSwarm, MissionProfile
    from sentinel_x.coordination import TaskAllocator, SwarmReconfigurationManager

    swarm = FederatedSwarm(num_spacecraft=4, action_dim=4,
                           mission_profile=MissionProfile(MissionProfile.BALANCED))

    tasks = [
        {"id": "img_0", "type": "imaging",   "priority": 0.9, "position": (0.0, 200.0)},
        {"id": "rel_0", "type": "relay",     "priority": 0.6, "position": (500.0, 0.0)},
        {"id": "sci_0", "type": "science",   "priority": 0.8, "position": (100.0, 100.0)},
    ]
    allocator = TaskAllocator(swarm)
    assignments = allocator.allocate(tasks)
    print(assignments)   # {task_id: spacecraft_index}

    reconfig = SwarmReconfigurationManager(swarm)
    event = reconfig.check_and_reconfigure()
    print(event)         # None or {"action": ..., "detail": ...}

Theory
------
The auction follows the Greedy Sequential Assignment algorithm:

1. Sort tasks by descending priority.
2. For each task, compute each healthy spacecraft's *bid*:
      bid_i = health_i * proximity_weight(task, agent_i)
3. The spacecraft with the highest bid wins.
4. Remove the winning agent from the bidding pool if ``exclusive=True``
   (one task per agent per allocation round).

Reconfiguration thresholds (all configurable):

* **fault_threshold** – fraction of failed spacecraft that triggers action.
* **promote_leader** – if the current leader (index 0) fails, index *k* is
  promoted (first healthy spacecraft).
* **reshape_formation** – once >40 % of spacecraft have failed, switch to
  a tighter ``"line"`` formation to reduce coordination overhead.
* **degraded_mode** – once >60 % have failed, broadcast a degraded-mode
  event and reduce federation frequency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _health_score(spacecraft) -> float:
    """Return a normalised [0, 1] health score for a spacecraft."""
    try:
        state = spacecraft.state_vector()
        # Use the first element (memory health) and the power level (index 3)
        mem_ok    = float(state[0] == 0)          # 0 errors = healthy
        power_ok  = float(np.clip(state[3], 0.0, 1.0))
        thermal_ok = float(np.clip(1.0 - state[1], 0.0, 1.0))
        return (mem_ok + power_ok + thermal_ok) / 3.0
    except Exception:
        try:
            return float(spacecraft.is_operational())
        except Exception:
            return 0.0


def _proximity_weight(
    task_position: Tuple[float, float],
    agent_position: Tuple[float, float],
    scale: float = 500.0,
) -> float:
    """Gaussian proximity weight ∈ (0, 1] between agent and task positions."""
    dx = task_position[0] - agent_position[0]
    dy = task_position[1] - agent_position[1]
    dist = (dx ** 2 + dy ** 2) ** 0.5
    return float(np.exp(-(dist ** 2) / (2.0 * scale ** 2)))


# ---------------------------------------------------------------------------
# Task allocation
# ---------------------------------------------------------------------------

@dataclass
class AllocationResult:
    """
    Result of a single task-allocation round.

    Attributes
    ----------
    assignments : dict
        ``{task_id: spacecraft_index}`` – winning agent for each task.
    unassigned : list of str
        Task IDs that could not be assigned (no healthy bidder).
    bids : dict
        ``{task_id: {agent_idx: bid_value}}`` – full bid matrix.
    """
    assignments: Dict[str, int] = field(default_factory=dict)
    unassigned:  List[str]      = field(default_factory=list)
    bids:        Dict[str, Dict[int, float]] = field(default_factory=dict)


class TaskAllocator:
    """
    Distributed greedy-auction task allocator.

    Each healthy spacecraft bids on available tasks.  The bid for agent *i*
    on task *t* is:

        bid(i, t) = health_score(i) * proximity_weight(t.position, pos_i)
                    * priority(t)

    Tasks are processed in descending priority order.  Once an agent wins a
    task, it is removed from subsequent bids (``exclusive=True``) or allowed
    to win multiple tasks (``exclusive=False``).

    Parameters
    ----------
    swarm : FederatedSwarm
        The swarm to allocate tasks for.
    agent_positions : list of (float, float), optional
        2-D positions for each spacecraft (metres).  If not provided,
        positions are set to ``(i * 100.0, 0.0)`` for spacecraft *i*.
    exclusive : bool
        If True (default) each spacecraft can win at most one task per round.
    proximity_scale : float
        Gaussian width (metres) for the proximity kernel (default 500 m).

    Attributes
    ----------
    history : list of AllocationResult
        One entry per allocation round (call to :meth:`allocate`).
    """

    def __init__(
        self,
        swarm,
        agent_positions: Optional[List[Tuple[float, float]]] = None,
        exclusive: bool = True,
        proximity_scale: float = 500.0,
    ) -> None:
        self._swarm  = swarm
        self._excl   = exclusive
        self._scale  = float(proximity_scale)
        n = len(swarm.spacecraft)
        if agent_positions is not None:
            if len(agent_positions) != n:
                raise ValueError(
                    f"agent_positions length {len(agent_positions)} != n_spacecraft {n}"
                )
            self._positions: List[Tuple[float, float]] = list(agent_positions)
        else:
            self._positions = [(float(i) * 100.0, 0.0) for i in range(n)]
        self.history: List[AllocationResult] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def update_position(self, agent_idx: int, position: Tuple[float, float]) -> None:
        """Update the 2-D position of a specific spacecraft."""
        if not 0 <= agent_idx < len(self._positions):
            raise IndexError(f"agent_idx {agent_idx} out of range.")
        self._positions[agent_idx] = (float(position[0]), float(position[1]))

    def allocate(
        self,
        tasks: List[Dict[str, Any]],
    ) -> AllocationResult:
        """
        Run one allocation round for *tasks*.

        Parameters
        ----------
        tasks : list of dict
            Each task must have:

            * ``"id"`` (str) – unique task identifier.
            * ``"priority"`` (float ∈ [0, 1]) – task priority.
            * ``"position"`` (tuple[float, float]) – 2-D task location (m).
            * ``"type"`` (str, optional) – task type label (not used in
              bid computation but stored in the result).

        Returns
        -------
        AllocationResult
            Winning assignments and full bid matrix.
        """
        result = AllocationResult()
        n      = len(self._swarm.spacecraft)
        available_agents: set = {
            i for i in range(n)
            if self._swarm.spacecraft[i].is_operational()
        }

        # Sort by descending priority
        sorted_tasks = sorted(
            tasks,
            key=lambda t: float(t.get("priority", 0.0)),
            reverse=True,
        )

        for task in sorted_tasks:
            tid      = str(task["id"])
            priority = float(task.get("priority", 0.5))
            pos      = (float(task["position"][0]), float(task["position"][1]))

            # Compute bids
            bids: Dict[int, float] = {}
            for i in available_agents:
                h  = _health_score(self._swarm.spacecraft[i])
                pw = _proximity_weight(pos, self._positions[i], self._scale)
                bids[i] = h * pw * priority

            result.bids[tid] = bids

            if not bids:
                result.unassigned.append(tid)
                continue

            winner = max(bids, key=lambda i: bids[i])
            result.assignments[tid] = winner
            if self._excl:
                available_agents.discard(winner)

        self.history.append(result)
        return result

    def reallocate_failed(
        self,
        previous: AllocationResult,
        tasks: List[Dict[str, Any]],
    ) -> AllocationResult:
        """
        Re-auction tasks whose winning agent has since failed.

        Parameters
        ----------
        previous : AllocationResult
            The result of the previous allocation round.
        tasks : list of dict
            Full task list (same format as :meth:`allocate`).

        Returns
        -------
        AllocationResult
            New allocation for the affected tasks only.
        """
        task_map = {str(t["id"]): t for t in tasks}
        orphaned = [
            task_map[tid]
            for tid, agent_idx in previous.assignments.items()
            if tid in task_map
            and not self._swarm.spacecraft[agent_idx].is_operational()
        ]
        if not orphaned:
            return AllocationResult()
        return self.allocate(orphaned)

    def summary(self) -> Dict[str, Any]:
        """Return aggregate statistics over all allocation rounds."""
        if not self.history:
            return {"rounds": 0, "total_assigned": 0, "total_unassigned": 0}
        total_assigned   = sum(len(r.assignments) for r in self.history)
        total_unassigned = sum(len(r.unassigned)  for r in self.history)
        return {
            "rounds":           len(self.history),
            "total_assigned":   total_assigned,
            "total_unassigned": total_unassigned,
            "assignment_rate":  (
                total_assigned / max(total_assigned + total_unassigned, 1)
            ),
        }


# ---------------------------------------------------------------------------
# Swarm reconfiguration
# ---------------------------------------------------------------------------

@dataclass
class ReconfigurationEvent:
    """
    A single reconfiguration action taken by the manager.

    Attributes
    ----------
    action : str
        Action taken.  One of ``"leader_promoted"``, ``"formation_reshaped"``,
        ``"degraded_mode"``, ``"recovered"``.
    detail : str
        Human-readable description.
    healthy_fraction : float
        Fraction of operational spacecraft at the time of the event.
    new_leader : int or None
        New leader index if a leader promotion occurred.
    new_formation : str or None
        New formation type if a reshape occurred.
    """
    action:           str
    detail:           str
    healthy_fraction: float
    new_leader:       Optional[int] = None
    new_formation:    Optional[str] = None


class SwarmReconfigurationManager:
    """
    Monitors swarm health and triggers reconfiguration actions.

    Three independent reconfiguration triggers are evaluated in
    :meth:`check_and_reconfigure` (called each episode or on demand):

    1. **Leader promotion** – if spacecraft 0 has failed, promote the first
       healthy spacecraft as the new *logical* leader (recorded in
       :attr:`current_leader`).

    2. **Formation reshape** – once the healthy fraction drops below
       *reshape_threshold* (default 0.6), switch the suggested formation
       type to ``"line"`` to reduce coordination overhead.

    3. **Degraded mode** – once the healthy fraction drops below
       *degraded_threshold* (default 0.4), emit a ``"degraded_mode"``
       event and record the reduced federation interval suggestion.

    All events are appended to :attr:`event_log` regardless of whether a
    :class:`~sentinel_x.formation.FormationController` is actually
    attached.  Callers can inspect the log or register a callback.

    Parameters
    ----------
    swarm : FederatedSwarm
        The swarm to monitor.
    reshape_threshold : float
        Healthy-agent fraction below which a formation reshape is triggered
        (default 0.6).
    degraded_threshold : float
        Healthy-agent fraction below which degraded-mode is declared
        (default 0.4).
    recovery_threshold : float
        Healthy-agent fraction above which a ``"recovered"`` event is emitted
        after a previous degraded-mode state (default 0.7).
    on_event : callable, optional
        Optional callback ``fn(event: ReconfigurationEvent)`` called each
        time a new event is generated.

    Attributes
    ----------
    current_leader : int
        Index of the current logical leader (initially 0).
    suggested_formation : str
        Current suggested formation type.
    is_degraded : bool
        True if the swarm is currently in degraded mode.
    event_log : list of ReconfigurationEvent
        All reconfiguration events in chronological order.
    """

    _FORMATIONS = ("circular", "v", "diamond", "line")

    def __init__(
        self,
        swarm,
        reshape_threshold: float = 0.6,
        degraded_threshold: float = 0.4,
        recovery_threshold: float = 0.7,
        on_event=None,
    ) -> None:
        if not 0.0 < degraded_threshold < reshape_threshold < 1.0:
            raise ValueError(
                "Thresholds must satisfy 0 < degraded_threshold "
                "< reshape_threshold < 1."
            )
        self._swarm      = swarm
        self._reshape_t  = float(reshape_threshold)
        self._degraded_t = float(degraded_threshold)
        self._recovery_t = float(recovery_threshold)
        self._on_event   = on_event

        self.current_leader:      int  = 0
        self.suggested_formation: str  = "circular"
        self.is_degraded:         bool = False
        self.suggested_fed_interval: int = 10
        self.event_log: List[ReconfigurationEvent] = []

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def healthy_fraction(self) -> float:
        """Current fraction of operational spacecraft."""
        n  = len(self._swarm.spacecraft)
        ok = sum(1 for sc in self._swarm.spacecraft if sc.is_operational())
        return float(ok) / max(n, 1)

    def check_and_reconfigure(self) -> Optional[ReconfigurationEvent]:
        """
        Evaluate all reconfiguration triggers and act on the most severe.

        Called each episode.  Returns the highest-severity event generated
        (or ``None`` if no action was taken).

        Returns
        -------
        ReconfigurationEvent or None
        """
        hf = self.healthy_fraction
        event: Optional[ReconfigurationEvent] = None

        # ── 1. Degraded-mode transition ────────────────────────────────────
        if not self.is_degraded and hf < self._degraded_t:
            self.is_degraded = True
            self.suggested_fed_interval = 20   # reduce federation frequency
            event = ReconfigurationEvent(
                action="degraded_mode",
                detail=(
                    f"Swarm degraded: {hf:.0%} healthy "
                    f"(threshold {self._degraded_t:.0%}).  "
                    f"Federation interval → {self.suggested_fed_interval}."
                ),
                healthy_fraction=hf,
            )
            self._emit(event)
            return event

        # ── 2. Recovery from degraded mode ────────────────────────────────
        if self.is_degraded and hf >= self._recovery_t:
            self.is_degraded = False
            self.suggested_fed_interval = 10
            event = ReconfigurationEvent(
                action="recovered",
                detail=(
                    f"Swarm recovered: {hf:.0%} healthy "
                    f"(threshold {self._recovery_t:.0%})."
                ),
                healthy_fraction=hf,
            )
            self._emit(event)
            return event

        # ── 3. Formation reshape ───────────────────────────────────────────
        if hf < self._reshape_t and self.suggested_formation != "line":
            old_formation           = self.suggested_formation
            self.suggested_formation = "line"
            event = ReconfigurationEvent(
                action="formation_reshaped",
                detail=(
                    f"Formation {old_formation!r} → 'line' "
                    f"({hf:.0%} healthy, threshold {self._reshape_t:.0%})."
                ),
                healthy_fraction=hf,
                new_formation="line",
            )
            self._emit(event)
            return event

        # ── 4. Leader promotion ────────────────────────────────────────────
        leader_alive = self._swarm.spacecraft[self.current_leader].is_operational()
        if not leader_alive:
            new_leader = self._find_healthy_spacecraft()
            if new_leader is not None and new_leader != self.current_leader:
                old_leader          = self.current_leader
                self.current_leader = new_leader
                event = ReconfigurationEvent(
                    action="leader_promoted",
                    detail=(
                        f"Leader {old_leader} failed.  "
                        f"Spacecraft {new_leader} promoted as new leader."
                    ),
                    healthy_fraction=hf,
                    new_leader=new_leader,
                )
                self._emit(event)
                return event

        return None

    def force_reconfiguration(self, formation_type: str) -> ReconfigurationEvent:
        """
        Manually override the formation type (e.g., from ground control).

        Parameters
        ----------
        formation_type : str
            New formation.  One of ``"line"``, ``"v"``, ``"diamond"``,
            ``"circular"``.

        Returns
        -------
        ReconfigurationEvent
        """
        if formation_type not in self._FORMATIONS:
            raise ValueError(
                f"Unknown formation_type {formation_type!r}.  "
                f"Choose from {self._FORMATIONS}."
            )
        old = self.suggested_formation
        self.suggested_formation = formation_type
        hf = self.healthy_fraction
        event = ReconfigurationEvent(
            action="formation_reshaped",
            detail=f"Manual override: {old!r} → {formation_type!r}.",
            healthy_fraction=hf,
            new_formation=formation_type,
        )
        self._emit(event)
        return event

    def summary(self) -> Dict[str, Any]:
        """Return event-log summary statistics."""
        counts: Dict[str, int] = {}
        for ev in self.event_log:
            counts[ev.action] = counts.get(ev.action, 0) + 1
        return {
            "total_events":      len(self.event_log),
            "event_counts":      counts,
            "current_leader":    self.current_leader,
            "suggested_formation": self.suggested_formation,
            "is_degraded":       self.is_degraded,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _find_healthy_spacecraft(self) -> Optional[int]:
        """Return the index of the first healthy spacecraft, or None."""
        for i, sc in enumerate(self._swarm.spacecraft):
            if sc.is_operational():
                return i
        return None

    def _emit(self, event: ReconfigurationEvent) -> None:
        self.event_log.append(event)
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                pass
