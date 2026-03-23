#!/usr/bin/env python3
"""
scripts/dashboard.py – SENTINEL-X Operator Dashboard
=====================================================

A terminal-based real-time operator dashboard for monitoring spacecraft
swarm health, agent actions, and mission statistics.

Two rendering backends are provided:

* **Rich TUI** (default) – uses the ``rich`` library for a colourful,
  well-structured terminal dashboard.  Refreshes every second.
  Install: ``pip install rich``

* **Plain-text fallback** – used automatically when ``rich`` is not
  installed (e.g., minimal CI environments).  Prints a text table to
  stdout on each refresh.

Features
--------
* Per-spacecraft health status (✓ / ✗), fault counts, and last action.
* Swarm-level statistics: mean reward, override count, federated rounds.
* Formation coherence (if a ``FormationController`` is attached).
* Safety override count and LTL violation count.
* Live episode counter and elapsed time.
* Press **Q** to quit gracefully.

Usage
-----
::

    # Live dashboard with a simulated swarm (default)
    python scripts/dashboard.py

    # Watch a specific number of episodes then exit
    python scripts/dashboard.py --episodes 20

    # Adjust swarm size and refresh rate
    python scripts/dashboard.py --spacecraft 6 --refresh 0.5

    # Plain-text mode (no rich required)
    python scripts/dashboard.py --plain

    # Stream live telemetry from a connected STM32 / RPi
    python scripts/dashboard.py --port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import sys
import time
import threading
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional rich import
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


# ---------------------------------------------------------------------------
# Shared dashboard state
# ---------------------------------------------------------------------------

class DashboardState:
    """
    Thread-safe container for dashboard state.

    Attributes updated by the training thread; read by the rendering thread.
    """

    def __init__(self, n_spacecraft: int) -> None:
        self._lock          = threading.Lock()
        self.n_spacecraft   = n_spacecraft
        self.episode        = 0
        self.step           = 0
        self.elapsed_s      = 0.0
        self.rewards: List[float] = []
        self.overrides      = 0
        self.ltl_penalties  = 0.0
        self.fed_rounds     = 0
        self.formation_coherence: Optional[float] = None

        # Per-spacecraft
        self.health:   List[bool]  = [True]  * n_spacecraft
        self.actions:  List[int]   = [0]     * n_spacecraft
        self.fault_counts: List[int] = [0]   * n_spacecraft
        self.stopped    = False

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "episode":    self.episode,
                "step":       self.step,
                "elapsed_s":  self.elapsed_s,
                "rewards":    list(self.rewards),
                "overrides":  self.overrides,
                "ltl":        self.ltl_penalties,
                "fed_rounds": self.fed_rounds,
                "health":     list(self.health),
                "actions":    list(self.actions),
                "fault_counts": list(self.fault_counts),
                "coherence":  self.formation_coherence,
            }


# ---------------------------------------------------------------------------
# Training thread
# ---------------------------------------------------------------------------

ACTION_LABELS = ["DO_NOTHING", "RESTART", "SWITCH_RED.", "SAFE_MODE"]
_FAULT_PROB = 0.05   # per-step fault injection probability


def _training_thread(
    state: DashboardState,
    n_episodes: int,
    port: Optional[str] = None,
    baud: int = 115_200,
) -> None:
    """Run training loop in a background thread, updating DashboardState.

    When *port* is given the loop reads live sensor telemetry from the
    connected MCU (via :class:`hardware.stm32_driver.STM32Driver`) and
    injects it into the first spacecraft's fault model for each step.
    """
    import sentinel_x_advanced as sx

    # Optional real-hardware driver
    hw_driver = None
    if port:
        try:
            from hardware.stm32_driver import STM32Driver
            hw_driver = STM32Driver(port=port, baud_rate=baud)
            hw_driver.open()
        except Exception as exc:  # noqa: BLE001
            print(f"[dashboard] WARNING: could not open {port}: {exc}",
                  file=sys.stderr)
            hw_driver = None

    swarm = sx.FederatedSwarm(
        num_spacecraft=state.n_spacecraft,
        action_dim=4,
        mission_profile=sx.MissionProfile(sx.MissionProfile.BALANCED),
        safety_monitor=sx.SafetyMonitor(),
        cooperative_bonus=0.5,
    )

    t0 = time.monotonic()
    for ep in range(n_episodes):
        if state.stopped:
            break

        # When real hardware is connected, read one sensor frame and use the
        # health_flag from the MCU to override the first spacecraft's state.
        if hw_driver is not None:
            try:
                reading = hw_driver.read_sensors()
                hw_vec  = reading.to_state_vector()
                # Inject telemetry: mark spacecraft 0 as faulty if health_flag
                if reading.health_flag and swarm.spacecraft:
                    swarm.spacecraft[0].inject_fault("hardware_flag")
            except Exception:  # noqa: BLE001
                pass  # Don't crash the dashboard on transient UART errors

        reward = swarm.train_episode(max_steps=100)

        # Collect per-spacecraft state
        health      = []
        actions_ep  = []
        fault_total = []
        for i, sc in enumerate(swarm.spacecraft):
            h = sc.is_operational()
            health.append(bool(h))
            # Latest action: read from agent's last act
            a = int(swarm.agents[i].act(sc.get_state(), training=False))
            actions_ep.append(a)
            fc = getattr(sc, "_fault_count", 0)
            fault_total.append(int(fc))

        state.update(
            episode   = ep + 1,
            elapsed_s = time.monotonic() - t0,
            rewards   = getattr(state, "rewards", []) + [reward],
            overrides = swarm.last_override_count,
            ltl       = swarm.last_ltl_penalty,
            fed_rounds= swarm._episode_count,
            health    = health,
            actions   = actions_ep,
            fault_counts = fault_total,
        )

    state.stopped = True
    if hw_driver is not None:
        try:
            hw_driver.close()
        except Exception:  # noqa: BLE001
            pass
# ---------------------------------------------------------------------------

def _make_spacecraft_table(snap: Dict) -> Table:
    tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    tbl.add_column("ID",      justify="right",  width=4)
    tbl.add_column("Health",  justify="center", width=8)
    tbl.add_column("Faults",  justify="right",  width=7)
    tbl.add_column("Action",  justify="left",   width=14)

    for i in range(snap["n_spacecraft"] if "n_spacecraft" in snap else len(snap["health"])):
        h = snap["health"][i] if i < len(snap["health"]) else True
        a = snap["actions"][i] if i < len(snap["actions"]) else 0
        fc = snap["fault_counts"][i] if i < len(snap["fault_counts"]) else 0
        health_str = Text("✓ OK",   style="green") if h else Text("✗ FAULT", style="red bold")
        action_str = Text(ACTION_LABELS[a], style="yellow" if a > 0 else "dim")
        tbl.add_row(str(i), health_str, str(fc), action_str)
    return tbl


def _make_stats_panel(snap: Dict) -> Panel:
    ep       = snap["episode"]
    elapsed  = snap["elapsed_s"]
    rewards  = snap["rewards"]
    mean_rew = float(np.mean(rewards[-10:])) if rewards else 0.0
    overrides = snap["overrides"]
    fed_rounds = snap["fed_rounds"]
    coherence  = snap.get("coherence")
    ltl        = snap.get("ltl", 0.0)

    lines = [
        f"[bold]Episode[/bold]      {ep}",
        f"[bold]Elapsed[/bold]      {elapsed:.1f} s",
        f"[bold]Mean Reward[/bold]  {mean_rew:+.2f}  (last 10 ep)",
        f"[bold]Overrides[/bold]    {overrides}",
        f"[bold]LTL penalty[/bold]  {ltl:.3f}",
        f"[bold]Fed. rounds[/bold]  {fed_rounds}",
    ]
    if coherence is not None:
        lines.append(f"[bold]Formation[/bold]    {coherence:.2%}")

    return Panel("\n".join(lines), title="[bold blue]Mission Statistics", border_style="blue")


def _run_rich_dashboard(state: DashboardState, refresh: float) -> None:
    console = Console()

    def build() -> Layout:
        snap = state.snapshot()
        snap["n_spacecraft"] = state.n_spacecraft

        layout = Layout()
        layout.split_column(
            Layout(name="header",  size=3),
            Layout(name="body"),
            Layout(name="footer",  size=2),
        )
        layout["body"].split_row(
            Layout(name="spacecraft", ratio=3),
            Layout(name="stats",      ratio=2),
        )

        ep = snap["episode"]
        layout["header"].update(
            Panel(f"[bold yellow]SENTINEL-X Operator Dashboard[/bold yellow]"
                  f"  |  Episode {ep}  |  Press Ctrl+C to quit",
                  style="on dark_blue")
        )
        layout["spacecraft"].update(
            Panel(_make_spacecraft_table(snap),
                  title="[bold cyan]Spacecraft Status", border_style="cyan")
        )
        layout["stats"].update(_make_stats_panel(snap))

        healthy_count = sum(1 for h in snap["health"] if h)
        total         = state.n_spacecraft
        layout["footer"].update(
            Panel(f"[green]{healthy_count}/{total} healthy[/green]"
                  f"  |  [dim]Refreshing every {refresh:.1f}s[/dim]",
                  border_style="dim")
        )
        return layout

    with Live(build(), refresh_per_second=int(1.0 / max(refresh, 0.1)),
              console=console, screen=True) as live:
        while not state.stopped:
            try:
                time.sleep(refresh)
                live.update(build())
            except KeyboardInterrupt:
                break
    state.stopped = True


# ---------------------------------------------------------------------------
# Plain-text renderer
# ---------------------------------------------------------------------------

def _run_plain_dashboard(state: DashboardState, refresh: float) -> None:
    while not state.stopped:
        try:
            snap = state.snapshot()
            rewards  = snap["rewards"]
            mean_rew = float(np.mean(rewards[-10:])) if rewards else 0.0

            sep = "-" * 52
            print(sep)
            print(f"SENTINEL-X Dashboard  |  Episode {snap['episode']}"
                  f"  |  {snap['elapsed_s']:.1f}s elapsed")
            print(sep)
            print(f"{'ID':>3}  {'Health':^10}  {'Faults':>6}  {'Last Action':<14}")
            print(sep)
            for i in range(state.n_spacecraft):
                h  = snap["health"][i]  if i < len(snap["health"])  else True
                a  = snap["actions"][i] if i < len(snap["actions"]) else 0
                fc = snap["fault_counts"][i] if i < len(snap["fault_counts"]) else 0
                h_str = " OK    " if h else "FAULT  "
                print(f"{i:>3}  {h_str:^10}  {fc:>6}  {ACTION_LABELS[a]:<14}")
            print(sep)
            print(f"Mean Reward (last 10): {mean_rew:+.2f}  "
                  f"Overrides: {snap['overrides']}  "
                  f"Fed rounds: {snap['fed_rounds']}")
            if snap.get("coherence") is not None:
                print(f"Formation coherence: {snap['coherence']:.2%}")
            print()
            time.sleep(refresh)
        except KeyboardInterrupt:
            break
    state.stopped = True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SENTINEL-X real-time operator dashboard"
    )
    parser.add_argument("--spacecraft", type=int,   default=4,
                        help="Number of spacecraft in the swarm (default 4)")
    parser.add_argument("--episodes",   type=int,   default=1000,
                        help="Number of training episodes to run (default 1000)")
    parser.add_argument("--refresh",    type=float, default=1.0,
                        help="Dashboard refresh interval in seconds (default 1.0)")
    parser.add_argument("--plain",      action="store_true",
                        help="Force plain-text output (no rich required)")
    parser.add_argument("--port",       default=None,
                        help=(
                            "Serial port for live hardware telemetry "
                            "(e.g. /dev/ttyUSB0). "
                            "When provided, sensor readings are streamed from "
                            "the connected MCU instead of the simulator."
                        ))
    parser.add_argument("--baud",       type=int,   default=115_200,
                        help="UART baud rate when --port is used (default 115200)")
    args = parser.parse_args(argv)

    use_rich = _HAS_RICH and not args.plain

    state = DashboardState(n_spacecraft=args.spacecraft)

    # Start training in background thread
    train_thread = threading.Thread(
        target=_training_thread,
        args=(state, args.episodes),
        kwargs={"port": args.port, "baud": args.baud},
        daemon=True,
    )
    train_thread.start()

    # Run dashboard in main thread
    try:
        if use_rich:
            _run_rich_dashboard(state, args.refresh)
        else:
            _run_plain_dashboard(state, args.refresh)
    except KeyboardInterrupt:
        state.stopped = True

    train_thread.join(timeout=5.0)
    print("\nDashboard closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
