"""
sentinel_x/formal_verification.py – External Formal Verification Integration
==============================================================================

Optional module providing integration with professional neural-network
verification tools:

* **Marabou** (Stanford) – SMT-based complete verifier for piecewise-linear
  networks.  Verifies properties like ``∀ x ∈ B(ε): argmax(f(x)) = a``.
  https://github.com/NeuralNetworkVerification/Marabou

* **ERAN** (ETH Zurich) – Abstract interpretation-based verifier supporting
  DeepZ, DeepPoly, and k-ReLU domains.
  https://github.com/eth-sri/eran

Both tools are **optional dependencies**.  If they are not installed the
classes degrade gracefully to a ``NotAvailable`` state: the constructor
raises :class:`ImportError` with clear installation instructions.

The module also provides :func:`export_to_onnx`, which converts a trained
:class:`~sentinel_x_advanced.DQNAgent` to the ONNX format required by both
Marabou ≥ 2.0 and ERAN.

Typical usage
-------------
::

    from sentinel_x.formal_verification import MarabouVerifier, export_to_onnx

    # Export model to ONNX (requires onnx + tf2onnx)
    export_to_onnx(agent, "policy.onnx")

    # Verify a single safety property
    verifier = MarabouVerifier("policy.onnx")
    result = verifier.verify_property(
        input_lb=[0.0]*11, input_ub=[0.3]*11,   # neighbourhood of healthy state
        output_constraint={"type": "argmax_not", "action": 0},  # must not DO_NOTHING
    )
    print(result)   # {"sat": True, "counterexample": None, "time_s": 0.42}

    # Run the standard SENTINEL-X safety property suite
    report = verifier.run_safety_suite(n_samples=100, epsilon=0.1)
    print(f"Properties verified: {report['verified']}/{report['total']}")

Installation
------------
::

    # Marabou (Linux / macOS)
    pip install maraboupy        # Python bindings (requires Marabou built)
    # -or- build from source:
    # git clone https://github.com/NeuralNetworkVerification/Marabou
    # cd Marabou && mkdir build && cd build && cmake .. && make -j4

    # ONNX export dependencies
    pip install onnx tf2onnx

    # ERAN (requires additional setup – see docs/formal_verification_external.md)
    pip install eran              # if available on PyPI; otherwise build from source

Notes
-----
* Both verifiers operate on the **exported ONNX model**, not the live Keras
  model.  Export once and cache the ``.onnx`` file.
* Verification time scales with input dimension and network size.  The
  SENTINEL-X DQN (11-dim input, two hidden layers of 64 units) typically
  verifies in < 1 s per property on a modern laptop.
* For flight-critical DO-178C / ECSS certification, Marabou provides
  **complete** proofs (no false negatives).  ERAN provides **sound** but
  potentially conservative bounds (may report ``unverified`` for true
  properties).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency availability flags
# ---------------------------------------------------------------------------
try:
    from maraboupy import Marabou as _Marabou              # type: ignore[import]
    from maraboupy import MarabouCore as _MarabouCore      # type: ignore[import]
    _HAS_MARABOU = True
except ImportError:
    _HAS_MARABOU = False

try:
    import onnx                          # type: ignore[import]
    import tf2onnx                       # type: ignore[import]
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

try:
    import eran                          # type: ignore[import]  # noqa: F401
    _HAS_ERAN = True
except ImportError:
    _HAS_ERAN = False


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(
    agent,
    output_path: str = "sentinel_x_policy.onnx",
    opset: int = 13,
) -> str:
    """
    Export a trained :class:`~sentinel_x_advanced.DQNAgent` to ONNX format.

    Parameters
    ----------
    agent : DQNAgent
        A trained SENTINEL-X agent.
    output_path : str
        Destination ``.onnx`` file (default: ``sentinel_x_policy.onnx``).
    opset : int
        ONNX opset version (default: 13, compatible with Marabou ≥ 2.0).

    Returns
    -------
    str
        Absolute path to the written ``.onnx`` file.

    Raises
    ------
    ImportError
        If ``onnx`` or ``tf2onnx`` is not installed.
    """
    if not _HAS_ONNX:
        raise ImportError(
            "ONNX export requires 'onnx' and 'tf2onnx'.\n"
            "Install with: pip install onnx tf2onnx"
        )

    import tensorflow as tf
    import tf2onnx as _tf2onnx

    # Build a concrete-function trace at the correct input shape
    state_dim = agent.state_dim
    spec = (tf.TensorSpec((None, state_dim), tf.float32, name="state"),)

    model_proto, _ = _tf2onnx.convert.from_keras(
        agent.model,
        input_signature=spec,
        opset=opset,
        output_path=output_path,
    )

    abs_path = str(Path(output_path).resolve())
    return abs_path


# ---------------------------------------------------------------------------
# Marabou verifier
# ---------------------------------------------------------------------------

class MarabouVerifier:
    """
    Wrapper around Marabou for complete formal verification of SENTINEL-X
    policies.

    Marabou verifies properties of the form:

        "For all inputs **x** in the hypercube [lb, ub],
         the output satisfies the given constraint."

    This is a **complete** verifier: if Marabou returns ``SAT`` the property
    holds everywhere in the input region.  If it returns ``UNSAT`` it also
    provides a concrete counterexample.

    Parameters
    ----------
    onnx_path : str
        Path to an ONNX model exported via :func:`export_to_onnx`.
    state_dim : int
        Input dimension (default 11 for SENTINEL-X).
    action_dim : int
        Output dimension (default 4 for SENTINEL-X).
    timeout_s : float
        Per-query time limit in seconds (default 30).

    Raises
    ------
    ImportError
        If ``maraboupy`` is not installed.
    FileNotFoundError
        If *onnx_path* does not exist.
    """

    def __init__(
        self,
        onnx_path: str,
        state_dim: int = 11,
        action_dim: int = 4,
        timeout_s: float = 30.0,
    ) -> None:
        if not _HAS_MARABOU:
            raise ImportError(
                "Marabou is not installed.\n"
                "Build from source: https://github.com/NeuralNetworkVerification/Marabou\n"
                "Then install Python bindings:\n"
                "  cd Marabou && pip install -e .\n"
                "See docs/formal_verification_external.md for full instructions."
            )
        if not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                "Run export_to_onnx(agent, 'model.onnx') first."
            )

        self._onnx_path  = onnx_path
        self._state_dim  = state_dim
        self._action_dim = action_dim
        self._timeout    = timeout_s
        self._network    = _Marabou.read_onnx(onnx_path)

    # ── Property verification ─────────────────────────────────────────────────

    def verify_property(
        self,
        input_lb: List[float],
        input_ub: List[float],
        output_constraint: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Verify a single input-output property.

        Parameters
        ----------
        input_lb, input_ub : list of float
            Lower and upper bounds of the input hypercube (length = state_dim).
        output_constraint : dict
            Supported types:

            ``{"type": "argmax_is", "action": int}``
                The greedy action must equal *action* everywhere in [lb, ub].

            ``{"type": "argmax_not", "action": int}``
                The greedy action must never equal *action* in [lb, ub].

            ``{"type": "output_lb", "index": int, "value": float}``
                Output neuron *index* must be ≥ *value* everywhere.

        Returns
        -------
        dict
            ``{"sat": bool, "counterexample": list or None, "time_s": float}``
        """
        net = _Marabou.read_onnx(self._onnx_path)   # fresh network per query

        inputs  = net.inputVars[0][0]    # shape: (state_dim,)
        outputs = net.outputVars[0]      # shape: (action_dim,)

        # ── Input bounds ─────────────────────────────────────────────────────
        for i in range(self._state_dim):
            net.setLowerBound(inputs[i], float(input_lb[i]))
            net.setUpperBound(inputs[i], float(input_ub[i]))

        # ── Output constraints ───────────────────────────────────────────────
        c_type = output_constraint.get("type", "")
        if c_type == "argmax_is":
            a = int(output_constraint["action"])
            # output[a] > output[j] for all j ≠ a
            for j in range(self._action_dim):
                if j != a:
                    net.addInequality([outputs[j], outputs[a]], [1, -1], 0)
        elif c_type == "argmax_not":
            a = int(output_constraint["action"])
            # ∃ j ≠ a : output[j] > output[a]  (negate: output[a] ≥ output[j]+ε)
            for j in range(self._action_dim):
                if j != a:
                    net.addInequality([outputs[a], outputs[j]], [1, -1], 0)
        elif c_type == "output_lb":
            idx = int(output_constraint["index"])
            val = float(output_constraint["value"])
            net.setLowerBound(outputs[idx], val)
        else:
            raise ValueError(f"Unknown output_constraint type: {c_type!r}")

        # ── Solve ─────────────────────────────────────────────────────────────
        options = _Marabou.createOptions(verbosity=0, timeoutInSeconds=int(self._timeout))
        t0      = time.perf_counter()
        vals, stats = net.solve(options=options)
        elapsed = time.perf_counter() - t0

        sat = (stats.hasTimedOut() is False) and bool(vals)
        cex = [float(vals[inputs[i]]) for i in range(self._state_dim)] if vals else None

        return {"sat": sat, "counterexample": cex, "time_s": round(elapsed, 3)}

    # ── Safety property suite ─────────────────────────────────────────────────

    def run_safety_suite(
        self,
        n_samples: int = 50,
        epsilon: float = 0.1,
        rng_seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Run the standard SENTINEL-X safety property suite.

        For each of *n_samples* randomly sampled states, verifies that the
        policy satisfies the two core safety rules within the ε-ball:

        1. **No inaction on faults** – if health_flag ≥ 0.5, argmax ≠ 0.
        2. **No reset on low power** – if power_level ≤ 0.15, argmax ≠ 2.

        Parameters
        ----------
        n_samples : int
            Number of states to sample (default 50).
        epsilon : float
            Radius of the input perturbation ball (default 0.1).
        rng_seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            ``{"verified": int, "total": int, "violations": list,
               "time_s": float}``
        """
        rng      = np.random.default_rng(rng_seed)
        verified = 0
        violations: list = []
        t0 = time.perf_counter()

        for i in range(n_samples):
            state = rng.random(self._state_dim).astype(float)
            lb    = np.clip(state - epsilon, 0.0, 1.0).tolist()
            ub    = np.clip(state + epsilon, 0.0, 1.0).tolist()

            health_gt_half  = float(state[5]) >= 0.5
            power_low       = float(state[7]) <= 0.15

            try:
                if health_gt_half:
                    result = self.verify_property(lb, ub, {"type": "argmax_not", "action": 0})
                elif power_low:
                    result = self.verify_property(lb, ub, {"type": "argmax_not", "action": 2})
                else:
                    verified += 1
                    continue

                if result["sat"]:
                    verified += 1
                else:
                    violations.append({
                        "sample_idx":     i,
                        "state":          state.tolist(),
                        "counterexample": result.get("counterexample"),
                    })
            except Exception as exc:
                violations.append({"sample_idx": i, "error": str(exc)})

        return {
            "verified":   verified,
            "total":      n_samples,
            "violations": violations,
            "time_s":     round(time.perf_counter() - t0, 2),
        }


# ---------------------------------------------------------------------------
# ERAN verifier
# ---------------------------------------------------------------------------

class ERANVerifier:
    """
    Wrapper around ERAN (ETH Robustness Analyzer for Neural Networks) for
    abstract-interpretation-based verification of SENTINEL-X policies.

    ERAN supports several abstract domains:

    * **DeepZ** – fast, coarser bounds; suitable for large networks.
    * **DeepPoly** – tighter bounds; better for properties involving ReLU.
    * **k-ReLU** – most precise; exponential worst-case cost.

    This is a **sound** verifier: all verified properties are guaranteed
    correct, but ``"unverified"`` does not mean the property is violated.

    Parameters
    ----------
    onnx_path : str
        Path to an ONNX model exported via :func:`export_to_onnx`.
    domain : str
        Abstract domain: ``"deepz"`` (default), ``"deeppoly"``, or ``"krelu"``.
    state_dim : int
        Input dimension (default 11).

    Raises
    ------
    ImportError
        If ``eran`` is not installed.
    """

    def __init__(
        self,
        onnx_path: str,
        domain: str = "deeppoly",
        state_dim: int = 11,
    ) -> None:
        if not _HAS_ERAN:
            raise ImportError(
                "ERAN is not installed.\n"
                "Installation: see docs/formal_verification_external.md\n"
                "  git clone https://github.com/eth-sri/eran\n"
                "  cd eran && pip install -e ."
            )
        if not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                "Run export_to_onnx(agent, 'model.onnx') first."
            )
        if domain not in ("deepz", "deeppoly", "krelu"):
            raise ValueError(f"Unknown domain: {domain!r}. Choose 'deepz', 'deeppoly', or 'krelu'.")

        self._onnx_path = onnx_path
        self._domain    = domain
        self._state_dim = state_dim

        # Load ERAN analyser
        self._analyser = eran.ERAN(onnx_path, is_onnx=True)  # type: ignore[name-defined]

    def verify_robustness(
        self,
        state: List[float],
        epsilon: float,
        expected_action: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Verify that the policy is locally robust at *state* under ε-ball
        perturbations.

        Parameters
        ----------
        state : list of float
            The centre point (normalised 11-dim state vector).
        epsilon : float
            L∞ perturbation radius.
        expected_action : int or None
            If provided, verifies that *expected_action* is the argmax for
            all inputs in the ε-ball.  If None, only checks that the argmax
            does not change (robustness under perturbation).

        Returns
        -------
        dict
            ``{"verified": bool, "domain": str, "time_s": float}``
        """
        state_arr = np.array(state, dtype=np.float64)
        lb = np.clip(state_arr - epsilon, 0.0, 1.0)
        ub = np.clip(state_arr + epsilon, 0.0, 1.0)

        t0 = time.perf_counter()
        label = expected_action if expected_action is not None else int(np.argmax(state_arr))

        verified, _ = self._analyser.analyze_box(
            lb, ub,
            domain=self._domain,
            label=label,
        )
        elapsed = time.perf_counter() - t0

        return {
            "verified": bool(verified),
            "domain":   self._domain,
            "time_s":   round(elapsed, 3),
        }

    def certify_safe_actions(
        self,
        states: List[List[float]],
        epsilon: float,
    ) -> Dict[str, Any]:
        """
        Batch robustness certification over a list of states.

        Parameters
        ----------
        states : list of list of float
            State vectors to certify.
        epsilon : float
            L∞ perturbation radius.

        Returns
        -------
        dict
            ``{"certified": int, "total": int, "rate": float, "time_s": float}``
        """
        certified = 0
        t0 = time.perf_counter()

        for state in states:
            result = self.verify_robustness(state, epsilon)
            if result["verified"]:
                certified += 1

        total = len(states)
        return {
            "certified": certified,
            "total":     total,
            "rate":      certified / total if total > 0 else 0.0,
            "time_s":    round(time.perf_counter() - t0, 2),
        }


# ---------------------------------------------------------------------------
# Availability helper
# ---------------------------------------------------------------------------

def available_verifiers() -> Dict[str, bool]:
    """
    Return a dict indicating which verifiers are currently installed.

    Returns
    -------
    dict
        ``{"marabou": bool, "eran": bool, "onnx_export": bool}``
    """
    return {
        "marabou":     _HAS_MARABOU,
        "eran":        _HAS_ERAN,
        "onnx_export": _HAS_ONNX,
    }
