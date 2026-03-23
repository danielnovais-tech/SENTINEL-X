# Formal Verification with External Tools (Marabou / ERAN)

SENTINEL-X includes a built-in lightweight verifier (`PolicyVerifier`) that
uses probabilistic sampling.  For **flight-critical certification** (DO-178C,
ECSS-E-ST-40C) a *complete* or *sound* formal verifier is required.

This guide covers two professional tools:

| Tool | Approach | Completeness | Best for |
|------|----------|-------------|----------|
| **Marabou** (Stanford) | SMT / LP-based | Complete | Small networks; hard safety proofs |
| **ERAN** (ETH Zurich) | Abstract interpretation | Sound | Larger networks; robustness certification |

Both are integrated in `sentinel_x/formal_verification.py` with **graceful
fallback** – if neither tool is installed, importing `sentinel_x` still works.

---

## Installation

### Step 1 – Export dependencies

Both tools require the model in **ONNX format**:

```bash
pip install onnx tf2onnx
```

### Step 2a – Install Marabou

```bash
# Option A: pip (if a prebuilt wheel exists for your platform)
pip install maraboupy

# Option B: build from source (recommended for reproducibility)
git clone https://github.com/NeuralNetworkVerification/Marabou
cd Marabou
mkdir build && cd build
cmake .. -DENABLE_PYTHON=ON
make -j$(nproc)
cd ..
pip install -e .
```

Verify installation:
```python
from maraboupy import Marabou
print("Marabou OK")
```

### Step 2b – Install ERAN

```bash
git clone https://github.com/eth-sri/eran
cd eran
# Follow platform-specific setup in README (GUROBI or ELINA dependency)
pip install -e .
```

Verify installation:
```python
import eran
print("ERAN OK")
```

---

## Checking availability

```python
from sentinel_x.formal_verification import available_verifiers
print(available_verifiers())
# {"marabou": False, "eran": False, "onnx_export": True}
```

---

## Step 1 – Export to ONNX

```python
from sentinel_x.formal_verification import export_to_onnx
from sentinel_x_advanced import DQNAgent, FederatedSwarm, MissionProfile

# ... train agent ...
export_to_onnx(agent, "sentinel_x_policy.onnx")
```

---

## Using Marabou

### Verify a single safety property

```python
from sentinel_x.formal_verification import MarabouVerifier

verifier = MarabouVerifier("sentinel_x_policy.onnx", timeout_s=60)

# Verify: for all states near a healthy baseline,
# the policy must NOT choose DO_NOTHING when health_flag = 1
result = verifier.verify_property(
    input_lb=[0.0, 0.0, 0.0, 0.0, 0.5, 0.9, 0.0, 0.8, 0.0, 0.8, 0.0],
    input_ub=[0.1, 0.0, 0.1, 0.0, 0.6, 1.0, 0.0, 0.9, 0.1, 0.9, 0.1],
    output_constraint={"type": "argmax_not", "action": 0},
)
print(result)
# {"sat": True, "counterexample": None, "time_s": 0.32}
# sat=True means the property HOLDS (no counterexample found)
```

### Run the full safety property suite

```python
report = verifier.run_safety_suite(n_samples=100, epsilon=0.1)
print(f"Properties verified: {report['verified']}/{report['total']}")
print(f"Time: {report['time_s']:.1f} s")
if report['violations']:
    print("⚠ Violations found:")
    for v in report['violations']:
        print(f"  Sample {v['sample_idx']}: cex={v.get('counterexample')}")
```

### Property reference

| Property | `output_constraint` |
|----------|---------------------|
| Argmax must be action *a* | `{"type": "argmax_is", "action": a}` |
| Argmax must not be action *a* | `{"type": "argmax_not", "action": a}` |
| Output neuron *i* ≥ value *v* | `{"type": "output_lb", "index": i, "value": v}` |

---

## Using ERAN

### Certify robustness at a single state

```python
from sentinel_x.formal_verification import ERANVerifier

eran_v = ERANVerifier("sentinel_x_policy.onnx", domain="deeppoly")

state = [0.0, 0.0, 0.05, 0.0, 0.5, 0.0, 0.0, 0.9, 0.05, 0.9, 0.0]
result = eran_v.verify_robustness(state, epsilon=0.05, expected_action=0)
print(result)
# {"verified": True, "domain": "deeppoly", "time_s": 0.18}
```

### Batch certification

```python
import numpy as np
states = np.random.default_rng(0).random((200, 11)).tolist()
cert   = eran_v.certify_safe_actions(states, epsilon=0.05)
print(f"Certified: {cert['certified']}/{cert['total']} ({cert['rate']:.1%})")
```

### Domain comparison

| Domain | Speed | Precision | Use case |
|--------|-------|-----------|----------|
| `deepz` | Fastest | Coarsest | Screening large numbers of states |
| `deeppoly` | Medium | Good | Standard certification reports |
| `krelu` | Slowest | Tightest | High-assurance certification |

---

## Interpreting results

| Result | Meaning |
|--------|---------|
| Marabou `sat=True`, `counterexample=None` | Property **proved** for entire input region |
| Marabou `sat=False`, `counterexample=[...]` | Property **violated** – counterexample is the offending input |
| ERAN `verified=True` | Property **certified** (sound) |
| ERAN `verified=False` | Inconclusive – may be a true violation or a verification gap |

---

## Integration with CI

Add formal verification as a nightly CI job (not every commit – tools are slow):

```yaml
# .github/workflows/formal_verify.yml
name: Formal Verification
on:
  schedule:
    - cron: "0 2 * * 1"   # Monday 02:00 UTC
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install onnx tf2onnx maraboupy
      - run: python -c "
          import sentinel_x as sx, numpy as np
          from sentinel_x.formal_verification import export_to_onnx, MarabouVerifier
          swarm = sx.FederatedSwarm(2, 4, sx.MissionProfile(sx.MissionProfile.BALANCED))
          export_to_onnx(swarm.agents[0], 'policy.onnx')
          v = MarabouVerifier('policy.onnx')
          r = v.run_safety_suite(n_samples=20, epsilon=0.1)
          assert r['violations'] == [], r
          print('Verification passed')
          "
```

---

## Further reading

- [Marabou documentation](https://neuralnetworkverification.github.io/Marabou/)
- [ERAN documentation](https://eth-sri.github.io/eran)
- [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) – alternative complete verifier
- [PolicyVerifier](../sentinel_x_advanced.py) – SENTINEL-X built-in sampling-based verifier
