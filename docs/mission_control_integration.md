# Mission Control Integration (NASA F´ / ESA TASTE)

SENTINEL-X can forward telemetry events and receive commands from
industry-standard flight-software stacks via the
`sentinel_x.mission_control` module.

Two concrete adapters are provided, both with **graceful fallback**
(raising `ImportError` if the underlying library is not installed):

| Adapter | Framework | Transport | Optional dependency |
|---------|-----------|-----------|---------------------|
| `FPrimeAdapter` | NASA F´ (F Prime) | TCP GDS socket | `fprime-gds` |
| `TASTEAdapter` | ESA TASTE / ASN.1 | File-based VFB | `asn1tools` |
| `MissionControlBridge` (base) | Any / none | JSON-lines log | – (none) |

---

## Installation

### NASA F´

```bash
# Install the F´ Ground Data System
pip install fprime-gds

# Start the GDS (requires an fprime-gds dictionary JSON)
fprime-gds -d path/to/SentinelX.json --gui-addr 127.0.0.1 --gui-port 50050
```

### ESA TASTE / ASN.1

```bash
pip install asn1tools
```

TASTE itself is a toolchain (not a Python library); see
https://taste.tools/ for installation instructions.

---

## Base bridge (no external dependencies)

```python
from sentinel_x.mission_control import MissionControlBridge

# Subclass and override _publish_impl / _command_impl as needed
class MyBridge(MissionControlBridge):
    pass

bridge = MyBridge(log_path="telemetry.jsonl")
bridge.connect()

bridge.publish_telemetry({
    "spacecraft_id": 0,
    "step": 42,
    "action": 2,
    "action_label": "SWITCH_REDUNDANT",
    "reward": 3.5,
    "health": True,
})

bridge.publish_episode_summary(episode=5, reward=12.3, overrides=1)
bridge.send_command("RESET_OVERRIDE_COUNT", {})

print(f"TX frames: {bridge.tx_count}")  # 3
bridge.disconnect()
```

---

## NASA F´ adapter

```python
from sentinel_x.mission_control import FPrimeAdapter

bridge = FPrimeAdapter(
    gds_host="127.0.0.1",
    gds_port=50050,
    component_name="SentinelX.FaultDetector",
    log_path="telemetry.jsonl",
)

with bridge:   # connect() / disconnect() via context manager
    bridge.publish_telemetry({
        "spacecraft_id": 0,
        "action":  2,
        "reward":  3.5,
        "health":  True,
        "overrides": 0,
    })
    bridge.send_command("RESET_OVERRIDE_COUNT", {})
```

### F´ channel mapping

| SENTINEL-X key | F´ channel suffix |
|----------------|-------------------|
| `spacecraft_id` | `SpacecraftId` |
| `action` | `LastAction` |
| `reward` | `EpisodeReward` |
| `health` | `HealthStatus` |
| `overrides` | `SafetyOverrides` |

---

## ESA TASTE adapter

```python
from sentinel_x.mission_control import TASTEAdapter

bridge = TASTEAdapter(
    schema_path=None,          # use built-in SentinelXTelemetry schema
    output_dir="taste_telemetry",
    log_path="telemetry.jsonl",
)

with bridge:
    bridge.publish_telemetry({
        "spacecraft_id": 1,
        "action": 1,
        "reward": -0.5,
        "health": False,
    })
# → taste_telemetry/frame_000000.ber (ASN.1 DER-encoded)
```

### Custom ASN.1 schema

```asn1
-- my_schema.asn
MY-TELEMETRY DEFINITIONS AUTOMATIC TAGS ::= BEGIN

SentinelXTelemetry ::= SEQUENCE {
    timestamp    INTEGER (0..MAX),
    spacecraftId INTEGER (0..63),
    action       INTEGER (0..3),
    reward       INTEGER (-1000..1000),
    healthFlag   BOOLEAN,
    overrides    INTEGER (0..MAX)
}

END
```

```python
bridge = TASTEAdapter(schema_path="my_schema.asn")
```

---

## Checking availability

```python
from sentinel_x.mission_control import available_adapters
print(available_adapters())
# {"fprime": False, "taste": False}
```

---

## Integration with the training loop

```python
import sentinel_x_advanced as sx
from sentinel_x.mission_control import MissionControlBridge

class LogBridge(MissionControlBridge):
    pass

swarm  = sx.FederatedSwarm(4, 4, sx.MissionProfile(sx.MissionProfile.BALANCED))
bridge = LogBridge(log_path="mission_log.jsonl")
bridge.connect()

for ep in range(100):
    reward = swarm.train_episode()
    bridge.publish_episode_summary(
        episode=ep,
        reward=reward,
        overrides=swarm.last_override_count,
    )

bridge.disconnect()
print(f"Published {bridge.tx_count} telemetry frames")
```

---

## NASA F´ component registration

To register SENTINEL-X as a native F´ component (requires a full F´ build
environment):

```python
# SentinelXComponent.fpp  (F´ FPP syntax)
#
# component SentinelXFaultDetector {
#   telemetry SpacecraftId: U8  id 1
#   telemetry LastAction:   U8  id 2
#   telemetry EpisodeReward: F32 id 3
#   telemetry HealthStatus: bool id 4
#   telemetry SafetyOverrides: U32 id 5
#   command RESET_OVERRIDE_COUNT opcode 0x01
# }
```

See [NASA F´ component model documentation](https://nasa.github.io/fprime/UsersGuide/user/full-intro.html)
for the complete FPP component tutorial.

---

## Further reading

- [Mission control source](../sentinel_x/mission_control.py)
- [NASA F´ GDS documentation](https://nasa.github.io/fprime/GettingStarted/gds.html)
- [ESA TASTE documentation](https://taste.tools/)
- [asn1tools documentation](https://asn1tools.readthedocs.io/)
