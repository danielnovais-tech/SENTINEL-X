# SENTINEL-X Flight Deployment Roadmap

This guide describes the engineering steps required to take SENTINEL-X from a
ground-tested prototype to a flight-ready spacecraft component.  It covers
hardware selection, software hardening, flight software integration,
verification and validation, operational considerations, and deployment
milestones.

---

## 1  Flight Hardware Selection

### 1.1  Processor and Memory

| Property | Recommendation |
|----------|---------------|
| MCU | STMicroelectronics **STM32H7-R** (Cortex-M7, radiation-tolerant) or Microchip **SAMRH71** (Cortex-M7, space heritage) |
| Operating frequency | 400–480 MHz |
| Flash / FRAM | ECC-protected NOR flash or FRAM (e.g., Cypress CY15B256Q) |
| SRAM | ECC-protected SRAM (on-chip or external with EDAC controller) |
| Redundancy | Cold-spare dual-CPU for mission-critical applications |

**Why these devices?**  Both carry ECSS qualification heritage and have
published SEU / SEL (single-event latch-up) characterisation data from
radiation campaigns.

### 1.2  Power and Interfaces

- **DC-DC converter:** Use a space-qualified converter (e.g., Vicor DCM)
  with latching current limit and undervoltage lockout (UVLO).
- **I/O protection:** All UART, I²C, and SPI pins must have ESD and
  transient-voltage suppression (TVS) diodes to prevent latch-up.
- **Power monitoring:** Measure bus voltage and current in every telemetry
  frame; the `PowerSubsystem` model already tracks this.

### 1.3  Hardware Watchdog

The external hardware watchdog (HW WDT) is the last resort against a
software hang:

- Must be **external to the CPU** (e.g., dedicated MAX706 or Maxim DS1315)
  so that a CPU halt cannot disable it.
- **Cannot be disabled by software** after boot – the disable pin must be
  strapped to prevent software override.
- Recommended timeout: **200 ms** (2× the 100 Hz control-loop period).

In `hardware/freertos_task.c`, map `SENTINEL_WDT_REFRESH()` to your BSP call:

```c
/* In your project's main.h or a project-specific header: */
extern IWDG_HandleTypeDef hiwdg;
#define SENTINEL_WDT_REFRESH()  HAL_IWDG_Refresh(&hiwdg)
```

The task refreshes the watchdog at the top of every 10 ms control-loop tick
(100 Hz).  If the inference hangs (e.g., due to a memory fault), the watchdog
fires and resets the CPU.

---

## 2  Software Hardening for Space

### 2.1  Error Detection and Correction

| Measure | Implementation |
|---------|---------------|
| ECC RAM | Enabled via MCU memory-controller configuration (EDAC bit in flash option bytes on STM32) |
| Stack canaries | Compile with `-fstack-protector-strong` and enable MPU stack overflow detection in FreeRTOS |
| Model CRC-32 at boot | `sentinel_verify_model_crc()` in `freertos_task.c` – checked before each power-on |
| Periodic model re-check | Re-run `sentinel_verify_model_crc()` every N minutes in a background task |
| Double-checked function returns | All `HAL_*` and `interpreter->Invoke()` return values checked; failure paths call `sentinel_enter_safe_mode()` |

**Computing the expected CRC before flight:**

```bash
python3 - <<'EOF'
import zlib, pathlib
data = pathlib.Path("sentinel_x_model_int8.tflite").read_bytes()
print(f"#define SENTINEL_MODEL_CRC32_EXPECTED  0x{zlib.crc32(data) & 0xFFFFFFFF:08X}U")
EOF
```

Copy the output into `hardware/freertos_task.c` before the flight build.
The current placeholder `0x00000000U` intentionally causes `sentinel_tflite_init()`
to fail so that developers cannot accidentally fly an unchecked build.

### 2.2  Safe Mode and Recovery

Safe mode is entered via `sentinel_enter_safe_mode(reason)`:

| Trigger | `reason` string |
|---------|----------------|
| Watchdog reset limit reached | `"wdt_limit"` |
| Model CRC mismatch or TFLite init failure | `"tflite_init"` |
| Manual ground command | `"ground_cmd"` |
| Unrecoverable power brownout detected at runtime | `"brownout"` |

In safe mode the function:
1. Disables non-critical peripherals (payload, heaters, actuators).
2. Emits a **distress beacon** over UART every 5 s containing the reset
   counter and reason string.
3. Continues refreshing the hardware watchdog (so the system stays alive
   and reachable by ground).
4. Spins forever until a ground reset command is received.

**Ground recovery:** The ground operator sends a reset command via the
`MissionControlBridge` (F′ or TASTE adapter).  On receipt the task resets
`sentinel_wdt_reset_count` to zero and restarts the inference loop.

### 2.3  Watchdog Reset Tracking

The reset counter is stored in a `.noinit` RAM region that **survives a soft
reset** (the RAM content is preserved across watchdog- and software-triggered
resets but cleared on a full power cycle):

```c
__attribute__((section(".noinit")))
static volatile uint32_t sentinel_wdt_reset_count;
```

Add the section to your linker script:

```ld
.noinit (NOLOAD) : {
    KEEP(*(.noinit))
} > RAM
```

On a true power-on, a magic word (`0xDEADBEEF`) is used to distinguish the
uninitialised region from a watchdog reset and the counter is set to zero.

---

## 3  Integration with Flight Software Frameworks

### 3.1  NASA F′ (F Prime)

1. Add `FPrimeAdapter` as an F′ component in your deployment topology.
2. Map the five telemetry channels to F′ telemetry points (see
   `docs/mission_control_integration.md`).
3. Add a ground command handler that calls `sentinel_enter_safe_mode("ground_cmd")`
   or resets `sentinel_wdt_reset_count` to resume from safe mode.
4. Run the adapter in a dedicated F′ active component thread; communicate
   with the FreeRTOS inference task via a protected queue.

### 3.2  ESA TASTE

1. Use the `TASTEAdapter` with the provided ASN.1 schema
   (`SentinelXTelemetry`).
2. Connect to the Virtual Function Bus in the TASTE interface view.
3. Validate the complete TASTE deployment with the TASTE software simulator
   before first hardware integration.

### 3.3  NASA Core Flight System (cFS) — Optional

Create a cFS application that:
- Uses `hardware/hal.py` as the template for the cFS HAL.
- Implements a `CFE_SB_SendMsg()` wrapper around the inference result.
- Subscribes to a ground-command message to toggle safe mode.

---

## 4  Verification and Validation

### 4.1  Environmental Testing

| Test | Conditions | Pass criterion |
|------|-----------|----------------|
| Thermal vacuum | −40 °C to +85 °C, 10 cycles | No crash, recovery rate ≥ 90 % |
| Vibration | Sine sweep 20–2 000 Hz; random per ECSS-E-ST-10-03 | No structural damage, full boot after |
| Radiation (SEU) | Heavy-ion beam at TAMU Cyclotron or CERN CHARM; LET 10–80 MeV·cm²/mg | ECC corrects all SEUs; no uncorrected MBU in TFLite model area |

### 4.2  Fault Injection Campaign

Use an FPGA-based fault injection board (e.g., MicroSemi RTG4 or Xilinx
Zynq with a fault injection IP) to:

1. Flip individual bits in the sensor frame while the system runs.
2. Corrupt model weights in flash (single bytes).
3. Inject UART noise and out-of-sequence bytes.
4. Force a power brownout via controllable load.

Expected results:
- Bit flips in the sensor frame: `SafetyMonitor` ensures no dangerous action.
- Model corruption: CRC check detects corruption at next boot → safe mode.
- UART noise: Frame start/end markers (`0xAB` / `0xCD`) and length check
  reject malformed frames.
- Brownout: Power monitor triggers `sentinel_enter_safe_mode("brownout")`.

### 4.3  Formal Verification for Certification

Run the full formal verification suite on the **final int8 model** before
the flight build:

```bash
# Marabou safety property verification
python -c "
from sentinel_x_advanced import PolicyVerifier, DQNAgent
agent    = DQNAgent.load('sentinel_x_model_int8.tflite')
verifier = PolicyVerifier(agent)
report   = verifier.verify_policy(n_samples=10000)
assert report['ltl_pass']  and report['safety_pass'], report
print('Formal verification: PASS')
"

# ERAN L∞ robustness (ε = 0.1)
python docs/formal_verification_external.md  # see guide for full command
```

Archive the output as the **verification certificate** for inclusion in the
Software Verification and Validation Report (SVVR).

See [`docs/formal_verification_external.md`](formal_verification_external.md)
and [`docs/decision_tree_certification.md`](decision_tree_certification.md)
for the complete procedures.

### 4.4  Compliance with Standards

| Standard | Scope | Status |
|----------|-------|--------|
| ECSS-E-ST-40C | Space software | Align SDP, SRS, test plan |
| DO-178C / DO-333 | Avionics (formal methods supplement) | Use DT surrogate as certified policy |
| ISO 26262 (informative) | Safety concept | Reference for hazard analysis |

Deliverables required:
- **Software Development Plan (SDP)**
- **Software Requirements Specification (SRS)**
- **Software Verification and Validation Report (SVVR)**

---

## 5  Operational Considerations

### 5.1  Ground Interface

All actions, telemetry, and safety vetoes are logged via the `JSONLogger`
base class and downlinked regularly.  The ground system should:

| Capability | Command / API |
|------------|--------------|
| Disable RL agent, revert to rule-based FDIR | `OP_MODE=FALLBACK` command via F′ / TASTE |
| Upload new model weights | `MODEL_UPLOAD` command + `sentinel_verify_model_crc()` check after upload |
| Upload new mission config | `CONFIG_UPLOAD` command (updates `sentinel_x_config.yaml`) |
| Reset from safe mode | `RESET_SAFE_MODE` → sets `sentinel_wdt_reset_count = 0` |
| Query system health | `HEALTH_REQUEST` → returns last 10 telemetry frames |

### 5.2  In-Flight Policy Updates

The agent does **not** update its weights in flight (inference only).  The
workflow for uploading a new policy is:

1. Collect telemetry + replay buffer via downlink.
2. Train an improved model on the ground using the federated learning
   framework (see `docs/continuous_learning.md`).
3. Verify the new model with `PolicyVerifier` and Marabou.
4. Uplink the new `.tflite` flatbuffer via the `MODEL_UPLOAD` command.
5. The flight software verifies the CRC, backs up the old model to the
   redundant flash bank, and activates the new model on next reboot.

---

## 6  Deployment Milestones

| Milestone | Activities | Gate criteria |
|-----------|-----------|---------------|
| **M1 — Hardware Selection** | Choose CPU, memory, power; procure engineering model | PDR (Preliminary Design Review) complete |
| **M2 — Software Porting** | Adapt HAL drivers; integrate with F′/TASTE/cFS; port C code | All unit tests passing on EM hardware |
| **M3 — Verification** | Environmental testing; fault injection; formal verification; SVVR draft | Recovery rate ≥ 92 %, all LTL properties verified |
| **M4 — Flight Model Build** | Assemble flight-grade units; acceptance test | Acceptance test report signed |
| **M5 — System Integration** | Install on spacecraft; full system-level I&T | Mission CDR (Critical Design Review) complete |
| **M6 — Launch** | Fairing integration; launch; LEOP | First telemetry downlink received |
| **M7 — In-Orbit Commissioning** | Switch from rule-based FDIR to SENTINEL-X; monitor for 30 days | Recovery rate ≥ 92 % in orbit |

---

## 7  Resources and References

### Hardware
- STMicroelectronics [STM32H7 series](https://www.st.com/en/microcontrollers-microprocessors/stm32h7-series.html)
- Microchip [SAMRH71](https://www.microchip.com/en-us/product/SAMRH71)

### Flight Software
- NASA [F Prime (F′)](https://nasa.github.io/fprime/)
- ESA [TASTE](https://taste.tools/)
- NASA [Core Flight System (cFS)](https://cfs.gsfc.nasa.gov/)

### Radiation Testing Facilities
- Texas A&M University [Cyclotron Institute](https://cyclotron.tamu.edu/)
- CERN [CHARM facility](https://charm.web.cern.ch/)

### Formal Verification Tools
- [Marabou](https://github.com/NeuralNetworkVerification/Marabou)
- [ERAN](https://github.com/eth-sri/eran)

### Standards
- ECSS-E-ST-40C — *Space Engineering: Software*
- RTCA DO-178C / DO-333 — *Formal Methods Supplement*
- ECSS-Q-ST-80C — *Software Product Assurance*

---

*See also:*
- [`docs/hardware_integration.md`](hardware_integration.md) — wiring, UART protocol, embedded deployment
- [`docs/hardware_deployment.md`](hardware_deployment.md) — step-by-step RPi + STM32 guide
- [`docs/rtos_integration.md`](rtos_integration.md) — FreeRTOS and Zephyr RTOS build guide
- [`docs/formal_verification_external.md`](formal_verification_external.md) — Marabou and ERAN integration
- [`docs/continuous_learning.md`](continuous_learning.md) — long-term autonomy and in-flight updates
- [`ROADMAP.md`](../ROADMAP.md) — full 10-item enhancement roadmap
