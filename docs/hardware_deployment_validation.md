# Hardware Deployment Validation

After writing and flashing the drivers, use `scripts/hardware_timing_validator.py`
to confirm the SENTINEL-X pipeline meets real-time timing requirements on the
actual target hardware.

The validator runs four test suites and produces a JSON report with PASS/FAIL
results against configurable budgets.

---

## What is validated

| Test | What is measured | Default budget |
|------|-----------------|---------------|
| **Inference latency** | TFLite int8 policy inference on target | 10 ms (100 Hz) |
| **State normalisation** | `SensorReading.to_state_vector()` | 0.5 ms |
| **SafetyMonitor veto** | `SafetyMonitor.veto()` rule check | 0.1 ms |
| **UART round-trip** *(optional)* | 22-byte request+response to MCU | 5 ms |

---

## Quick start (simulation mode)

No hardware needed – validates inference and software paths:

```bash
python scripts/hardware_timing_validator.py
```

Expected output (development laptop):
```
SENTINEL-X Hardware Timing Validator
==================================================

[1] TFLite int8 inference latency
  ✓ PASS  Inference
         median=0.312 ms  p95=0.419 ms  p99=0.511 ms  (budget=10.0 ms)

[2] State-vector normalisation latency
  ✓ PASS  Normalisation
         median=0.004 ms  p95=0.006 ms  p99=0.009 ms  (budget=0.5 ms)

[3] SafetyMonitor veto latency
  ✓ PASS  SafetyMonitor
         median=0.002 ms  p95=0.004 ms  p99=0.005 ms  (budget=0.1 ms)

==================================================
Result: ALL TESTS PASSED
Report: timing_report.json
```

---

## With real hardware (UART)

Connect your MCU via USB-UART and pass the port:

```bash
# Linux
python scripts/hardware_timing_validator.py --port /dev/ttyUSB0

# macOS
python scripts/hardware_timing_validator.py --port /dev/cu.usbserial-110

# Windows
python scripts/hardware_timing_validator.py --port COM3
```

Expected MCU round-trip on STM32H7 @ 115200 baud:
```
[4] UART round-trip (/dev/ttyUSB0 @ 115200 baud)
  ✓ PASS  UART round-trip
         median=2.1 ms  p95=2.4 ms  p99=2.8 ms  (budget=5.0 ms)
```

---

## Customising budgets

```bash
# Stricter budget for 200 Hz loop (5 ms deadline)
python scripts/hardware_timing_validator.py \
    --inference-budget-ms 5.0 \
    --uart-budget-ms 3.0 \
    --runs 1000

# Quiet JSON output only (for CI integration)
python scripts/hardware_timing_validator.py \
    --quiet \
    --output /tmp/timing_report.json
```

---

## JSON report format

```json
{
  "inference": {
    "stats": {
      "min_ms": 0.28,  "p50_ms": 0.31,  "p95_ms": 0.42,
      "p99_ms": 0.51,  "max_ms": 0.89,  "mean_ms": 0.32,  "std_ms": 0.04
    },
    "budget_ms": 10.0,
    "passed": true
  },
  "normalisation": { ... },
  "safety_monitor": { ... },
  "uart_roundtrip": { "skipped": true, "reason": "No --port specified" },
  "summary": {
    "all_passed": true,
    "runs": 200,
    "platform": "linux"
  }
}
```

---

## CI integration

Add the timing validator as a nightly CI job:

```yaml
# .github/workflows/timing.yml
name: Hardware Timing
on:
  schedule:
    - cron: "0 3 * * 1"   # Monday 03:00 UTC
jobs:
  timing:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: python scripts/hardware_timing_validator.py --quiet
      - run: python -c "
          import json; r = json.load(open('timing_report.json'))
          assert r['summary']['all_passed'], r
          print('All timing budgets met')
          "
```

---

## Board-specific results

Results from reference hardware runs:

| Board | MCU | Frequency | Inference | UART RTT |
|-------|-----|-----------|-----------|----------|
| STM32H7 Nucleo | STM32H743 | 480 MHz | ~0.5 ms | ~2.1 ms |
| STM32F4 Nucleo | STM32F446 | 180 MHz | ~1.5 ms | ~2.2 ms |
| Raspberry Pi 4 | Cortex-A72 | 1.8 GHz | ~0.3 ms | ~1.8 ms |
| ESP32-S3 DevKit | Xtensa LX7 | 240 MHz | ~1.0 ms | ~3.5 ms |

All meet the 10 ms (100 Hz) default budget.  For 200 Hz applications
(5 ms budget) the STM32H7 and Raspberry Pi 4 are the recommended targets.

---

## See also

- [hardware/hal.py](../hardware/hal.py) – Hardware Abstraction Layer
- [hardware/rpi_driver.py](../hardware/rpi_driver.py) – Raspberry Pi driver
- [hardware/stm32_driver.py](../hardware/stm32_driver.py) – STM32 driver
- [hardware/freertos_task.c](../hardware/freertos_task.c) – FreeRTOS task
- [docs/hardware_integration.md](hardware_integration.md) – Wiring guide
- [docs/rtos_integration.md](rtos_integration.md) – RTOS guide
