# Running SENTINEL-X on Real Hardware

This guide explains how to deploy SENTINEL-X on a **Raspberry Pi** (as the
main host) and an **STM32** (as the MCU running the TFLite Micro policy).
The same approach works with any board supported by the hardware abstraction
layer.

---

## 1. Prepare the Hardware

### Raspberry Pi (any model with GPIO and USB-UART)

1. Install **Raspberry Pi OS** (Lite or Desktop).
2. Enable UART (`/dev/ttyAMA0` or `/dev/ttyS0`) and I²C/SPI if needed.
3. Ensure **Python 3.9+** and `pip` are installed.

### STM32 (e.g., STM32F4 Discovery, STM32H7 Nucleo)

1. Flash the provided C code: `hardware/freertos_task.c`.

   This code implements:
   - UART framing (request/response with 22-byte packets).
   - TFLite Micro inference (int8 model).
   - `SafetyMonitor` veto logic (mirrors Python rules).
   - Watchdog ping.

2. Build using **STM32CubeIDE** or a Makefile with FreeRTOS.

3. Connect the STM32 to the Raspberry Pi via a USB-to-UART adapter or
   direct UART pins.

---

## 2. Install SENTINEL-X on the Raspberry Pi

Clone the repository and install dependencies:

```bash
git clone https://github.com/danielnovais-tech/SENTINEL-X.git
cd SENTINEL-X
pip install -r requirements.txt
```

- **`pyserial`** – UART communication (already in requirements).
- **`tflite-runtime`** – on-Pi inference (install separately if needed:
  `pip install tflite-runtime`).
- **`rich`** – terminal dashboard (optional: `pip install rich`).

---

## 3. Test the UART Connection

### Option A – Using the MCU emulator (no hardware needed)

```bash
# Terminal 1: start the emulator as a TCP server
python scripts/mcu_emulator.py --server --port 8765

# Terminal 2: test as client (simulates the Pi)
python scripts/mcu_emulator.py --client --port 8765 --steps 5
```

### Option B – Using real hardware

Ensure the STM32 is powered and connected.  On the Pi, find the serial port:

```bash
ls /dev/ttyUSB*
```

Then ping the MCU:

```bash
python -m sentinel_x.hardware.stm32_driver --port /dev/ttyUSB0 --baud 115200 --ping
```

Expected output:

```
Using STM32Driver on /dev/ttyUSB0 @ 115200 baud

SensorReading:
  timestamp_s          = 1711148224.123
  memory_error_count   = 0
  parity_error_flag    = False
  sensor_deviation     = 0.002
  health_flag          = False
  thermal_fault        = False
  power_level_v        = 3.299 V
  attitude_rate_dps    = 0.010 dps
  comm_quality_db      = -62.5 dB

Normalised state vector: [0.0, 0.0, 0.0002, 0.0, 0.0, 0.0, 0.0, 0.66, 0.0001, 0.475, 0.0]
```

Run timing measurements to verify UART latency:

```bash
python -m sentinel_x.hardware.stm32_driver --port /dev/ttyUSB0 --timing-runs 200
```

---

## 4. Run the Full Hardware Validation

`scripts/collect_hardware_perf.py` automates the entire pipeline.  It will:

1. Load the trained int8 model.
2. Communicate with the STM32 (if `--port` is given) to measure UART round-trip.
3. Run inference, state normalisation, and safety veto timing tests.
4. Evaluate the policy on fault scenarios and compute recovery rates.
5. Generate a **JSON report** and a **Markdown summary**.

**With a real STM32:**

```bash
python scripts/collect_hardware_perf.py \
    --port /dev/ttyUSB0 \
    --timing-runs 1000 \
    --train-episodes 50 \
    --output results.json
```

**Simulation mode** (no hardware needed, suitable for CI):

```bash
python scripts/collect_hardware_perf.py --quiet --max-steps 10 --train-episodes 2
```

The output files (`hardware_perf_results.json` and `hardware_perf_report.md`)
contain all the metrics needed for the technical paper.

---

## 5. Monitor the Swarm in Real Time

Use the terminal dashboard to see live statistics while the system is running:

```bash
# Simulation (no hardware needed)
python scripts/dashboard.py

# Stream live telemetry from a connected MCU
python scripts/dashboard.py --port /dev/ttyUSB0
```

This displays per-spacecraft health, actions, rewards, and federation
information.  Press **Ctrl+C** to quit gracefully.

---

## 6. Customise the Deployment

| What to change | Where |
|---|---|
| Fault injection profile | `sentinel_x_config.yaml` |
| Safety monitor rules | `sentinel_x/utils/safety.py` or `hardware/freertos_task.c` |
| RTOS target | `docs/rtos_integration.md` (Zephyr and other RTOS templates) |
| Formation type | `sentinel_x/formation.py` `FormationController(formation_type=…)` |

---

## Expected Results (Reference Board)

On an STM32H7 with TFLite Micro you can expect:

| Metric | Value |
|---|---|
| Inference latency (p50) | 0.54 ms |
| State normalisation | 0.02 ms |
| SafetyMonitor veto | 0.01 ms |
| UART round-trip | 0.3 ms |
| Recovery rate | 92.3 % |
| Federated improvement | +12.4 % |

Full reference tables are in `docs/hardware_performance_results.md`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| UART not responding | Check baud rate (115200), wiring, and that the STM32 is running the correct firmware |
| Missing Python packages | `pip install -r requirements.txt` |
| `Permission denied on /dev/ttyUSB0` | `sudo usermod -a -G dialout $USER` (log out and back in) |
| Short frame (< 22 bytes) warning | Reduce baud rate or check USB-UART adapter driver |
| `ModuleNotFoundError: No module named 'numpy'` | `pip install numpy` |

---

*Part of the [SENTINEL-X](https://github.com/danielnovais-tech/SENTINEL-X)
open-source spacecraft fault-detection platform (Apache 2.0).*
