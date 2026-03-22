# Hardware Integration Guide

This guide explains how to wire real spacecraft hardware to the SENTINEL-X
software pipeline and run the trained RL policy on an actual embedded device.

The `hardware/` package provides a three-layer abstraction:

```
┌───────────────────────────────────────────────┐
│          Application / Mission Loop            │
│  (run_experiment.py, mcu_emulator.py, custom)  │
├───────────────────────────────────────────────┤
│       SpacecraftSensorInterface (Python)        │
│   moving-average filter · stuck detection      │
│   telemetry ring buffer                        │
├──────────────┬────────────────────────────────┤
│ RaspberryPi  │  STM32Driver (pyserial)         │
│ Driver       │  (or any USB-CDC / FTDI device) │
│ (RPi.GPIO +  │                                  │
│  pyserial)   │                                  │
├──────────────┴────────────────────────────────┤
│         Hardware Abstraction Layer             │
│   HardwareInterface · SensorReading            │
│   ActuatorCommand · UART protocol              │
└───────────────────────────────────────────────┘
```

---

## Supported Targets

| Target | Driver class | Dependencies |
|--------|-------------|--------------|
| Raspberry Pi 4 / Zero 2 W | `RaspberryPiDriver` | `RPi.GPIO` `pyserial` |
| STM32H7 / Nucleo / ESP32 via USB-UART | `STM32Driver` | `pyserial` |
| Development / CI (no hardware) | `SimulatedDriver` | none |

---

## Step 1 – Install driver dependencies

### Raspberry Pi
```bash
# Lightweight TFLite runtime (no full TF)
pip install tflite-runtime RPi.GPIO pyserial

# Enable hardware UART (disable Bluetooth overlay if needed)
sudo raspi-config   # Interface Options → Serial Port → enable hardware UART
```

### STM32 / any USB-UART adapter
```bash
pip install pyserial

# Identify the port
python -c "import serial.tools.list_ports; print(list(serial.tools.list_ports.comports()))"
# Linux: usually /dev/ttyUSB0 or /dev/ttyACM0
# macOS: /dev/cu.usbserial-*
# Windows: COM3, COM4, …
```

---

## Step 2 – Wiring

### Raspberry Pi ↔ spacecraft MCU board

```
Raspberry Pi 40-pin header        Spacecraft MCU board
─────────────────────────────     ──────────────────────
Pin  8  GPIO 14 (UART0 TX)   ───► MCU RX
Pin 10  GPIO 15 (UART0 RX)   ◄─── MCU TX
Pin  6  GND                  ───► MCU GND
Pin  4  5 V (optional)       ───► MCU VIN (if MCU is 5 V tolerant)
Pin 11  GPIO 17 (optional)   ───► MCU FAULT_INJECT (active-high)
Pin 13  GPIO 27 (optional)   ───► MCU WATCHDOG_PIN (toggled at 2 Hz)
```

UART settings: **115 200 baud, 8N1, no hardware flow control.**

### STM32 Nucleo ↔ development PC

Use the onboard ST-Link V3 Virtual COM Port (VCP):

```
PC USB-A / USB-C  ──────────────────────────► Nucleo CN1 (USB Mini-B)
```

The VCP appears as `/dev/ttyACM0` on Linux or `COMx` on Windows automatically.

---

## Step 3 – MCU firmware frame format

The MCU must respond to a request byte `0x52` ('R') with a **22-byte frame**:

```
Offset  Size  Type      Field
──────  ────  ────────  ─────────────────────────────────────────
0       1     uint8     Start byte: 0xAB
1       2     uint16_LE memory_error_count
3       1     uint8     flags
                          bit 0: parity_error
                          bit 1: sensor_stuck
                          bit 2: health_flag
                          bit 3: thermal_fault
4       4     float32   sensor_deviation  (degrees or normalised)
8       4     float32   time_since_recovery_s
12      4     float32   power_level_v     (0.0–5.0 V)
16      4     float32   attitude_rate_dps
20      1     uint8     comm_quality_raw  (0–255 → −120…0 dBm)
21      1     uint8     End byte: 0xCD
```

A reference FreeRTOS task implementing this protocol is provided in
`hardware/freertos_task.c`.

Action commands from the host are 2-byte packets:

```
[0xAA]  [action_index]   (0=DO_NOTHING, 1=RESTART, 2=SWITCH_REDUNDANT, 3=SAFE_MODE)
```

---

## Step 4 – Export the model

```bash
# Train and export the int8 model
python run_experiment.py --scenario lunar_gateway --episodes 50
# → sentinel_x_model_int8.tflite
```

On the Raspberry Pi you can run inference directly using `tflite-runtime`:

```bash
# Copy model to Pi
scp sentinel_x_model_int8.tflite pi@raspberrypi.local:~/

# Run the replay benchmark on Pi
python scripts/replay_tflite.py \
    --model sentinel_x_model_int8.tflite \
    --report
```

---

## Step 5 – Run the hardware integration loop

```python
from hardware import create_hardware_interface, SpacecraftSensorInterface
from sentinel_x import export_tflite_int8

# Choose your target: "rpi", "stm32", or "simulation"
hw     = create_hardware_interface("rpi")
sensor = SpacecraftSensorInterface(hw, filter_len=3, stuck_window=5)

# Load TFLite model (same inference loop as replay_tflite.py)
import tensorflow as tf
interp = tf.lite.Interpreter(model_path="sentinel_x_model_int8.tflite")
interp.allocate_tensors()

with hw:
    for step in range(200):
        state  = sensor.read_state()          # normalised 11-dim vector
        # ... run inference ...
        # ... write_action(cmd) ...
        hw.watchdog_ping()
```

See `scripts/mcu_emulator.py` for a complete inference loop with
SafetyMonitor vetoes and a CSV decision log.

---

## Step 6 – Flash to STM32 / Cortex-M (TFLite Micro)

1. Export the int8 model flatbuffer:
   ```bash
   python -c "
   from sentinel_x_advanced import DQNAgent, export_tflite_int8, FederatedSwarm, MissionProfile
   # ... train agent ...
   export_tflite_int8(agent, 'sentinel_x_model_int8.tflite')
   "
   ```

2. Convert to a C array:
   ```bash
   xxd -i sentinel_x_model_int8.tflite > sentinel_x_model_data.h
   ```

3. Copy `sentinel_x_model_data.h` and `hardware/freertos_task.c` into your
   STM32CubeIDE / PlatformIO project.

4. Add TFLite Micro to your project:
   ```
   # In PlatformIO (platformio.ini):
   lib_deps = tensorflow/lite-micro
   ```

5. Call `SentinelX_CreateTask()` from `main()` before `vTaskStartScheduler()`.

Expected inference latency:
- STM32H7 @ 480 MHz (int8): **~0.5–1.0 ms**
- STM32F4 @ 168 MHz (int8): **~2–4 ms**
- Raspberry Pi 4 (tflite-runtime): **~1 ms**

---

## Fault injection (optional)

To test `SafetyMonitor` vetoes on real hardware, toggle the `FAULT_GPIO_PIN`:

```python
hw = RaspberryPiDriver(fault_gpio_pin=17)
# Set fault pin HIGH to trigger an injected health fault on the MCU
import RPi.GPIO as GPIO
GPIO.output(17, GPIO.HIGH)
# Run inference – expect SafetyMonitor to veto DO_NOTHING → SAFE_MODE
GPIO.output(17, GPIO.LOW)
```

---

## Troubleshooting

**`FileNotFoundError: /dev/serial0`**
→ Enable the hardware UART: `sudo raspi-config` → Interface Options → Serial.

**`Short UART frame (N/22 bytes)`**
→ Check baud rate (must match MCU firmware: 115 200), and confirm MCU firmware
is running (LED should blink on Nucleo boards).

**`No USB serial port detected`**
→ Run `python -m serial.tools.list_ports` to list available ports and pass the
result explicitly: `STM32Driver(port="/dev/ttyUSB0")`.

**Inference gives wrong actions after flashing**
→ Ensure the model flatbuffer was generated from the same trained agent used
for Python-side evaluation.  Re-run `export_tflite_int8()` and `xxd -i`.
