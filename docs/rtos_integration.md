# RTOS Integration Guide

This guide shows how to integrate the SENTINEL-X TFLite policy into a
real-time operating system (RTOS) for deployment on spacecraft avionics
hardware.

Two popular open-source RTOSes are covered:

| RTOS | Target MCUs | SENTINEL-X support |
|------|------------|-------------------|
| **FreeRTOS** | STM32H7, STM32F4, Cortex-M0–M33, ESP32 | Full task template (`hardware/freertos_task.c`) |
| **Zephyr RTOS** | STM32, nRF52, i.MX RT, RISC-V | Configuration snippet in this guide |

> **Note:** RTOS integration is not required for simulation or Raspberry Pi
> deployment.  Use `scripts/mcu_emulator.py` and `scripts/replay_tflite.py`
> to validate the full pipeline without an RTOS.

---

## FreeRTOS

### What is provided

`hardware/freertos_task.c` implements a complete FreeRTOS inference task:

- Reads a 22-byte sensor frame from the host/sensors over UART.
- Builds the normalised 11-dim state vector.
- Runs the int8 TFLite Micro policy.
- Applies hard `SafetyMonitor` rules (mirrors the Python implementation).
- Transmits the chosen action back over UART.
- Runs at 100 Hz (10 ms task period).

### Build instructions

#### 1. Generate the model header

```bash
# On your development machine (after training)
python run_experiment.py --scenario lunar_gateway --episodes 50
# → sentinel_x_model_int8.tflite

# Convert to C array
xxd -i sentinel_x_model_int8.tflite > sentinel_x_model_data.h
```

#### 2. Add files to your STM32CubeIDE project

```
YourProject/
├── Core/
│   └── Src/
│       ├── main.c
│       └── sentinel_x_task.c      ← copy hardware/freertos_task.c here
├── Core/
│   └── Inc/
│       └── sentinel_x_model_data.h  ← copy generated header here
└── Middlewares/
    └── Third_Party/
        └── tflite-micro/            ← TFLite Micro source tree
```

#### 3. Configure TFLite Micro

Add the TFLite Micro sources to your project.  Minimum required operators for
the SENTINEL-X DQN:

```cpp
static tflite::MicroMutableOpResolver<4> resolver;
resolver.AddFullyConnected();
resolver.AddRelu();
resolver.AddQuantize();
resolver.AddDequantize();
```

If you add PPO (softmax output) also add:
```cpp
resolver.AddSoftmax();
```

#### 4. Register the task

In `main.c`, before `vTaskStartScheduler()`:

```c
#include "freertos_task.h"  // or add the prototype manually

int main(void)
{
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART2_UART_Init();   // must match SENTINEL_UART in freertos_task.c

    SentinelX_CreateTask();  // ← add this line

    osKernelStart();
    vTaskStartScheduler();
    for (;;);
}
```

#### 5. Memory requirements

| Component | Flash (KB) | SRAM (KB) |
|-----------|-----------|---------|
| TFLite Micro runtime | ~30 | – |
| int8 model flatbuffer | ~12 | – |
| Tensor arena (configurable) | – | 32 |
| Task stack | – | 4 |
| **Total** | **~42** | **~36** |

Adjust `SENTINEL_TENSOR_ARENA_KB` in `freertos_task.c` if allocation fails.

#### 6. Expected performance

| MCU | Frequency | Inference time |
|-----|-----------|----------------|
| STM32H743 | 480 MHz | ~0.5 ms |
| STM32F446 | 180 MHz | ~1.5 ms |
| STM32L432 | 80 MHz | ~3.5 ms |
| ESP32-S3 (Xtensa LX7) | 240 MHz | ~1 ms |

All well within the 10 ms task period.

---

## Zephyr RTOS

Zephyr requires a slightly different build system integration.

### 1. Add TFLite Micro as a Zephyr module

In your `west.yml` manifest:

```yaml
manifest:
  projects:
    - name: tflite-micro
      url: https://github.com/tensorflow/tflite-micro
      revision: main
      path: modules/tflite-micro
```

### 2. Enable TFLite Micro in Kconfig

```kconfig
# prj.conf
CONFIG_TFLITE_MICRO=y
CONFIG_TFLITE_MICRO_FULLY_CONNECTED=y
CONFIG_TFLITE_MICRO_RELU=y
CONFIG_TFLITE_MICRO_QUANTIZE=y
CONFIG_TFLITE_MICRO_DEQUANTIZE=y
CONFIG_HEAP_MEM_POOL_SIZE=65536   # 64 KB for tensor arena + stack
CONFIG_MAIN_STACK_SIZE=4096
```

### 3. Zephyr inference thread

```c
#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "sentinel_x_model_data.h"

#define TENSOR_ARENA_SIZE (32 * 1024)
static uint8_t tensor_arena[TENSOR_ARENA_SIZE] __aligned(16);

void sentinel_x_thread(void *p1, void *p2, void *p3)
{
    /* Identical inference logic to freertos_task.c */
    /* Replace HAL_UART_* calls with Zephyr uart_* API */
    const struct device *uart = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));

    /* ... initialise interpreter, loop, read UART frame, invoke, write action ... */
}

K_THREAD_DEFINE(sentinel_x_tid, 4096,
                sentinel_x_thread, NULL, NULL, NULL,
                K_PRIO_PREEMPT(7), 0, 0);
```

### 4. UART device tree overlay

```dts
/* boards/stm32h743xi.overlay */
&usart2 {
    status = "okay";
    current-speed = <115200>;
    pinctrl-0 = <&usart2_tx_pa2 &usart2_rx_pa3>;
    pinctrl-names = "default";
};

/ {
    chosen {
        zephyr,sentinel-x-uart = &usart2;
    };
};
```

---

## Common RTOS deployment checklist

- [ ] Tensor arena size verified (`AllocateTensors()` returns `kTfLiteOk`)
- [ ] Inference latency measured < control-loop deadline (default 10 ms)
- [ ] SafetyMonitor veto rules verified in unit test (see `tests/test_sentinel_x.py::TestMcuEmulator`)
- [ ] Watchdog timer enabled and pinged at ≥ 2× watchdog period
- [ ] UART framing validated against Python `_parse_uart_frame()`
- [ ] int8 model tested with `scripts/replay_tflite.py --report` before flashing
- [ ] Stack overflow detection enabled (`configCHECK_FOR_STACK_OVERFLOW = 2` in FreeRTOS)
- [ ] Model flatbuffer CRC check implemented in firmware boot loader

---

## Further reading

- [TFLite Micro documentation](https://github.com/tensorflow/tflite-micro)
- [FreeRTOS documentation](https://www.freertos.org/Documentation/RTOS_book.html)
- [Zephyr RTOS documentation](https://docs.zephyrproject.org/latest/)
- [STM32CubeIDE TFLite Micro tutorial](https://wiki.st.com/stm32mcu/wiki/AI:TensorFlow_Lite_for_Microcontrollers)
- [hardware/freertos_task.c](../hardware/freertos_task.c) – Reference FreeRTOS task implementation
- [docs/hardware_integration.md](hardware_integration.md) – Hardware driver guide
