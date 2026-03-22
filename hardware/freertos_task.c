/**
 * @file   freertos_task.c
 * @brief  SENTINEL-X inference task for FreeRTOS (STM32H7 / Cortex-M7)
 *
 * This FreeRTOS task runs on-device inference using TensorFlow Lite Micro.
 * It reads sensor telemetry from the system, runs the int8 TFLite policy,
 * applies hard safety rules (matching SafetyMonitor), and commands actuators
 * via UART and GPIO.
 *
 * Dependencies
 * ------------
 * - TensorFlow Lite for Microcontrollers (tflite-micro)
 * - STM32 HAL (for UART, GPIO, RTC)
 * - FreeRTOS (any version ≥ 10.x)
 *
 * Build
 * -----
 * 1. Copy sentinel_x_model_int8.tflite → convert to C array:
 *        xxd -i sentinel_x_model_int8.tflite > sentinel_x_model_data.h
 * 2. Add this file and sentinel_x_model_data.h to your STM32CubeIDE project.
 * 3. Add tflite-micro sources to the project (see TFLite Micro docs).
 * 4. Enable UART and GPIO clocks in MX_Init().
 *
 * UART Protocol
 * -------------
 * Host sends:  0x52 ("R") → request sensor frame
 * MCU returns: 22-byte frame (see SentinelXFrame struct below)
 * Host sends:  0xAA, action_index → action command
 *
 * State Vector (11 floats, normalised to [0,1])
 * ---------------------------------------------
 * [0]  memory_error_ratio       [1]  parity_flag
 * [2]  sensor_deviation_norm    [3]  sensor_stuck_flag
 * [4]  time_since_recovery_norm [5]  health_flag
 * [6]  thermal_fault_flag       [7]  power_level_norm
 * [8]  attitude_rate_norm       [9]  comm_quality_norm
 * [10] peer_sensor_deviation
 *
 * Action mapping
 * --------------
 * 0  DO_NOTHING        – no actuator change
 * 1  RESTART           – cycle power to the affected subsystem
 * 2  SWITCH_REDUNDANT  – switch to redundant unit
 * 3  SAFE_MODE         – enter minimum-power safe mode
 */

#include "FreeRTOS.h"
#include "task.h"
#include "cmsis_os.h"
#include "main.h"

/* TFLite Micro headers (generated / copied from tflite-micro) */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* Generated from the .tflite flatbuffer via xxd -i */
#include "sentinel_x_model_data.h"

/* ───────────────────────────── Constants ─────────────────────────────────── */

#define SENTINEL_STATE_DIM      11
#define SENTINEL_ACTION_DIM      4
#define SENTINEL_TENSOR_ARENA_KB 32          /* adjust if inference fails     */
#define SENTINEL_UART            huart2      /* change to your UART handle    */
#define SENTINEL_TASK_STACK_WORDS 1024
#define SENTINEL_TASK_PRIORITY   osPriorityNormal

/* Safety monitor thresholds (must match Python SafetyMonitor) */
#define HEALTH_FLAG_INDEX        5
#define POWER_LEVEL_INDEX        7
#define POWER_CRITICAL_THRESHOLD 0.15f   /* < 15% → refuse SWITCH_REDUNDANT  */
#define ACTION_DO_NOTHING        0
#define ACTION_RESTART           1
#define ACTION_SWITCH_REDUNDANT  2
#define ACTION_SAFE_MODE         3
#define UART_FRAME_START         0xAB
#define UART_FRAME_END           0xCD
#define UART_FRAME_LEN           22
#define UART_REQUEST_BYTE        0x52     /* 'R' */
#define UART_ACTION_HEADER       0xAA

/* ───────────────────────────── Types ────────────────────────────────────── */

/** 22-byte sensor telemetry frame (same encoding as Python _parse_uart_frame) */
#pragma pack(push, 1)
typedef struct {
    uint8_t  start;                  /* 0xAB */
    uint16_t memory_error_count;
    uint8_t  flags;                  /* bit0=parity,bit1=stuck,bit2=health,bit3=thermal */
    float    sensor_deviation;
    float    time_since_recovery_s;
    float    power_level_v;
    float    attitude_rate_dps;
    uint8_t  comm_quality_raw;       /* 0–255 → −120…0 dBm */
    uint8_t  end;                    /* 0xCD */
} SentinelXFrame;
#pragma pack(pop)

_Static_assert(sizeof(SentinelXFrame) == UART_FRAME_LEN,
               "Frame size mismatch – check struct packing");

/* ───────────────────────────── Globals ──────────────────────────────────── */

static uint8_t  tensor_arena[SENTINEL_TENSOR_ARENA_KB * 1024] __attribute__((aligned(16)));
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor*             input_tensor  = nullptr;
static TfLiteTensor*             output_tensor = nullptr;

extern UART_HandleTypeDef SENTINEL_UART;

/* ───────────────────────────── Helpers ──────────────────────────────────── */

/**
 * @brief  Normalise raw sensor values into the 11-dim state vector.
 * @param  frame   Pointer to a decoded SentinelXFrame.
 * @param  state   Output buffer (11 floats).
 */
static void sentinel_build_state(const SentinelXFrame *frame, float *state)
{
    const float MAX_MEM_ERR    = 10.0f;
    const float MAX_SENSOR_DEV = 10.0f;
    const float MAX_RECOVERY_S = 3600.0f;
    const float MAX_POWER_V    = 5.0f;
    const float MAX_RATE_DPS   = 180.0f;

#define CLIP01(x) ( (x) < 0.0f ? 0.0f : ((x) > 1.0f ? 1.0f : (x)) )

    state[0] = CLIP01((float)frame->memory_error_count / MAX_MEM_ERR);
    state[1] = (frame->flags & 0x01) ? 1.0f : 0.0f;  /* parity */
    state[2] = CLIP01(fabsf(frame->sensor_deviation) / MAX_SENSOR_DEV);
    state[3] = (frame->flags & 0x02) ? 1.0f : 0.0f;  /* stuck */
    state[4] = CLIP01(frame->time_since_recovery_s / MAX_RECOVERY_S);
    state[5] = (frame->flags & 0x04) ? 1.0f : 0.0f;  /* health */
    state[6] = (frame->flags & 0x08) ? 1.0f : 0.0f;  /* thermal */
    state[7] = CLIP01(frame->power_level_v / MAX_POWER_V);
    state[8] = CLIP01(fabsf(frame->attitude_rate_dps) / MAX_RATE_DPS);
    /* comm: −120…0 dBm → 0…1 */
    state[9] = CLIP01(((float)frame->comm_quality_raw / 255.0f));
    state[10] = 0.0f;  /* peer_deviation (set from swarm message if available) */

#undef CLIP01
}

/**
 * @brief  Apply hard safety rules (mirrors Python SafetyMonitor).
 * @param  action  Proposed action from the RL policy.
 * @param  state   Current normalised state vector.
 * @return         Final (possibly overridden) action.
 */
static int sentinel_safety_monitor(int action, const float *state)
{
    /* Rule 1: never do nothing on an active health fault */
    if (state[HEALTH_FLAG_INDEX] >= 0.5f && action == ACTION_DO_NOTHING)
        return ACTION_SAFE_MODE;

    /* Rule 2: never switch to redundant hardware when power is critical */
    if (state[POWER_LEVEL_INDEX] < POWER_CRITICAL_THRESHOLD
            && action == ACTION_SWITCH_REDUNDANT)
        return ACTION_SAFE_MODE;

    return action;
}

/**
 * @brief  Find argmax of a float array.
 */
static int argmax_f(const float *arr, int n)
{
    int   best_idx = 0;
    float best_val = arr[0];
    for (int i = 1; i < n; ++i) {
        if (arr[i] > best_val) { best_val = arr[i]; best_idx = i; }
    }
    return best_idx;
}

/* ───────────────────────────── TFLite setup ─────────────────────────────── */

/**
 * @brief  Load the TFLite model and allocate tensors.
 * @return pdTRUE on success, pdFALSE on failure.
 */
static BaseType_t sentinel_tflite_init(void)
{
    tflite::InitializeTarget();

    const tflite::Model *model = tflite::GetModel(sentinel_x_model_int8_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        /* Schema mismatch – re-generate the model data header */
        return pdFALSE;
    }

    /* Register only the ops used by the SENTINEL-X DQN model */
    static tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter static_interp(
        model, resolver, tensor_arena, sizeof(tensor_arena));
    interpreter = &static_interp;

    if (interpreter->AllocateTensors() != kTfLiteOk)
        return pdFALSE;

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);
    return pdTRUE;
}

/* ───────────────────────────── Main task ────────────────────────────────── */

/**
 * @brief  SENTINEL-X inference task (runs at configTICK_RATE_HZ).
 *
 * Loop:
 *   1. Read sensor frame from host / sensors.
 *   2. Build normalised state vector.
 *   3. Run TFLite inference.
 *   4. Apply SafetyMonitor vetoes.
 *   5. Transmit action to host and drive actuators.
 */
void SentinelX_InferenceTask(void *pvParameters)
{
    (void)pvParameters;

    if (sentinel_tflite_init() != pdTRUE) {
        /* Hang in safe mode if model fails to load */
        for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    uint8_t      rx_buf[UART_FRAME_LEN];
    SentinelXFrame frame;
    float        state[SENTINEL_STATE_DIM];
    float        q_values[SENTINEL_ACTION_DIM];
    uint8_t      tx_buf[2];

    for (;;)
    {
        /* ── 1. Wait for a request byte from the host ── */
        uint8_t req;
        if (HAL_UART_Receive(&SENTINEL_UART, &req, 1, 20) == HAL_OK
                && req == UART_REQUEST_BYTE)
        {
            /* ── 2. Send sensor frame ── */
            /* Populate frame from real sensors here (ADC, I2C, SPI …) */
            frame.start = UART_FRAME_START;
            frame.end   = UART_FRAME_END;
            /* TODO: fill frame fields from actual sensor reads */
            HAL_UART_Transmit(&SENTINEL_UART, (uint8_t*)&frame,
                              UART_FRAME_LEN, 10);
        }

        /* ── 3. Receive action command from host ── */
        if (HAL_UART_Receive(&SENTINEL_UART, rx_buf, 2, 20) == HAL_OK
                && rx_buf[0] == UART_ACTION_HEADER)
        {
            int host_action = (int)(rx_buf[1] & 0x03);

            /* Build state from last sent frame and run local inference too */
            sentinel_build_state(&frame, state);

            /* Run TFLite inference */
            if (input_tensor->type == kTfLiteInt8) {
                for (int i = 0; i < SENTINEL_STATE_DIM; ++i) {
                    float scale = input_tensor->params.scale;
                    int   zp    = input_tensor->params.zero_point;
                    if (scale == 0.0f) scale = 1.0f;
                    input_tensor->data.int8[i] =
                        (int8_t)roundf(state[i] / scale + zp);
                }
            } else {
                for (int i = 0; i < SENTINEL_STATE_DIM; ++i)
                    input_tensor->data.f[i] = state[i];
            }

            interpreter->Invoke();

            /* Dequantise output if int8 */
            if (output_tensor->type == kTfLiteInt8) {
                float scale = output_tensor->params.scale;
                int   zp    = output_tensor->params.zero_point;
                if (scale == 0.0f) scale = 1.0f;
                for (int i = 0; i < SENTINEL_ACTION_DIM; ++i)
                    q_values[i] = (output_tensor->data.int8[i] - zp) * scale;
            } else {
                for (int i = 0; i < SENTINEL_ACTION_DIM; ++i)
                    q_values[i] = output_tensor->data.f[i];
            }

            int local_action = argmax_f(q_values, SENTINEL_ACTION_DIM);
            int final_action  = sentinel_safety_monitor(local_action, state);

            /* ── 4. Actuate ── */
            /* TODO: drive GPIO / relay / solenoid based on final_action */
            (void)final_action;

            /* ── 5. Echo final action back to host ── */
            tx_buf[0] = UART_ACTION_HEADER;
            tx_buf[1] = (uint8_t)(final_action & 0xFF);
            HAL_UART_Transmit(&SENTINEL_UART, tx_buf, 2, 10);
        }

        vTaskDelay(pdMS_TO_TICKS(10));   /* 100 Hz control loop */
    }
}

/* ───────────────────────────── Registration ─────────────────────────────── */

/**
 * @brief  Create the inference task.  Call from main() before vTaskStartScheduler().
 */
void SentinelX_CreateTask(void)
{
    xTaskCreate(
        SentinelX_InferenceTask,
        "SentinelX",
        SENTINEL_TASK_STACK_WORDS,
        NULL,
        SENTINEL_TASK_PRIORITY,
        NULL
    );
}
