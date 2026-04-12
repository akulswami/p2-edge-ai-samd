#include <cstdio>
#include <cstring>
#include <vector>

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/tensorflow/classification.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"

#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"

#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace coralmicro {
namespace {

const char* kModelPath =
    "/models/mobilenet_v1_1.0_224_quant_edgetpu.tflite";

constexpr int kTensorArenaSize = 8 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

constexpr int kNumIterations = 2000;

void FillInput(uint8_t* input, int size) {
  for (int i = 0; i < size; ++i) {
    input[i] = static_cast<uint8_t>((i * 7) % 255);
  }
}

[[noreturn]] void Main() {
  printf("E0 Infer Baseline Start\r\n");

  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("ERROR: model load failed\r\n");
    vTaskSuspend(nullptr);
  }

  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    printf("ERROR: TPU open failed\r\n");
    vTaskSuspend(nullptr);
  }

  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  tflite::MicroInterpreter interpreter(
      tflite::GetModel(model.data()),
      resolver,
      tensor_arena,
      kTensorArenaSize,
      &error_reporter);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors failed\r\n");
    vTaskSuspend(nullptr);
  }

  if (interpreter.inputs().size() != 1) {
    printf("ERROR: Model must have one input tensor\r\n");
    vTaskSuspend(nullptr);
  }

  auto* input = interpreter.input_tensor(0);
  if (!input) {
    printf("ERROR: input tensor null\r\n");
    vTaskSuspend(nullptr);
  }

  printf("Model ready\r\n");

  for (int i = 0; i < kNumIterations; ++i) {
    std::memset(input->data.uint8, 0, input->bytes);
    FillInput(input->data.uint8, input->bytes);

    if (interpreter.Invoke() != kTfLiteOk) {
      printf("infer,%d,ERROR\r\n", i);
      continue;
    }

    auto results = tensorflow::GetClassificationResults(&interpreter, 0.0f, 1);

    if (!results.empty()) {
      const auto& r = results[0];
      printf("infer,%d,%d,%.6f\r\n", i, r.id, r.score);
    } else {
      printf("infer,%d,NONE\r\n", i);
    }

    LedSet(Led::kUser, (i % 2) == 0);
  }

  printf("E0 Infer Baseline End\r\n");

  while (true) {
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
