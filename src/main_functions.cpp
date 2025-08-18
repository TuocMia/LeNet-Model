#include "main_functions.h"
#include "constants.h"
#include "output_handler.hpp"
#include "model-LeNet.h"
#include "test_image.h"

#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/schema/schema_generated.h>

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


void setup(void) {
  MicroPrintf("Setup TFLM LeNet (quant int8)");

  model = tflite::GetModel(model_LeNet_quant);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal "
                "to supported version %d.",
                model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddPad();
  resolver.AddConv2D();
  resolver.AddAveragePool2D();
  resolver.AddTanh();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  MicroPrintf("Setup done. input bytes=%d, output bytes=%d", input->bytes, output->bytes);
}


void loop(void) {
  extern const uint8_t test_image[28*28];  

  // Chuẩn bị input: float → int8
  for (int i = 0; i < kImageSize; i++) {
    float x = test_image[i] / 255.0f;
    int8_t q = static_cast<int8_t>(x / kInputScale + kInputZeroPoint);
    input->data.int8[i] = q;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke failed!");
    return;
  }

  // Gọi output handler
  HandleOutput(output->data.int8, kNumClasses);
}
