#include "main_functions.h"
#include "constants.h"
#include "output_handler.hpp"
#include "model-LeNet.h"

#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <zephyr/kernel.h>   // để dùng k_cycle_get_32 và k_sleep

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

  uint32_t start = k_cycle_get_32();
  TfLiteStatus invoke_status = interpreter->Invoke();
	if (invoke_status != kTfLiteOk) {
		MicroPrintf("Invoke failed");
		return;
	}

  uint32_t end = k_cycle_get_32();

  uint64_t duration_ns = k_cyc_to_ns_floor64(end - start);

  MicroPrintf("Inference time = %llu us", duration_ns / 1000);

}
