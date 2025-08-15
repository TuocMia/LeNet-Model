#include "main_functions.h"
// #include "constants.h"
#include "output_handler.hpp"

// model header của bạn
#include "model-LeNet.h"    

#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/schema/schema_generated.h>
// #include <tensorflow/lite/version.h>

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Arena nên căn 16 byte để tránh lỗi canh hàng SIMD trên một số MCU
  	constexpr int kTensorArenaSize = 32000;
	uint8_t tensor_arena[kTensorArenaSize];

  // helper: nạp ảnh 28x28 (uint8 0..255) vào input int8 (scale~1/255, zp=-128)
  void FeedImage28x28U8ToInt8(const uint8_t* img_u8, TfLiteTensor* in) {
    // Netron của bạn cho thấy: real = 0.00392156 * (q + 128)
    // => zero_point ≈ -128. Với đầu vào uint8 [0..255], cách nhanh nhất là q = u8 - 128.
    const int bytes = in->bytes;
    for (int i = 0; i < bytes; ++i) {
      in->data.int8[i] = static_cast<int8_t>(static_cast<int>(img_u8[i]) - 128);
    }
  }
}  // namespace

void setup() {
  MicroPrintf("Setup TFLM LeNet (quant int8)");

  model = tflite::GetModel(model_LeNet_quant);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Bad schema: model %d vs runtime %d",
                model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Cần đúng các op theo sơ đồ bạn gửi
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

	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		MicroPrintf("AllocateTensors() failed");
		return;
	}

  input  = interpreter->input(0);   // int8, shape [1,28,28,1]
  output = interpreter->output(0);  // int8, shape [1,10]
  MicroPrintf("Setup done. input bytes=%d, output bytes=%d", input->bytes, output->bytes);
}

void loop() {
  // TODO: cung cấp ảnh thực tế 28x28 (grayscale 0..255).
  // Tạm thời tạo dữ liệu dummy: toàn 0 (màu đen).
  static uint8_t dummy_img[28 * 28] = {0};

  // Nếu input là [1,28,28,1], bytes = 784
  FeedImage28x28U8ToInt8(dummy_img, input);

  // Chạy suy luận
  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  // In kết quả
  HandleOutput(output);
}
