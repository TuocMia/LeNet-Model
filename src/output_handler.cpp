// output_handler.cpp
#include "output_handler.hpp"
#include <tensorflow/lite/micro/micro_log.h>
#include <cmath>

// In probabilities & Top-1 từ tensor output (int8, shape [1,10])
void HandleOutput(const TfLiteTensor* output) {
  const int n = output->dims->data[output->dims->size - 1];

  int top_idx = -1;
  int8_t top_q = INT8_MIN;

  // In logits (đã dequantize) và tìm Top-1
  for (int i = 0; i < n; ++i) {
    const int8_t q = output->data.int8[i];
    float y = (q - output->params.zero_point) * output->params.scale; // dequant
    if (q > top_q) { top_q = q; top_idx = i; }

    MicroPrintf("logit[%d] = %f", i, static_cast<double>(y));
  }

  MicroPrintf("===> Top-1 class: %d", top_idx);
}
