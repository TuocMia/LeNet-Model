#ifndef OUTPUT_HANDLER_HPP
#define OUTPUT_HANDLER_HPP

#include <tensorflow/lite/c/common.h>

void HandleOutput(const TfLiteTensor* output);

#endif  // OUTPUT_HANDLER_HPP