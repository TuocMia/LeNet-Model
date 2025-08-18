#ifndef OUTPUT_HANDLER_HPP
#define OUTPUT_HANDLER_HPP

#include <tensorflow/lite/c/common.h>

void HandleOutput(const int8_t* output_data, int length);

#endif  // OUTPUT_HANDLER_HPP