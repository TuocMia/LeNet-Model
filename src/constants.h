#ifndef CONSTANTS_H_
#define CONSTANTS_H_

// Tensor arena
const int kTensorArenaSize = 32000;

// Input MNIST: 28x28, int8 quant
const int kImageSize = 28 * 28;
const float kInputScale = 0.0039215689f;   // ~1/255
const int kInputZeroPoint = -128;

// Output: 10 lớp (digit 0-9)
const int kNumClasses = 10;
const float kOutputScale = 0.00390625f;
const int kOutputZeroPoint = -128;

#endif  // CONSTANTS_H_
