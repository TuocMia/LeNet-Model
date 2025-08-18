// output_handler.cpp
#include "output_handler.hpp"
#include <tensorflow/lite/micro/micro_log.h>
#include <cmath>
#include "constants.h"

// In probabilities & Top-1 từ tensor output (int8, shape [1,10])
void HandleOutput(const int8_t* output_data, int length) {
    int predicted = -1;
    float max_val = -1e9;

    for (int i = 0; i < length; i++) {
        float val = kOutputScale * (output_data[i] - kOutputZeroPoint);
        MicroPrintf("Class %d = %f", i, static_cast<double>(val));

        if (val > max_val) {
            max_val = val;
            predicted = i;
        }
    }

    MicroPrintf("Predicted digit = %d", predicted);
}
