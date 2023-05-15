#include "simd_delta.hpp"

#include "simd_util.hpp"

void decompress_scalar(const int8_t *__restrict input, uint32_t start_value, size_t input_size,
                       uint32_t *__restrict output) {
    output[0] = start_value;
    for (int i = 1; i < input_size; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

size_t scan_scalar(uint32_t predicate_low, uint32_t predicate_high, int8_t *__restrict input,
                   uint32_t start_value, size_t input_size, uint32_t *__restrict output) {
    uint32_t previous = start_value;
    size_t count = 0;
    for (int i = 0; i < input_size; ++i) {
        uint32_t decompressedValue = previous + input[i];
        previous = decompressedValue;
        if (decompressedValue >= predicate_low && decompressedValue <= predicate_high) {
            output[count] = decompressedValue;
            count++;
        }
    }
    return count;
}

// might be useful later for SIMD
//size_t scan_scalar(uint32_t predicate_low, uint32_t predicate_high, int8_t *__restrict input,
//                   uint32_t start_value, size_t input_size, uint32_t *__restrict output) {
//    std::vector<uint32_t> test(input_size);
//    test[0] = start_value;
//    for (int i = 1; i < input_size; ++i) {
//        test[i] = test[i - 1] + input[i];
//    }
//    int t = 0;
//    size_t count = 0;
//    for (int i = 0; i < input_size; ++i) {
//        if (test[i] >= predicate_low && test[i] <= predicate_high) {
//            output[t] = test[i];
//            t++;
//            count++;
//        }
//    }
//    return count;
//}