#include "simd_delta.hpp"

#include "simd_util.hpp"

void decompress_scalar(const int8_t* __restrict input, uint32_t start_value, size_t input_size, uint32_t* __restrict output) {
  // TODO: implement
}

size_t scan_scalar(uint32_t predicate_low, uint32_t predicate_high, int8_t* __restrict input,
                   uint32_t start_value, size_t input_size, uint32_t* __restrict output) {
  // TODO: implement
  return 0;
}
