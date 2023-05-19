#include <immintrin.h>

#include "simd_delta.hpp"

void decompress_avx512(const int8_t* __restrict input, uint32_t start_value, size_t input_size,
                       uint32_t* __restrict output) {
    //_mm256_loadu_epi8 (might be good)
  // TODO: implement
}

size_t scan_avx512(uint32_t predicate_low, uint32_t predicate_high, int8_t* __restrict input,
                   uint32_t start_value, size_t input_size, uint32_t* __restrict output) {
  // TODO: implement
  return 0;
}
