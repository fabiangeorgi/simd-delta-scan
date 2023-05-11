#include <cassert>
#include <cstdlib>

#include "simd_delta.hpp"

int8_t* compress_input(const std::vector<uint32_t>& input) {
  assert(input.size() > 2 && "Must have >2 elements.");

  const size_t data_size = input.size() + 64;                   // 64 byte padding
  const size_t alloc_size = data_size + 64 - (data_size % 64);  // ensure multiple of 64
  auto* out = reinterpret_cast<int8_t*>(std::aligned_alloc(64, alloc_size));
  assert(out && "Bad alloc. This should not happen. Please let the teaching team know.");

  int64_t current = input[0];
  for (int i = 0; i < input.size(); ++i) {
    assert(std::labs(input[i] - current) >> 8 == 0 &&
           "Diff too large. This should not happen. Please let the teaching team know.");
    out[i] = static_cast<int8_t>(input[i] - current);
    current = input[i];
  }
  return out;
}
