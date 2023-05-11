#include "simd_util.hpp"

#include <bitset>

namespace {

// Simple method to reverse the bits in a uint8_t.
// Taken from https://stackoverflow.com/a/2602885/4505331
uint8_t reverse(uint8_t b) {
  b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
  b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
  b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
  return b;
}

}  // namespace

void print_bits_left_to_right(const void* data, size_t num_bytes, std::ostream& os) {
  auto* bytes = reinterpret_cast<const uint8_t*>(data);
  for (size_t offset = 0; offset < num_bytes; ++offset) {
    os << std::bitset<8>(reverse(bytes[offset])) << ' ';
  }
  os << std::endl;
}

void print_bits_right_to_left(const void* data, size_t num_bytes, std::ostream& os) {
  auto* bytes = reinterpret_cast<const uint8_t*>(data);
  for (size_t offset = num_bytes; offset > 0; --offset) {
    os << std::bitset<8>(bytes[offset - 1]) << ' ';
  }
  os << std::endl;
}

void print_ints(const void* data, size_t num_ints, std::ostream& os) {
  if (reinterpret_cast<uint64_t>(data) % sizeof(uint32_t) != 0) {
    throw std::runtime_error{"Cannot print unaligned integers!"};
  }

  auto* ints = reinterpret_cast<const uint32_t*>(data);
  for (size_t offset = 0; offset < num_ints; ++offset) {
    os << ints[offset] << ' ';
  }
  os << std::endl;
}

std::string variant_to_string(size_t i) {
  switch (i) {
    case 0:
      return "scalar";
    case 1:
      return "sse";
    case 2:
      return "avx2";
    case 3:
      return "avx512";
    default:
      return "unknown" + std::to_string(i);
  };
}
