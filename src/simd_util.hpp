#pragma once

#include <iostream>
#include <ostream>
#include <string>

/** Print `num_bytes` as binary to `os` stating at `data` in left to right order, i.e., the least
 *significant bit (and byte #0) is left and the most significant is right.
 *
 * Example: Four bytes 0, 1, 2, 3 are represented as:
 *         00000000 10000000 01000000 11000000
 * Byte:   0        1        2        3
 **/
void print_bits_left_to_right(const void* data, size_t num_bytes, std::ostream& os = std::cout);

/** Print `num_bytes` as binary to `os` stating at `data` in right to left order, i.e., the least
 *significant bit (and byte #0) is right and the most significant is left. This is the more
 *"natural" way of viewing binary numbers.
 *
 * Example: Four bytes 0, 1, 2, 3 are represented as:
 *         00000011 00000010 00000001 00000000
 * Byte:          3        2        1        0
 **/
void print_bits_right_to_left(const void* data, size_t num_bytes, std::ostream& os = std::cout);

/** Print `num_ints` (4 Byte/32 Bit) as integers to `os` stating at `data` */
void print_ints(const void* data, size_t num_ints, std::ostream& os = std::cout);

/**
 * Convenience method to print the contents of a SIMD register via the print_bits_* methods above.
 *
 * @param reg a reference to the __m128/256/512i register to print
 * @param os the output stream you want to print to, probably std::cout
 * @param right_to_left whether to print with the right_to_left (=true) or left_to_right (=false)
 * method
 */
template <typename SimdRegister>
void print_register(const SimdRegister& reg, std::ostream& os = std::cout,
                    bool right_to_left = true) {
  constexpr size_t num_bytes = sizeof(SimdRegister);
  if (right_to_left) {
    return print_bits_right_to_left(&reg, num_bytes, os);
  }
  return print_bits_left_to_right(&reg, num_bytes, os);
}

// Just a small helper for us to use in tests and the benchmark to print the correct name.
std::string variant_to_string(size_t i);
