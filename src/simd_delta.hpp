#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "simd_util.hpp"

/**
 * SIMD DECOMPRESS - decompresses the compressed input
 * Input: compressed "delta" integers, input size (#compressed integers)
 * Output: uncompressed array of integers
 * Return: void
 *
 * You need to implement this method four times.
 *  - scalar: You may not use SIMD instructions.
 *  - sse:    You should use SIMD instructions available in the SSE family.
 *  - avx2:   You should use SIMD instructions available in the AVX/AVX2 family.
 *  - avx512: You should use SIMD instructions available in the AVX512 family.
 *
 *  Note that each of them is implemented in a separate compilation unit and only with the available
 * flags for the target "family". Trying to use, e.g., AVX512 in the _sse version will cause
 * compilation errors.
 */
// clang-format off
void decompress_scalar(const int8_t* input, uint32_t start_value, size_t input_size, uint32_t* output);
void    decompress_sse(const int8_t* input, uint32_t start_value, size_t input_size, uint32_t* output);
void   decompress_avx2(const int8_t* input, uint32_t start_value, size_t input_size, uint32_t* output);
void decompress_avx512(const int8_t* input, uint32_t start_value, size_t input_size, uint32_t* output);
// clang-format on

/**
 * SIMD SCAN - Scans compressed input for a range of predicates:
 * (predicate_low<=key<=predicate_high) Input: low & high predicate, compressed bitstream, start
 * value of encoding, input size (#compressed integers) Output: uncompressed and filtered array of
 * integers Return: number of tuples found in that range.
 *
 * You need to implement this together with the respective decompress_* method. The same
 * restrictions to the use of instructions apply.
 */
size_t scan_scalar(uint32_t predicate_low, uint32_t predicate_high, int8_t* input,
                   uint32_t start_value, size_t input_size, uint32_t* output);

size_t scan_sse(uint32_t predicate_low, uint32_t predicate_high, int8_t* input,
                uint32_t start_value, size_t input_size, uint32_t* output);

size_t scan_avx2(uint32_t predicate_low, uint32_t predicate_high, int8_t* input,
                 uint32_t start_value, size_t input_size, uint32_t* output);

size_t scan_avx512(uint32_t predicate_low, uint32_t predicate_high, int8_t* input,
                   uint32_t start_value, size_t input_size, uint32_t* output);

////////////////////////////////////
/// IMPLEMENTED BY US BELOW HERE ///
////////////////////////////////////
/**
 * This is implemented by us in simd_compress.cpp. Tou do not need to change this.
 *
 * COMPRESS - Compresses an input vector so that each number is represented as the delta to the
 *            previous value, i.e., input[i] - input[i - 1].
 * Input: uncompressed vector of integers
 * Return: Compressed stream of 8-bit "delta" integers (64-Byte aligned)
 *
 * Example:
 *   input:  10, 12,  9,  9,  33   <-- 32-bit unsigned integers
 *   output:  0,  2, -3,  0,  24   <--  8-bit   signed integers
 *
 * Note that the start vale (10 in this example) is given separately to the decompress methods.
 * The first delta is always 0.
 */
int8_t* compress_input(const std::vector<uint32_t>& input);
