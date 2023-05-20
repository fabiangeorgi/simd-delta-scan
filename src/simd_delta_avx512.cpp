#include <immintrin.h>

#include "simd_delta.hpp"

const __m512i ZEROSET = _mm512_setzero_si512();

void decompress_avx512(const int8_t *__restrict input, uint32_t start_value, size_t input_size,
                       uint32_t *__restrict output) {
    auto previousHighest = _mm512_set1_epi32(start_value);
    for (int i = 0; i < input_size; i += 16) {
        // read 16 * int8_t numbers
        auto inputValues = _mm_load_epi32(input + i);
        // convert them to 16 * 32-bit numbers -> full lane
        auto inputRegister = _mm512_cvtepi8_epi32(inputValues);

        // inspired by http://www.cs.columbia.edu/~kar/pubsk/ADMS2020.pdf
        auto offset = _mm512_alignr_epi32(inputRegister, ZEROSET, 16 - 1);
        auto prefixSum = _mm512_add_epi32(inputRegister, offset);
        offset = _mm512_alignr_epi32(prefixSum, ZEROSET, 16 - 2);
        prefixSum = _mm512_add_epi32(prefixSum, offset);
        offset = _mm512_alignr_epi32(prefixSum, ZEROSET, 16 - 4);
        prefixSum = _mm512_add_epi32(prefixSum, offset);
        offset = _mm512_alignr_epi32(prefixSum, ZEROSET, 16 - 8);
        prefixSum = _mm512_add_epi32(prefixSum, offset);

        // add with offest from previous round and save
        auto result = _mm512_add_epi32(prefixSum, previousHighest);
        _mm512_store_si512(output + i, result);

        // extract from 512 bit register the top 256 bit register and extract the highest number
        // basically the last element in the register and set1 to generate the offset for our next iteration
        // auto offsetForNextIteration = _mm_extract_epi32(
        // _mm256_extracti32x4_epi32(_mm512_extracti32x8_epi32(result, 1), 1), 3);

        // read directly from output, and we don't need that stuff right here
        previousHighest = _mm512_set1_epi32(output[i + 15]);
    }
}

size_t scan_avx512(uint32_t predicate_low, uint32_t predicate_high, int8_t *__restrict input,
                   uint32_t start_value, size_t input_size, uint32_t *__restrict output) {
    size_t count = 0;
    auto previousHighest = _mm512_set1_epi32(start_value);
    auto predicateHighSet = _mm512_set1_epi32(predicate_high);
    auto predicateLowSet = _mm512_set1_epi32(predicate_low);
    for (int i = 0; i < input_size; i += 16) {
        // read 16 * int8_t numbers
        auto inputValues = _mm_load_epi32(input + i);
        // convert them to 16 * 32-bit numbers -> full lane
        auto inputRegister = _mm512_cvtepi8_epi32(inputValues);

        // inspired by http://www.cs.columbia.edu/~kar/pubsk/ADMS2020.pdf
        auto offset = _mm512_alignr_epi32(inputRegister, ZEROSET, 16 - 1);
        auto prefixSum = _mm512_add_epi32(inputRegister, offset);
        offset = _mm512_alignr_epi32(prefixSum, ZEROSET, 16 - 2);
        prefixSum = _mm512_add_epi32(prefixSum, offset);
        offset = _mm512_alignr_epi32(prefixSum, ZEROSET, 16 - 4);
        prefixSum = _mm512_add_epi32(prefixSum, offset);
        offset = _mm512_alignr_epi32(prefixSum, ZEROSET, 16 - 8);
        prefixSum = _mm512_add_epi32(prefixSum, offset);

        // add with offest from previous round and save
        auto result = _mm512_add_epi32(prefixSum, previousHighest);

        // filter out elements
        __mmask16 highMask = _mm512_cmple_epi32_mask(result, predicateHighSet);
        __mmask16 lowMask = _mm512_cmpge_epi32_mask(result, predicateLowSet);
        __mmask16 filterMask = _mm512_kand(highMask, lowMask);

        _mm512_mask_compressstoreu_epi32(output + count, filterMask, result);
        // add count/offset depending on how many elements we stored
        count += __builtin_popcount(filterMask);

        // extract from 512 bit register the top 256 bit register and extract the highest number
        // basically the last element in the register and set1 to generate the offset for our next iteration
        // auto offsetForNextIteration = _mm_extract_epi32(
        //  _mm256_extracti32x4_epi32(_mm512_extracti32x8_epi32(result, 1), 1), 3);
        // print_register(previousHighest);
        // try this for hopeful improvement
        auto nextIterationElementIndex = count ? count : 0;
        previousHighest = _mm512_set1_epi32(nextIterationElementIndex);
    }
    return count;
}
