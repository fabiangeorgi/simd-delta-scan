#include <immintrin.h>

#include "simd_delta.hpp"

void decompress_avx2(const int8_t *__restrict input, uint32_t start_value, size_t input_size,
                     uint32_t *__restrict output) {
    auto previous = (int32_t) start_value;

    for (int i = 0; i < input_size; i += 8) {
        // note: Might be a better way to load this directly but not sure how
        auto laneOne = _mm_cvtepi8_epi32(_mm_loadu_si32(reinterpret_cast<const __m128i *>(input + i)));
        auto laneTwo = _mm_cvtepi8_epi32(_mm_loadu_si32(reinterpret_cast<const __m128i *>(input + 4 + i)));

        // now we have a full 8 * 32-bit integer lane
        auto fullLane = _mm256_set_m128i(laneTwo, laneOne);

        // these operations only work on the two separate 128 bit lanes -> so we have to broadcast between them
        auto prefixSum = _mm256_add_epi32(fullLane, _mm256_slli_si256(fullLane, 4));
        prefixSum = _mm256_add_epi32(prefixSum, _mm256_slli_si256(prefixSum, 8));

        // we extract the last element in the first lane -> so 6 in the first run (0, 1, 3, [6], 4, 9, 15, 22)
        auto offset = _mm256_set1_epi32(_mm256_extract_epi32(prefixSum, 3));
        // we blend out the first part -> 0, 0, 0, 0, 6, 6, 6, 6
        // then we can easily add them to our second 128 bit lane
        offset = _mm256_blend_epi32 (offset, _mm256_set1_epi32(0), 0x0F);
        // add the offset to our second lane
        auto result = _mm256_add_epi32(prefixSum, offset);
        // add previous offset (from iteration before -> the cycle before in our loop)
        result = _mm256_add_epi32(result, _mm256_set1_epi32(previous));
        // store new iteration for next iteration
        previous = _mm256_extract_epi32(result, 7);

        _mm256_storeu_si256(reinterpret_cast<__m256i_u *>(output + i), result);
    }
}

size_t scan_avx2(uint32_t predicate_low, uint32_t predicate_high, int8_t *__restrict input,
                 uint32_t start_value, size_t input_size, uint32_t *__restrict output) {
    size_t count = 0;
    auto previous = (int32_t) start_value;

    for (int i = 0; i < input_size; i += 8) {
        // note: Might be a better way to load this directly but not sure how
        auto laneOne = _mm_cvtepi8_epi32(_mm_loadu_si32(reinterpret_cast<const __m128i *>(input + i)));
        auto laneTwo = _mm_cvtepi8_epi32(_mm_loadu_si32(reinterpret_cast<const __m128i *>(input + 4 + i)));

        // now we have a full 8 * 32-bit integer lane
        auto fullLane = _mm256_set_m128i(laneTwo, laneOne);

        // these operations only work on the two separate 128 bit lanes -> so we have to broadcast between them
        auto prefixSum = _mm256_add_epi32(fullLane, _mm256_slli_si256(fullLane, 4));
        prefixSum = _mm256_add_epi32(prefixSum, _mm256_slli_si256(prefixSum, 8));

        // we extract the last element in the first lane -> so 6 in the first run (0, 1, 3, [6], 4, 9, 15, 22)
        auto offset = _mm256_set1_epi32(_mm256_extract_epi32(prefixSum, 3));
        // we blend out the first part -> 0, 0, 0, 0, 6, 6, 6, 6
        // then we can easily add them to our second 128 bit lane
        offset = _mm256_blend_epi32 (offset, _mm256_set1_epi32(0), 0x0F);
        // add the offset to our second lane
        auto result = _mm256_add_epi32(prefixSum, offset);
        // add previous offset (from iteration before -> the cycle before in our loop)
        result = _mm256_add_epi32(result, _mm256_set1_epi32(previous));
        // store new iteration for next iteration
        previous = _mm256_extract_epi32(result, 7);

        auto lowerFilterSet = _mm256_set1_epi32(predicate_low - 1);
        auto higherFilterSet = _mm256_set1_epi32(predicate_high + 1);
        auto lowerFilter = _mm256_cmpgt_epi32(result, lowerFilterSet);
        auto higherFilter = _mm256_cmpgt_epi32(higherFilterSet, result);
        auto returnMask = _mm256_and_si256(lowerFilter, higherFilter);

        // I don't really know why this does not work?
        // _mm256_maskstore_epi32(reinterpret_cast<int *>(output), returnMask, result);
        // so have to take the manual route
        alignas(32) int32_t v[8];
        alignas(32) int32_t r[8];
        _mm256_store_si256((__m256i *) v, returnMask);
        _mm256_store_si256((__m256i *) r, result);
        for (int j = 0; j < 8; j++) {
            if (v[j] != 0) {
                output[count] = r[j];
                count++;
            }
        }
    }
    return count;
}
