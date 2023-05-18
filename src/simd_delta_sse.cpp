#include <nmmintrin.h>

#include "simd_delta.hpp"

#define DEBUG 1

void decompress_sse(const int8_t *__restrict input, uint32_t start_value, size_t input_size,
                    uint32_t *__restrict output) {
#ifdef DEBUG
    for (int i = 0; i < input_size; ++i) {
        std::cout << +input[i] << std::endl;
    }
    std::cout << start_value << std::endl;
#endif

    auto previous = (int32_t) start_value;

    for (int i = 0; i < input_size; i += 4) {
        auto registerOne = _mm_loadu_si32(input + i);
        auto registerTwo = _mm_cvtepi8_epi32(registerOne);
        print_register(registerTwo);
        // prefix sum
        registerTwo = _mm_add_epi32(registerTwo, _mm_slli_si128(registerTwo, 4)); // moves register 32 bit
        registerTwo = _mm_add_epi32(registerTwo, _mm_slli_si128(registerTwo, 8)); // moves register 64 bit
        print_register(registerTwo);
        auto returnRegister = _mm_set1_epi32(previous);
        returnRegister = _mm_add_epi32(registerTwo, returnRegister);
        print_register(returnRegister);
        _mm_storeu_si128(reinterpret_cast<__m128i_u *>(output + i), returnRegister);
        previous = _mm_extract_epi32(returnRegister, 3);
#ifdef DEBUG
        for (int j = 0; j < input_size; ++j) {
            std::cout << +output[j] << std::endl;
        }
        std::cout << previous << std::endl;
#endif
    }
    // register: 128 bits
    // delta: 8 bit
    // 128 register mit 32 bit align damit end result einfacher
    // 128 / 32 = 4 end result
}

size_t scan_sse(uint32_t predicate_low, uint32_t predicate_high, int8_t *__restrict input,
                uint32_t start_value, size_t input_size, uint32_t *__restrict output) {
    size_t count = 0;
    auto previous = (int32_t) start_value;
    auto lowFilter = _mm_set1_epi32((int32_t) predicate_low);
    auto highFilter = _mm_set1_epi32((int32_t) predicate_high);

    for (int i = 0; i < input_size; i += 4) {
        auto registerOne = _mm_loadu_si32(input + i);
        auto registerTwo = _mm_cvtepi8_epi32(registerOne);
        // prefix sum
        registerTwo = _mm_add_epi32(registerTwo, _mm_slli_si128(registerTwo, 4)); // moves register 32 bit
        registerTwo = _mm_add_epi32(registerTwo, _mm_slli_si128(registerTwo, 8)); // moves register 64 bit
        auto returnRegister = _mm_set1_epi32(previous);
        returnRegister = _mm_add_epi32(registerTwo, returnRegister);

        // compare results and filter
        __m128i lowBoundFilter = _mm_cmpgt_epi32(returnRegister, lowFilter);
        __m128i lowBoundEqualFilter = _mm_cmpeq_epi32(returnRegister, lowFilter);
        __m128i highBoundEqualFilter = _mm_cmpeq_epi32(returnRegister, highFilter);
        __m128i highBoundFilter = _mm_cmplt_epi32(returnRegister, highFilter);

        // greater than lower or equal
        auto lowerMask = _mm_or_si128(lowBoundFilter, lowBoundEqualFilter);
        // lower than higher or equal
        auto higherMask = _mm_or_si128(highBoundEqualFilter, highBoundFilter);

        // must be in bounds of the two
        auto resultMask = _mm_and_si128(lowerMask, higherMask);
#ifdef DEBUG
        print_register(resultMask);
        print_register(returnRegister);
#endif
        previous = _mm_extract_epi32(returnRegister, 3);

        alignas(16) int32_t v[4];
        alignas(16) int32_t r[4];
        _mm_store_si128((__m128i *) v, resultMask);
        _mm_store_si128((__m128i *) r, returnRegister);
        for (int j = 0; j < 4; j++) {
            if (v[j] != 0) {
                output[count] = r[j];
                count++;
            }
        }
    }

    return count;
}
