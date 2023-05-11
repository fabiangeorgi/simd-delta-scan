#include <algorithm>
#include <numeric>
#include <span>
#include <vector>

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include "simd_delta.hpp"

namespace {

using DecompressFn = std::function<decltype(decompress_scalar)>;
using ScanFn = std::function<decltype(scan_scalar)>;

class DecompressTestFixture : public ::testing::TestWithParam<DecompressFn> {};
class ScanTestFixture : public ::testing::TestWithParam<ScanFn> {};

}  // namespace

TEST_P(DecompressTestFixture, TestDecompress1) {
  constexpr int NUM_VALUES = 16;
  std::vector<uint32_t> input_numbers(NUM_VALUES);
  for (int i = 1; i < input_numbers.size(); ++i) {
    input_numbers[i] = input_numbers[i - 1] + i;
  }

  int8_t* compressed_data = compress_input(input_numbers);

  alignas(64) std::array<uint32_t, NUM_VALUES> result_buffer{};
  DecompressFn fn = GetParam();
  fn(compressed_data, input_numbers[0], input_numbers.size(), result_buffer.data());

  EXPECT_THAT(result_buffer, ::testing::ElementsAreArray(input_numbers));
  delete compressed_data;
}

TEST_P(DecompressTestFixture, TestDecompress2) {
  constexpr int NUM_VALUES = 128;
  std::vector<uint32_t> input_numbers(NUM_VALUES);
  for (int i = 1; i < input_numbers.size(); ++i) {
    input_numbers[i] = input_numbers[i - 1] + 42;
  }

  int8_t* compressed_data = compress_input(input_numbers);

  alignas(64) std::array<uint32_t, NUM_VALUES> result_buffer{};
  DecompressFn fn = GetParam();
  fn(compressed_data, input_numbers[0], input_numbers.size(), result_buffer.data());

  EXPECT_THAT(result_buffer, ::testing::ElementsAreArray(input_numbers));
  delete compressed_data;
}

TEST_P(ScanTestFixture, TestScanRange) {
  std::vector<uint32_t> input_numbers(16);
  std::iota(input_numbers.begin(), input_numbers.end(), 1);

  int8_t* compressed_data = compress_input(input_numbers);

  alignas(64) std::array<uint32_t, 6> result_buffer{};
  ScanFn fn = GetParam();
  size_t qualified_tuples =
      fn(3, 8, compressed_data, input_numbers[0], input_numbers.size(), result_buffer.data());

  EXPECT_EQ(qualified_tuples, 6);
  EXPECT_THAT(result_buffer, ::testing::ElementsAre(3, 4, 5, 6, 7, 8));

  delete compressed_data;
}

TEST_P(ScanTestFixture, TestScanEq) {
  std::vector<uint32_t> input_numbers(32);
  for (int i = 0; i < input_numbers.size(); ++i) {
    input_numbers[i] = i % 6;
  }

  int8_t* compressed_data = compress_input(input_numbers);

  alignas(64) std::array<uint32_t, 5> result_buffer{};
  ScanFn fn = GetParam();
  size_t qualified_tuples =
      fn(4, 4, compressed_data, input_numbers[0], input_numbers.size(), result_buffer.data());

  EXPECT_EQ(qualified_tuples, 5);
  EXPECT_THAT(result_buffer, ::testing::ElementsAreArray(std::vector<uint32_t>(5, 4)));

  delete compressed_data;
}

INSTANTIATE_TEST_SUITE_P(DecompressTests, DecompressTestFixture,
                         ::testing::Values(decompress_scalar, decompress_sse, decompress_avx2,
                                           decompress_avx512),
                         [](const auto& info) { return variant_to_string(info.index); });

INSTANTIATE_TEST_SUITE_P(ScanTests, ScanTestFixture,
                         ::testing::Values(scan_scalar, scan_sse, scan_avx2, scan_avx512),
                         [](const auto& info) { return variant_to_string(info.index); });

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
