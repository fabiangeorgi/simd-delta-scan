cmake_minimum_required(VERSION 3.16)
project(simd_delta)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

option(CI_BUILD "Set to ON for complete build in CI." OFF)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type specified. Defaulting to Debug.
                    Use `cmake -DCMAKE_BUILD_TYPE=Release ..` to build an optimized version.")
endif ()

if (CMAKE_BUILD_TYPE MATCHES Release)
  message(STATUS "Building an optimized release version.")
  add_compile_options(-O3)
endif ()

set(
    TASK_SOURCES

    # Implemented by you.
    src/simd_delta_scalar.cpp
    src/simd_delta_sse.cpp
    src/simd_delta_avx2.cpp
    src/simd_delta_avx512.cpp

    # Implemented by us.
    src/simd_compress.cpp
    src/simd_delta.hpp
    src/simd_util.cpp
    src/simd_util.hpp
)

# All SIMD-related flags relevant to this task.
set(SSE_FLAGS "-mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2")
set(AVX2_FLAGS "-mavx -mavx2")
set(AVX_512_FLAGS "-mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512vnni")

set_source_files_properties(src/simd_delta_sse.cpp PROPERTIES COMPILE_FLAGS ${SSE_FLAGS})
set_source_files_properties(src/simd_delta_avx2.cpp PROPERTIES COMPILE_FLAGS "${SSE_FLAGS} ${AVX2_FLAGS}")
set_source_files_properties(src/simd_delta_avx512.cpp PROPERTIES COMPILE_FLAGS "${SSE_FLAGS} ${AVX2_FLAGS} ${AVX_512_FLAGS}")

add_library(simd_delta ${TASK_SOURCES})
target_include_directories(simd_delta INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

enable_testing()
if (NOT ${CI_BUILD})
  FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG release-1.10.0
  )

  set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
  set(BUILD_GTEST ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  set(TEST_LIBS gtest gmock)
else ()
  find_package(GTest REQUIRED)
  set(TEST_LIBS gtest gmock -pthread)
  include_directories(${GTEST_INCLUDE_DIRS})
endif ()

add_executable(basic_test test/basic.cpp)
add_test(basic_test basic_test)
target_link_libraries(basic_test simd_delta ${TEST_LIBS})

if (${CI_BUILD})
  # Build advanced tests and benchmark only in CI.
  add_executable(advanced_test test/advanced.cpp)
  add_test(advanced_test advanced_test)
  target_link_libraries(advanced_test simd_delta ${TEST_LIBS})

  add_executable(hdp_benchmark test/benchmark.cpp)
  target_link_libraries(hdp_benchmark simd_delta)
endif ()

