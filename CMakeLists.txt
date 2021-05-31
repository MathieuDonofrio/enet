#
# ENET
#

cmake_minimum_required(VERSION 3.14)

#
# Project Configuration
#

project(
  ENET
  VERSION 1.0.0
  DESCRIPTION "Fast evolutionary artificial neural network library"
  HOMEPAGE_URL "https://github.com/MathieuDonofrio/enet"
  LANGUAGES CXX
)

#
# Compiler Options
#

if(MSVC)
  add_compile_options(
    /W4
    $<$<CONFIG:Debug>:/Od>
    $<$<CONFIG:Release>:/O2>
  )
else()
  add_compile_options(
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:Release>:-O3>
  )
endif()

#
# Target
#

add_library(ENET INTERFACE)
target_include_directories(ENET INTERFACE ${ENET_SOURCE_DIR}/src)
target_compile_features(ENET INTERFACE cxx_std_17)

#
# Tests
#

option(ENET_BUILD_TESTING "Enable building with tests" OFF)

if(ENET_BUILD_TESTING)
  include(CTest)
  enable_testing()
  add_subdirectory(test)
endif()

#
# Benchmarks
#

option(ENET_BUILD_BENCHMARKING "Enable building with benchmarks" OFF)

if(ENET_BUILD_BENCHMARKING)
  add_subdirectory(bench)
endif()