cmake_minimum_required(VERSION 3.20)

# This core can be built and tested independently, without NInfer or CUDA.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(kvmem_core LANGUAGES CXX)
  include(CTest)
endif()

add_library(kvmem_core STATIC src/block_retrieval.cpp)
target_compile_features(kvmem_core PUBLIC cxx_std_20)
target_include_directories(kvmem_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(BUILD_TESTING)
  add_executable(kvmem_core_test tests/test_core.cpp)
  target_link_libraries(kvmem_core_test PRIVATE kvmem_core)
  add_test(NAME kvmem_core_test COMMAND kvmem_core_test)
endif()
