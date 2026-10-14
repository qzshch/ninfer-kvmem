cmake_minimum_required(VERSION 3.28)

# A staged build of the existing physical Core, not a replacement inference build.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx928 CACHE STRING "Hygon K100_AI target")
endif()
if(NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx928")
  message(FATAL_ERROR "This port qualification targets gfx928 only")
endif()
project(ninfer_k100_core_qualification LANGUAGES CXX HIP)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(hip REQUIRED CONFIG)
find_package(Threads REQUIRED)
include(CTest)
get_filename_component(NINFER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

# Compile current v3/KVMem physical storage code directly. Graph execution and
# mathematical operators are deliberately outside this first qualification.
set(core_sources
  arena.cu device.cu dtype.cpp tensor.cpp layout.cpp weight_view.cpp
  host_worker_pool.cpp gdn_replay_records.cpp cyclic_kv_cache.cpp
  host_context_arena.cpp host_kv_arena.cpp paged_kv_cache.cpp
  linear_attention_state.cpp)
list(TRANSFORM core_sources PREPEND "${NINFER_ROOT}/src/core/")
set_source_files_properties("${NINFER_ROOT}/src/core/arena.cu"
  "${NINFER_ROOT}/src/core/device.cu" PROPERTIES LANGUAGE CXX)
add_library(ninfer_k100_physical_core STATIC ${core_sources})
# DTK uses this HIP platform-selection macro; it is not the device vendor name.
target_compile_definitions(ninfer_k100_physical_core PUBLIC NINFER_HYGON_DTK=1 __HIP_PLATFORM_AMD__=1)
target_include_directories(ninfer_k100_physical_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include" "${NINFER_ROOT}/src" "${NINFER_ROOT}/include")
target_link_libraries(ninfer_k100_physical_core PUBLIC hip::host Threads::Threads)

# Reuse the public Op wrapper and substitute a DTK mathematical implementation.
add_library(ninfer_k100_norm STATIC
  "${NINFER_ROOT}/src/ops/wrapper/rmsnorm.cpp"
  "${NINFER_ROOT}/src/ops/launcher/rmsnorm_hip.hip")
target_link_libraries(ninfer_k100_norm PUBLIC ninfer_k100_physical_core)

set_source_files_properties("${NINFER_ROOT}/src/ops/linear/portable/launch.cu" PROPERTIES LANGUAGE HIP)
add_library(ninfer_k100_projection STATIC
  "${NINFER_ROOT}/src/ops/linear/linear.cpp"
  "${NINFER_ROOT}/src/ops/linear/fp32.cpp"
  "${NINFER_ROOT}/src/ops/linear/portable/launch.cu")
target_link_libraries(ninfer_k100_projection PUBLIC ninfer_k100_physical_core)

add_subdirectory("${NINFER_ROOT}/kvmem" "${CMAKE_CURRENT_BINARY_DIR}/kvmem")

if(BUILD_TESTING)
  foreach(name IN ITEMS device arena tensor host_context_arena kv_cache state_store gdn_replay_records)
    add_executable(k100_${name}_test "${NINFER_ROOT}/tests/test_${name}.cpp")
    target_link_libraries(k100_${name}_test PRIVATE ninfer_k100_physical_core)
    foreach(device RANGE 0 2)
      add_test(NAME k100_${name}_device${device} COMMAND k100_${name}_test)
      set_tests_properties(k100_${name}_device${device} PROPERTIES
        ENVIRONMENT "HIP_VISIBLE_DEVICES=${device}"
        ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:"
        RUN_SERIAL TRUE TIMEOUT 120)
    endforeach()
  endforeach()
  add_executable(k100_peer_transfer_test tests/peer_transfer.hip)
  target_link_libraries(k100_peer_transfer_test PRIVATE ninfer_k100_physical_core)
  add_test(NAME k100_peer_transfer COMMAND k100_peer_transfer_test)
  set_tests_properties(k100_peer_transfer PROPERTIES RUN_SERIAL TRUE TIMEOUT 120)
  add_executable(k100_rmsnorm_test tests/rmsnorm.cpp)
  target_link_libraries(k100_rmsnorm_test PRIVATE ninfer_k100_norm)
  foreach(device RANGE 0 2)
    add_test(NAME k100_rmsnorm_device${device} COMMAND k100_rmsnorm_test)
    set_tests_properties(k100_rmsnorm_device${device} PROPERTIES
      ENVIRONMENT "HIP_VISIBLE_DEVICES=${device}"
      ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:"
      RUN_SERIAL TRUE TIMEOUT 120)
  endforeach()
  add_executable(k100_linear_test tests/linear.cpp)
  target_link_libraries(k100_linear_test PRIVATE ninfer_k100_projection)
  set(NINFER_K100_REAL_WEIGHT_SAMPLE "" CACHE FILEPATH "64x5120 original MXFP4 rows for the real-source oracle")
  foreach(device RANGE 0 2)
    add_test(NAME k100_linear_device${device} COMMAND k100_linear_test ${NINFER_K100_REAL_WEIGHT_SAMPLE})
    set_tests_properties(k100_linear_device${device} PROPERTIES
      ENVIRONMENT "HIP_VISIBLE_DEVICES=${device}"
      ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:" RUN_SERIAL TRUE TIMEOUT 180)
  endforeach()
endif()
