cmake_minimum_required(VERSION 3.28)

# Finite Hygon DTK Core/Ops and resident GLM public Engine build.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx928 CACHE STRING "Hygon K100_AI target")
endif()
if(NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx928")
  message(FATAL_ERROR "This port qualification targets gfx928 only")
endif()
project(ninfer_k100_core_qualification LANGUAGES C CXX HIP)
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

# Compile current v3/KVMem physical storage code directly; Ops/Model/Program/Engine
# are separate consumers below. Core owns real HIP device Graph resources.
set(core_sources
  arena.cu device.cu dtype.cpp tensor.cpp layout.cpp weight_view.cpp
  host_worker_pool.cpp gdn_replay_records.cpp cyclic_kv_cache.cpp
  host_context_arena.cpp host_kv_arena.cpp paged_kv_cache.cpp
  linear_attention_state.cpp decode_graph.cpp indexed_transfer.hip)
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
  "${NINFER_ROOT}/src/ops/linear/portable/indexed_hip.hip"
  "${NINFER_ROOT}/src/ops/linear/portable/grouped_hip.hip"
  "${NINFER_ROOT}/src/ops/linear/portable/launch.cu")
target_link_libraries(ninfer_k100_projection PUBLIC ninfer_k100_physical_core)
target_include_directories(ninfer_k100_projection PRIVATE "${NINFER_ROOT}/third_party")

add_library(ninfer_k100_kda STATIC
  "${NINFER_ROOT}/src/ops/linear_attention/kimi_delta_attention/kimi_delta_attention.cpp"
  "${NINFER_ROOT}/src/ops/linear_attention/kimi_delta_attention/chunked_hip.hip"
  "${NINFER_ROOT}/src/ops/linear_attention/kimi_delta_attention/recurrent_hip.hip")
target_link_libraries(ninfer_k100_kda PUBLIC ninfer_k100_physical_core)

add_library(ninfer_k100_hyper_connection STATIC "${NINFER_ROOT}/src/ops/hyper_connection/launch_hip.hip")
target_link_libraries(ninfer_k100_hyper_connection PUBLIC ninfer_k100_physical_core)
add_library(ninfer_k100_resident_transform STATIC "${NINFER_ROOT}/src/ops/resident_transform/launch_hip.hip"
  "${NINFER_ROOT}/src/ops/resident_transform/parallel_hip.hip")
target_link_libraries(ninfer_k100_resident_transform PUBLIC ninfer_k100_physical_core)
add_library(ninfer_k100_sparse_latent STATIC "${NINFER_ROOT}/src/ops/sparse_latent_attention/launch_hip.hip")
target_link_libraries(ninfer_k100_sparse_latent PUBLIC ninfer_k100_physical_core)
add_library(ninfer_k100_dflash_block STATIC "${NINFER_ROOT}/src/ops/dflash_block/launch_hip.hip")
target_link_libraries(ninfer_k100_dflash_block PUBLIC ninfer_k100_physical_core)
set(artifact_sources schema.cpp formats.cpp layouts.cpp file_io.cpp reader.cpp binder.cpp materializer.cpp views.cpp)
list(TRANSFORM artifact_sources PREPEND "${NINFER_ROOT}/src/artifact/")
add_library(ninfer_k100_artifact STATIC ${artifact_sources})
target_include_directories(ninfer_k100_artifact PUBLIC "${NINFER_ROOT}/third_party")
target_link_libraries(ninfer_k100_artifact PUBLIC ninfer_k100_physical_core)
add_library(ninfer_k100_glm_model STATIC "${NINFER_ROOT}/src/models/glm5_next/model.cpp")
target_link_libraries(ninfer_k100_glm_model PUBLIC ninfer_k100_artifact)
add_subdirectory("${NINFER_ROOT}/third_party/llama-jinja" "${CMAKE_CURRENT_BINARY_DIR}/jinja")
add_library(ninfer_k100_unicode STATIC "${NINFER_ROOT}/src/text/unicode.cpp"
  "${NINFER_ROOT}/third_party/utf8proc/utf8proc.c")
target_include_directories(ninfer_k100_unicode PUBLIC "${NINFER_ROOT}/src" "${NINFER_ROOT}/third_party")
target_include_directories(ninfer_jinja PRIVATE "${NINFER_ROOT}/src" "${NINFER_ROOT}/third_party")
target_link_libraries(ninfer_jinja PRIVATE ninfer_k100_unicode)
add_library(ninfer_k100_text STATIC "${NINFER_ROOT}/src/text/jinja.cpp")
target_include_directories(ninfer_k100_text PUBLIC "${NINFER_ROOT}/src" "${NINFER_ROOT}/include" "${NINFER_ROOT}/third_party")
target_link_libraries(ninfer_k100_text PUBLIC ninfer_jinja ninfer_k100_unicode)
add_library(ninfer_k100_glm_frontend STATIC "${NINFER_ROOT}/src/models/glm5_next/frontend.cpp"
  "${NINFER_ROOT}/src/runtime/engine/public_types.cpp")
find_library(PCRE2_LIBRARY NAMES pcre2-8 REQUIRED)
target_link_libraries(ninfer_k100_glm_frontend PUBLIC ninfer_k100_glm_model ninfer_k100_text "${PCRE2_LIBRARY}")

add_subdirectory("${NINFER_ROOT}/kvmem" "${CMAKE_CURRENT_BINARY_DIR}/kvmem")
add_library(ninfer_k100_glm_program STATIC "${NINFER_ROOT}/src/models/glm5_next/program.cpp"
  "${NINFER_ROOT}/src/models/glm5_next/program_parallel.cpp")
target_link_libraries(ninfer_k100_glm_program PUBLIC ninfer_k100_glm_model ninfer_k100_projection
  ninfer_k100_kda ninfer_k100_hyper_connection ninfer_k100_resident_transform ninfer_k100_sparse_latent ninfer_k100_dflash_block kvmem_core)
add_library(ninfer_k100_engine STATIC "${NINFER_ROOT}/src/runtime/engine/engine_dtk.cpp"
  "${NINFER_ROOT}/src/runtime/contract/sampling.cpp")
target_link_libraries(ninfer_k100_engine PUBLIC ninfer_k100_glm_program ninfer_k100_glm_frontend Threads::Threads)
add_executable(ninfer_k100_generate ninfer_generate.cpp)
target_link_libraries(ninfer_k100_generate PRIVATE ninfer_k100_engine)

if(BUILD_TESTING)
  add_executable(k100_row_materialization_test tests/row_materialization.cpp)
  target_include_directories(k100_row_materialization_test PRIVATE "${NINFER_ROOT}/tests")
  target_link_libraries(k100_row_materialization_test PRIVATE ninfer_k100_artifact)
  foreach(device RANGE 0 2)
    add_test(NAME k100_row_materialization_device${device} COMMAND k100_row_materialization_test)
    set_tests_properties(k100_row_materialization_device${device} PROPERTIES
      ENVIRONMENT "HIP_VISIBLE_DEVICES=${device}" RUN_SERIAL TRUE TIMEOUT 120)
  endforeach()
  add_executable(k100_glm_frames_test "${NINFER_ROOT}/tests/runtime/test_glm_frames.cpp")
  target_include_directories(k100_glm_frames_test PRIVATE "${NINFER_ROOT}/src")
  add_test(NAME k100_glm_frames COMMAND k100_glm_frames_test)
  add_executable(k100_token_batch_test "${NINFER_ROOT}/tests/runtime/test_token_batch.cpp")
  add_executable(k100_kda_chunk_plan_test "${NINFER_ROOT}/tests/ops/test_kda_chunk_plan.cpp")
  target_include_directories(k100_kda_chunk_plan_test PRIVATE "${NINFER_ROOT}/src")
  add_test(NAME k100_kda_chunk_plan COMMAND k100_kda_chunk_plan_test)
  target_include_directories(k100_token_batch_test PRIVATE "${NINFER_ROOT}/src")
  add_test(NAME k100_token_batch COMMAND k100_token_batch_test)
  # Real artifact qualification is opt-in, never an implicit 145-GiB CTest load.
  add_executable(ninfer_k100_parallel_check tests/program_parallel.cpp)
  target_link_libraries(ninfer_k100_parallel_check PRIVATE ninfer_k100_glm_program ninfer_k100_glm_frontend)
  add_executable(ninfer_k100_prefill_check tests/program_prefill.cpp)
  target_link_libraries(ninfer_k100_prefill_check PRIVATE ninfer_k100_glm_program ninfer_k100_glm_frontend)
  add_executable(ninfer_k100_engine_check tests/engine_batch.cpp)
  target_link_libraries(ninfer_k100_engine_check PRIVATE ninfer_k100_engine)
  add_executable(k100_resident_batch_test tests/resident_batch.cpp)
  target_link_libraries(k100_resident_batch_test PRIVATE ninfer_k100_sparse_latent kvmem_core)
  foreach(device RANGE 0 2)
    add_test(NAME k100_resident_batch_device${device} COMMAND k100_resident_batch_test)
    set_tests_properties(k100_resident_batch_device${device} PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=${device}" RUN_SERIAL TRUE TIMEOUT 180)
  endforeach()
  add_executable(k100_tokenizer_test tests/tokenizer.cpp)
  target_link_libraries(k100_tokenizer_test PRIVATE ninfer_k100_glm_frontend)
  set(NINFER_K100_TOKENIZER_FIXTURE "" CACHE FILEPATH "Pinned official CPU tokenizer/Jinja parity fixture")
  if(NINFER_K100_TOKENIZER_FIXTURE)
    add_test(NAME k100_tokenizer COMMAND k100_tokenizer_test "${NINFER_K100_TOKENIZER_FIXTURE}")
  endif()
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
  add_executable(k100_weight_decode_test tests/weight_decode.hip)
  target_include_directories(k100_weight_decode_test PRIVATE "${NINFER_ROOT}/third_party")
  target_link_libraries(k100_weight_decode_test PRIVATE ninfer_k100_physical_core)
  foreach(device RANGE 0 2)
    add_test(NAME k100_weight_decode_device${device} COMMAND k100_weight_decode_test)
    set_tests_properties(k100_weight_decode_device${device} PROPERTIES
      ENVIRONMENT "HIP_VISIBLE_DEVICES=${device}"
      ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:" RUN_SERIAL TRUE TIMEOUT 60)
  endforeach()
  add_executable(k100_kda_test tests/kda.cpp)
  target_include_directories(k100_kda_test PRIVATE "${NINFER_ROOT}/tests")
  target_link_libraries(k100_kda_test PRIVATE ninfer_k100_kda)
  foreach(device RANGE 0 2)
    add_test(NAME k100_kda_device${device} COMMAND k100_kda_test)
    set_tests_properties(k100_kda_device${device} PROPERTIES
      ENVIRONMENT "HIP_VISIBLE_DEVICES=${device}"
      ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:" RUN_SERIAL TRUE TIMEOUT 180)
  endforeach()
  find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
  add_executable(k100_dflash_block_test tests/dflash_block.cpp)
  target_link_libraries(k100_dflash_block_test PRIVATE ninfer_k100_dflash_block)
  foreach(device RANGE 0 2)
    add_test(NAME k100_dflash_block_device${device} COMMAND k100_dflash_block_test)
    set_tests_properties(k100_dflash_block_device${device} PROPERTIES
      ENVIRONMENT "HIP_VISIBLE_DEVICES=${device}"
      ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:" RUN_SERIAL TRUE TIMEOUT 180)
  endforeach()
  add_executable(k100_attention_test tests/attention.cpp)
  target_link_libraries(k100_attention_test PRIVATE ninfer_k100_sparse_latent ninfer_k100_dflash_block)
  foreach(device RANGE 0 2)
    add_test(NAME k100_attention_device${device} COMMAND k100_attention_test)
    set_tests_properties(k100_attention_device${device} PROPERTIES
      ENVIRONMENT "HIP_VISIBLE_DEVICES=${device}"
      ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:" RUN_SERIAL TRUE TIMEOUT 180)
  endforeach()
  add_executable(k100_resident_transform_test tests/resident_transform.cpp)
  target_link_libraries(k100_resident_transform_test PRIVATE ninfer_k100_resident_transform)
  foreach(device RANGE 0 2)
    add_test(NAME k100_resident_transform_device${device} COMMAND k100_resident_transform_test)
    set_tests_properties(k100_resident_transform_device${device} PROPERTIES
      ENVIRONMENT "HIP_VISIBLE_DEVICES=${device}"
      ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:" RUN_SERIAL TRUE TIMEOUT 180)
  endforeach()
  add_executable(k100_hyper_connection_test tests/hyper_connection.cpp)
  target_link_libraries(k100_hyper_connection_test PRIVATE ninfer_k100_hyper_connection)
  foreach(device RANGE 0 2)
    add_test(NAME k100_hyper_connection_device${device} COMMAND k100_hyper_connection_test)
    set_tests_properties(k100_hyper_connection_device${device} PROPERTIES
      ENVIRONMENT "HIP_VISIBLE_DEVICES=${device}"
      ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:" RUN_SERIAL TRUE TIMEOUT 180)
  endforeach()
  set(block_fixture_dir "${CMAKE_CURRENT_BINARY_DIR}/block-fixtures")
  add_custom_command(OUTPUT "${block_fixture_dir}/stamp"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/block_fixtures.py" "${block_fixture_dir}"
    DEPENDS tests/block_fixtures.py "${NINFER_ROOT}/tools/artifact/codecs/block256.py"
      "${NINFER_ROOT}/third_party/ggml-blocks/iq3s_grid.inc")
  add_custom_target(k100_block_fixtures DEPENDS "${block_fixture_dir}/stamp")
  add_executable(k100_block_projection_test tests/block_projection.cpp "${NINFER_ROOT}/src/artifact/formats.cpp")
  target_include_directories(k100_block_projection_test PRIVATE "${NINFER_ROOT}/third_party")
  target_link_libraries(k100_block_projection_test PRIVATE ninfer_k100_projection)
  add_dependencies(k100_block_projection_test k100_block_fixtures)
  foreach(device RANGE 0 2)
    add_test(NAME k100_block_projection_device${device} COMMAND k100_block_projection_test "${block_fixture_dir}")
    set_tests_properties(k100_block_projection_device${device} PROPERTIES
      ENVIRONMENT "HIP_VISIBLE_DEVICES=${device}"
      ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:" RUN_SERIAL TRUE TIMEOUT 180)
  endforeach()
endif()
