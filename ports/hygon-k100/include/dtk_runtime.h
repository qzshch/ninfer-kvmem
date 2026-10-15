// Hygon DTK physical-Core adapter. Project modifications: Apache-2.0.
// The narrow CUDA-name mapping approach follows the Apache-2.0 NInfer HIP port:
// https://github.com/JCraigWasTaken/ninfer-gfx906
// revision ad0f68a48dc5135a11b08b33afe02a711980f797, src/core/hip_compat.h.
// Only runtime calls needed by the current physical Core are mapped. No CUDA
// instructions, FP4/FP8 codecs, warp operations or graph behavior are emulated.
#pragma once

#if !defined(NINFER_HYGON_DTK)
#error "This include surface belongs only to the Hygon DTK qualification target"
#endif
#include <hip/hip_runtime.h>

#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaSuccess hipSuccess
#define cudaErrorNoDevice hipErrorNoDevice
#define cudaErrorInsufficientDriver hipErrorInsufficientDriver
#define cudaErrorNotReady hipErrorNotReady
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventRecordExternal hipEventRecordExternal
#define cudaDeviceScheduleAuto hipDeviceScheduleAuto
#define cudaDeviceScheduleSpin hipDeviceScheduleSpin
#define cudaDeviceScheduleYield hipDeviceScheduleYield
#define cudaDeviceScheduleBlockingSync hipDeviceScheduleBlockingSync
#define cudaDeviceScheduleMask hipDeviceScheduleMask
#define cudaDevAttrMaxPitch hipDeviceAttributeMaxPitch
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaSetDeviceFlags hipSetDeviceFlags
#define cudaGetDeviceFlags hipGetDeviceFlags
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocDefault)
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemset2DAsync hipMemset2DAsync
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventRecordWithFlags hipEventRecordWithFlags
#define cudaEventQuery hipEventQuery
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphUpload hipGraphUpload
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
