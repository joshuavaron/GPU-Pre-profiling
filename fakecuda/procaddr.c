#include "fakecuda.h"
#include <string.h>

#define CUDA_SUCCESS 0
#define CUDA_ERROR_NOT_FOUND 500

#define DRIVER_SYMS                                                         \
  X(cuInit, cuInit, 0)                                                      \
  X(cuDriverGetVersion, cuDriverGetVersion, 0)                              \
  X(cuDeviceGetCount, cuDeviceGetCount, 0)                                  \
  X(cuDeviceGet, cuDeviceGet, 0)                                            \
  X(cuDeviceGetAttribute, cuDeviceGetAttribute, 0)                          \
  X(cuDeviceGetUuid, cuDeviceGetUuid, 0)                                    \
  X(cuDevicePrimaryCtxGetState, cuDevicePrimaryCtxGetState, 0)              \
  X(cuDeviceGetName, cuDeviceGetName, 0)                                    \
  X(cuDeviceTotalMem, cuDeviceTotalMem_v2, 3020)                            \
  X(cuCtxGetLimit, cuCtxGetLimit, 0)                                        \
  X(cuGetErrorString, cuGetErrorString, 0)                                  \
  X(cuGetErrorName, cuGetErrorName, 0)                                      \
  X(cuDevicePrimaryCtxRetain, cuDevicePrimaryCtxRetain, 0)                  \
  X(cuDevicePrimaryCtxRelease, cuDevicePrimaryCtxRelease_v2, 0)             \
  X(cuCtxGetCurrent, cuCtxGetCurrent, 0)                                    \
  X(cuCtxSetCurrent, cuCtxSetCurrent, 0)                                    \
  X(cuCtxPushCurrent, cuCtxPushCurrent_v2, 0)                               \
  X(cuCtxPopCurrent, cuCtxPopCurrent_v2, 0)                                 \
  X(cuCtxGetDevice, cuCtxGetDevice, 0)                                      \
  X(cuCtxSynchronize, cuCtxSynchronize, 0)                                  \
  X(cuGetExportTable, cuGetExportTable, 0)                                  \
  X(cuMemAlloc, cuMemAlloc_v2, 0)                                           \
  X(cuMemFree, cuMemFree_v2, 0)                                             \
  X(cuMemGetInfo, cuMemGetInfo_v2, 0)                                       \
  X(cuModuleGetLoadingMode, cuModuleGetLoadingMode, 0)                      \
  X(cuModuleLoadData, cuModuleLoadData, 0)                                  \
  X(cuModuleLoadDataEx, cuModuleLoadDataEx, 0)                              \
  X(cuModuleUnload, cuModuleUnload, 0)                                      \
  X(cuModuleGetFunction, cuModuleGetFunction, 0)                            \
  X(cuModuleGetGlobal, cuModuleGetGlobal_v2, 3020)                          \
  X(cuLibraryLoadData, cuLibraryLoadData, 0)                                \
  X(cuLibraryUnload, cuLibraryUnload, 0)                                    \
  X(cuLibraryGetModule, cuLibraryGetModule, 0)                              \
  X(cuLibraryGetKernel, cuLibraryGetKernel, 0)                              \
  X(cuKernelGetFunction, cuKernelGetFunction, 0)                            \
  X(cuKernelGetAttribute, cuKernelGetAttribute, 0)                          \
  X(cuKernelSetAttribute, cuKernelSetAttribute, 0)                          \
  X(cuFuncGetAttribute, cuFuncGetAttribute, 0)                              \
  X(cuFuncSetAttribute, cuFuncSetAttribute, 0)                              \
  X(cuFuncSetCacheConfig, cuFuncSetCacheConfig, 0)                          \
  X(cuOccupancyMaxActiveBlocksPerMultiprocessor,                            \
    cuOccupancyMaxActiveBlocksPerMultiprocessor, 0)                         \
  X(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,                   \
    cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, 0)                \
  X(cuOccupancyAvailableDynamicSMemPerBlock, cuOccupancyAvailableDynamicSMemPerBlock, 0) \
  X(cuOccupancyMaxActiveClusters, cuOccupancyMaxActiveClusters, 0)          \
  X(cuLaunchKernel, cuLaunchKernel, 0)                                      \
  X(cuLaunchKernelEx, cuLaunchKernelEx, 0)                                  \
  X(cuDeviceCanAccessPeer, cuDeviceCanAccessPeer, 0)                        \
  X(cuCtxEnablePeerAccess, cuCtxEnablePeerAccess, 0)                        \
  X(cuCtxDisablePeerAccess, cuCtxDisablePeerAccess, 0)                      \
  X(cuStreamCreate, cuStreamCreate, 0)                                      \
  X(cuStreamDestroy, cuStreamDestroy_v2, 0)                                 \
  X(cuStreamSynchronize, cuStreamSynchronize, 0)                            \
  X(cuStreamWaitEvent, cuStreamWaitEvent, 0)                                \
  X(cuStreamGetCtx, cuStreamGetCtx, 0)                                      \
  X(cuStreamIsCapturing, cuStreamIsCapturing, 0)                            \
  X(cuStreamGetCaptureInfo, cuStreamGetCaptureInfo_v2, 11030)               \
  X(cuStreamGetCaptureInfo, cuStreamGetCaptureInfo_v3, 12030)               \
  X(cuThreadExchangeStreamCaptureMode, cuThreadExchangeStreamCaptureMode, 0) \
  X(cuStreamWaitValue32, cuStreamWaitValue32, 0)                            \
  X(cuStreamBatchMemOp, cuStreamBatchMemOp, 0)                              \
  X(cuEventCreate, cuEventCreate, 0)                                        \
  X(cuEventDestroy, cuEventDestroy_v2, 0)                                   \
  X(cuEventRecord, cuEventRecord, 0)                                        \
  X(cuEventQuery, cuEventQuery, 0)                                          \
  X(cuEventSynchronize, cuEventSynchronize, 0)                              \
  X(cuEventElapsedTime, cuEventElapsedTime, 0)                              \
  X(cuMemcpy, cuMemcpy, 0)                                                  \
  X(cuMemcpyAsync, cuMemcpyAsync, 0)                                        \
  X(cuMemcpyHtoD, cuMemcpyHtoD_v2, 3020)                                    \
  X(cuMemcpyDtoDAsync, cuMemcpyDtoDAsync_v2, 3020)                          \
  X(cuMemcpy2DAsync, cuMemcpy2DAsync_v2, 3020)                              \
  X(cuMemsetD8, cuMemsetD8_v2, 3020)                                        \
  X(cuMemPrefetchAsync, cuMemPrefetchAsync, 0)                              \
  X(cuMemPrefetchAsync, cuMemPrefetchAsync_v2, 12020)                       \
  X(cuPointerGetAttribute, cuPointerGetAttribute, 0)                        \
  X(cuPointerGetAttributes, cuPointerGetAttributes, 0)

#define X(name, impl, ver) void impl(void);
DRIVER_SYMS
#undef X
int cuGetProcAddress(const char *symbol, void **pfn, int version, unsigned long long flags);
int cuGetProcAddress_v2(const char *symbol, void **pfn, int version, unsigned long long flags,
                        int *status);

static const struct { const char *name; void *fn; int since; } driver_syms[] = {
#define X(name, impl, ver) {#name, (void *)impl, ver},
    DRIVER_SYMS
#undef X
};

int fake_get_proc_address(const char *symbol, void **pfn, int version,
                          unsigned long long flags, int *status) {
  void *fn = NULL;
  int best = -1;
  if (strcmp(symbol, "cuGetProcAddress") == 0)
    fn = version >= 12000 ? (void *)cuGetProcAddress_v2 : (void *)cuGetProcAddress;
  for (size_t i = 0; !fn && i < sizeof driver_syms / sizeof *driver_syms; i++)
    if (strcmp(symbol, driver_syms[i].name) == 0 && driver_syms[i].since <= version &&
        driver_syms[i].since > best) {
      best = driver_syms[i].since;
      fn = driver_syms[i].fn;
    }
  for (size_t i = 0; !fn && best < 0 && i < sizeof driver_syms / sizeof *driver_syms; i++)
    if (strcmp(symbol, driver_syms[i].name) == 0) best = -2;

  if (fn) {
    FAKE_LOG("cuGetProcAddress(%s, v%d, flags=%llu) -> emulated", symbol, version, flags);
  } else {
    FAKE_LOG("MISSING (looked up): %s (v%d)%s", symbol, version,
             best == -2 ? " [only newer versions emulated]" : "");
    fn = fake_stub_for(symbol);
  }

  *pfn = fn;
  if (status) *status = fn ? 0 : 1;
  return fn ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}

int cuGetProcAddress(const char *symbol, void **pfn, int version, unsigned long long flags) {
  return fake_get_proc_address(symbol, pfn, version, flags, NULL);
}

int cuGetProcAddress_v2(const char *symbol, void **pfn, int version, unsigned long long flags,
                        int *status) {
  return fake_get_proc_address(symbol, pfn, version, flags, status);
}
