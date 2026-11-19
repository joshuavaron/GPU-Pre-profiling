#include "fakecuda.h"
#include <cuda.h>
#include <string.h>

#undef cuDevicePrimaryCtxRelease

CUresult cuInit(unsigned flags) {
  FAKE_LOG("cuInit(%u) = 0", flags);
  return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int *v) {
  *v = FAKECUDA_CUDA_VERSION;
  FAKE_LOG("cuDriverGetVersion() = 0  %d", *v);
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int *count) {
  *count = fake_device_count;
  FAKE_LOG("cuDeviceGetCount() = 0  count=%d", *count);
  return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice *dev, int ordinal) {
  CUresult r = CUDA_ERROR_INVALID_DEVICE;
  if (fake_valid(ordinal)) {
    *dev = ordinal;
    r = CUDA_SUCCESS;
  }
  FAKE_LOG("cuDeviceGet(ordinal=%d) = %d", ordinal, r);
  return r;
}

CUresult cuDeviceGetAttribute(int *value, CUdevice_attribute attr, CUdevice dev) {
  CUresult r = CUDA_ERROR_INVALID_DEVICE;
  if (fake_valid(dev)) {
    *value = fake_device_attr(attr, dev);
    r = CUDA_SUCCESS;
  }
  FAKE_LOG("cuDeviceGetAttribute(attr=%d, dev=%d) = %d  value=%d", attr, dev, r,
           r == CUDA_SUCCESS ? *value : -1);
  return r;
}

CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
  CUresult r = CUDA_ERROR_INVALID_DEVICE;
  if (fake_valid(dev)) {
    memcpy(uuid->bytes, fake_devices[dev].uuid.bytes, sizeof uuid->bytes);
    r = CUDA_SUCCESS;
  }
  FAKE_LOG("cuDeviceGetUuid(dev=%d) = %d", dev, r);
  return r;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned *flags, int *active) {
  CUresult r = CUDA_ERROR_INVALID_DEVICE;
  if (fake_valid(dev)) {
    *flags = 0;
    *active = __atomic_load_n(&fake_ctx_active[dev], __ATOMIC_RELAXED);
    r = CUDA_SUCCESS;
  }
  FAKE_LOG("cuDevicePrimaryCtxGetState(dev=%d) = %d  active=%d", dev, r, r ? -1 : *active);
  return r;
}

CUresult cuGetErrorString(CUresult err, const char **s) {
  *s = err == CUDA_SUCCESS ? "no error" : "fakecuda: operation not emulated";
  FAKE_LOG("cuGetErrorString(%d) = 0", err);
  return CUDA_SUCCESS;
}

CUresult cuGetErrorName(CUresult err, const char **s) {
  *s = err == CUDA_SUCCESS ? "CUDA_SUCCESS" : "CUDA_ERROR_NOT_SUPPORTED";
  FAKE_LOG("cuGetErrorName(%d) = 0", err);
  return CUDA_SUCCESS;
}

static char primary_ctx[FAKECUDA_MAX_DEVICES];
static __thread int ctx_stack[16];
static __thread int ctx_depth;

static CUcontext ctx_of(int dev) { return (CUcontext)&primary_ctx[dev]; }
static int dev_of(CUcontext ctx) {
  int dev = (int)((char *)ctx - primary_ctx);
  return ctx && fake_valid(dev) ? dev : -1;
}
void *fake_primary_ctx(int dev) { return ctx_of(dev); }
int fake_ctx_device(void *ctx) { return dev_of(ctx); }

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  if (!fake_valid(dev)) return CUDA_ERROR_INVALID_DEVICE;
  fake_ctx_touch(dev);
  *pctx = ctx_of(dev);
  FAKE_LOG("cuDevicePrimaryCtxRetain(dev=%d) = 0  %p", dev, (void *)*pctx);
  return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
  FAKE_LOG("cuDevicePrimaryCtxRelease(dev=%d) = 0", dev);
  return fake_valid(dev) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_DEVICE;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) { return cuDevicePrimaryCtxRelease_v2(dev); }

CUresult cuCtxGetCurrent(CUcontext *pctx) {
  int dev = fake_current_device;
  *pctx = __atomic_load_n(&fake_ctx_active[dev], __ATOMIC_RELAXED) ? ctx_of(dev) : NULL;
  FAKE_LOG("cuCtxGetCurrent() = 0  %p", (void *)*pctx);
  return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
  int dev = dev_of(ctx);
  FAKE_LOG("cuCtxSetCurrent(%p) = %d  dev=%d", (void *)ctx,
           ctx && dev < 0 ? CUDA_ERROR_INVALID_VALUE : 0, dev);
  if (!ctx) return CUDA_SUCCESS;
  if (dev < 0) return CUDA_ERROR_INVALID_VALUE;
  fake_current_device = dev;
  return CUDA_SUCCESS;
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
  int dev = dev_of(ctx);
  if (dev < 0 || ctx_depth == 16) return CUDA_ERROR_INVALID_VALUE;
  ctx_stack[ctx_depth++] = fake_current_device;
  fake_current_device = dev;
  FAKE_LOG("cuCtxPushCurrent(dev=%d) = 0", dev);
  return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
  if (ctx_depth == 0) return CUDA_ERROR_INVALID_VALUE;
  if (pctx) *pctx = ctx_of(fake_current_device);
  fake_current_device = ctx_stack[--ctx_depth];
  FAKE_LOG("cuCtxPopCurrent() = 0");
  return CUDA_SUCCESS;
}

CUresult cuCtxGetDevice(CUdevice *dev) {
  *dev = fake_current_device;
  FAKE_LOG("cuCtxGetDevice() = 0  dev=%d", *dev);
  return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
  FAKE_LOG("cuCtxSynchronize() = 0");
  return CUDA_SUCCESS;
}

static CUresult mem_error(int r) {
  return r == FAKE_MEM_OK ? CUDA_SUCCESS
       : r == FAKE_MEM_OOM ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t size) {
  void *p;
  CUresult r = mem_error(fake_mem_alloc(fake_current_device, size, &p));
  *dptr = (CUdeviceptr)p;
  FAKE_LOG("cuMemAlloc(%zu) = %d  dev=%d ptr=%p", size, r, fake_current_device, p);
  return r;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
  CUresult r = mem_error(fake_mem_free((void *)dptr));
  FAKE_LOG("cuMemFree(%p) = %d", (void *)dptr, r);
  return r;
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
  *total = fake_devices[fake_current_device].totalGlobalMem;
  *free = *total - fake_mem_used(fake_current_device);
  FAKE_LOG("cuMemGetInfo() = 0  dev=%d free=%zu total=%zu", fake_current_device, *free, *total);
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
  if (!fake_valid(dev)) return CUDA_ERROR_INVALID_DEVICE;
  if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
  snprintf(name, len, "%s", fake_devices[dev].name);
  FAKE_LOG("cuDeviceGetName(dev=%d) = 0  %s", dev, name);
  return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
  if (!fake_valid(dev)) return CUDA_ERROR_INVALID_DEVICE;
  *bytes = fake_devices[dev].totalGlobalMem;
  FAKE_LOG("cuDeviceTotalMem(dev=%d) = 0  %zu", dev, *bytes);
  return CUDA_SUCCESS;
}

static size_t ctx_limits[16] = {1024, 1 << 20, 8 << 20, 2, 2048, 0, 0};

CUresult cuCtxGetLimit(size_t *value, CUlimit limit) {
  if ((unsigned)limit >= 16) return CUDA_ERROR_INVALID_VALUE;
  *value = ctx_limits[limit];
  FAKE_LOG("cuCtxGetLimit(%d) = 0  %zu", limit, *value);
  return CUDA_SUCCESS;
}

CUresult cuDeviceCanAccessPeer(int *can, CUdevice dev, CUdevice peer) {
  if (!fake_valid(dev) || !fake_valid(peer)) return CUDA_ERROR_INVALID_DEVICE;
  *can = dev != peer;
  return CUDA_SUCCESS;
}

static char peer_enabled[FAKECUDA_MAX_DEVICES][FAKECUDA_MAX_DEVICES];

CUresult cuCtxEnablePeerAccess(CUcontext peer_ctx, unsigned flags) {
  int peer = dev_of(peer_ctx), dev = fake_current_device;
  if (peer < 0 || flags) return CUDA_ERROR_INVALID_VALUE;
  if (peer_enabled[dev][peer]) return CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
  peer_enabled[dev][peer] = 1;
  FAKE_LOG("cuCtxEnablePeerAccess(dev=%d -> %d) = 0", dev, peer);
  return CUDA_SUCCESS;
}

CUresult cuCtxDisablePeerAccess(CUcontext peer_ctx) {
  int peer = dev_of(peer_ctx), dev = fake_current_device;
  if (peer < 0) return CUDA_ERROR_INVALID_VALUE;
  if (!peer_enabled[dev][peer]) return CUDA_ERROR_PEER_ACCESS_NOT_ENABLED;
  peer_enabled[dev][peer] = 0;
  return CUDA_SUCCESS;
}
