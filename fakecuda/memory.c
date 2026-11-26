#include "fakecuda.h"
#include <cuda.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

struct alloc {
  char *ptr;
  size_t size;
  int device;
};

static struct alloc *allocs;
static size_t n_allocs, cap_allocs;
static size_t used[FAKECUDA_MAX_DEVICES];
static pthread_mutex_t mem_lock = PTHREAD_MUTEX_INITIALIZER;

static void track_add(char *ptr, size_t size, int device) {
  if (n_allocs == cap_allocs) {
    cap_allocs = cap_allocs ? 2 * cap_allocs : 256;
    allocs = realloc(allocs, cap_allocs * sizeof *allocs);
  }
  allocs[n_allocs++] = (struct alloc){ptr, size, device};
  if (device >= 0) used[device] += size;
}

static int track_remove(void *ptr, int want_host, size_t *size, int *device) {
  for (size_t i = 0; i < n_allocs; i++) {
    if (allocs[i].ptr != ptr || (allocs[i].device < 0) != want_host) continue;
    *size = allocs[i].size;
    *device = allocs[i].device;
    if (allocs[i].device >= 0) used[allocs[i].device] -= allocs[i].size;
    allocs[i] = allocs[--n_allocs];
    return 1;
  }
  return 0;
}

static void *reserve(size_t size) {
  void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                 -1, 0);
  return p == MAP_FAILED ? NULL : p;
}

int fake_mem_alloc(int device, size_t size, void **out) {
  *out = NULL;
  if (size == 0) return FAKE_MEM_OK;
  pthread_mutex_lock(&mem_lock);
  int r = FAKE_MEM_OOM;
  fake_ctx_touch(device);
  if (size <= fake_devices[device].totalGlobalMem - used[device] && (*out = reserve(size))) {
    track_add(*out, size, device);
    r = FAKE_MEM_OK;
  }
  pthread_mutex_unlock(&mem_lock);

  struct fake_event e;
  memset(&e, 0, sizeof e);
  if (r == FAKE_MEM_OK)
    e.type = FAKE_EVENT_ALLOC;
  else
    e.type = FAKE_EVENT_ALLOC_FAILED;
  e.api = "fake_mem_alloc";
  e.device = device;
  e.ptr = *out;
  e.size = size;
  fire_callback(&e);
  return r;
}

int fake_mem_free(void *ptr) {
  if (!ptr) return FAKE_MEM_OK;
  size_t size;
  int device;
  pthread_mutex_lock(&mem_lock);
  int found = track_remove(ptr, 0, &size, &device);
  pthread_mutex_unlock(&mem_lock);
  if (!found) return FAKE_MEM_INVALID;
  munmap(ptr, size);

  struct fake_event e;
  memset(&e, 0, sizeof e);
  e.type = FAKE_EVENT_FREE;
  e.api = "fake_mem_free";
  e.device = device;
  e.ptr = ptr;
  e.size = size;
  fire_callback(&e);
  return FAKE_MEM_OK;
}

int fake_host_alloc(size_t size, void **out) {
  *out = NULL;
  if (size == 0) return FAKE_MEM_OK;
  if (!(*out = reserve(size))) return FAKE_MEM_OOM;
  pthread_mutex_lock(&mem_lock);
  track_add(*out, size, -1);
  pthread_mutex_unlock(&mem_lock);
  return FAKE_MEM_OK;
}

int fake_host_free(void *ptr) {
  if (!ptr) return FAKE_MEM_OK;
  size_t size;
  int device;
  pthread_mutex_lock(&mem_lock);
  int found = track_remove(ptr, 1, &size, &device);
  pthread_mutex_unlock(&mem_lock);
  if (!found) return FAKE_MEM_INVALID;
  munmap(ptr, size);
  return FAKE_MEM_OK;
}

size_t fake_mem_used(int device) {
  pthread_mutex_lock(&mem_lock);
  size_t u = used[device];
  pthread_mutex_unlock(&mem_lock);
  return u;
}

int fake_mem_range(const void *ptr, int *device, void **base, size_t *size) {
  int found = 0;
  pthread_mutex_lock(&mem_lock);
  for (size_t i = 0; i < n_allocs && !found; i++) {
    const struct alloc *a = &allocs[i];
    if ((const char *)ptr >= a->ptr && (const char *)ptr < a->ptr + a->size) {
      if (device) *device = a->device;
      if (base) *base = a->ptr;
      if (size) *size = a->size;
      found = 1;
    }
  }
  pthread_mutex_unlock(&mem_lock);
  return found;
}

int fake_mem_lookup(const void *ptr, int *device) { return fake_mem_range(ptr, device, NULL, NULL); }

#undef cuStreamWaitValue32
#undef cuStreamBatchMemOp

static CUresult copy(const char *api, void *dst, const void *src, size_t n) {
  if (n && (!dst || !src)) return CUDA_ERROR_INVALID_VALUE;
  memmove(dst, src, n);
  FAKE_LOG("%s(%p <- %p, %zu) = 0", api, dst, src, n);
  return CUDA_SUCCESS;
}

#define P(x) ((void *)(uintptr_t)(x))

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t n) {
  return copy("cuMemcpy", P(dst), P(src), n);
}
CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream s) {
  (void)s;
  return copy("cuMemcpyAsync", P(dst), P(src), n);
}
CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t n) {
  return copy("cuMemcpyHtoD", P(dst), src, n);
}
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream s) {
  (void)s;
  return copy("cuMemcpyDtoDAsync", P(dst), P(src), n);
}

static const char *side(CUmemorytype t, const void *host, CUdeviceptr dev) {
  return t == CU_MEMORYTYPE_HOST ? host : t == CU_MEMORYTYPE_DEVICE || t == CU_MEMORYTYPE_UNIFIED
                                              ? P(dev) : NULL;
}

CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *c, CUstream s) {
  (void)s;
  if (!c) return CUDA_ERROR_INVALID_VALUE;
  const char *src = side(c->srcMemoryType, c->srcHost, c->srcDevice);
  char *dst = (char *)side(c->dstMemoryType, c->dstHost, c->dstDevice);
  if (!src || !dst) {
    fprintf(stderr, "[fakecuda] cuMemcpy2DAsync: CUDA arrays are not emulated\n");
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  for (size_t y = 0; y < c->Height; y++)
    memmove(dst + (c->dstY + y) * c->dstPitch + c->dstXInBytes,
            src + (c->srcY + y) * c->srcPitch + c->srcXInBytes, c->WidthInBytes);
  FAKE_LOG("cuMemcpy2DAsync(%p <- %p, %zu x %zu) = 0", (void *)dst, (const void *)src, c->WidthInBytes,
           c->Height);
  return CUDA_SUCCESS;
}

static CUresult fill(const char *api, CUdeviceptr dst, unsigned v, size_t elem, size_t n) {
  char *d = P(dst);
  if (n && !d) return CUDA_ERROR_INVALID_VALUE;
  if (elem == 1) {
    memset(d, (int)v, n);
  } else {
    for (size_t i = 0; i < n; i++) memcpy(d + i * elem, &v, elem);
  }
  FAKE_LOG("%s(%p, %#x, %zu) = 0", api, (void *)d, v, n);
  return CUDA_SUCCESS;
}

CUresult cuMemsetD8_v2(CUdeviceptr d, unsigned char v, size_t n) { return fill("cuMemsetD8", d, v, 1, n); }

CUresult cuMemPrefetchAsync(CUdeviceptr p, size_t n, CUdevice dev, CUstream s) {
  (void)s;
  FAKE_LOG("cuMemPrefetchAsync(%p, %zu, dev=%d) = 0  no-op", P(p), n, dev);
  return CUDA_SUCCESS;
}

static CUresult pointer_attr(void *data, CUpointer_attribute a, CUdeviceptr p) {
  void *base = NULL;
  size_t size = 0;
  int dev = -1, known = fake_mem_range(P(p), &dev, &base, &size);
  int d = dev >= 0 ? dev : fake_current_device;
  switch (a) {
  case CU_POINTER_ATTRIBUTE_CONTEXT:
    if (!known) return CUDA_ERROR_INVALID_VALUE;
    *(CUcontext *)data = fake_primary_ctx(d);
    break;
  case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
    if (!known) return CUDA_ERROR_INVALID_VALUE;
    *(CUmemorytype *)data = dev >= 0 ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
    break;
  case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
    if (!known) return CUDA_ERROR_INVALID_VALUE;
    *(CUdeviceptr *)data = p;
    break;
  case CU_POINTER_ATTRIBUTE_HOST_POINTER:
    if (!known || dev >= 0) return CUDA_ERROR_INVALID_VALUE;
    *(void **)data = P(p);
    break;
  case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS: *(int *)data = 1; break;
  case CU_POINTER_ATTRIBUTE_BUFFER_ID:
  case CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID:
    if (!known) return CUDA_ERROR_INVALID_VALUE;
    *(unsigned long long *)data = (uintptr_t)base;
    break;
  case CU_POINTER_ATTRIBUTE_IS_MANAGED:
  case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
  case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
  case CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE:
    *(int *)data = 0;
    break;
  case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
    if (!known) return CUDA_ERROR_INVALID_VALUE;
    *(int *)data = d;
    break;
  case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
  case CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR:
    if (!known) return CUDA_ERROR_INVALID_VALUE;
    *(void **)data = base;
    break;
  case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
  case CU_POINTER_ATTRIBUTE_MAPPING_SIZE:
    if (!known) return CUDA_ERROR_INVALID_VALUE;
    *(size_t *)data = size;
    break;
  case CU_POINTER_ATTRIBUTE_MAPPED: *(int *)data = known; break;
  case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES: *(unsigned long long *)data = 0; break;
  case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS:
    *(int *)data = known ? CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE : 0;
    break;
  case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE: *(void **)data = NULL; break;
  default:
    fprintf(stderr, "[fakecuda] cuPointerGetAttribute: attribute %d not emulated\n", a);
    return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}

CUresult cuPointerGetAttribute(void *data, CUpointer_attribute a, CUdeviceptr p) {
  CUresult r = data ? pointer_attr(data, a, p) : CUDA_ERROR_INVALID_VALUE;
  FAKE_LOG("cuPointerGetAttribute(%p, attr=%d) = %d", P(p), a, r);
  return r;
}

CUresult cuPointerGetAttributes(unsigned n, CUpointer_attribute *attrs, void **data, CUdeviceptr p) {
  if (!attrs || !data) return CUDA_ERROR_INVALID_VALUE;
  for (unsigned i = 0; i < n; i++) {
    CUresult r = pointer_attr(data[i], attrs[i], p);
    if (r == CUDA_ERROR_INVALID_VALUE && attrs[i] == CU_POINTER_ATTRIBUTE_MEMORY_TYPE)
      *(CUmemorytype *)data[i] = 0;
    else if (r == CUDA_ERROR_INVALID_VALUE && attrs[i] == CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL)
      *(int *)data[i] = -2;
    else if (r == CUDA_ERROR_INVALID_VALUE)
      memset(data[i], 0, attrs[i] == CU_POINTER_ATTRIBUTE_SYNC_MEMOPS ? sizeof(int) : sizeof(void *));
  }
  FAKE_LOG("cuPointerGetAttributes(%p, n=%u) = 0", P(p), n);
  return CUDA_SUCCESS;
}

static int wait_holds(const void *addr, uint64_t v, unsigned flags, int is64) {
  uint64_t x = is64 ? *(const uint64_t *)addr : *(const uint32_t *)addr;
  if (!is64) v = (uint32_t)v;
  switch (flags & 3) {
  case CU_STREAM_WAIT_VALUE_GEQ: return is64 ? (int64_t)(x - v) >= 0 : (int32_t)(x - v) >= 0;
  case CU_STREAM_WAIT_VALUE_EQ: return x == v;
  case CU_STREAM_WAIT_VALUE_AND: return (x & v) != 0;
  default: return ~(x | ~v) != 0;
  }
}

static CUresult wait_value(const char *api, CUdeviceptr a, uint64_t v, unsigned flags, int is64) {
  if (!a) return CUDA_ERROR_INVALID_VALUE;
  if (!wait_holds(P(a), v, flags, is64))
    fprintf(stderr, "[fakecuda] %s(%p, %#llx, flags=%#x): condition not met and nothing else will "
            "write it; continuing\n", api, P(a), (unsigned long long)v, flags);
  return CUDA_SUCCESS;
}

CUresult cuStreamWaitValue32(CUstream s, CUdeviceptr a, cuuint32_t v, unsigned flags) {
  (void)s;
  return wait_value("cuStreamWaitValue32", a, v, flags, 0);
}

CUresult cuStreamBatchMemOp(CUstream s, unsigned n, CUstreamBatchMemOpParams *ops, unsigned flags) {
  (void)flags;
  for (unsigned i = 0; i < n; i++) {
    CUstreamBatchMemOpParams *o = &ops[i];
    CUresult r = CUDA_SUCCESS;
    switch (o->operation) {
    case CU_STREAM_MEM_OP_WAIT_VALUE_32:
      r = cuStreamWaitValue32(s, o->waitValue.address, o->waitValue.value, o->waitValue.flags);
      break;
    case CU_STREAM_MEM_OP_WAIT_VALUE_64:
      r = wait_value("cuStreamBatchMemOp", o->waitValue.address, o->waitValue.value64,
                     o->waitValue.flags, 1);
      break;
    case CU_STREAM_MEM_OP_WRITE_VALUE_32:
      if (!o->writeValue.address) return CUDA_ERROR_INVALID_VALUE;
      *(volatile uint32_t *)P(o->writeValue.address) = o->writeValue.value;
      break;
    case CU_STREAM_MEM_OP_WRITE_VALUE_64:
      if (!o->writeValue.address) return CUDA_ERROR_INVALID_VALUE;
      *(volatile uint64_t *)P(o->writeValue.address) = o->writeValue.value64;
      break;
    case CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES:
    case CU_STREAM_MEM_OP_BARRIER: break;
    default: r = CUDA_ERROR_NOT_SUPPORTED; break;
    }
    if (r) return r;
  }
  return CUDA_SUCCESS;
}
