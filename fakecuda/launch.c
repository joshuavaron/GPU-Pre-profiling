#include "fakecuda.h"
#include <cuda.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct reg {
  const void *host;
  const char *name;
  char *pretty;
  void *var;
  size_t size;
};

static struct reg *regs;
static size_t reg_cap, reg_count;
static pthread_mutex_t reg_lock = PTHREAD_MUTEX_INITIALIZER;

static size_t slot(const void *p, size_t cap) { return ((size_t)p >> 4) * 0x9E3779B97F4A7C15ull & (cap - 1); }

static struct reg *reg_find_locked(const void *host) {
  if (!reg_cap) return NULL;
  for (size_t i = slot(host, reg_cap);; i = (i + 1) & (reg_cap - 1)) {
    if (regs[i].host == host) return &regs[i];
    if (!regs[i].host) return NULL;
  }
}

static void reg_add(struct reg r) {
  pthread_mutex_lock(&reg_lock);
  if (2 * (reg_count + 1) > reg_cap) {
    size_t old_cap = reg_cap;
    struct reg *old = regs;
    reg_cap = reg_cap ? 2 * reg_cap : 1 << 14;
    regs = calloc(reg_cap, sizeof *regs);
    for (size_t i = 0; i < old_cap; i++) {
      if (!old[i].host) continue;
      size_t j = slot(old[i].host, reg_cap);
      while (regs[j].host) j = (j + 1) & (reg_cap - 1);
      regs[j] = old[i];
    }
    free(old);
  }
  struct reg *e = reg_find_locked(r.host);
  if (!e) {
    size_t j = slot(r.host, reg_cap);
    while (regs[j].host) j = (j + 1) & (reg_cap - 1);
    e = &regs[j];
    reg_count++;
  }
  *e = r;
  pthread_mutex_unlock(&reg_lock);
}

static struct reg *reg_find(const void *host) {
  pthread_mutex_lock(&reg_lock);
  struct reg *e = reg_find_locked(host);
  pthread_mutex_unlock(&reg_lock);
  return e;
}

static const char *pretty_name(struct reg *e) {
  if (!e) return "<unregistered>";
  if (!e->pretty) {
    static char *(*demangle)(const char *, char *, size_t *, int *);
    if (!demangle) {
      void *h = dlopen("libstdc++.so.6", RTLD_LAZY | RTLD_NOLOAD);
      if (!h) h = dlopen("libstdc++.so.6", RTLD_LAZY);
      if (h) demangle = (char *(*)(const char *, char *, size_t *, int *))dlsym(h, "__cxa_demangle");
    }
    int status = -1;
    char *d = demangle ? demangle(e->name, NULL, NULL, &status) : NULL;
    e->pretty = status == 0 && d ? d : strdup(e->name);
    if (strlen(e->pretty) > 160) strcpy(e->pretty + 157, "...");
  }
  return e->pretty;
}

void **__cudaRegisterFatBinary(void *fatCubin) {
  void **handle = malloc(sizeof *handle);
  *handle = fatCubin;
  return handle;
}

void __cudaRegisterFatBinaryEnd(void **handle) { (void)handle; }

void __cudaUnregisterFatBinary(void **handle) { free(handle); }

char __cudaInitModule(void **handle) {
  (void)handle;
  return 1;
}

void __cudaRegisterFunction(void **handle, const char *hostFun, char *deviceFun,
                            const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
  (void)handle, (void)deviceFun, (void)thread_limit, (void)tid, (void)bid, (void)bDim, (void)gDim,
      (void)wSize;
  reg_add((struct reg){.host = hostFun, .name = deviceName});
}

void __cudaRegisterVar(void **handle, char *hostVar, char *deviceAddress, const char *deviceName,
                       int ext, size_t size, int constant, int global) {
  (void)handle, (void)deviceAddress, (void)ext, (void)constant, (void)global;
  reg_add((struct reg){.host = hostVar, .name = deviceName, .var = calloc(1, size ? size : 1),
                       .size = size});
}

void __cudaRegisterManagedVar(void **handle, void **hostVarPtrAddress, char *deviceAddress,
                              const char *deviceName, int ext, size_t size, int constant,
                              int global) {
  (void)handle, (void)deviceAddress, (void)ext, (void)constant, (void)global;
  void *mem = calloc(1, size ? size : 1);
  *hostVarPtrAddress = mem;
  reg_add((struct reg){.host = hostVarPtrAddress, .name = deviceName, .var = mem, .size = size});
}

void __cudaRegisterHostVar(void) {}
void __cudaRegisterUnifiedTable(void) {}

struct call_config {
  dim3 grid, block;
  size_t shmem;
  cudaStream_t stream;
};
static __thread struct call_config config_stack[8];
static __thread int config_depth;

unsigned __cudaPushCallConfiguration(dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
  if (config_depth == 8) return 1;
  config_stack[config_depth++] = (struct call_config){grid, block, shmem, stream};
  return 0;
}

cudaError_t __cudaPopCallConfiguration(dim3 *grid, dim3 *block, size_t *shmem, void *stream) {
  if (config_depth == 0) return cudaErrorMissingConfiguration;
  struct call_config c = config_stack[--config_depth];
  *grid = c.grid;
  *block = c.block;
  *shmem = c.shmem;
  *(cudaStream_t *)stream = c.stream;
  return cudaSuccess;
}

/* RANDOM RESULT: no kernel is ever executed.  A launch is only validated and logged, so whatever
 * the kernel would have written to device memory is left as it was (garbage/zeros). */
int fake_launch(const char *api, const char *name, dim3 g, dim3 b, size_t shmem,
                size_t max_dyn_shmem, const void *stream) {
  const struct cudaDeviceProp *p = &fake_devices[fake_current_device];
  int ok = (unsigned long long)b.x * b.y * b.z <= (unsigned)p->maxThreadsPerBlock && b.x &&
           b.y && b.z && b.x <= (unsigned)p->maxThreadsDim[0] &&
           b.y <= (unsigned)p->maxThreadsDim[1] && b.z <= (unsigned)p->maxThreadsDim[2] && g.x &&
           g.y && g.z && g.x <= (unsigned)p->maxGridSize[0] && g.y <= (unsigned)p->maxGridSize[1] &&
           g.z <= (unsigned)p->maxGridSize[2] && shmem <= p->sharedMemPerBlockOptin &&
           shmem <= max_dyn_shmem;
  if (fake_log || !ok)
    fprintf(stderr, "[fakecuda] %s %s grid=(%u,%u,%u) block=(%u,%u,%u) shmem=%zu stream=%p = %s\n",
            api, name, g.x, g.y, g.z, b.x, b.y, b.z, shmem, stream, ok ? "ok (not executed)" : "invalid config");
  if (ok) {
    fake_ctx_touch(fake_current_device);

    struct fake_event e;
    memset(&e, 0, sizeof e);
    e.type = FAKE_EVENT_LAUNCH;
    e.api = api;
    e.device = fake_current_device;
    e.kernel = name;
    e.grid.x = g.x;
    e.grid.y = g.y;
    e.grid.z = g.z;
    e.block.x = b.x;
    e.block.y = b.y;
    e.block.z = b.z;
    fire_callback(&e);
  }
  return ok;
}

/* RANDOM RESULT: the kernel is not executed (see fake_launch); its outputs are left as-is. */
static cudaError_t launch(const char *api, const void *func, dim3 g, dim3 b, size_t shmem,
                          cudaStream_t stream) {
  struct reg *e = reg_find(func);
  if (!e || e->var) {
    fprintf(stderr, "[fakecuda] %s %s: not a registered kernel\n", api, pretty_name(e));
    return cudaErrorInvalidDeviceFunction;
  }
  return fake_launch(api, pretty_name(e), g, b, shmem, fake_devices[fake_current_device].sharedMemPerBlockOptin,
                     stream) ? cudaSuccess : cudaErrorInvalidConfiguration;
}

/* RANDOM RESULT: the kernel is not executed (see fake_launch); its outputs are left as-is. */
cudaError_t cudaLaunchKernel(const void *func, dim3 grid, dim3 block, void **args, size_t shmem,
                             cudaStream_t stream) {
  (void)args;
  return launch("cudaLaunchKernel", func, grid, block, shmem, stream);
}

/* RANDOM RESULT: the kernel is not executed (see fake_launch); its outputs are left as-is. */
cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 grid, dim3 block, void **args,
                                        size_t shmem, cudaStream_t stream) {
  (void)args;
  return launch("cudaLaunchCooperativeKernel", func, grid, block, shmem, stream);
}

/* RANDOM RESULT: the kernel is not executed (see fake_launch); its outputs are left as-is. */
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args) {
  (void)args;
  return launch("cudaLaunchKernelExC", func, config->gridDim, config->blockDim,
                config->dynamicSmemBytes, config->stream);
}

cudaError_t __cudaGetKernel(cudaKernel_t *kernel, const void *func) {
  *kernel = (cudaKernel_t)func;
  return reg_find(func) ? cudaSuccess : cudaErrorInvalidDeviceFunction;
}

/* RANDOM RESULT: the kernel is not executed (see fake_launch); its outputs are left as-is. */
cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block, void **args,
                               size_t shmem, cudaStream_t stream) {
  (void)args;
  return launch("launch", (const void *)kernel, grid, block, shmem, stream);
}

/* RANDOM RESULT: the kernel is not executed (see fake_launch); its outputs are left as-is. */
cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 grid, dim3 block, void **args,
                                    size_t shmem, cudaStream_t stream) {
  return __cudaLaunchKernel(kernel, grid, block, args, shmem, stream);
}

/* RANDOM RESULT: kernel attributes are placeholders, not read from the kernel binary. */
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {
  struct reg *e = reg_find(func);
  if (!e || e->var) return cudaErrorInvalidDeviceFunction;
  memset(attr, 0, sizeof *attr);
  attr->maxThreadsPerBlock = fake_devices[fake_current_device].maxThreadsPerBlock;
  attr->numRegs = 32;
  attr->ptxVersion = 90;
  attr->binaryVersion = 90;
  attr->maxDynamicSharedSizeBytes = 48 << 10;
  attr->preferredShmemCarveout = -1;
  FAKE_LOG("cudaFuncGetAttributes(%s) = 0", pretty_name(e));
  return cudaSuccess;
}

cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value) {
  FAKE_LOG("cudaFuncSetAttribute(%p, attr=%d, %d) = 0", func, attr, value);
  return reg_find(func) ? cudaSuccess : cudaErrorInvalidDeviceFunction;
}

cudaError_t cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache config) {
  FAKE_LOG("cudaFuncSetCacheConfig(%p, %d) = 0", func, config);
  return reg_find(func) ? cudaSuccess : cudaErrorInvalidDeviceFunction;
}

/* RANDOM RESULT: occupancy from a simple thread/shared-memory model, ignoring real register use. */
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func,
                                                                  int blockSize, size_t shmem,
                                                                  unsigned int flags) {
  (void)flags;
  if (!reg_find(func)) return cudaErrorInvalidDeviceFunction;
  const struct cudaDeviceProp *p = &fake_devices[fake_current_device];
  if (blockSize <= 0 || blockSize > p->maxThreadsPerBlock) return cudaErrorInvalidValue;
  int warps = (blockSize + p->warpSize - 1) / p->warpSize;
  int n = p->maxThreadsPerMultiProcessor / (warps * p->warpSize);
  if (n > p->maxBlocksPerMultiProcessor) n = p->maxBlocksPerMultiProcessor;
  if (shmem && (int)(p->sharedMemPerMultiprocessor / shmem) < n)
    n = (int)(p->sharedMemPerMultiprocessor / shmem);
  *numBlocks = n;
  FAKE_LOG("cudaOccupancyMaxActiveBlocksPerMultiprocessor(block=%d, shmem=%zu) = 0  %d", blockSize,
           shmem, n);
  return cudaSuccess;
}

/* RANDOM RESULT: see cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags. */
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func,
                                                          int blockSize, size_t shmem) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, shmem, 0);
}

static struct reg *find_var(const void *symbol) {
  struct reg *e = reg_find(symbol);
  return e && e->var ? e : NULL;
}

cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
  struct reg *e = find_var(symbol);
  if (!e) return cudaErrorInvalidSymbol;
  *devPtr = e->var;
  return cudaSuccess;
}

cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol) {
  struct reg *e = find_var(symbol);
  if (!e) return cudaErrorInvalidSymbol;
  *size = e->size;
  return cudaSuccess;
}

cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset,
                               enum cudaMemcpyKind kind) {
  (void)kind;
  struct reg *e = find_var(symbol);
  if (!e) return cudaErrorInvalidSymbol;
  if (offset + count > e->size) return cudaErrorInvalidValue;
  memmove((char *)e->var + offset, src, count);
  FAKE_LOG("cudaMemcpyToSymbol(%s, %zu, offset=%zu) = 0", e->name, count, offset);
  return cudaSuccess;
}

cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset,
                                 enum cudaMemcpyKind kind) {
  (void)kind;
  struct reg *e = find_var(symbol);
  if (!e) return cudaErrorInvalidSymbol;
  if (offset + count > e->size) return cudaErrorInvalidValue;
  memmove(dst, (char *)e->var + offset, count);
  return cudaSuccess;
}

cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset,
                                    enum cudaMemcpyKind kind, cudaStream_t stream) {
  (void)stream;
  return cudaMemcpyToSymbol(symbol, src, count, offset, kind);
}

cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset,
                                      enum cudaMemcpyKind kind, cudaStream_t stream) {
  (void)stream;
  return cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
}

#define MOD_MAGIC 0x6d6f6466u
#define FUNC_MAGIC 0x636e7566u

struct CUfunc_st {
  unsigned magic;
  struct CUmod_st *mod;
  struct CUfunc_st *next;
  char *name;
  int max_dyn_shmem, carveout, cache_config, non_portable;
};

struct CUmod_st {
  unsigned magic;
  const void *image;
  struct CUfunc_st *funcs;
  struct global *globals;
};

struct global {
  struct global *next;
  char *name;
  void *ptr;
  size_t size;
};

static pthread_mutex_t mod_lock = PTHREAD_MUTEX_INITIALIZER;

static struct CUmod_st *new_module(const void *image) {
  struct CUmod_st *m = calloc(1, sizeof *m);
  m->magic = MOD_MAGIC;
  m->image = image;
  return m;
}

static struct CUmod_st *as_mod(const void *h) {
  const struct CUmod_st *m = h;
  return m && m->magic == MOD_MAGIC ? (struct CUmod_st *)m : NULL;
}

static struct CUfunc_st *as_func(const void *h) {
  const struct CUfunc_st *f = h;
  return f && f->magic == FUNC_MAGIC ? (struct CUfunc_st *)f : NULL;
}

static void free_module(struct CUmod_st *m) {
  for (struct CUfunc_st *f = m->funcs, *n; f; f = n) {
    n = f->next;
    f->magic = 0;
    free(f->name);
    free(f);
  }
  for (struct global *g = m->globals, *n; g; g = n) {
    n = g->next;
    fake_mem_free(g->ptr);
    free(g->name);
    free(g);
  }
  m->magic = 0;
  free(m);
}

static struct CUfunc_st *get_func(struct CUmod_st *m, const char *name) {
  pthread_mutex_lock(&mod_lock);
  struct CUfunc_st *f = m->funcs;
  while (f && strcmp(f->name, name) != 0) f = f->next;
  if (!f) {
    f = calloc(1, sizeof *f);
    *f = (struct CUfunc_st){FUNC_MAGIC, m, m->funcs, strdup(name), 48 << 10, -1, 0, 0};
    m->funcs = f;
  }
  pthread_mutex_unlock(&mod_lock);
  return f;
}


CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode *mode) {
  *mode = CU_MODULE_EAGER_LOADING;
  FAKE_LOG("cuModuleGetLoadingMode() = 0  eager");
  return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
  if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
  *module = (CUmodule)new_module(image);
  FAKE_LOG("cuModuleLoadData(%p) = 0  %p", image, (void *)*module);
  return CUDA_SUCCESS;
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned n, CUjit_option *o,
                            void **v) {
  (void)n, (void)o, (void)v;
  return cuModuleLoadData(module, image);
}

CUresult cuModuleUnload(CUmodule module) {
  struct CUmod_st *m = as_mod(module);
  FAKE_LOG("cuModuleUnload(%p) = %d", (void *)module, m ? 0 : CUDA_ERROR_INVALID_HANDLE);
  if (!m) return CUDA_ERROR_INVALID_HANDLE;
  free_module(m);
  return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction *func, CUmodule module, const char *name) {
  struct CUmod_st *m = as_mod(module);
  if (!m || !func || !name) return !m ? CUDA_ERROR_INVALID_HANDLE : CUDA_ERROR_INVALID_VALUE;
  *func = (CUfunction)get_func(m, name);
  FAKE_LOG("cuModuleGetFunction(%p, %s) = 0  %p", (void *)module, name, (void *)*func);
  return CUDA_SUCCESS;
}

/* RANDOM RESULT: the global's initial contents are zeros, not the image's initializer. */
CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule module, const char *name) {
  struct CUmod_st *m = as_mod(module);
  if (!m || !name) return !m ? CUDA_ERROR_INVALID_HANDLE : CUDA_ERROR_INVALID_VALUE;
  pthread_mutex_lock(&mod_lock);
  struct global *g = m->globals;
  while (g && strcmp(g->name, name) != 0) g = g->next;
  if (!g && (g = calloc(1, sizeof *g))) {
    g->name = strdup(name);
    g->size = 4096;
    fake_mem_alloc(fake_current_device, g->size, &g->ptr);
    g->next = m->globals;
    m->globals = g;
  }
  pthread_mutex_unlock(&mod_lock);
  if (dptr) *dptr = (CUdeviceptr)g->ptr;
  if (bytes) *bytes = g->size;
  FAKE_LOG("cuModuleGetGlobal(%p, %s) = 0  %p (size unknown, gave %zu)", (void *)module, name,
           g->ptr, g->size);
  return CUDA_SUCCESS;
}


CUresult cuLibraryLoadData(CUlibrary *library, const void *code, CUjit_option *jo, void **jv,
                           unsigned njo, CUlibraryOption *lo, void **lv, unsigned nlo) {
  (void)jo, (void)jv, (void)njo, (void)lo, (void)lv, (void)nlo;
  if (!library || !code) return CUDA_ERROR_INVALID_VALUE;
  *library = (CUlibrary)new_module(code);
  FAKE_LOG("cuLibraryLoadData(%p) = 0  %p", code, (void *)*library);
  return CUDA_SUCCESS;
}

CUresult cuLibraryUnload(CUlibrary library) { return cuModuleUnload((CUmodule)library); }

CUresult cuLibraryGetModule(CUmodule *module, CUlibrary library) {
  if (!as_mod(library)) return CUDA_ERROR_INVALID_HANDLE;
  *module = (CUmodule)library;
  return CUDA_SUCCESS;
}

CUresult cuLibraryGetKernel(CUkernel *kernel, CUlibrary library, const char *name) {
  return cuModuleGetFunction((CUfunction *)kernel, (CUmodule)library, name);
}

CUresult cuKernelGetFunction(CUfunction *func, CUkernel kernel) {
  if (!as_func(kernel)) return CUDA_ERROR_INVALID_HANDLE;
  *func = (CUfunction)kernel;
  return CUDA_SUCCESS;
}


/* RANDOM RESULT: register count, local/const/static shared sizes and versions are placeholders;
 * the kernel binary is never inspected. */
CUresult cuFuncGetAttribute(int *value, CUfunction_attribute attr, CUfunction func) {
  struct CUfunc_st *f = as_func(func);
  if (!f || !value) return !f ? CUDA_ERROR_INVALID_HANDLE : CUDA_ERROR_INVALID_VALUE;
  switch (attr) {
  case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
    *value = fake_devices[fake_current_device].maxThreadsPerBlock;
    break;
  case CU_FUNC_ATTRIBUTE_NUM_REGS: *value = 32; break;
  case CU_FUNC_ATTRIBUTE_PTX_VERSION:
  case CU_FUNC_ATTRIBUTE_BINARY_VERSION: *value = 90; break;
  case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: *value = f->max_dyn_shmem; break;
  case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: *value = f->carveout; break;
  case CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED: *value = f->non_portable; break;
  default: *value = 0; break;
  }
  FAKE_LOG("cuFuncGetAttribute(%s, attr=%d) = 0  %d", f->name, attr, *value);
  return CUDA_SUCCESS;
}

CUresult cuFuncSetAttribute(CUfunction func, CUfunction_attribute attr, int value) {
  struct CUfunc_st *f = as_func(func);
  if (!f) return CUDA_ERROR_INVALID_HANDLE;
  CUresult r = CUDA_SUCCESS;
  if (attr == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) {
    if (value < 0 || (size_t)value > fake_devices[fake_current_device].sharedMemPerBlockOptin)
      r = CUDA_ERROR_INVALID_VALUE;
    else
      f->max_dyn_shmem = value;
  } else if (attr == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT) {
    f->carveout = value;
  } else if (attr == CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED) {
    f->non_portable = value;
  }
  FAKE_LOG("cuFuncSetAttribute(%s, attr=%d, %d) = %d", f->name, attr, value, r);
  return r;
}

CUresult cuFuncSetCacheConfig(CUfunction func, CUfunc_cache config) {
  struct CUfunc_st *f = as_func(func);
  if (!f) return CUDA_ERROR_INVALID_HANDLE;
  f->cache_config = config;
  return CUDA_SUCCESS;
}

/* RANDOM RESULT: placeholder attributes, see cuFuncGetAttribute. */
CUresult cuKernelGetAttribute(int *value, CUfunction_attribute attr, CUkernel kernel, CUdevice dev) {
  if (!fake_valid(dev)) return CUDA_ERROR_INVALID_DEVICE;
  return cuFuncGetAttribute(value, attr, (CUfunction)kernel);
}

CUresult cuKernelSetAttribute(CUfunction_attribute attr, int value, CUkernel kernel, CUdevice dev) {
  if (!fake_valid(dev)) return CUDA_ERROR_INVALID_DEVICE;
  return cuFuncSetAttribute((CUfunction)kernel, attr, value);
}

/* RANDOM RESULT: occupancy from a simple thread/shared-memory model, ignoring real register use. */
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *num, CUfunction func, int block,
                                                              size_t shmem, unsigned flags) {
  (void)flags;
  const struct cudaDeviceProp *p = &fake_devices[fake_current_device];
  if (!as_func(func)) return CUDA_ERROR_INVALID_HANDLE;
  if (block <= 0 || block > p->maxThreadsPerBlock) return CUDA_ERROR_INVALID_VALUE;
  int warps = (block + p->warpSize - 1) / p->warpSize;
  int n = p->maxThreadsPerMultiProcessor / (warps * p->warpSize);
  if (n > p->maxBlocksPerMultiProcessor) n = p->maxBlocksPerMultiProcessor;
  if (shmem && (int)(p->sharedMemPerMultiprocessor / shmem) < n)
    n = (int)(p->sharedMemPerMultiprocessor / shmem);
  *num = n;
  return CUDA_SUCCESS;
}

/* RANDOM RESULT: assumes the kernel uses no static shared memory (its binary isn't read). */
CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *size, CUfunction func, int blocks,
                                                 int block) {
  const struct cudaDeviceProp *p = &fake_devices[fake_current_device];
  if (!as_func(func)) return CUDA_ERROR_INVALID_HANDLE;
  if (!size || blocks <= 0 || block <= 0 || block > p->maxThreadsPerBlock)
    return CUDA_ERROR_INVALID_VALUE;
  size_t per_block = p->sharedMemPerMultiprocessor / blocks;
  per_block = per_block > p->reservedSharedMemPerBlock ? per_block - p->reservedSharedMemPerBlock : 0;
  *size = per_block < p->sharedMemPerBlockOptin ? per_block : p->sharedMemPerBlockOptin;
  FAKE_LOG("cuOccupancyAvailableDynamicSMemPerBlock(blocks=%d, block=%d) = 0  %zu", blocks, block,
           *size);
  return CUDA_SUCCESS;
}

/* RANDOM RESULT: see cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags. */
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *num, CUfunction func, int block,
                                                     size_t shmem) {
  return cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(num, func, block, shmem, 0);
}

static unsigned cluster_size(const CUlaunchConfig *c) {
  for (unsigned i = 0; c && i < c->numAttrs; i++)
    if (c->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
      const CUlaunchAttributeValue *v = &c->attrs[i].value;
      return v->clusterDim.x * v->clusterDim.y * v->clusterDim.z;
    }
  return 1;
}

static const int gpc_sms[] = {18, 18, 16, 16, 16, 16, 16, 16};

/* RANDOM RESULT: cluster occupancy from the simple per-SM model above and the made-up GPC layout,
 * not from the kernel or a real GPU. */
CUresult cuOccupancyMaxActiveClusters(int *num, CUfunction func, const CUlaunchConfig *c) {
  struct CUfunc_st *f = as_func(func);
  if (!f) return CUDA_ERROR_INVALID_HANDLE;
  if (!num || !c) return CUDA_ERROR_INVALID_VALUE;
  unsigned cs = cluster_size(c);
  if (cs == 0 || cs > (f->non_portable ? 16u : 8u)) return CUDA_ERROR_INVALID_CLUSTER_SIZE;
  int per_sm;
  CUresult r = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &per_sm, func, (int)(c->blockDimX * c->blockDimY * c->blockDimZ), c->sharedMemBytes, 0);
  if (r) return r;
  *num = 0;
  if (c->sharedMemBytes <= (unsigned)f->max_dyn_shmem)
    for (size_t g = 0; g < sizeof gpc_sms / sizeof *gpc_sms; g++)
      *num += gpc_sms[g] * per_sm / (int)cs;
  FAKE_LOG("cuOccupancyMaxActiveClusters(%s, cluster=%u, block=%u, shmem=%u) = 0  %d", f->name, cs,
           c->blockDimX * c->blockDimY * c->blockDimZ, c->sharedMemBytes, *num);
  return CUDA_SUCCESS;
}


/* RANDOM RESULT: see fake_launch(); the kernel is validated and logged but never executed, so its
 * outputs keep whatever bytes were in device memory. */
static CUresult drv_launch(const char *api, CUfunction func, dim3 g, dim3 b, unsigned shmem,
                       CUstream stream) {
  struct CUfunc_st *f = as_func(func);
  if (!f) {
    fprintf(stderr, "[fakecuda] %s %p: not a fakecuda function handle\n", api, (void *)func);
    return CUDA_ERROR_INVALID_HANDLE;
  }
  return fake_launch(api, f->name, g, b, shmem, f->max_dyn_shmem, stream)
             ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

/* RANDOM RESULT: the kernel is not executed (see fake_launch); its outputs are left as-is. */
CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                        unsigned by, unsigned bz, unsigned shmem, CUstream stream, void **params,
                        void **extra) {
  (void)params, (void)extra;
  return drv_launch("cuLaunchKernel", f, (dim3){gx, gy, gz}, (dim3){bx, by, bz}, shmem, stream);
}

/* RANDOM RESULT: the kernel is not executed (see fake_launch); its outputs are left as-is. */
CUresult cuLaunchKernelEx(const CUlaunchConfig *c, CUfunction f, void **params, void **extra) {
  (void)params, (void)extra;
  if (!c) return CUDA_ERROR_INVALID_VALUE;
  return drv_launch("cuLaunchKernelEx", f, (dim3){c->gridDimX, c->gridDimY, c->gridDimZ},
                (dim3){c->blockDimX, c->blockDimY, c->blockDimZ}, c->sharedMemBytes, c->hStream);
}
