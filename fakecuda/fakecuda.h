#pragma once
#include <cuda_runtime_api.h>
#include <stdio.h>

extern int fake_log;
#define FAKE_LOG(fmt, ...) \
  do { if (fake_log) fprintf(stderr, "[fakecuda] " fmt "\n", ##__VA_ARGS__); } while (0)

#define FAKECUDA_MAX_DEVICES 64
#define FAKECUDA_CUDA_VERSION 12060
#define FAKECUDA_DRIVER_VERSION "560.35.03"

extern struct cudaDeviceProp fake_devices[FAKECUDA_MAX_DEVICES];
extern int fake_device_count;

static inline int fake_valid(int dev) { return dev >= 0 && dev < fake_device_count; }

extern __thread int fake_current_device;

extern int fake_ctx_active[FAKECUDA_MAX_DEVICES];
static inline void fake_ctx_touch(int dev) { __atomic_store_n(&fake_ctx_active[dev], 1, __ATOMIC_RELAXED); }

int fake_device_attr(int attr, int dev);

void *fake_primary_ctx(int dev);
int fake_ctx_device(void *ctx);

int fake_get_proc_address(const char *symbol, void **pfn, int version,
                          unsigned long long flags, int *status);

void *fake_stub_for(const char *name);

enum { FAKE_MEM_OK, FAKE_MEM_OOM, FAKE_MEM_INVALID };
int fake_mem_alloc(int device, size_t size, void **out);
int fake_mem_free(void *ptr);
int fake_host_alloc(size_t size, void **out);
int fake_host_free(void *ptr);
size_t fake_mem_used(int device);
int fake_mem_lookup(const void *ptr, int *device);
int fake_mem_range(const void *ptr, int *device, void **base, size_t *size);

int fake_launch(const char *api, const char *name, dim3 grid, dim3 block, size_t shmem,
                size_t max_dyn_shmem, const void *stream);

void *fake_stream_object(int dev, cudaStream_t s, int per_thread);
unsigned long long fake_stream_id(void *object);
