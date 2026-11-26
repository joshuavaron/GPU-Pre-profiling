#pragma once
/*
 * Records allocation and free events and keeps per-device current and peak
 * usage. It does not depend on CUDA or fakecuda; the caller reports events.
 *
 * All functions are thread-safe. Passing a NULL profiler is a no-op (queries
 * return -1 or zeroed results). Bookkeeping errors in the event stream never
 * abort: they are counted and appear in the report as anomalies.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct memprof memprof_t;

typedef enum {
  MEMPROF_DEVICE = 0,      /* device memory (cudaMalloc, cuMemAlloc): counts toward peaks */
  MEMPROF_HOST_PINNED = 1, /* pinned host memory (cudaHostAlloc): tracked separately;
                              the device argument is ignored */
  MEMPROF_MANAGED = 2,     /* managed memory (cudaMallocManaged): tracked per device,
                              separately from device memory */
} memprof_memory_kind;

/* num_devices >= 0. Returns NULL on a negative count or out of memory. */
memprof_t *memprof_create(int num_devices);
void memprof_destroy(memprof_t *m);

/* ---- events ---- */

/* A successful allocation. (ptr == NULL, size == 0) is a no-op, like cudaMalloc(0).
 * Allocating a pointer that is already live counts as a duplicate_alloc anomaly;
 * the old allocation is treated as freed and replaced by the new one. */
void memprof_alloc(memprof_t *m, int device, const void *ptr, size_t size, memprof_memory_kind kind);

/* Frees the live allocation starting at ptr. free(NULL) is a no-op. */
void memprof_free(memprof_t *m, const void *ptr);

/* An allocation request the allocator rejected (for example, out of memory). */
void memprof_alloc_failed(memprof_t *m, int device, size_t size);

/* Marks the run as incomplete: some allocations were not reported (for
 * example, an unsupported allocation API was called), so peaks are lower
 * bounds. Repeated reasons are de-duplicated and counted. */
void memprof_mark_incomplete(memprof_t *m, const char *reason);

/* ---- queries ---- */

typedef struct {
  size_t current, peak;                  /* MEMPROF_DEVICE bytes */
  size_t managed_current, managed_peak;  /* MEMPROF_MANAGED bytes */
  uint64_t n_allocs, n_frees;            /* device + managed events on this device */
  uint64_t n_failed;
  size_t largest_failed;
} memprof_device_stats;

typedef struct {
  int num_devices;
  size_t current_total, peak_total;  /* MEMPROF_DEVICE bytes summed across devices; peak_total
                                        is the peak of the sum, not the sum of peaks */
  size_t host_pinned_current, host_pinned_peak;
  uint64_t live_allocs;              /* all kinds */

  /* anomalies */
  uint64_t unknown_frees;    /* free of a pointer that is not live */
  uint64_t invalid_args;     /* out-of-range device or unknown kind; the event is ignored */
  uint64_t duplicate_allocs; /* alloc of a pointer that is already live */
  uint64_t null_allocs;      /* alloc with ptr == NULL and size > 0; ignored */
  uint64_t internal_errors;  /* the profiler itself ran out of memory; marks the run incomplete */

  int incomplete;            /* nonzero: peaks are lower bounds */
} memprof_summary;

/* Returns 0, or -1 if the device index is out of range. */
int memprof_get_device_stats(const memprof_t *m, int device, memprof_device_stats *out);
void memprof_get_summary(const memprof_t *m, memprof_summary *out);

/* ---- reports ---- */

/* Each returns 0, or -1 on a write error. */
int memprof_write_json(const memprof_t *m, FILE *out);
int memprof_write_text(const memprof_t *m, FILE *out);

#ifdef __cplusplus
}
#endif
