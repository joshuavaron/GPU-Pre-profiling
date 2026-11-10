# memprofiler

Records GPU memory allocation events and reports per-device peak usage.
It is a standalone C library with no dependency on CUDA or fakecuda: the caller
(eventually fakecuda's `memory.c`) reports each allocation and free.

## Usage

```bash
make         # build/libmemprof.a
make test    # unit tests + JSON validity check of an example report
```

```c
#include "memprof.h"

memprof_t *m = memprof_create(num_devices);
memprof_alloc(m, device, ptr, size, MEMPROF_DEVICE);   // after a successful allocation
memprof_free(m, ptr);                                   // after a successful free
memprof_alloc_failed(m, device, size);                  // allocator returned OOM
memprof_mark_incomplete(m, "cudaMallocAsync");          // an untracked alloc path was used

memprof_write_json(m, stdout);
memprof_write_text(m, stderr);
memprof_destroy(m);
```

Link with `build/libmemprof.a -pthread`. All functions are thread-safe.

## What it tracks

- **Per device:** current and peak bytes of device memory, current and peak
  managed memory, alloc/free counts, and failed allocations (count and the
  largest failed request).
- **Combined peak:** the peak of device memory summed across devices. This is
  the peak of the sum, which can be less than the sum of per-device peaks.
- **Host pinned memory:** current and peak, kept separate from device memory.
- **Incomplete runs:** `memprof_mark_incomplete` records why some allocations
  weren't seen. The report then says peaks are lower bounds.
- **Anomalies** (counted and reported; the profiled job never aborts):
  - `unknown_frees`: a pointer that isn't live was freed, including double frees.
  - `invalid_args`: out-of-range device or unknown kind. The event is ignored.
  - `duplicate_allocs`: a pointer that is already live was allocated again.
    The old allocation is treated as freed and replaced by the new one.
  - `null_allocs`: `ptr == NULL` with `size > 0`. The event is ignored.

  `alloc(NULL, 0)` and `free(NULL)` are no-ops, matching CUDA.

The profiler keeps its own pointer → allocation map, so `free` takes only a
pointer, the same as `cudaFree`.

## Report formats

`memprof_write_text`:

```
memprof: 3 devices, combined peak 38.0 GiB  [INCOMPLETE: peaks are lower bounds]
  device         peak      current    allocs     frees  failed  largest failed
       0     26.0 GiB     20.0 GiB         2         1       0               -
       1     18.0 GiB     18.0 GiB         2         0       1       150.0 GiB
       2          0 B          0 B         0         0       0               -
  managed: device 1 peak 1.0 GiB, current 1.0 GiB
  host pinned: peak 2.0 GiB, current 2.0 GiB
  incomplete: cudaMallocAsync (x1)
  anomalies: 1 unknown frees, 0 invalid args, 0 duplicate allocs, 0 null allocs
```

`memprof_write_json` produces the same data with exact byte counts. The
top-level keys are `format_version`, `complete`, `num_devices`,
`peak_total_bytes`, `current_total_bytes`, `live_allocations`, `devices[]`,
`host_pinned`, `incomplete_reasons[]`, `incomplete_reasons_dropped` and
`anomalies`. Run `make test` and look at `build/example_report.json` for a
full example.
