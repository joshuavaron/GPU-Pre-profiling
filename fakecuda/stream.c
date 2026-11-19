#include "fakecuda.h"
#include <stdlib.h>
#include <time.h>

struct CUstream_st {
  int device, priority;
  unsigned flags;
  unsigned long long id;
};

static int is_default(cudaStream_t s) { return (size_t)s <= 2; }

static struct CUstream_st default_streams[FAKECUDA_MAX_DEVICES][2];
static unsigned long long next_stream_id = 2 * FAKECUDA_MAX_DEVICES + 1;

void *fake_stream_object(int dev, cudaStream_t s, int per_thread) {
  if (!is_default(s)) return s;
  if (!fake_valid(dev)) return NULL;
  int pt = s == cudaStreamPerThread || (s == 0 && per_thread);
  struct CUstream_st *d = &default_streams[dev][pt];
  if (!d->id) *d = (struct CUstream_st){dev, 0, 0, (unsigned long long)(2 * dev + pt + 1)};
  return d;
}

unsigned long long fake_stream_id(void *object) { return ((struct CUstream_st *)object)->id; }

struct CUevent_st {
  unsigned flags;
  int recorded;
  double ms;
};


static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t *stream, unsigned int flags, int priority) {
  struct CUstream_st *s = malloc(sizeof *s);
  *s = (struct CUstream_st){fake_current_device, priority, flags,
                            __atomic_fetch_add(&next_stream_id, 1, __ATOMIC_RELAXED)};
  fake_ctx_touch(fake_current_device);
  *stream = s;
  FAKE_LOG("cudaStreamCreate(flags=%u, priority=%d) = 0  %p", flags, priority, (void *)s);
  return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags) {
  return cudaStreamCreateWithPriority(stream, flags, 0);
}

cudaError_t cudaStreamCreate(cudaStream_t *stream) {
  return cudaStreamCreateWithPriority(stream, cudaStreamDefault, 0);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  if (is_default(stream)) return cudaErrorInvalidResourceHandle;
  FAKE_LOG("cudaStreamDestroy(%p) = 0", (void *)stream);
  free(stream);
  return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
  FAKE_LOG("cudaStreamQuery(%p) = 0  idle", (void *)stream);
  return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
  FAKE_LOG("cudaStreamWaitEvent(%p, %p, %u) = 0", (void *)stream, (void *)event, flags);
  return cudaSuccess;
}

cudaError_t cudaStreamGetPriority(cudaStream_t stream, int *priority) {
  *priority = is_default(stream) ? 0 : stream->priority;
  return cudaSuccess;
}

cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags) {
  *flags = is_default(stream) ? 0 : stream->flags;
  return cudaSuccess;
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData) {
  FAKE_LOG("cudaLaunchHostFunc(%p) = 0  running now", (void *)stream);
  fn(userData);
  return cudaSuccess;
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback,
                                  void *userData, unsigned int flags) {
  (void)flags;
  FAKE_LOG("cudaStreamAddCallback(%p) = 0  running now", (void *)stream);
  callback(stream, cudaSuccess, userData);
  return cudaSuccess;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *status) {
  *status = cudaStreamCaptureStatusNone;
  FAKE_LOG("cudaStreamIsCapturing(%p) = 0  none", (void *)stream);
  return cudaSuccess;
}

cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus *status,
                                        unsigned long long *id, cudaGraph_t *graph,
                                        const cudaGraphNode_t **deps, size_t *numDeps) {
  *status = cudaStreamCaptureStatusNone;
  if (id) *id = 0;
  if (graph) *graph = NULL;
  if (deps) *deps = NULL;
  if (numDeps) *numDeps = 0;
  FAKE_LOG("cudaStreamGetCaptureInfo(%p) = 0  none", (void *)stream);
  return cudaSuccess;
}

cudaError_t cudaStreamGetCaptureInfo_v3(cudaStream_t stream, enum cudaStreamCaptureStatus *status,
                                        unsigned long long *id, cudaGraph_t *graph,
                                        const cudaGraphNode_t **deps,
                                        const cudaGraphEdgeData **edgeData, size_t *numDeps) {
  if (edgeData) *edgeData = NULL;
  return cudaStreamGetCaptureInfo_v2(stream, status, id, graph, deps, numDeps);
}

static __thread enum cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeGlobal;

cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode) {
  enum cudaStreamCaptureMode old = capture_mode;
  capture_mode = *mode;
  *mode = old;
  return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
  struct CUevent_st *e = calloc(1, sizeof *e);
  e->flags = flags;
  *event = e;
  FAKE_LOG("cudaEventCreate(flags=%u) = 0  %p", flags, (void *)e);
  return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t *event) {
  return cudaEventCreateWithFlags(event, cudaEventDefault);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
  if (!event) return cudaErrorInvalidResourceHandle;
  FAKE_LOG("cudaEventDestroy(%p) = 0", (void *)event);
  free(event);
  return cudaSuccess;
}

cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) {
  (void)flags;
  if (!event) return cudaErrorInvalidResourceHandle;
  event->recorded = 1;
  event->ms = now_ms();
  FAKE_LOG("cudaEventRecord(%p, stream=%p) = 0", (void *)event, (void *)stream);
  return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return cudaEventRecordWithFlags(event, stream, 0);
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
  return event ? cudaSuccess : cudaErrorInvalidResourceHandle;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return event ? cudaSuccess : cudaErrorInvalidResourceHandle;
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
  if (!start || !end) return cudaErrorInvalidResourceHandle;
  if ((start->flags | end->flags) & cudaEventDisableTiming) return cudaErrorInvalidResourceHandle;
  if (!start->recorded || !end->recorded) return cudaErrorInvalidResourceHandle;
  *ms = (float)(end->ms - start->ms);
  FAKE_LOG("cudaEventElapsedTime() = 0  %.3f ms", *ms);
  return cudaSuccess;
}


int cuStreamCreate(cudaStream_t *s, unsigned flags) { return cudaStreamCreateWithPriority(s, flags, 0); }
int cuStreamDestroy_v2(cudaStream_t s) { return cudaStreamDestroy(s); }
int cuStreamSynchronize(cudaStream_t s) {
  FAKE_LOG("cuStreamSynchronize(%p) = 0", (void *)s);
  return cudaSuccess;
}
int cuStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned flags) {
  return cudaStreamWaitEvent(s, e, flags);
}
int cuStreamGetCtx(cudaStream_t s, void **ctx) {
  *ctx = fake_primary_ctx(is_default(s) ? fake_current_device : s->device);
  return cudaSuccess;
}
int cuStreamIsCapturing(cudaStream_t s, enum cudaStreamCaptureStatus *status) {
  return cudaStreamIsCapturing(s, status);
}
int cuStreamGetCaptureInfo_v2(cudaStream_t s, enum cudaStreamCaptureStatus *status,
                              unsigned long long *id, cudaGraph_t *graph,
                              const cudaGraphNode_t **deps, size_t *n) {
  return cudaStreamGetCaptureInfo_v2(s, status, id, graph, deps, n);
}
int cuThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode) {
  return cudaThreadExchangeStreamCaptureMode(mode);
}

int cuEventCreate(cudaEvent_t *e, unsigned flags) { return cudaEventCreateWithFlags(e, flags); }
int cuEventDestroy_v2(cudaEvent_t e) { return cudaEventDestroy(e); }
int cuEventRecord(cudaEvent_t e, cudaStream_t s) { return cudaEventRecordWithFlags(e, s, 0); }
int cuEventQuery(cudaEvent_t e) { return cudaEventQuery(e); }
int cuEventSynchronize(cudaEvent_t e) { return cudaEventSynchronize(e); }
int cuEventElapsedTime(float *ms, cudaEvent_t a, cudaEvent_t b) { return cudaEventElapsedTime(ms, a, b); }
