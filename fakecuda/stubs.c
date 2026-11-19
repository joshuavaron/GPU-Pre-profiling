#include "fakecuda.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define FAKE_STUB_SLOTS 1024
#define CUDA_ERROR_NOT_SUPPORTED 801

static const char *stub_names[FAKE_STUB_SLOTS];
static int stub_called[FAKE_STUB_SLOTS];
static int stub_used;
static pthread_mutex_t stub_lock = PTHREAD_MUTEX_INITIALIZER;

static int stub_hit(int i) {
  if (!__atomic_exchange_n(&stub_called[i], 1, __ATOMIC_RELAXED))
    fprintf(stderr, "[fakecuda] MISSING (CALLED): %s\n", stub_names[i]);
  else
    FAKE_LOG("MISSING (CALLED): %s", stub_names[i]);
  return CUDA_ERROR_NOT_SUPPORTED;
}

#define S16(p, q) S1(p, q, 0) S1(p, q, 1) S1(p, q, 2) S1(p, q, 3) S1(p, q, 4) S1(p, q, 5) \
  S1(p, q, 6) S1(p, q, 7) S1(p, q, 8) S1(p, q, 9) S1(p, q, a) S1(p, q, b) S1(p, q, c)  \
  S1(p, q, d) S1(p, q, e) S1(p, q, f)
#define S256(p) S16(p, 0) S16(p, 1) S16(p, 2) S16(p, 3) S16(p, 4) S16(p, 5) S16(p, 6) \
  S16(p, 7) S16(p, 8) S16(p, 9) S16(p, a) S16(p, b) S16(p, c) S16(p, d) S16(p, e) S16(p, f)
#define STUB_POOL S256(0) S256(1) S256(2) S256(3)

#define S1(p, q, r) static int stub_##p##q##r(void) { return stub_hit(0x##p##q##r); }
STUB_POOL
#undef S1

#define S1(p, q, r) stub_##p##q##r,
static int (*const stubs[FAKE_STUB_SLOTS])(void) = {STUB_POOL};
#undef S1

void *fake_stub_for(const char *name) {
  void *fn = NULL;
  pthread_mutex_lock(&stub_lock);
  for (int i = 0; i < stub_used; i++)
    if (strcmp(stub_names[i], name) == 0) { fn = (void *)stubs[i]; break; }
  if (!fn && stub_used < FAKE_STUB_SLOTS) {
    stub_names[stub_used] = strdup(name);
    fn = (void *)stubs[stub_used++];
  }
  pthread_mutex_unlock(&stub_lock);
  if (!fn) fprintf(stderr, "[fakecuda] stub pool exhausted, cannot fake %s\n", name);
  return fn;
}

int fake_runtime_missing(int *reported, const char *name) {
  if (!__atomic_exchange_n(reported, 1, __ATOMIC_RELAXED))
    fprintf(stderr, "[fakecuda] MISSING (CALLED): %s\n", name);
  else
    FAKE_LOG("MISSING (CALLED): %s", name);
  return cudaErrorNotSupported;
}

static int driver_missing(int *reported, const char *name) {
  if (!__atomic_exchange_n(reported, 1, __ATOMIC_RELAXED))
    fprintf(stderr, "[fakecuda] MISSING (CALLED): %s\n", name);
  else
    FAKE_LOG("MISSING (CALLED): %s", name);
  return CUDA_ERROR_NOT_SUPPORTED;
}

#define DSTUB(name) \
  int name(void) { static int reported; return driver_missing(&reported, #name); }
DSTUB(cuLinkCreate_v2)
DSTUB(cuLinkAddData_v2)
DSTUB(cuLinkComplete)
DSTUB(cuLinkDestroy)
DSTUB(cuStreamBeginCapture_v2)
DSTUB(cuStreamEndCapture)
DSTUB(cuGraphInstantiate)
DSTUB(cuGraphLaunch)
DSTUB(cuGraphDestroy)
DSTUB(cuGraphExecDestroy)
DSTUB(cuGraphGetNodes)
DSTUB(cuGraphNodeGetType)
DSTUB(cuGraphKernelNodeGetParams)
DSTUB(cuGraphExecKernelNodeSetParams)
DSTUB(cuMemPrefetchAsync_v2)
DSTUB(cuStreamGetCaptureInfo_v3)
