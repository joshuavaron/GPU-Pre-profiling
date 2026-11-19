#define _GNU_SOURCE
#include "fakecuda.h"
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_NOT_SUPPORTED 801

#define TRAP_SLOTS 1024
#define DISCOVERY_SLOTS 64

typedef uintptr_t word;

struct dark_slot {
  int index;
  void *fn;
};
#define DARK_NULL ((void *)1)

struct dark_table {
  const char *name;
  unsigned char uuid[16];
  int nslots;
  int size_header;
  const struct dark_slot *impl;
};

/* from zluda */
static int cudart_get_primary_ctx(void **pctx, int dev) {
  int r = CUDA_ERROR_INVALID_VALUE;
  if (!fake_valid(dev)) r = 101;
  else if (pctx) *pctx = fake_primary_ctx(dev), r = CUDA_SUCCESS;
  FAKE_LOG("dark CUDART_INTERFACE[2] get_primary_ctx(dev=%d) = %d", dev, r);
  return r;
}

static const struct dark_slot cudart_interface[] = {
    {2, cudart_get_primary_ctx}, {10, DARK_NULL}, {-1, NULL}};

/* from zluda */
static uint64_t runtime_cb_enabled[0x400], runtime_cb_enabled2[0xe];

static void hooks_get_buffer1(void **ptr, size_t *count) {
  *ptr = runtime_cb_enabled;
  *count = 0x400;
  FAKE_LOG("dark TOOLS_RUNTIME_CALLBACK_HOOKS[2] get_buffer() = %p x %zu", *ptr, *count);
}

static void hooks_get_buffer2(void **ptr, size_t *count) {
  *ptr = runtime_cb_enabled2;
  *count = 0xe;
  FAKE_LOG("dark TOOLS_RUNTIME_CALLBACK_HOOKS[6] get_buffer() = %p x %zu", *ptr, *count);
}

static const struct dark_slot tools_runtime_callback_hooks[] = {
    {2, hooks_get_buffer1}, {6, hooks_get_buffer2}, {-1, NULL}};

static int tools_tls_get_ctx(void **out) {
  int dev = fake_current_device;
  if (!out) return CUDA_ERROR_INVALID_VALUE;
  *out = __atomic_load_n(&fake_ctx_active[dev], __ATOMIC_RELAXED) ? fake_primary_ctx(dev) : NULL;
  FAKE_LOG("dark TOOLS_TLS[2] get_current_ctx() = 0  %p", *out);
  return CUDA_SUCCESS;
}

static const struct dark_slot tools_tls[] = {{2, tools_tls_get_ctx}, {-1, NULL}};

/* from zluda */
struct cls_node {
  struct cls_node *next;
  void *ctx, *key, *value, *dtor;
};
static struct cls_node *cls_head;
static pthread_mutex_t cls_lock = PTHREAD_MUTEX_INITIALIZER;

static int cls_ctx(void **ctx) {
  if (!*ctx) {
    int dev = fake_current_device;
    if (!__atomic_load_n(&fake_ctx_active[dev], __ATOMIC_RELAXED)) return 201;
    *ctx = fake_primary_ctx(dev);
  }
  return fake_ctx_device(*ctx) < 0 ? 201 : CUDA_SUCCESS;
}

static struct cls_node **cls_find(void *ctx, void *key) {
  struct cls_node **p = &cls_head;
  while (*p && ((*p)->ctx != ctx || (*p)->key != key)) p = &(*p)->next;
  return p;
}

static int cls_put(void *ctx, void *key, void *value, void *dtor) {
  int r = cls_ctx(&ctx);
  if (r) return r;
  pthread_mutex_lock(&cls_lock);
  struct cls_node **p = cls_find(ctx, key), *n;
  if (*p) r = 400;
  else if (!(n = malloc(sizeof *n))) r = 2;
  else *n = (struct cls_node){cls_head, ctx, key, value, dtor}, cls_head = n;
  pthread_mutex_unlock(&cls_lock);
  FAKE_LOG("dark CONTEXT_LOCAL_STORAGE[0] put(ctx=%p, key=%p, value=%p) = %d", ctx, key, value, r);
  return r;
}

static int cls_delete(void *ctx, void *key) {
  int r = cls_ctx(&ctx);
  if (r) return r;
  pthread_mutex_lock(&cls_lock);
  struct cls_node **p = cls_find(ctx, key), *n = *p;
  if (n) *p = n->next, free(n);
  else r = 400;
  pthread_mutex_unlock(&cls_lock);
  FAKE_LOG("dark CONTEXT_LOCAL_STORAGE[1] delete(ctx=%p, key=%p) = %d", ctx, key, r);
  return r;
}

static int cls_get(void **value, void *ctx, void *key) {
  int r = cls_ctx(&ctx);
  if (r) return r;
  pthread_mutex_lock(&cls_lock);
  struct cls_node *n = *cls_find(ctx, key);
  if (n) *value = n->value;
  else r = 400;
  pthread_mutex_unlock(&cls_lock);
  FAKE_LOG("dark CONTEXT_LOCAL_STORAGE[2] get(ctx=%p, key=%p) = %d", ctx, key, r);
  return r;
}

static const struct dark_slot context_local_storage[] = {
    {0, cls_put}, {1, cls_delete}, {2, cls_get}, {3, DARK_NULL}, {-1, NULL}};

/* from zluda */
static unsigned context_checks_flag(void) {
  FAKE_LOG("dark CONTEXT_CHECKS[3] flag() = 0");
  return 0;
}

/* from zluda */
/* RANDOM RESULT */
static int context_checks_kind(void *ctx, int *kind, void **parent) {
  (void)parent;
  if (!kind) return CUDA_ERROR_INVALID_VALUE;
  int r = cls_ctx(&ctx);
  if (r == CUDA_SUCCESS) *kind = 0;
  FAKE_LOG("dark CONTEXT_CHECKS[2] context_kind(ctx=%p) = %d  kind=0", ctx, r);
  return r;
}

static const struct dark_slot context_checks[] = {
    {2, context_checks_kind}, {3, context_checks_flag}, {-1, NULL}};

/* from zluda */
static const unsigned char md2_s[256] = {
    0x29, 0x2e, 0x43, 0xc9, 0xa2, 0xd8, 0x7c, 0x01, 0x3d, 0x36, 0x54, 0xa1, 0xec, 0xf0, 0x06, 0x13,
    0x62, 0xa7, 0x05, 0xf3, 0xc0, 0xc7, 0x73, 0x8c, 0x98, 0x93, 0x2b, 0xd9, 0xbc, 0x4c, 0x82, 0xca,
    0x1e, 0x9b, 0x57, 0x3c, 0xfd, 0xd4, 0xe0, 0x16, 0x67, 0x42, 0x6f, 0x18, 0x8a, 0x17, 0xe5, 0x12,
    0xbe, 0x4e, 0xc4, 0xd6, 0xda, 0x9e, 0xde, 0x49, 0xa0, 0xfb, 0xf5, 0x8e, 0xbb, 0x2f, 0xee, 0x7a,
    0xa9, 0x68, 0x79, 0x91, 0x15, 0xb2, 0x07, 0x3f, 0x94, 0xc2, 0x10, 0x89, 0x0b, 0x22, 0x5f, 0x21,
    0x80, 0x7f, 0x5d, 0x9a, 0x5a, 0x90, 0x32, 0x27, 0x35, 0x3e, 0xcc, 0xe7, 0xbf, 0xf7, 0x97, 0x03,
    0xff, 0x19, 0x30, 0xb3, 0x48, 0xa5, 0xb5, 0xd1, 0xd7, 0x5e, 0x92, 0x2a, 0xac, 0x56, 0xaa, 0xc6,
    0x4f, 0xb8, 0x38, 0xd2, 0x96, 0xa4, 0x7d, 0xb6, 0x76, 0xfc, 0x6b, 0xe2, 0x9c, 0x74, 0x04, 0xf1,
    0x45, 0x9d, 0x70, 0x59, 0x64, 0x71, 0x87, 0x20, 0x86, 0x5b, 0xcf, 0x65, 0xe6, 0x2d, 0xa8, 0x02,
    0x1b, 0x60, 0x25, 0xad, 0xae, 0xb0, 0xb9, 0xf6, 0x1c, 0x46, 0x61, 0x69, 0x34, 0x40, 0x7e, 0x0f,
    0x55, 0x47, 0xa3, 0x23, 0xdd, 0x51, 0xaf, 0x3a, 0xc3, 0x5c, 0xf9, 0xce, 0xba, 0xc5, 0xea, 0x26,
    0x2c, 0x53, 0x0d, 0x6e, 0x85, 0x28, 0x84, 0x09, 0xd3, 0xdf, 0xcd, 0xf4, 0x41, 0x81, 0x4d, 0x52,
    0x6a, 0xdc, 0x37, 0xc8, 0x6c, 0xc1, 0xab, 0xfa, 0x24, 0xe1, 0x7b, 0x08, 0x0c, 0xbd, 0xb1, 0x4a,
    0x78, 0x88, 0x95, 0x8b, 0xe3, 0x63, 0xe8, 0x6d, 0xe9, 0xcb, 0xd5, 0xfe, 0x3b, 0x00, 0x1d, 0x39,
    0xf2, 0xef, 0xb7, 0x0e, 0x66, 0x58, 0xd0, 0xe4, 0xa6, 0x77, 0x72, 0xf8, 0xeb, 0x75, 0x4b, 0x0a,
    0x31, 0x44, 0x50, 0xb4, 0x8f, 0xed, 0x1f, 0x1a, 0xdb, 0x99, 0x8d, 0x33, 0x9f, 0x11, 0x83, 0x14,
};

struct md2 {
  unsigned char x[48], c[16], buf[16], n, l;
};

static void md2_byte(struct md2 *m, unsigned char b) {
  m->buf[m->n++] = b;
  if (m->n < 16) return;
  m->n = 0;
  for (int j = 0; j < 16; j++) {
    m->x[16 + j] = m->buf[j];
    m->x[32 + j] = m->buf[j] ^ m->x[j];
    m->l = m->c[j] ^= md2_s[m->buf[j] ^ m->l];
  }
  unsigned t = 0;
  for (int round = 0; round < 18; round++) {
    for (int k = 0; k < 48; k++) t = m->x[k] ^= md2_s[t];
    t = (t + round) & 0xff;
  }
}

static void md2_bytes(struct md2 *m, const void *p, size_t n, unsigned char mask) {
  for (size_t i = 0; i < n; i++) md2_byte(m, ((const unsigned char *)p)[i] ^ mask);
}

static void md2_final(struct md2 *m, unsigned char out[16]) {
  unsigned char pad = 16 - m->n, c[16];
  for (int i = 0; i < pad; i++) md2_byte(m, pad);
  memcpy(c, m->c, 16);
  md2_bytes(m, c, 16, 0);
  memcpy(out, m->x, 16);
}

/* from zluda */
static const unsigned char integrity_key[16] = {0x14, 0x6a, 0xdd, 0xae, 0x53, 0xa9, 0xa7, 0x52,
                                                0xaa, 0x08, 0x41, 0x36, 0x0b, 0xf5, 0x5a, 0x9f};
static const uint64_t integrity_const[2] = {0x3341181c03cb675cull, 0x8ed383aa1f4cd1e8ull};
static word *table_named(const char *name);
int cuGetExportTable(const void **table, const unsigned char *uuid);

static int integrity_check_fn(unsigned version, uint64_t unix_seconds, uint64_t out[2]) {
  if (version % 10 < 2) {
    memcpy(out, integrity_const, 16);
    if (version % 10 == 1) ((unsigned char *)out)[7] = 0x18;
  } else {
    struct {
      uint32_t driver_version, version, pid, tid;
      void *cudart_table, *integrity_table, *fn;
      uint64_t unix_seconds;
    } msg = {FAKECUDA_CUDA_VERSION, version, (uint32_t)getpid(), (uint32_t)pthread_self(),
             table_named("CUDART_INTERFACE"), table_named("INTEGRITY_CHECK"),
             (void *)integrity_check_fn, unix_seconds};
    struct md2 m = {0};
    unsigned char inner[16];
    md2_bytes(&m, integrity_key, 16, 0x36);
    md2_bytes(&m, &msg, sizeof msg, 0);
    for (int i = 0; i < fake_device_count; i++) {
      struct { unsigned char uuid[16]; uint32_t domain, bus, device; } d;
      memcpy(d.uuid, fake_devices[i].uuid.bytes, 16);
      d.domain = fake_devices[i].pciDomainID;
      d.bus = fake_devices[i].pciBusID;
      d.device = fake_devices[i].pciDeviceID;
      md2_bytes(&m, &d, sizeof d, 0);
    }
    md2_final(&m, inner);
    memset(&m, 0, sizeof m);
    md2_bytes(&m, integrity_key, 16, 0x5c);
    md2_bytes(&m, inner, 16, 0);
    md2_final(&m, (unsigned char *)out);
  }
  FAKE_LOG("dark INTEGRITY_CHECK[1] integrity_check(version=%u, t=%llu) = 0", version,
           (unsigned long long)unix_seconds);
  return CUDA_SUCCESS;
}

static const struct dark_slot integrity_check[] = {{1, integrity_check_fn}, {-1, NULL}};

/* RANDOM RESULT */
static int ctx_get_id(void *ctx, uint64_t *id) {
  if (!id) return 999;
  *id = 0;
  if (ctx) {
    int dev = fake_ctx_device(ctx);
    if (dev < 0) return 201;
    *id = (uint64_t)dev + 1;
  }
  FAKE_LOG("dark 21318c60[4] get_ctx_id(%p) = 0  %llu", ctx, (unsigned long long)*id);
  return CUDA_SUCCESS;
}

static int get_stream_object(void *ctx, uintptr_t handle, void **object, char per_thread) {
  if (!object) return 1;
  if (handle == (uintptr_t)-1) return 400;
  int dev = handle > 2 ? fake_current_device : ctx ? fake_ctx_device(ctx) : -1;
  if (dev < 0) return 400;
  *object = fake_stream_object(dev, (cudaStream_t)handle, per_thread);
  FAKE_LOG("dark 21318c60[51] get_stream_object(ctx=%p, stream=%#lx) = 0  %p", ctx,
           (unsigned long)handle, *object);
  return *object ? CUDA_SUCCESS : 400;
}

/* RANDOM RESULT */
static int get_stream_id(void *ctx, void *object, uint64_t *id) {
  (void)ctx;
  if (!object || !id) return 1;
  *id = fake_stream_id(object);
  FAKE_LOG("dark 21318c60[39] get_stream_id(%p) = 0  %llu", object, (unsigned long long)*id);
  return CUDA_SUCCESS;
}

static const struct dark_slot table_21318c60[] = {
    {4, ctx_get_id}, {39, get_stream_id}, {51, get_stream_object}, {-1, NULL}};

static int get_device_blobs(void **blob1, void **blob2) {
  (void)blob1, (void)blob2;
  FAKE_LOG("dark 1734dc26[4] get_device_blobs() = 801 (not provided by this fake GPU)");
  return CUDA_ERROR_NOT_SUPPORTED;
}

static const struct dark_slot table_1734dc26[] = {{4, get_device_blobs}, {-1, NULL}};

static const struct dark_table dark_tables[] = {
    {"CUDART_INTERFACE",
     {0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a, 0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9},
     12, 1, cudart_interface},
    {"TOOLS_RUNTIME_CALLBACK_HOOKS",
     {0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74, 0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66},
     7, 1, tools_runtime_callback_hooks},
    {"TOOLS_TLS",
     {0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47, 0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc},
     3, 1, tools_tls},
    {"CONTEXT_LOCAL_STORAGE",
     {0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11, 0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93},
     4, 0, context_local_storage},
    {"CONTEXT_CHECKS",
     {0x26, 0x3e, 0x88, 0x60, 0x7c, 0xd2, 0x61, 0x43, 0x92, 0xf6, 0xbb, 0xd5, 0x00, 0x6d, 0xfa, 0x7e},
     5, 1, context_checks},
    {"INTEGRITY_CHECK",
     {0xd4, 0x08, 0x20, 0x55, 0xbd, 0xe6, 0x70, 0x4b, 0x8d, 0x34, 0xba, 0x12, 0x3c, 0x66, 0xe1, 0xf2},
     3, 1, integrity_check},
    {"21318c60",
     {0x21, 0x31, 0x8c, 0x60, 0x97, 0x14, 0x32, 0x48, 0x8c, 0xa6, 0x41, 0xff, 0x73, 0x24, 0xc8, 0xf2},
     92, 1, table_21318c60},
    {"1734dc26",
     {0x17, 0x34, 0xdc, 0x26, 0x80, 0x0d, 0x47, 0x45, 0x87, 0x26, 0xc0, 0xf1, 0xe7, 0xdd, 0x8b, 0xca},
     17, 1, table_1734dc26},
};

struct trap_info {
  const unsigned char *uuid;
  const char *table;
  int slot;
};
static struct trap_info trap_info[TRAP_SLOTS];
static int trap_used;

static void uuid_str(char out[37], const unsigned char *u) {
  snprintf(out, 37, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0],
           u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14],
           u[15]);
}

static void where(char *out, size_t n, void *addr) {
  Dl_info di;
  if (dladdr(addr, &di) && di.dli_fname) {
    const char *base = strrchr(di.dli_fname, '/');
    snprintf(out, n, "%s+0x%lx", base ? base + 1 : di.dli_fname,
             (unsigned long)((char *)addr - (char *)di.dli_fbase));
  } else {
    snprintf(out, n, "%p", addr);
  }
}

__attribute__((noreturn)) static void trap_hit(int i, void *ret, word a0, word a1, word a2,
                                               word a3, word a4, word a5) {
  char u[37], caller[256];
  uuid_str(u, trap_info[i].uuid);
  where(caller, sizeof caller, ret);
  fprintf(stderr,
          "[fakecuda] DARK API TRAP: table %s (%s) slot %d called from %s\n"
          "[fakecuda]   rdi=%#lx rsi=%#lx rdx=%#lx rcx=%#lx r8=%#lx r9=%#lx\n"
          "[fakecuda]   this slot is not implemented; aborting (see docs/darkapi.md)\n",
          u, trap_info[i].table, trap_info[i].slot, caller, (unsigned long)a0, (unsigned long)a1,
          (unsigned long)a2, (unsigned long)a3, (unsigned long)a4, (unsigned long)a5);
  abort();
}

#define S16(p, q) S1(p, q, 0) S1(p, q, 1) S1(p, q, 2) S1(p, q, 3) S1(p, q, 4) S1(p, q, 5) \
  S1(p, q, 6) S1(p, q, 7) S1(p, q, 8) S1(p, q, 9) S1(p, q, a) S1(p, q, b) S1(p, q, c)  \
  S1(p, q, d) S1(p, q, e) S1(p, q, f)
#define S256(p) S16(p, 0) S16(p, 1) S16(p, 2) S16(p, 3) S16(p, 4) S16(p, 5) S16(p, 6) \
  S16(p, 7) S16(p, 8) S16(p, 9) S16(p, a) S16(p, b) S16(p, c) S16(p, d) S16(p, e) S16(p, f)
#define TRAP_POOL S256(0) S256(1) S256(2) S256(3)

#define S1(p, q, r)                                                                    \
  __attribute__((noinline)) static word trap_##p##q##r(word a0, word a1, word a2, word a3, \
                                                       word a4, word a5) {              \
    trap_hit(0x##p##q##r, __builtin_return_address(0), a0, a1, a2, a3, a4, a5);        \
  }
TRAP_POOL
#undef S1

#define S1(p, q, r) (void *)trap_##p##q##r,
static void *const traps[TRAP_SLOTS] = {TRAP_POOL};
#undef S1

struct built {
  unsigned char uuid[16];
  const char *name;
  word *words;
};
static struct built built[TRAP_SLOTS / 8];
static int n_built;
static pthread_mutex_t dark_lock = PTHREAD_MUTEX_INITIALIZER;

static void *new_trap(const unsigned char *uuid, const char *table, int slot) {
  if (trap_used == TRAP_SLOTS) return NULL;
  trap_info[trap_used] = (struct trap_info){uuid, table, slot};
  return traps[trap_used++];
}

static word *build(const struct dark_table *t, const unsigned char *uuid) {
  struct built *b = &built[n_built];
  int n = t ? t->nslots : DISCOVERY_SLOTS;
  word *w = calloc(n, sizeof *w);
  memcpy(b->uuid, uuid, 16);
  b->name = t ? t->name : "unknown";
  for (int i = 0; i < n; i++) {
    if (t && t->size_header && i == 0) {
      w[0] = n * sizeof(word);
      continue;
    }
    void *fn = NULL;
    for (const struct dark_slot *s = t ? t->impl : NULL; s && s->fn; s++)
      if (s->index == i) fn = s->fn;
    if (fn == DARK_NULL) continue;
    if (!fn && !(fn = new_trap(b->uuid, b->name, i))) {
      free(w);
      return NULL;
    }
    w[i] = (word)fn;
  }
  b->words = w;
  n_built++;
  return w;
}

static word *table_named(const char *name) {
  for (size_t i = 0; i < sizeof dark_tables / sizeof *dark_tables; i++)
    if (strcmp(dark_tables[i].name, name) == 0) {
      const void *t = NULL;
      cuGetExportTable(&t, dark_tables[i].uuid);
      return (word *)t;
    }
  return NULL;
}

int cuGetExportTable(const void **table, const unsigned char *uuid) {
  char u[37];
  if (!table || !uuid) return CUDA_ERROR_INVALID_VALUE;
  uuid_str(u, uuid);
  const char *e = getenv("FAKECUDA_DARKAPI_TRAP");
  int trap_mode = e && *e && strcmp(e, "0") != 0;

  const struct dark_table *t = NULL;
  for (size_t i = 0; i < sizeof dark_tables / sizeof *dark_tables; i++)
    if (memcmp(dark_tables[i].uuid, uuid, 16) == 0) t = &dark_tables[i];

  word *w = NULL;
  pthread_mutex_lock(&dark_lock);
  for (int i = 0; i < n_built && !w; i++)
    if (memcmp(built[i].uuid, uuid, 16) == 0) w = built[i].words;
  if (!w && (t || trap_mode) && n_built < (int)(sizeof built / sizeof *built)) w = build(t, uuid);
  pthread_mutex_unlock(&dark_lock);

  *table = w;
  if (!w) {
    fprintf(stderr,
            "[fakecuda] MISSING (CALLED): cuGetExportTable(%s), private driver interface%s\n", u,
            t || trap_mode ? " (trap pool exhausted)" : "");
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  FAKE_LOG("cuGetExportTable(%s) = 0  %s table %p%s", u, t ? t->name : "unknown", (void *)w,
           t ? "" : " (all slots trap)");
  return CUDA_SUCCESS;
}
