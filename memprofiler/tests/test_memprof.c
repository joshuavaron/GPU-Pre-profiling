#define _POSIX_C_SOURCE 200809L
#include "memprof.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks, failures;

#define CHECK(c)                                                              \
  do {                                                                        \
    checks++;                                                                 \
    if (!(c)) {                                                               \
      failures++;                                                             \
      fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c);   \
    }                                                                         \
  } while (0)

#define CHECK_EQ(a, b)                                                        \
  do {                                                                        \
    unsigned long long a_ = (unsigned long long)(a), b_ = (unsigned long long)(b); \
    checks++;                                                                 \
    if (a_ != b_) {                                                           \
      failures++;                                                             \
      fprintf(stderr, "%s:%d: %s == %s failed: %llu != %llu\n", __FILE__,      \
              __LINE__, #a, #b, a_, b_);                                      \
    }                                                                         \
  } while (0)

/* Fake pointers: the profiler never dereferences them. */
#define P(x) ((const void *)(uintptr_t)(x))

static memprof_device_stats dev(const memprof_t *m, int d) {
  memprof_device_stats s;
  memset(&s, 0, sizeof s);
  CHECK_EQ(memprof_get_device_stats(m, d, &s), 0);
  return s;
}

static memprof_summary sum(const memprof_t *m) {
  memprof_summary s;
  memprof_get_summary(m, &s);
  return s;
}

static char *capture(int (*writer)(const memprof_t *, FILE *), const memprof_t *m) {
  char *buf = NULL;
  size_t len = 0;
  FILE *f = open_memstream(&buf, &len);
  if (!f) {
    perror("open_memstream");
    exit(2);
  }
  CHECK_EQ(writer(m, f), 0);
  fclose(f);
  return buf;
}

static void test_basic(void) {
  memprof_t *m = memprof_create(2);
  memprof_alloc(m, 0, P(0x1000), 100, MEMPROF_DEVICE);
  memprof_alloc(m, 0, P(0x2000), 200, MEMPROF_DEVICE);
  CHECK_EQ(dev(m, 0).current, 300);
  CHECK_EQ(dev(m, 0).peak, 300);

  memprof_free(m, P(0x1000));
  CHECK_EQ(dev(m, 0).current, 200);
  CHECK_EQ(dev(m, 0).peak, 300);

  memprof_alloc(m, 0, P(0x3000), 50, MEMPROF_DEVICE);
  CHECK_EQ(dev(m, 0).current, 250);
  CHECK_EQ(dev(m, 0).peak, 300);

  memprof_alloc(m, 1, P(0x4000), 1000, MEMPROF_DEVICE);
  CHECK_EQ(dev(m, 1).current, 1000);
  CHECK_EQ(sum(m).current_total, 1250);
  CHECK_EQ(sum(m).peak_total, 1250);
  CHECK_EQ(sum(m).live_allocs, 3);

  memprof_free(m, P(0x2000));
  memprof_free(m, P(0x3000));
  memprof_free(m, P(0x4000));
  memprof_device_stats d0 = dev(m, 0);
  CHECK_EQ(d0.current, 0);
  CHECK_EQ(d0.n_allocs, 3);
  CHECK_EQ(d0.n_frees, 3);
  CHECK_EQ(sum(m).current_total, 0);
  CHECK_EQ(sum(m).live_allocs, 0);
  CHECK(!sum(m).incomplete);

  memprof_device_stats unused;
  CHECK_EQ(memprof_get_device_stats(m, 2, &unused), -1);
  CHECK_EQ(memprof_get_device_stats(m, -1, &unused), -1);
  memprof_destroy(m);
}

/* Devices peak at different times, so the combined peak is less than the sum of peaks. */
static void test_peak_total_is_not_sum_of_peaks(void) {
  memprof_t *m = memprof_create(2);
  memprof_alloc(m, 0, P(0x1000), 100, MEMPROF_DEVICE);
  memprof_free(m, P(0x1000));
  memprof_alloc(m, 1, P(0x2000), 80, MEMPROF_DEVICE);
  CHECK_EQ(dev(m, 0).peak, 100);
  CHECK_EQ(dev(m, 1).peak, 80);
  CHECK_EQ(sum(m).peak_total, 100);
  memprof_destroy(m);
}

static void test_anomalies(void) {
  memprof_t *m = memprof_create(2);

  memprof_free(m, P(0xdead));
  CHECK_EQ(sum(m).unknown_frees, 1);

  memprof_free(m, NULL); /* no-op, like cudaFree(0) */
  memprof_alloc(m, 0, NULL, 0, MEMPROF_DEVICE); /* no-op, like cudaMalloc(0) */
  CHECK_EQ(sum(m).unknown_frees, 1);
  CHECK_EQ(sum(m).null_allocs, 0);
  CHECK_EQ(sum(m).live_allocs, 0);

  memprof_alloc(m, 0, NULL, 64, MEMPROF_DEVICE);
  CHECK_EQ(sum(m).null_allocs, 1);

  memprof_alloc(m, 2, P(0x1000), 64, MEMPROF_DEVICE);
  memprof_alloc(m, -1, P(0x1000), 64, MEMPROF_MANAGED);
  memprof_alloc(m, 0, P(0x1000), 64, (memprof_memory_kind)7);
  memprof_alloc_failed(m, 9, 64);
  CHECK_EQ(sum(m).invalid_args, 4);
  CHECK_EQ(sum(m).live_allocs, 0);
  CHECK_EQ(sum(m).current_total, 0);

  /* A size-0 allocation with a real pointer is tracked, so its later free is not "unknown". */
  memprof_alloc(m, 0, P(0x5000), 0, MEMPROF_DEVICE);
  CHECK_EQ(sum(m).live_allocs, 1);
  memprof_free(m, P(0x5000));
  CHECK_EQ(sum(m).unknown_frees, 1);

  /* Duplicate alloc: the old allocation is treated as freed and replaced. */
  memprof_alloc(m, 0, P(0x2000), 100, MEMPROF_DEVICE);
  memprof_alloc(m, 1, P(0x2000), 30, MEMPROF_DEVICE);
  CHECK_EQ(sum(m).duplicate_allocs, 1);
  CHECK_EQ(sum(m).live_allocs, 1);
  CHECK_EQ(dev(m, 0).current, 0);
  CHECK_EQ(dev(m, 0).peak, 100);
  CHECK_EQ(dev(m, 1).current, 30);
  CHECK_EQ(sum(m).current_total, 30);

  memprof_free(m, P(0x2000));
  CHECK_EQ(sum(m).current_total, 0);
  memprof_free(m, P(0x2000)); /* double free */
  CHECK_EQ(sum(m).unknown_frees, 2);

  /* Anomalies are reported but don't mark the run incomplete. */
  CHECK(!sum(m).incomplete);
  memprof_destroy(m);
}

static void test_kinds(void) {
  memprof_t *m = memprof_create(2);
  memprof_alloc(m, 12345, P(0x1000), 500, MEMPROF_HOST_PINNED); /* device is ignored */
  memprof_alloc(m, 1, P(0x2000), 300, MEMPROF_MANAGED);
  memprof_summary s = sum(m);
  CHECK_EQ(s.host_pinned_current, 500);
  CHECK_EQ(s.host_pinned_peak, 500);
  CHECK_EQ(s.current_total, 0);
  CHECK_EQ(s.peak_total, 0);
  CHECK_EQ(s.live_allocs, 2);
  CHECK_EQ(s.invalid_args, 0);
  memprof_device_stats d1 = dev(m, 1);
  CHECK_EQ(d1.managed_current, 300);
  CHECK_EQ(d1.managed_peak, 300);
  CHECK_EQ(d1.current, 0);
  CHECK_EQ(d1.n_allocs, 1);

  memprof_free(m, P(0x1000));
  memprof_free(m, P(0x2000));
  s = sum(m);
  CHECK_EQ(s.host_pinned_current, 0);
  CHECK_EQ(s.host_pinned_peak, 500);
  CHECK_EQ(dev(m, 1).managed_current, 0);
  CHECK_EQ(dev(m, 1).n_frees, 1);
  CHECK_EQ(s.live_allocs, 0);
  memprof_destroy(m);
}

static void test_failed_allocs(void) {
  memprof_t *m = memprof_create(1);
  memprof_alloc_failed(m, 0, (size_t)1 << 40);
  memprof_alloc_failed(m, 0, 10);
  CHECK_EQ(dev(m, 0).n_failed, 2);
  CHECK_EQ(dev(m, 0).largest_failed, (size_t)1 << 40);
  CHECK_EQ(dev(m, 0).current, 0);
  memprof_destroy(m);
}

static void test_incomplete(void) {
  memprof_t *m = memprof_create(1);
  memprof_mark_incomplete(m, "cudaMallocAsync");
  memprof_mark_incomplete(m, "cuMemCreate");
  memprof_mark_incomplete(m, "cudaMallocAsync");
  memprof_mark_incomplete(m, "cudaMallocAsync");
  memprof_mark_incomplete(m, NULL);
  CHECK(sum(m).incomplete);

  char *json = capture(memprof_write_json, m);
  CHECK(strstr(json, "\"complete\": false") != NULL);
  CHECK(strstr(json, "{\"reason\": \"cudaMallocAsync\", \"count\": 3}") != NULL);
  CHECK(strstr(json, "{\"reason\": \"cuMemCreate\", \"count\": 1}") != NULL);
  CHECK(strstr(json, "{\"reason\": \"(unspecified)\", \"count\": 1}") != NULL);
  CHECK(strstr(json, "\"incomplete_reasons_dropped\": 0") != NULL);
  free(json);

  char *text = capture(memprof_write_text, m);
  CHECK(strstr(text, "INCOMPLETE") != NULL);
  CHECK(strstr(text, "incomplete: cudaMallocAsync (x3)") != NULL);
  free(text);

  /* Distinct reasons beyond the table size are counted, not stored. */
  for (int i = 0; i < 40; i++) {
    char reason[32];
    snprintf(reason, sizeof reason, "reason-%d", i);
    memprof_mark_incomplete(m, reason);
  }
  json = capture(memprof_write_json, m);
  CHECK(strstr(json, "\"incomplete_reasons_dropped\": 11") != NULL); /* 32 slots - 3 used = 29 stored */
  free(json);
  memprof_destroy(m);
}

static void test_reports(void) {
  memprof_t *m = memprof_create(2);
  memprof_alloc(m, 0, P(0x1000), (size_t)3 << 30, MEMPROF_DEVICE);
  memprof_alloc(m, 1, P(0x2000), 512, MEMPROF_DEVICE);
  memprof_mark_incomplete(m, "has \"quotes\", a \\ and a\nnewline");

  char *json = capture(memprof_write_json, m);
  CHECK(strstr(json, "\"peak_total_bytes\": 3221225984") != NULL);
  CHECK(strstr(json, "{\"device\": 0, \"peak_bytes\": 3221225472,") != NULL);
  CHECK(strstr(json, "has \\\"quotes\\\", a \\\\ and a\\u000anewline") != NULL);
  free(json);

  char *text = capture(memprof_write_text, m);
  CHECK(strstr(text, "memprof: 2 devices, combined peak 3.0 GiB") != NULL);
  CHECK(strstr(text, "512 B") != NULL);
  free(text);
  memprof_destroy(m);

  /* Zero devices still produces valid output. */
  m = memprof_create(0);
  json = capture(memprof_write_json, m);
  CHECK(strstr(json, "\"devices\": [],") != NULL);
  free(json);
  text = capture(memprof_write_text, m);
  CHECK(strstr(text, "memprof: 0 devices") != NULL);
  free(text);
  memprof_destroy(m);

  CHECK(memprof_create(-1) == NULL);
}

/* Random alloc/free stream compared against a simple model. Exercises map
 * growth, backward-shift deletion and address reuse. */
static uint64_t rng_state = 88172645463325252ULL;
static uint64_t rng(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}

enum { NDEV = 4, OPS = 300000, MAXLIVE = 40000 };

struct model_alloc {
  const void *p;
  size_t size;
  int dev;
};

struct model {
  struct model_alloc live[MAXLIVE];
  size_t n_live;
  const void *freed[MAXLIVE];
  size_t n_freed;
  size_t cur[NDEV], peak[NDEV], cur_total, peak_total;
};

static void model_compare(const memprof_t *m, const struct model *r) {
  memprof_summary s = sum(m);
  CHECK_EQ(s.live_allocs, r->n_live);
  CHECK_EQ(s.current_total, r->cur_total);
  CHECK_EQ(s.peak_total, r->peak_total);
  for (int d = 0; d < NDEV; d++) {
    memprof_device_stats st = dev(m, d);
    CHECK_EQ(st.current, r->cur[d]);
    CHECK_EQ(st.peak, r->peak[d]);
  }
}

static void test_random_against_model(void) {
  memprof_t *m = memprof_create(NDEV);
  struct model *r = calloc(1, sizeof *r);
  uint64_t next_id = 1;

  for (int op = 0; op < OPS; op++) {
    int do_alloc = r->n_live == 0 || (r->n_live < MAXLIVE && rng() % 100 < 55);
    if (do_alloc) {
      int d = (int)(rng() % NDEV);
      size_t size = 1 + rng() % (1 << 20);
      const void *p;
      if (r->n_freed && rng() % 100 < 30)
        p = r->freed[--r->n_freed]; /* reuse a freed address */
      else
        p = P(next_id++ * 0x200000); /* 2 MiB-aligned, like real device pointers */
      memprof_alloc(m, d, p, size, MEMPROF_DEVICE);
      r->live[r->n_live++] = (struct model_alloc){p, size, d};
      r->cur[d] += size;
      if (r->cur[d] > r->peak[d]) r->peak[d] = r->cur[d];
      r->cur_total += size;
      if (r->cur_total > r->peak_total) r->peak_total = r->cur_total;
    } else {
      size_t i = rng() % r->n_live;
      struct model_alloc a = r->live[i];
      memprof_free(m, a.p);
      r->live[i] = r->live[--r->n_live];
      r->cur[a.dev] -= a.size;
      r->cur_total -= a.size;
      if (r->n_freed < MAXLIVE) r->freed[r->n_freed++] = a.p;
    }
    if (op % 997 == 0) model_compare(m, r);
  }
  model_compare(m, r);

  while (r->n_live) {
    struct model_alloc a = r->live[--r->n_live];
    memprof_free(m, a.p);
    r->cur[a.dev] -= a.size;
    r->cur_total -= a.size;
  }
  model_compare(m, r);
  memprof_summary s = sum(m);
  CHECK_EQ(s.unknown_frees + s.duplicate_allocs + s.invalid_args + s.internal_errors, 0);
  free(r);
  memprof_destroy(m);
}

enum { THREADS = 8, PER_THREAD = 20000, TSIZE = 100 };

struct worker_arg {
  memprof_t *m;
  int t;
};

static const void *thread_ptr(int t, int i) {
  return P(((uint64_t)(t + 1) << 40) | ((uint64_t)i << 8));
}

static void *worker(void *arg) {
  struct worker_arg *a = arg;
  for (int i = 0; i < PER_THREAD; i++) {
    memprof_alloc(a->m, a->t % 2, thread_ptr(a->t, i), TSIZE, MEMPROF_DEVICE);
    if (i % 2) memprof_free(a->m, thread_ptr(a->t, i));
  }
  return NULL;
}

static void test_threads(void) {
  memprof_t *m = memprof_create(2);
  pthread_t th[THREADS];
  struct worker_arg args[THREADS];
  for (int t = 0; t < THREADS; t++) {
    args[t] = (struct worker_arg){m, t};
    pthread_create(&th[t], NULL, worker, &args[t]);
  }
  for (int t = 0; t < THREADS; t++) pthread_join(th[t], NULL);

  const size_t kept = THREADS * PER_THREAD / 2;
  memprof_summary s = sum(m);
  CHECK_EQ(s.live_allocs, kept);
  CHECK_EQ(s.current_total, kept * TSIZE);
  CHECK(s.peak_total >= s.current_total);
  CHECK_EQ(dev(m, 0).current, kept / 2 * TSIZE);
  CHECK_EQ(dev(m, 0).n_allocs, THREADS / 2 * PER_THREAD);
  CHECK_EQ(dev(m, 0).n_frees, THREADS / 2 * PER_THREAD / 2);

  for (int t = 0; t < THREADS; t++)
    for (int i = 0; i < PER_THREAD; i += 2) memprof_free(m, thread_ptr(t, i));
  s = sum(m);
  CHECK_EQ(s.live_allocs, 0);
  CHECK_EQ(s.current_total, 0);
  CHECK_EQ(s.unknown_frees, 0);
  memprof_destroy(m);
}

static void test_null_profiler(void) {
  memprof_alloc(NULL, 0, P(0x1000), 1, MEMPROF_DEVICE);
  memprof_free(NULL, P(0x1000));
  memprof_alloc_failed(NULL, 0, 1);
  memprof_mark_incomplete(NULL, "x");
  memprof_destroy(NULL);
  memprof_summary s;
  memprof_get_summary(NULL, &s);
  CHECK_EQ(s.num_devices, 0);
  memprof_device_stats d;
  CHECK_EQ(memprof_get_device_stats(NULL, 0, &d), -1);
  CHECK_EQ(memprof_write_json(NULL, stdout), -1);
  CHECK_EQ(memprof_write_text(NULL, stdout), -1);
}

/* Writes a sample report to `path` (checked by `make test` with a JSON parser)
 * and prints the text version. */
static void write_example(const char *path) {
  memprof_t *m = memprof_create(3);
  memprof_alloc(m, 0, P(0x10000000), (size_t)20 << 30, MEMPROF_DEVICE);
  memprof_alloc(m, 0, P(0x20000000), (size_t)6 << 30, MEMPROF_DEVICE);
  memprof_free(m, P(0x20000000));
  memprof_alloc(m, 1, P(0x30000000), (size_t)18 << 30, MEMPROF_DEVICE);
  memprof_alloc(m, 1, P(0x40000000), (size_t)1 << 30, MEMPROF_MANAGED);
  memprof_alloc(m, 0, P(0x50000000), (size_t)2 << 30, MEMPROF_HOST_PINNED);
  memprof_alloc_failed(m, 1, (size_t)150 << 30);
  memprof_mark_incomplete(m, "cudaMallocAsync");
  memprof_mark_incomplete(m, "reason with \"quotes\"\tand a tab");
  memprof_free(m, P(0xbad));

  FILE *f = fopen(path, "w");
  if (!f) {
    perror(path);
    failures++;
  } else {
    CHECK_EQ(memprof_write_json(m, f), 0);
    fclose(f);
  }
  printf("--- example text report ---\n");
  CHECK_EQ(memprof_write_text(m, stdout), 0);
  printf("---------------------------\n");
  memprof_destroy(m);
}

int main(int argc, char **argv) {
  test_basic();
  test_peak_total_is_not_sum_of_peaks();
  test_anomalies();
  test_kinds();
  test_failed_allocs();
  test_incomplete();
  test_reports();
  test_random_against_model();
  test_threads();
  test_null_profiler();
  if (argc > 1) write_example(argv[1]);

  printf("%d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
