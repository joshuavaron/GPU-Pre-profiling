#include "../events.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned long long allocs;
static unsigned long long frees;
static unsigned long long failed;
static unsigned long long launches;
static size_t current;
static size_t peak;

static void on_alloc(const struct fake_event *e, void *user) {
  pthread_mutex_lock(&lock);
  allocs++;
  current += e->size;
  if (current > peak) peak = current;
  pthread_mutex_unlock(&lock);
}

static void on_free(const struct fake_event *e, void *user) {
  pthread_mutex_lock(&lock);
  frees++;
  current -= e->size;
  pthread_mutex_unlock(&lock);
}

static void on_failed(const struct fake_event *e, void *user) {
  pthread_mutex_lock(&lock);
  failed++;
  pthread_mutex_unlock(&lock);
}

static void on_launch(const struct fake_event *e, void *user) {
  pthread_mutex_lock(&lock);
  launches++;
  pthread_mutex_unlock(&lock);
}

static void report(void) {
  double gib = 1024.0 * 1024.0 * 1024.0;
  fprintf(stderr, "[example_counter] allocs=%llu frees=%llu failed=%llu launches=%llu "
          "current=%.3f GiB peak=%.3f GiB\n",
          allocs, frees, failed, launches, current / gib, peak / gib);
}

void fakecuda_plugin_init(void) {
  fake_register_callback(FAKE_EVENT_ALLOC, on_alloc, NULL);
  fake_register_callback(FAKE_EVENT_FREE, on_free, NULL);
  fake_register_callback(FAKE_EVENT_ALLOC_FAILED, on_failed, NULL);
  fake_register_callback(FAKE_EVENT_LAUNCH, on_launch, NULL);
  atexit(report);
}
