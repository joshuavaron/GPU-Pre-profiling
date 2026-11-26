#pragma once
#include <stddef.h>

enum fake_event_type {
  FAKE_EVENT_ALLOC,
  FAKE_EVENT_FREE,
  FAKE_EVENT_ALLOC_FAILED,
  FAKE_EVENT_LAUNCH,
  FAKE_EVENT_COUNT
};

struct fake_dim3 {
  unsigned x, y, z;
};

struct fake_event {
  enum fake_event_type type;
  const char *api;
  int device;
  void *ptr;
  size_t size;
  const char *kernel;
  struct fake_dim3 grid, block;
};

typedef void (*fake_callback)(const struct fake_event *e, void *user);

int fake_register_callback(enum fake_event_type type, fake_callback fn, void *user);
