#include "fakecuda.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

#define FAKE_MAX_CALLBACKS 16

struct callback {
  fake_callback fn;
  void *user;
};

static struct callback callbacks[FAKE_EVENT_COUNT][FAKE_MAX_CALLBACKS];
static int n_callbacks[FAKE_EVENT_COUNT];

int fake_register_callback(enum fake_event_type type, fake_callback fn, void *user) {
  if (type < 0 || type >= FAKE_EVENT_COUNT) return -1;
  if (fn == NULL) return -1;
  if (n_callbacks[type] == FAKE_MAX_CALLBACKS) {
    fprintf(stderr, "[fakecuda] fake_register_callback: too many callbacks for event %d\n", type);
    return -1;
  }
  int slot = n_callbacks[type];
  callbacks[type][slot].fn = fn;
  callbacks[type][slot].user = user;
  n_callbacks[type] = slot + 1;
  return 0;
}

void fire_callback(const struct fake_event *e) {
  int type = e->type;
  for (int i = 0; i < n_callbacks[type]; i++) {
    struct callback *c = &callbacks[type][i];
    c->fn(e, c->user);
  }
}

typedef void (*plugin_init_fn)(void);

static void load_plugin(const char *path) {
  void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) {
    fprintf(stderr, "[fakecuda] FAKECUDA_PROFILERS: cannot load %s: %s\n", path, dlerror());
    return;
  }
  plugin_init_fn init = (plugin_init_fn)dlsym(handle, "fakecuda_plugin_init");
  if (init == NULL) {
    fprintf(stderr, "[fakecuda] FAKECUDA_PROFILERS: %s has no fakecuda_plugin_init\n", path);
    dlclose(handle);
    return;
  }
  init();
  FAKE_LOG("loaded profiler %s", path);
}

void fake_load_plugins(void) {
  const char *env = getenv("FAKECUDA_PROFILERS");
  if (env == NULL || env[0] == '\0') return;
  char *list = strdup(env);
  char *save = NULL;
  char *path = strtok_r(list, ":", &save);
  while (path != NULL) {
    load_plugin(path);
    path = strtok_r(NULL, ":", &save);
  }
  free(list);
}
