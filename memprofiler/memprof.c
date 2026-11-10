#define _POSIX_C_SOURCE 200809L
#include "memprof.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define MAX_REASONS 32
#define MAP_MIN_CAP 1024

struct entry {
  const void *ptr; /* NULL marks an empty slot */
  size_t size;
  int device;      /* -1 for host-pinned */
  memprof_memory_kind kind;
};

struct reason {
  char *text;
  uint64_t count;
};

struct memprof {
  pthread_mutex_t lock;
  int num_devices;
  memprof_device_stats *devices;

  size_t current_total, peak_total;
  size_t host_current, host_peak;

  /* live allocations, keyed by pointer */
  struct entry *map;
  size_t map_cap, map_count;

  uint64_t unknown_frees, invalid_args, duplicate_allocs, null_allocs, internal_errors;

  struct reason reasons[MAX_REASONS];
  int n_reasons;
  uint64_t reasons_dropped;
};

/* Queries take a const profiler but still need its lock. */
static pthread_mutex_t *lock_of(const memprof_t *m) { return (pthread_mutex_t *)&m->lock; }

static int valid_device(const struct memprof *m, int device) {
  return device >= 0 && device < m->num_devices;
}

static int valid_kind(memprof_memory_kind kind) {
  int k = (int)kind;
  return k == MEMPROF_DEVICE || k == MEMPROF_HOST_PINNED || k == MEMPROF_MANAGED;
}

/* ---- pointer map: open addressing, linear probing, load factor <= 1/2 ---- */

/*
finds entry's home bucket
entries can be displaced from their home bucket under open addressing
*/
static size_t home(const void *p, size_t cap) {
  uint64_t h = (uint64_t)(uintptr_t)p;
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return (size_t)h & (cap - 1);
}

static size_t map_find(const struct memprof *m, const void *p) {
  for (size_t i = home(p, m->map_cap);; i = (i + 1) & (m->map_cap - 1)) {
    if (m->map[i].ptr == p) return i;
    if (!m->map[i].ptr) return SIZE_MAX;
  }
}

static void map_place(struct entry *map, size_t cap, struct entry e) {
  size_t i = home(e.ptr, cap);
  while (map[i].ptr) i = (i + 1) & (cap - 1);
  map[i] = e;
}

/* e.ptr must not already be in the map. Returns -1 if growing the map fails. */
static int map_insert(struct memprof *m, struct entry e) {
  if (2 * (m->map_count + 1) > m->map_cap) {
    size_t cap = 2 * m->map_cap;
    struct entry *map = calloc(cap, sizeof *map);
    if (!map) return -1;
    for (size_t i = 0; i < m->map_cap; i++)
      if (m->map[i].ptr) map_place(map, cap, m->map[i]);
    free(m->map);
    m->map = map;
    m->map_cap = cap;
  }
  map_place(m->map, m->map_cap, e);
  m->map_count++;
  return 0;
}

/* Backward-shift deletion: later entries in the probe run move back into the
 * hole, so lookups never need tombstones. */
static void map_remove_at(struct memprof *m, size_t i) {
  size_t mask = m->map_cap - 1;
  for (size_t j = (i + 1) & mask; m->map[j].ptr; j = (j + 1) & mask) {
    size_t k = home(m->map[j].ptr, m->map_cap);
    /* The entry at j must stay put if its home slot lies cyclically in (i, j]. */
    int stays = i <= j ? (i < k && k <= j) : (i < k || k <= j);
    if (stays) continue;
    m->map[i] = m->map[j];
    i = j;
  }
  m->map[i].ptr = NULL;
  m->map_count--;
}

/* ---- accounting ---- */

static void raise_peak(size_t *peak, size_t current) {
  if (current > *peak) *peak = current;
}

static void account_add(struct memprof *m, const struct entry *e) {
  if (e->kind == MEMPROF_HOST_PINNED) {
    m->host_current += e->size;
    raise_peak(&m->host_peak, m->host_current);
    return;
  }
  memprof_device_stats *d = &m->devices[e->device];
  d->n_allocs++;
  if (e->kind == MEMPROF_MANAGED) {
    d->managed_current += e->size;
    raise_peak(&d->managed_peak, d->managed_current);
    return;
  }
  d->current += e->size;
  raise_peak(&d->peak, d->current);
  m->current_total += e->size;
  raise_peak(&m->peak_total, m->current_total);
}

static void account_sub(struct memprof *m, const struct entry *e) {
  if (e->kind == MEMPROF_HOST_PINNED) {
    m->host_current -= e->size;
    return;
  }
  memprof_device_stats *d = &m->devices[e->device];
  d->n_frees++;
  if (e->kind == MEMPROF_MANAGED) {
    d->managed_current -= e->size;
    return;
  }
  d->current -= e->size;
  m->current_total -= e->size;
}

/* ---- lifecycle ---- */

memprof_t *memprof_create(int num_devices) {
  if (num_devices < 0) return NULL;
  struct memprof *m = calloc(1, sizeof *m);
  if (!m) return NULL;
  m->num_devices = num_devices;
  m->devices = calloc(num_devices ? (size_t)num_devices : 1, sizeof *m->devices);
  m->map_cap = MAP_MIN_CAP;
  m->map = calloc(m->map_cap, sizeof *m->map);
  if (!m->devices || !m->map || pthread_mutex_init(&m->lock, NULL) != 0) {
    free(m->devices);
    free(m->map);
    free(m);
    return NULL;
  }
  return m;
}

void memprof_destroy(memprof_t *m) {
  if (!m) return;
  for (int i = 0; i < m->n_reasons; i++) free(m->reasons[i].text);
  pthread_mutex_destroy(&m->lock);
  free(m->map);
  free(m->devices);
  free(m);
}

/* ---- events ---- */

void memprof_alloc(memprof_t *m, int device, const void *ptr, size_t size, memprof_memory_kind kind) {
  if (!m || (!ptr && size == 0)) return;
  pthread_mutex_lock(&m->lock);
  if (!ptr) {
    m->null_allocs++;
  } else if (!valid_kind(kind) || (kind != MEMPROF_HOST_PINNED && !valid_device(m, device))) {
    m->invalid_args++;
  } else {
    struct entry e = {ptr, size, kind == MEMPROF_HOST_PINNED ? -1 : device, kind};
    size_t i = map_find(m, ptr);
    if (i != SIZE_MAX) {
      /* A free was missed; the old allocation can't still exist at this address. */
      m->duplicate_allocs++;
      account_sub(m, &m->map[i]);
      m->map[i] = e;
      account_add(m, &e);
    } else if (map_insert(m, e) == 0) {
      account_add(m, &e);
    } else {
      m->internal_errors++;
    }
  }
  pthread_mutex_unlock(&m->lock);
}

void memprof_free(memprof_t *m, const void *ptr) {
  if (!m || !ptr) return;
  pthread_mutex_lock(&m->lock);
  size_t i = map_find(m, ptr);
  if (i == SIZE_MAX) {
    m->unknown_frees++;
  } else {
    struct entry e = m->map[i];
    map_remove_at(m, i);
    account_sub(m, &e);
  }
  pthread_mutex_unlock(&m->lock);
}

void memprof_alloc_failed(memprof_t *m, int device, size_t size) {
  if (!m) return;
  pthread_mutex_lock(&m->lock);
  if (!valid_device(m, device)) {
    m->invalid_args++;
  } else {
    memprof_device_stats *d = &m->devices[device];
    d->n_failed++;
    raise_peak(&d->largest_failed, size);
  }
  pthread_mutex_unlock(&m->lock);
}

void memprof_mark_incomplete(memprof_t *m, const char *reason) {
  if (!m) return;
  if (!reason) reason = "(unspecified)";
  pthread_mutex_lock(&m->lock);
  int i = 0;
  while (i < m->n_reasons && strcmp(m->reasons[i].text, reason) != 0) i++;
  if (i < m->n_reasons) {
    m->reasons[i].count++;
  } else if (m->n_reasons == MAX_REASONS) {
    m->reasons_dropped++;
  } else {
    char *text = strdup(reason);
    if (text)
      m->reasons[m->n_reasons++] = (struct reason){text, 1};
    else
      m->internal_errors++;
  }
  pthread_mutex_unlock(&m->lock);
}

/* ---- queries ---- */

static int incomplete_locked(const struct memprof *m) {
  return m->n_reasons > 0 || m->reasons_dropped > 0 || m->internal_errors > 0;
}

static void summary_locked(const struct memprof *m, memprof_summary *s) {
  *s = (memprof_summary){
      .num_devices = m->num_devices,
      .current_total = m->current_total,
      .peak_total = m->peak_total,
      .host_pinned_current = m->host_current,
      .host_pinned_peak = m->host_peak,
      .live_allocs = m->map_count,
      .unknown_frees = m->unknown_frees,
      .invalid_args = m->invalid_args,
      .duplicate_allocs = m->duplicate_allocs,
      .null_allocs = m->null_allocs,
      .internal_errors = m->internal_errors,
      .incomplete = incomplete_locked(m),
  };
}

int memprof_get_device_stats(const memprof_t *m, int device, memprof_device_stats *out) {
  if (!m || !out || !valid_device(m, device)) return -1;
  pthread_mutex_lock(lock_of(m));
  *out = m->devices[device];
  pthread_mutex_unlock(lock_of(m));
  return 0;
}

void memprof_get_summary(const memprof_t *m, memprof_summary *out) {
  if (!out) return;
  memset(out, 0, sizeof *out);
  if (!m) return;
  pthread_mutex_lock(lock_of(m));
  summary_locked(m, out);
  pthread_mutex_unlock(lock_of(m));
}

/* ---- reports ---- */

static void json_string(FILE *f, const char *s) {
  fputc('"', f);
  for (; *s; s++) {
    unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\')
      fprintf(f, "\\%c", c);
    else if (c < 0x20)
      fprintf(f, "\\u%04x", c);
    else
      fputc(c, f);
  }
  fputc('"', f);
}

int memprof_write_json(const memprof_t *m, FILE *f) {
  if (!m || !f) return -1;
  pthread_mutex_lock(lock_of(m));
  memprof_summary s;
  summary_locked(m, &s);

  fprintf(f, "{\n");
  fprintf(f, "  \"format_version\": 1,\n");
  fprintf(f, "  \"complete\": %s,\n", s.incomplete ? "false" : "true");
  fprintf(f, "  \"num_devices\": %d,\n", s.num_devices);
  fprintf(f, "  \"peak_total_bytes\": %zu,\n", s.peak_total);
  fprintf(f, "  \"current_total_bytes\": %zu,\n", s.current_total);
  fprintf(f, "  \"live_allocations\": %" PRIu64 ",\n", s.live_allocs);

  fprintf(f, "  \"devices\": [");
  for (int i = 0; i < m->num_devices; i++) {
    const memprof_device_stats *d = &m->devices[i];
    fprintf(f,
            "%s\n    {\"device\": %d, \"peak_bytes\": %zu, \"current_bytes\": %zu, "
            "\"managed_peak_bytes\": %zu, \"managed_current_bytes\": %zu, "
            "\"allocs\": %" PRIu64 ", \"frees\": %" PRIu64 ", \"failed_allocs\": %" PRIu64 ", "
            "\"largest_failed_bytes\": %zu}",
            i ? "," : "", i, d->peak, d->current, d->managed_peak, d->managed_current,
            d->n_allocs, d->n_frees, d->n_failed, d->largest_failed);
  }
  fprintf(f, "%s],\n", m->num_devices ? "\n  " : "");

  fprintf(f, "  \"host_pinned\": {\"peak_bytes\": %zu, \"current_bytes\": %zu},\n",
          s.host_pinned_peak, s.host_pinned_current);

  fprintf(f, "  \"incomplete_reasons\": [");
  for (int i = 0; i < m->n_reasons; i++) {
    fprintf(f, "%s\n    {\"reason\": ", i ? "," : "");
    json_string(f, m->reasons[i].text);
    fprintf(f, ", \"count\": %" PRIu64 "}", m->reasons[i].count);
  }
  fprintf(f, "%s],\n", m->n_reasons ? "\n  " : "");
  fprintf(f, "  \"incomplete_reasons_dropped\": %" PRIu64 ",\n", m->reasons_dropped);

  fprintf(f,
          "  \"anomalies\": {\"unknown_frees\": %" PRIu64 ", \"invalid_args\": %" PRIu64
          ", \"duplicate_allocs\": %" PRIu64 ", \"null_allocs\": %" PRIu64
          ", \"internal_errors\": %" PRIu64 "}\n",
          s.unknown_frees, s.invalid_args, s.duplicate_allocs, s.null_allocs, s.internal_errors);
  fprintf(f, "}\n");

  pthread_mutex_unlock(lock_of(m));
  return ferror(f) ? -1 : 0;
}

static const char *fmt_bytes(char *buf, size_t len, size_t v) {
  static const char *const units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB"};
  double x = (double)v;
  int u = 0;
  while (x >= 1024 && u < 5) {
    x /= 1024;
    u++;
  }
  if (u == 0)
    snprintf(buf, len, "%zu B", v);
  else
    snprintf(buf, len, "%.1f %s", x, units[u]);
  return buf;
}

int memprof_write_text(const memprof_t *m, FILE *f) {
  if (!m || !f) return -1;
  pthread_mutex_lock(lock_of(m));
  memprof_summary s;
  summary_locked(m, &s);
  char a[32], b[32], c[32];

  fprintf(f, "memprof: %d device%s, combined peak %s%s\n", s.num_devices,
          s.num_devices == 1 ? "" : "s", fmt_bytes(a, sizeof a, s.peak_total),
          s.incomplete ? "  [INCOMPLETE: peaks are lower bounds]" : "");

  if (m->num_devices > 0)
    fprintf(f, "  %6s  %11s  %11s  %8s  %8s  %6s  %14s\n", "device", "peak", "current", "allocs",
            "frees", "failed", "largest failed");
  for (int i = 0; i < m->num_devices; i++) {
    const memprof_device_stats *d = &m->devices[i];
    fprintf(f, "  %6d  %11s  %11s  %8" PRIu64 "  %8" PRIu64 "  %6" PRIu64 "  %14s\n", i,
            fmt_bytes(a, sizeof a, d->peak), fmt_bytes(b, sizeof b, d->current), d->n_allocs,
            d->n_frees, d->n_failed,
            d->n_failed ? fmt_bytes(c, sizeof c, d->largest_failed) : "-");
  }

  for (int i = 0; i < m->num_devices; i++) {
    const memprof_device_stats *d = &m->devices[i];
    if (d->managed_peak)
      fprintf(f, "  managed: device %d peak %s, current %s\n", i,
              fmt_bytes(a, sizeof a, d->managed_peak), fmt_bytes(b, sizeof b, d->managed_current));
  }
  if (s.host_pinned_peak)
    fprintf(f, "  host pinned: peak %s, current %s\n", fmt_bytes(a, sizeof a, s.host_pinned_peak),
            fmt_bytes(b, sizeof b, s.host_pinned_current));

  for (int i = 0; i < m->n_reasons; i++)
    fprintf(f, "  incomplete: %s (x%" PRIu64 ")\n", m->reasons[i].text, m->reasons[i].count);
  if (m->reasons_dropped)
    fprintf(f, "  incomplete: %" PRIu64 " more reasons not recorded\n", m->reasons_dropped);
  if (s.internal_errors)
    fprintf(f, "  incomplete: profiler ran out of memory %" PRIu64 " times\n", s.internal_errors);

  if (s.unknown_frees || s.invalid_args || s.duplicate_allocs || s.null_allocs)
    fprintf(f,
            "  anomalies: %" PRIu64 " unknown frees, %" PRIu64 " invalid args, %" PRIu64
            " duplicate allocs, %" PRIu64 " null allocs\n",
            s.unknown_frees, s.invalid_args, s.duplicate_allocs, s.null_allocs);
  else
    fprintf(f, "  anomalies: none\n");

  pthread_mutex_unlock(lock_of(m));
  return ferror(f) ? -1 : 0;
}
