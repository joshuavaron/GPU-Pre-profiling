#include <dlfcn.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

typedef int (*get_export_table_t)(const void **, const unsigned char *);
static get_export_table_t get_export_table;

static const unsigned char cudart_uuid[16] = {0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
                                              0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9};
static const unsigned char integrity_uuid[16] = {0xd4, 0x08, 0x20, 0x55, 0xbd, 0xe6, 0x70, 0x4b,
                                                 0x8d, 0x34, 0xba, 0x12, 0x3c, 0x66, 0xe1, 0xf2};
static const unsigned char unknown_uuid[16] = {0xde, 0xad, 0xbe, 0xef};

static int failures;
#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
  else printf("ok   %s\n", #cond); } while (0)

static int aborts_with(void (*fn)(void), const char *want) {
  int fds[2];
  if (pipe(fds)) return 0;
  pid_t pid = fork();
  if (pid == 0) {
    dup2(fds[1], 2);
    close(fds[0]);
    fn();
    _exit(0);
  }
  close(fds[1]);
  char buf[4096] = {0};
  size_t n = 0;
  ssize_t r;
  while (n < sizeof buf - 1 && (r = read(fds[0], buf + n, sizeof buf - 1 - n)) > 0) n += r;
  close(fds[0]);
  int status;
  waitpid(pid, &status, 0);
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT && strstr(buf, want);
}

static void call_unimplemented_slot(void) {
  const void *t;
  get_export_table(&t, cudart_uuid);
  ((int (*)(void *, void *))((void *const *)t)[1])(NULL, NULL);
}

static void call_discovery_slot(void) {
  const void *t;
  setenv("FAKECUDA_DARKAPI_TRAP", "1", 1);
  if (get_export_table(&t, unknown_uuid) != 0) _exit(3);
  ((int (*)(void))((void *const *)t)[5])();
}

int main(void) {
  void *h = dlopen("libcuda.so.1", RTLD_NOW);
  CHECK(h != NULL);
  if (!h) return 1;
  get_export_table = (get_export_table_t)dlsym(h, "cuGetExportTable");
  int (*cu_init)(unsigned) = (int (*)(unsigned))dlsym(h, "cuInit");
  CHECK(get_export_table && cu_init && cu_init(0) == 0);

  const void *t = NULL;
  CHECK(get_export_table(&t, cudart_uuid) == 0 && t);
  void *const *w = t;
  CHECK((uintptr_t)w[0] == 12 * sizeof(void *));
  CHECK(w[10] == NULL);
  const void *again = NULL;
  CHECK(get_export_table(&again, cudart_uuid) == 0 && again == t);

  int (*get_primary_ctx)(void **, int) = (int (*)(void **, int))w[2];
  void *ctx = NULL;
  CHECK(get_primary_ctx(&ctx, 0) == 0 && ctx != NULL);
  CHECK(get_primary_ctx(&ctx, 1000) == 101);

  CHECK(get_export_table(&t, integrity_uuid) == 0);
  int (*integrity)(unsigned, uint64_t, uint64_t *) =
      (int (*)(unsigned, uint64_t, uint64_t *))((void *const *)t)[1];
  uint64_t out[2], out2[2];
  CHECK(integrity(12080, 0, out) == 0 && out[0] == 0x3341181c03cb675cull &&
        out[1] == 0x8ed383aa1f4cd1e8ull);
  CHECK(integrity(12082, 1234, out) == 0 && integrity(12082, 1234, out2) == 0 &&
        memcmp(out, out2, 16) == 0 && out[0] != 0x3341181c03cb675cull);

  const void *u = (const void *)1;
  CHECK(get_export_table(&u, unknown_uuid) == 801 && u == NULL);
  CHECK(aborts_with(call_unimplemented_slot, "DARK API TRAP: table 6bd5fb6c"));
  CHECK(aborts_with(call_discovery_slot, "deadbeef-0000-0000-0000-000000000000 (unknown) slot 5"));

  if (failures) return 1;
  printf("PASS test_darkapi\n");
  return 0;
}
