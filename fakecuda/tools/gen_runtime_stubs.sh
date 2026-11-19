#!/bin/sh
set -e
cat <<'EOF'
int fake_runtime_missing(int *reported, const char *name);
#define STUB(name) \
  __attribute__((weak)) int name(void) { static int reported; return fake_runtime_missing(&reported, #name); }
EOF
nm -D --defined-only "$1" | awk '$2 == "T" { sub(/@.*/, "", $3); print $3 }' |
  grep '^cuda' | grep -vx -e cudaGetErrorString -e cudaGetErrorName | sort -u |
  sed 's/.*/STUB(&)/'
