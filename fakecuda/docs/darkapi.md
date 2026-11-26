# Dark API notes (`cuGetExportTable`)

Reverse-engineering notes behind `darkapi.c`. Everything here is a summary of observed behaviour,
written for interoperability. No NVIDIA code or binaries are in this repository.

## Versions analysed

| Component | Version | Where it came from |
|---|---|---|
| `libcuda.so` (callee side) | **570.124.04** (CUDA 12.8 driver) | `NVIDIA-Linux-x86_64-570.124.04.run --extract-only`. Not installed. |
| `libcublas.so.12`, `libcublasLt.so.12` (caller side) | **12.8.4.1** | pip `nvidia-cublas-cu12` in `../.venv` |
| static cudart inside both | 12.8 (asks for integrity versions 12080–12082) | linked into the cuBLAS libraries |
| ZLUDA (cross-reference) | `master` at time of writing | github.com/vosen/ZLUDA, `dark_api/src/lib.rs`, `zluda/src/impl/driver.rs` |

All libcuda addresses below are file virtual addresses in 570.124.04. Ghidra loads the `.so` at
image base 0x100000, so its listings show them +0x100000. Slot layouts and table sizes **change
between driver releases** (ZLUDA, tracking a newer driver, has `CUDART_INTERFACE[13]` and
`TOOLS_TLS[4]`; 570 has 12 and 3). fakecuda copies 570's layout, which cuBLAS 12.8.4.1 accepts.

## How cuBLAS reaches the driver

* `libcublas.so.12` and `libcublasLt.so.12` have **no** dynamic dependency on `libcudart` or
  `libcuda` (`ldd`, `nm -D`). Each statically links its own cudart, which `dlopen`s
  `libcuda.so.1` and resolves everything with `cuGetProcAddress`/`cuGetProcAddress_v2`. Since
  fakecuda's soname is `libcuda.so.1` and it is preloaded, the dlopen returns fakecuda. There are
  two independent cudart instances, one per library, so the init sequence below runs twice.
* `cublasLtCtxInit` **also** `dlsym`s about 85 driver symbols directly from `libcuda.so.1` by
  exported ABI name (`cuMemcpyHtoD_v2`, `cuEventDestroy_v2`, `cuStreamWaitValue32`, plain
  `cuDevicePrimaryCtxRelease`, `cuLinkCreate_v2`, `cuGraphLaunch`, ...). The list is a string table
  at libcublasLt 0xa232410, and the code is at 0x193ceb0. If any lookup returns NULL it logs
  `Driver API initialization failed` and `cublasCreate` returns 14. So fakecuda must *export*
  all of them. The ones it can't emulate (JIT linking, graphs, capture) are loud NOT_SUPPORTED
  stubs in `stubs.c`.

## Method

1. **Discovery (callee-agnostic).** `FAKECUDA_DARKAPI_TRAP=1` makes `cuGetExportTable` return,
   for any unknown UUID, 64 distinct trap stubs. Unimplemented slots of *known* tables always
   trap. A trap prints the UUID, slot, the six integer argument registers and the caller as
   `lib+offset`, then aborts. Each crash names the next slot needed.
2. **UUID → table map.** The exported `cuGetExportTable` (0x34e6f0) is a trampoline through a
   relocated pointer (0x4345cf8 → 0x35b090). The body linearly scans **105 entries** of
   `{const uuid *, const void *table}` at 0x426a820 and compares 16 bytes. All pointers are
   `R_X86_64_RELATIVE` relocations, so the map and each table's words were recovered from
   `readelf -r` and raw section bytes (a small script kept outside the repo).
3. **Layout.** Most tables start with a size in bytes (word 0 is a raw integer, not a
   relocation). `CONTEXT_LOCAL_STORAGE` has none. Some slots are NULL in the real table, and
   fakecuda keeps them NULL (`DARK_NULL` in the spec).
4. **Callee semantics.** Ghidra 11.2.1 headless, importing with `-noanalysis`, plus a script that
   creates and decompiles only the needed functions. Callees (`func_0x...`) are again +0x100000.
5. **Caller side.** `objdump` of the call sites in libcublas/libcublasLt. The trap gives the
   return address, and the code around it shows argument setup and how results are checked.
6. **Cross-check** against ZLUDA where it covers the table.

## Tables seen with cuBLAS 12.8.4.1 + PyTorch 2.11

"Requested by" means both cudart copies unless noted. Size is in words, including the header.

| UUID | Name (ours / ZLUDA) | Size | Requested by | Slots called | Status |
|---|---|---|---|---|---|
| `6bd5fb6c-5bf4-e74a-8987-d93912fd9df9` | CUDART_INTERFACE | 12 (hdr, slot 10 NULL) | cudart init | 2 | 2 implemented. Others trap. |
| `a094798c-2e74-2e74-93f2-0800200c0a66` | TOOLS_RUNTIME_CALLBACK_HOOKS | 7 (hdr) | cudart init | 2, 6 | 2, 6 implemented. 1, 3, 4, 5 trap. |
| `42d85a81-23f6-cb47-8298-f6e78a3aecdc` | TOOLS_TLS | 3 (hdr) | cudart init, cuBLASLt | 2 | 2 implemented. 1 traps. |
| `c693336e-1121-df11-a8c3-68f355d89593` | CONTEXT_LOCAL_STORAGE | 4 (no hdr, slot 3 NULL) | cudart (every API call) | 0, 1, 2 | all implemented |
| `263e8860-7cd2-6143-92f6-bbd5006dfa7e` | CONTEXT_CHECKS | 5 (hdr) | cudart | 2, 3 | 2 implemented (kind value GUESS), 3 implemented. 1, 4 trap. |
| `d4082055-bde6-704b-8d34-ba123c66e1f2` | INTEGRITY_CHECK | 3 (hdr) | cudart init | 1 | 1 implemented. 2 traps. |
| `21318c60-9714-3248-8ca6-41ff7324c8f2` | (unnamed, not in ZLUDA) | 92 (hdr) | cuBLASLt (algo selection), cuBLAS | 4, 39, 51 | those 3 implemented. 88 trap. |
| `1734dc26-800d-4745-8726-c0f1e7dd8bca` | (unnamed, not in ZLUDA) | 17 (hdr) | cuBLASLt (fp16/bf16 GEMM) | 4 | 4 returns 801 like a GPU without the hook. Others trap. |

Discovery mode found no other UUIDs on these paths (C SGEMM test; PyTorch fp32/bmm/fp16/bf16,
`F.linear` with bias, `preferred_blas_library("cublaslt")`).

## Slot notes

Return values are `CUresult`. "Init checks" means the driver's usual prologue: 3 if not
initialised, 4 if deinitialised, 800 if not permitted in this process state.

### CUDART_INTERFACE
* **[2] `get_primary_ctx(CUcontext *out, CUdevice dev)`**, libcuda 0x225820. After init checks,
  returns 101 for an out-of-range device; otherwise stores the device's primary-context handle.
  It does not retain or activate it. Called from cudart's per-device init loop (libcublas
  +0x53699c), which treats non-zero as failure and next queries name and total memory. ZLUDA:
  `cudart_interface_fn2`, same. **High.**
* [1], [6], [8] are, per ZLUDA, the "module from fatbin" loaders. cuBLAS 12.8 does not use them:
  its cudart loads with public `cuLibraryLoadData` (eager mode, as reported by
  `cuModuleGetLoadingMode`). They trap.

### TOOLS_RUNTIME_CALLBACK_HOOKS
How the static runtime reports its own API calls to a profiler attached to the driver.
* **[2], [6] `get_enable_array(void **arr, size_t *count)`**, 0x4aa760 / 0x4aa7e0. They return
  `.bss` arrays (0x43e4f60, 1024 entries; 0x43e4e20, 14 entries). These are all zero unless a
  tool subscribes. The caller (libcublas +0x53a35b) checks the size header is > 0x30 and requires
  count > 0x1e6 and > 0xd. fakecuda returns zeroed static arrays, which is the real "no profiler"
  state. Element width: GUESS u32 (ZLUDA). We allocate 8 bytes per entry. **High.**
* [1], [5] `notify(unsigned cbid, void *data)`, 0x4aa700 / 0x4aa780. They return 1 if the id
  is out of range (> 0x3ff / > 0xd). Otherwise they call the registered tool callback if there is
  one, and return 0. Understood, but never called (the enable arrays are zero), so they trap.
* [3] looks up an object by (handle, id) and returns a field. [4] returns a context's id
  (`ctx+0x38`). Both trap.

### TOOLS_TLS
* **[2] `get_current_ctx(CUcontext *out)`**, 0x4af710. Returns the top of the calling thread's
  context stack (NULL if empty), i.e. `cuCtxGetCurrent`. Called from libcublasLt +0x196e07c,
  which feeds the result to `21318c60[4]`. **High.**
* [1], 0x4b8d50: replaces the top of the thread's context stack with refcounting
  (`cuCtxSetCurrent`-like). Not seen called, so it traps.

### CONTEXT_LOCAL_STORAGE (no size header)
A per-context linked list of `{key, value, destructor}` that the static runtime uses to attach
its state to driver contexts. `ctx == NULL` means the current context.
* **[0] `put(ctx, key, value, dtor)`**, 0x224f50. Returns 400 if the key exists. Otherwise it
  pushes a node, returning 0 (2 on OOM).
* **[1] `delete(ctx, key)`**, 0x225150. Returns 400 if missing, else unlinks and frees.
* **[2] `get(void **value, ctx, key)`**, 0x225340. Returns 400 if missing.

The destructor would run when the context is destroyed. fakecuda never destroys primary
contexts. ZLUDA agrees on arguments and on 400. **High.**

### CONTEXT_CHECKS
* **[3] `flag(void) -> bool`**, 0x38b770. Returns bit 0 of a `.bss` flags word (0x43ebe58). The
  only direct writer of that bit is driver init, which keeps it 0 (0x2b4994). The other setters
  touch bits 2, 4 and 5. cuBLAS (+0x554909) calls it only when the driver version is > 12039,
  and stores `== 1`. ZLUDA: "always 0". Meaning unknown. **High that it is 0.**
* **[2] `context_kind(ctx, int *kind, void **parent)`**, 0x3946c0. Writes the context's type
  (first int of the driver's context object). For type 1 (a green context) it also writes the
  parent. cuBLAS (+0x54d134) only tests `kind == 1`. fakecuda answers 0. **The exact value for a
  regular context is GUESS.**
* [1] always returns 801. [4] fills a per-device info struct. Both trap.

### INTEGRITY_CHECK
cudart's check that it is talking to the genuine driver.
* **[1] `integrity_check(unsigned version, uint64_t unix_time, uint64_t out[2])`**, 0x3be540.
  * If `version % 10` is 0 or 1 (or a flag set by slot 2 is on), it writes a fixed 16-byte
    constant from `.rodata` 0x4016440. For `% 10 == 1`, byte 7 becomes 0x18.
  * Otherwise it writes **HMAC-MD2** (RFC 1319 MD2; 16-byte block = 16-byte key, pads 0x36/0x5c).
    The key is 16 bytes, de-obfuscated at runtime from four 16-byte `.rodata` tables (0x4016020).
    The message is:
    * `u32 0x2f30` (the driver's own CUDA version), `u32 version`, `u32 getpid()`,
      `u32 pthread_self()`
    * the addresses of the CUDART_INTERFACE table, the INTEGRITY_CHECK table and slot 1 itself
    * `u64 unix_time`
    * then, per device, `{uuid[16], u32 pci_domain, u32 pci_bus, u32 pci_device}` (device-struct
      offsets 0xc06, 0xc70, 0xc68)
  * The MD2 primitives are at 0x3aeb10 (update one byte), 0x3aeb20 (final) and 0x3aeaf0 (reset).
    The S-box at 0x4016060 is the standard PI_SUBST table.
  * cuBLAS's cudart (libcublas +0x546e88) calls it for 12080, 12081 and 12082 with the same time,
    recomputes the 12082 hash with its own MD2 (+0x5366c0) and fails `cublasCreate` on mismatch.
  * fakecuda reproduces it over its own table and function addresses and fake devices. The
    driver-version word must be the value `cuDriverGetVersion` returns (`FAKECUDA_CUDA_VERSION`).
    This was verified: hashing 12080 while reporting 12060 makes `cublasCreate` fail. ZLUDA has
    the same algorithm as "passes 1–7"; the key and S-box match byte for byte. **High (verified
    end to end).**
* [2], 0x3aeba0: sets the flag that forces the constant answer. Traps.

### 21318c60 (92 slots, unnamed)
* **[4] `get_ctx_id(CUcontext ctx, uint64_t *id)`**, 0x488000. Returns 999 if `id == NULL`.
  Otherwise `*id = 0`, and for a regular or green context, `*id = (u32) ctx->id` (offset 0x38).
  NULL ctx stays 0. cuBLASLt (+0x196e0bb) uses it as a per-context cache key. Our ids are
  `dev + 1` (made up, marked RANDOM RESULT). **High (semantics).**
* **[51] `get_stream_object(ctx, CUstream h, void **obj, bool per_thread)`**, 0x48c800. Returns
  1 if `obj == NULL` and 400 for `h == (CUstream)-1`. Real handles ignore ctx. The driver's
  stream resolver (0x2eee00) maps the handle to its internal stream object. For 0/1/2 that is the
  context's legacy or per-thread default stream. cuBLAS (+0x20a39b) passes the result only to
  [39]. **High.**
* **[39] `get_stream_id(ctx, void *obj, uint64_t *id)`**, 0x488090. Returns `obj->field_0x148`.
  The public `cuStreamGetId` (0x377a10) returns the same field, so it is the stream id. Slot 3 is
  [51] and [39] combined. fakecuda's ids come from its own stream counter (`fake_stream_id`).
  **High.**

### 1734dc26 (17 slots, unnamed)
* **[4] `get_device_blobs(void **a, void **b)`**, 0x225fc0. Needs a current context. Mallocs two
  buffers of a per-device size (+0xe94) and fills them through a per-GPU HAL hook
  (device+0x45d0); the caller owns them. **With no hook it returns 801.** cuBLASLt (+0x34ab8b2)
  handles any error or NULL result by caching "no data" and continuing. fakecuda returns 801, the
  same as a GPU without the hook. Contents and purpose of the blobs unknown. **High that 801 is a
  real outcome.** What cuBLASLt loses without them is unknown.

## Public driver API that cuBLAS needed

Beyond `cuGetExportTable`, `cublasCreate` and the GEMMs needed the following, all in
`procaddr.c`'s table and/or exported:

* `cuModuleGetLoadingMode`, `cuLibraryLoadData` / `GetModule` / `GetKernel`,
  `cuModuleGetFunction`, `cuKernelGetFunction`
* `cuFunc{Get,Set}Attribute`
* `cuOccupancyAvailableDynamicSMemPerBlock`, `cuOccupancyMaxActiveClusters`. cuBLAS probes a
  `nopClusterDiscoveryTool` kernel with cluster sizes 3..16 at full shared memory, i.e. it
  measures the GPC layout. fakecuda answers from an illustrative 2×18 + 6×16-SM layout.
* `cuLaunchKernelEx`: every GEMM launch, routed into `fake_launch`.
* full `cuDeviceGetAttribute` coverage (cudart builds `cudaDeviceProp` from ~130 attributes)
* the exported symbols cuBLASLt dlsyms (see above)

## Extending

Add a `struct dark_slot` array and a `dark_tables[]` row (name, UUID bytes, word count, whether
word 0 is a size, implementations) in `darkapi.c`. Put a comment above each implemented slot
giving the UUID, slot, libcuda address and decompiled behaviour, the caller and how it uses the
result, ZLUDA if any, and confidence. Write GUESS for anything inferred. Every slot left out of the
array traps.

## Open questions / risks

* **Version coupling.** Layouts are from driver 570.124.04 and validated only against cuBLAS
  12.8.4.1 / PyTorch 2.11. Other cuBLAS releases may want other slots or check table sizes
  differently. A newer cudart may require a bigger CUDART_INTERFACE (ZLUDA's newer driver has
  13). Re-run discovery mode after upgrading.
* `CONTEXT_CHECKS[2]` kind value, the hook-array element width, and the IDs we hand out are
  guesses that cuBLAS 12.8 does not distinguish.
* `1734dc26[4]`: cuBLASLt runs without the per-GPU blobs. The real driver on H100/H200 probably
  provides them. Kernel selection might differ from real hardware.
* fakecuda reports CUDA 12.6 (`cuDriverGetVersion` = 12060) while the tables mirror a 12.8
  driver. cuBLAS 12.8 accepts this. The integrity hash must use the reported value, so changing
  `FAKECUDA_CUDA_VERSION` is safe for the hash.
