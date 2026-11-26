"""Probe real CUDA 12 runtime and cuBLAS over fakecuda (Linux, CPU-only).

Usage: ./run python tools/probe_cublas.py /path/to/site-packages/nvidia
Kernel results are not validated: fakecuda does not execute GPU instructions.
"""
import ctypes as C
import sys
from pathlib import Path

root = Path(sys.argv[1])
runtime = C.CDLL(str(root / "cuda_runtime/lib/libcudart.so.12"))
blas = C.CDLL(str(root / "cublas/lib/libcublas.so.12"))

def bind(lib, name, args):
    fn = getattr(lib, name)
    fn.argtypes = args
    fn.restype = C.c_int
    return fn

def check(label, status):
    print(f"{label}: status={status}", flush=True)
    if status:
        raise SystemExit(1)

pointer = C.c_void_p
create = bind(blas, "cublasCreate_v2", [C.POINTER(pointer)])
destroy = bind(blas, "cublasDestroy_v2", [pointer])
allocate = bind(runtime, "cudaMalloc", [C.POINTER(pointer), C.c_size_t])
release = bind(runtime, "cudaFree", [pointer])
sync = bind(runtime, "cudaDeviceSynchronize", [])
gemm = bind(blas, "cublasSgemm_v2", [pointer, C.c_int, C.c_int,
    C.c_int, C.c_int, C.c_int, C.POINTER(C.c_float), pointer,
    C.c_int, pointer, C.c_int, C.POINTER(C.c_float), pointer, C.c_int])

handle = pointer()
print("Calling real cublasCreate_v2", flush=True)
check("cublasCreate_v2", create(C.byref(handle)))
allocated = []
try:
    for label in ("A", "B", "C"):
        p = pointer()
        check(f"cudaMalloc({label}, 16)", allocate(C.byref(p), 16))
        allocated.append(p)
    alpha, beta = C.c_float(1), C.c_float(0)
    print("Calling real cublasSgemm_v2: 2x2 float matrices", flush=True)
    check("cublasSgemm_v2", gemm(handle, 0, 0, 2, 2, 2,
        C.byref(alpha), allocated[0], 2, allocated[1], 2,
        C.byref(beta), allocated[2], 2))
    check("cudaDeviceSynchronize", sync())
finally:
    for p in allocated:
        print("cudaFree: status=", release(p), flush=True)
    print("cublasDestroy_v2: status=", destroy(handle), flush=True)
