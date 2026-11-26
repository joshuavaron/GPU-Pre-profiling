import torch
import torch.nn.functional as F


def run(name, fn):
    out = fn()
    torch.cuda.synchronize()
    host = out.cpu()
    print(f"ok   {name}: {tuple(host.shape)} {host.dtype}")


a = torch.randn(256, 256, device="cuda")
b = torch.randn(256, 256, device="cuda")
run("fp32 a @ b (cublasSgemm)", lambda: a @ b)
run("bmm", lambda: torch.bmm(a.expand(4, 256, 256), b.expand(4, 256, 256)))
run("fp16 matmul (cublasGemmEx)", lambda: a.half() @ b.half())
run("bf16 matmul", lambda: a.bfloat16() @ b.bfloat16())
bias = torch.randn(256, device="cuda")
run("linear + bias (cublasLt gemm_and_bias)", lambda: F.linear(a, b, bias))
torch.backends.cuda.preferred_blas_library("cublaslt")
run("fp32 a @ b, preferred_blas_library=cublaslt", lambda: a @ b)
run("fp16 a @ b, preferred_blas_library=cublaslt", lambda: a.half() @ b.half())
print("PASS test_matmul")
