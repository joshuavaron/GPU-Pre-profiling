import torch
import torch.nn.functional as F

print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
print("cuda.is_available():", torch.cuda.is_available())
print("cuda.device_count(): ", torch.cuda.device_count())

print("\n== Devices")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f"  [{i}] {p.name}  sm_{p.major}{p.minor}  {p.total_memory / 2**30:.1f} GiB  "
          f"{p.multi_processor_count} SMs  free {free / 2**30:.1f} GiB")

print("\n== Memory")
x = torch.arange(1_000_000, dtype=torch.float32)
d = x.to("cuda:0")
print("  host -> cuda:0 -> host matches:", torch.equal(d.cpu(), x))
print("  allocated on cuda:0:", torch.cuda.memory_allocated(0), "bytes")


print("\n== Kernels")
z = d + 1
torch.cuda.synchronize()
print("  (x + 1)[:5] =", z[:5].cpu().tolist())

print("\n== Matrix multiplication")
a = torch.randn(256, 256, device="cuda")
b = torch.randn(256, 256, device="cuda")
bias = torch.randn(256, device="cuda")

c = a @ b
print("  fp32 a @ b:", c.shape, c.dtype)

c = torch.bmm(a.expand(4, 256, 256), b.expand(4, 256, 256))
print("  bmm:", c.shape, c.dtype)

c = a.half() @ b.half()
print("  fp16 a @ b:", c.shape, c.dtype)

c = a.bfloat16() @ b.bfloat16()
print("  bf16 a @ b:", c.shape, c.dtype)

c = F.linear(a, b, bias)
print("  linear + bias:", c.shape, c.dtype)

torch.backends.cuda.preferred_blas_library("cublaslt")
c = a @ b
print("  fp32 a @ b (cuBLASLt):", c.shape, c.dtype)

torch.cuda.synchronize()
