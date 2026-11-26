#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct cublasContext *cublasHandle_t;
typedef int cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_OP_N 0
cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, int transa, int transb, int m, int n, int k,
                              const float *alpha, const float *A, int lda, const float *B, int ldb,
                              const float *beta, float *C, int ldc);
#define cublasCreate cublasCreate_v2
#define cublasDestroy cublasDestroy_v2
#define cublasSgemm cublasSgemm_v2

#define N 256

#define CK(call) do { cudaError_t e_ = (call); \
  if (e_ != cudaSuccess) { fprintf(stderr, "FAIL %s = %d\n", #call, e_); return 1; } \
  printf("ok   %s\n", #call); } while (0)
#define CB(call) do { cublasStatus_t s_ = (call); \
  if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "FAIL %s = %d\n", #call, s_); return 1; } \
  printf("ok   %s\n", #call); } while (0)

int main(void) {
  size_t bytes = (size_t)N * N * sizeof(float);
  float *h = malloc(bytes), *a, *b, *c;
  for (int i = 0; i < N * N; i++) h[i] = (float)(i % 7);
  CK(cudaMalloc((void **)&a, bytes));
  CK(cudaMalloc((void **)&b, bytes));
  CK(cudaMalloc((void **)&c, bytes));
  CK(cudaMemcpy(a, h, bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(b, h, bytes, cudaMemcpyHostToDevice));
  cublasHandle_t handle;
  CB(cublasCreate(&handle));
  float alpha = 1.0f, beta = 0.0f;
  CB(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, a, N, b, N, &beta, c, N));
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, c, bytes, cudaMemcpyDeviceToHost));
  printf("C[0]=%g C[last]=%g (not checked: fake GPU)\n", h[0], h[N * N - 1]);
  CB(cublasDestroy(handle));
  CK(cudaFree(a));
  CK(cudaFree(b));
  CK(cudaFree(c));
  free(h);
  printf("PASS test_sgemm\n");
  return 0;
}
