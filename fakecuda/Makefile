SP ?= ../.venv/lib/python3.12/site-packages/nvidia
CUDA_INC ?= -I$(SP)/cuda_runtime/include -I$(SP)/cuda_nvcc/include
CUDART ?= $(SP)/cuda_runtime/lib/libcudart.so.12
CUBLAS_LIB ?= $(SP)/cublas/lib
PYTHON ?= ../.venv/bin/python
SRCS = device.c runtime.c driver.c nvml.c memory.c launch.c stream.c stubs.c runtime_stubs.c \
       darkapi.c procaddr.c
TESTS = tests/test_darkapi tests/test_sgemm

all: libfakecuda.so lib/libnvidia-ml.so.1

libfakecuda.so: $(SRCS) fakecuda.h libcudart.map
	gcc -O2 -Wall -shared -fPIC $(CUDA_INC) -Wl,--version-script=libcudart.map,-soname,libcuda.so.1 -o $@ $(SRCS) -lpthread -ldl

runtime_stubs.c: tools/gen_runtime_stubs.sh
	sh tools/gen_runtime_stubs.sh $(CUDART) > $@

lib/libnvidia-ml.so.1:
	mkdir -p lib
	ln -sf ../libfakecuda.so $@

tests/test_darkapi: tests/test_darkapi.c
	gcc -O2 -Wall -o $@ $< -ldl

tests/test_sgemm: tests/test_sgemm.c
	gcc -O2 -Wall $(CUDA_INC) -o $@ $< $(CUBLAS_LIB)/libcublas.so.12 $(CUDART) \
	  -Wl,-rpath,$(abspath $(CUBLAS_LIB)):$(abspath $(dir $(CUDART)))

test: all $(TESTS)
	./run ./tests/test_darkapi
	./run ./tests/test_sgemm
	./run $(PYTHON) tests/test_matmul.py
	./run $(PYTHON) demo.py

clean:
	rm -rf *.so lib $(TESTS) runtime_stubs.c

.PHONY: all test clean
