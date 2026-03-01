# CUDA bridge build — Linux/Windows
# Requires: CUDA Toolkit (nvcc)
#
# NOTE: Not needed for POC. All GPU ops use cuBLAS via direct P/Invoke.
# Build this when custom kernels are added.
#
# Usage:
#   make            — build shared library
#   make clean      — remove artifacts

NVCC = nvcc
NVFLAGS = --shared --compiler-options '-fPIC' -O2

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    TARGET = libcuda_bridge.so
else
    TARGET = cuda_bridge.dll
endif

.PHONY: all clean

all: $(TARGET)

$(TARGET): cuda_bridge.cu cuda_bridge.h
	$(NVCC) $(NVFLAGS) -o $(TARGET) cuda_bridge.cu

clean:
	rm -f $(TARGET)
