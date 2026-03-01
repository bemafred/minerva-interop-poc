#ifndef CUDA_BRIDGE_H
#define CUDA_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * CUDA bridge for custom kernels.
 *
 * NOTE: For the POC, we don't need this. All GPU operations are handled
 * through direct P/Invoke to cudart and cublas shared libraries.
 *
 * This bridge becomes relevant when we need custom kernels that go beyond
 * what cuBLAS provides (e.g., fused attention, custom activation functions,
 * quantized operations for Minerva inference).
 *
 * When that time comes:
 *   1. Write kernels in .cu files
 *   2. Wrap __global__ functions in C host functions here
 *   3. Compile: nvcc --shared -o libcuda_bridge.so cuda_bridge.cu
 *   4. P/Invoke from C# via the same NativeResolver mechanism
 */

// Future: custom tensor element-wise add (when cuBLAS saxpy isn't sufficient)
int cuda_tensor_add(const float* a, const float* b, float* result,
                    uint32_t length);

// Future: fused operations for inference
int cuda_tensor_relu(const float* input, float* output, uint32_t length);
int cuda_tensor_softmax(const float* input, float* output,
                        uint32_t rows, uint32_t cols);

#ifdef __cplusplus
}
#endif

#endif // CUDA_BRIDGE_H
