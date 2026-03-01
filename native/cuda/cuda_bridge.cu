/*
 * CUDA kernels for tensor operations.
 *
 * POC STATUS: Not yet needed. cuBLAS provides sgemm and saxpy.
 * These stubs exist as a scaffold for when the Emergence phase
 * discovers operations that require custom kernels.
 *
 * Build: nvcc --shared -o libcuda_bridge.so cuda_bridge.cu
 */

#include "cuda_bridge.h"
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: element-wise addition
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_tensor_add(const float* a, const float* b, float* result, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        result[idx] = a[idx] + b[idx];
}

extern "C" int cuda_tensor_add(const float* a, const float* b, float* result, uint32_t length) {
    int blockSize = 256;
    int gridSize = (length + blockSize - 1) / blockSize;

    kernel_tensor_add<<<gridSize, blockSize>>>(a, b, result, length);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: ReLU activation
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_relu(const float* input, float* output, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] = fmaxf(0.0f, input[idx]);
}

extern "C" int cuda_tensor_relu(const float* input, float* output, uint32_t length) {
    int blockSize = 256;
    int gridSize = (length + blockSize - 1) / blockSize;

    kernel_relu<<<gridSize, blockSize>>>(input, output, length);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: softmax (naive per-row implementation — replace with warp-reduce later)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_softmax(const float* input, float* output, uint32_t rows, uint32_t cols) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // Find max for numerical stability
    float max_val = in_row[0];
    for (uint32_t j = 1; j < cols; j++)
        max_val = fmaxf(max_val, in_row[j]);

    // Compute exp and sum
    float sum = 0.0f;
    for (uint32_t j = 0; j < cols; j++) {
        out_row[j] = expf(in_row[j] - max_val);
        sum += out_row[j];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint32_t j = 0; j < cols; j++)
        out_row[j] *= inv_sum;
}

extern "C" int cuda_tensor_softmax(const float* input, float* output,
                                    uint32_t rows, uint32_t cols) {
    int blockSize = 256;
    int gridSize = (rows + blockSize - 1) / blockSize;

    kernel_softmax<<<gridSize, blockSize>>>(input, output, rows, cols);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}
