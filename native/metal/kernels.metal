#include <metal_stdlib>
using namespace metal;

/// Element-wise addition: result[i] = a[i] + b[i]
kernel void tensor_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    result[id] = a[id] + b[id];
}

/// Element-wise multiplication: result[i] = a[i] * b[i]
kernel void tensor_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    result[id] = a[id] * b[id];
}

/// ReLU activation: result[i] = max(0, a[i])
kernel void tensor_relu(
    device const float* a [[buffer(0)]],
    device float* result  [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    result[id] = max(0.0f, a[id]);
}
