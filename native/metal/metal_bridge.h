#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque context handle
typedef struct MetalContext MetalContext;

// Lifecycle
MetalContext* metal_create_context(void);
void metal_destroy_context(MetalContext* ctx);

// Memory — shared (UMA: CPU + GPU visible, zero-copy)
void* metal_alloc_shared(MetalContext* ctx, size_t bytes);
void metal_free_buffer(MetalContext* ctx, void* ptr);

// Compute operations (return 0 on success, non-zero on failure)
int metal_tensor_add(MetalContext* ctx,
                     const float* a, const float* b, float* result,
                     uint32_t length);

int metal_tensor_matmul(MetalContext* ctx,
                        const float* a, const float* b, float* c,
                        uint32_t m, uint32_t n, uint32_t k);

// Synchronization
void metal_synchronize(MetalContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // METAL_BRIDGE_H
