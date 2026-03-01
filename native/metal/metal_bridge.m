#import "metal_bridge.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>

// ─────────────────────────────────────────────────────────────────────────────
// Context: owns the Metal device, command queue, and compiled compute pipelines.
// One context per application lifetime is typical.
// ─────────────────────────────────────────────────────────────────────────────

struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> addPipeline;
};

// Track allocated buffers for cleanup.
// Maps raw pointer → MTLBuffer. Simple CFDictionary keyed on pointer value.
static NSMapTable* _bufferMap = nil;

static void register_buffer(void* ptr, id<MTLBuffer> buffer) {
    if (!_bufferMap)
        _bufferMap = [NSMapTable mapTableWithKeyOptions:NSPointerFunctionsOpaqueMemory | NSPointerFunctionsOpaquePersonality
                                          valueOptions:NSPointerFunctionsStrongMemory];
    [_bufferMap setObject:buffer forKey:(__bridge id)(void*)ptr];
}

static id<MTLBuffer> lookup_buffer(void* ptr) {
    return [_bufferMap objectForKey:(__bridge id)(void*)ptr];
}

static void unregister_buffer(void* ptr) {
    [_bufferMap removeObjectForKey:(__bridge id)(void*)ptr];
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedded Metal shader source for element-wise add.
// Compiled at context creation time — no external .metallib needed for basics.
// ─────────────────────────────────────────────────────────────────────────────

static NSString* const kShaderSource = @""
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "kernel void tensor_add(\n"
    "    device const float* a [[buffer(0)]],\n"
    "    device const float* b [[buffer(1)]],\n"
    "    device float* result  [[buffer(2)]],\n"
    "    uint id [[thread_position_in_grid]])\n"
    "{\n"
    "    result[id] = a[id] + b[id];\n"
    "}\n";

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

MetalContext* metal_create_context(void) {
    MetalContext* ctx = (MetalContext*)calloc(1, sizeof(MetalContext));
    if (!ctx) return NULL;

    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        free(ctx);
        return NULL;
    }

    ctx->queue = [ctx->device newCommandQueue];
    if (!ctx->queue) {
        free(ctx);
        return NULL;
    }

    // Compile the add kernel from source
    NSError* error = nil;
    id<MTLLibrary> library = [ctx->device newLibraryWithSource:kShaderSource
                                                       options:nil
                                                         error:&error];
    if (!library) {
        NSLog(@"Metal shader compilation failed: %@", error);
        free(ctx);
        return NULL;
    }

    id<MTLFunction> addFunction = [library newFunctionWithName:@"tensor_add"];
    if (addFunction) {
        ctx->addPipeline = [ctx->device newComputePipelineStateWithFunction:addFunction
                                                                     error:&error];
        if (!ctx->addPipeline) {
            NSLog(@"Failed to create add pipeline: %@", error);
        }
    }

    return ctx;
}

void metal_destroy_context(MetalContext* ctx) {
    if (!ctx) return;
    // ARC handles Objective-C object release
    ctx->device = nil;
    ctx->queue = nil;
    ctx->addPipeline = nil;
    free(ctx);
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory — Shared (UMA zero-copy)
// ─────────────────────────────────────────────────────────────────────────────

void* metal_alloc_shared(MetalContext* ctx, size_t bytes) {
    if (!ctx || !ctx->device || bytes == 0) return NULL;

    // StorageModeShared: CPU and GPU access the same physical memory on Apple Silicon.
    // This is the key UMA advantage — zero copy, no staging buffers.
    id<MTLBuffer> buffer = [ctx->device newBufferWithLength:bytes
                                                    options:MTLResourceStorageModeShared];
    if (!buffer) return NULL;

    void* ptr = [buffer contents];
    register_buffer(ptr, buffer);
    return ptr;
}

void metal_free_buffer(MetalContext* ctx, void* ptr) {
    if (!ptr) return;
    unregister_buffer(ptr);
    // ARC releases the MTLBuffer when removed from the map
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute: element-wise add
// ─────────────────────────────────────────────────────────────────────────────

int metal_tensor_add(MetalContext* ctx,
                     const float* a, const float* b, float* result,
                     uint32_t length) {
    if (!ctx || !ctx->addPipeline) return -1;

    id<MTLBuffer> bufA = lookup_buffer((void*)a);
    id<MTLBuffer> bufB = lookup_buffer((void*)b);
    id<MTLBuffer> bufR = lookup_buffer((void*)result);

    if (!bufA || !bufB || !bufR) return -2; // pointers not from metal_alloc_shared

    id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:ctx->addPipeline];
    [encoder setBuffer:bufA offset:0 atIndex:0];
    [encoder setBuffer:bufB offset:0 atIndex:1];
    [encoder setBuffer:bufR offset:0 atIndex:2];

    NSUInteger threadGroupSize = ctx->addPipeline.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > length) threadGroupSize = length;

    MTLSize gridSize = MTLSizeMake(length, 1, 1);
    MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
    [encoder endEncoding];

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    return (cmdBuf.error != nil) ? -3 : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute: matrix multiplication via MPS
// ─────────────────────────────────────────────────────────────────────────────

int metal_tensor_matmul(MetalContext* ctx,
                        const float* a, const float* b, float* c,
                        uint32_t m, uint32_t n, uint32_t k) {
    if (!ctx || !ctx->device || !ctx->queue) return -1;

    id<MTLBuffer> bufA = lookup_buffer((void*)a);
    id<MTLBuffer> bufB = lookup_buffer((void*)b);
    id<MTLBuffer> bufC = lookup_buffer((void*)c);

    if (!bufA || !bufB || !bufC) return -2;

    // Use Metal Performance Shaders for optimized matmul.
    // MPS uses row-major layout by default for matrix descriptors.
    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                                       columns:k
                                                                      rowBytes:k * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:k
                                                                       columns:n
                                                                      rowBytes:n * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                                       columns:n
                                                                      rowBytes:n * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    MPSMatrixMultiplication* matmul =
        [[MPSMatrixMultiplication alloc] initWithDevice:ctx->device
                                         transposeLeft:NO
                                        transposeRight:NO
                                            resultRows:m
                                         resultColumns:n
                                       interiorColumns:k
                                                 alpha:1.0
                                                  beta:0.0];

    id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
    [matmul encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    return (cmdBuf.error != nil) ? -3 : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Synchronization
// ─────────────────────────────────────────────────────────────────────────────

void metal_synchronize(MetalContext* ctx) {
    if (!ctx || !ctx->queue) return;

    // Submit an empty command buffer and wait — ensures all prior work is complete.
    id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}
