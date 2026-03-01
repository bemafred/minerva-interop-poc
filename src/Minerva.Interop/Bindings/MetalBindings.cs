using System.Runtime.InteropServices;

namespace Minerva.Interop.Bindings;

/// <summary>
/// Bindings to the Metal C bridge (metal_bridge.dylib).
/// macOS only. The bridge wraps Objective-C Metal API as plain C functions.
/// </summary>
internal static partial class MetalBindings
{
    [LibraryImport("tensor_gpu", EntryPoint = "metal_create_context")]
    internal static partial nint CreateContext();

    [LibraryImport("tensor_gpu", EntryPoint = "metal_destroy_context")]
    internal static partial void DestroyContext(nint ctx);

    /// <summary>
    /// Allocate shared memory buffer (CPU + GPU visible on UMA).
    /// Returns a CPU-dereferenceable pointer that is also GPU-accessible.
    /// </summary>
    [LibraryImport("tensor_gpu", EntryPoint = "metal_alloc_shared")]
    internal static unsafe partial void* AllocShared(nint ctx, nuint bytes);

    /// <summary>
    /// Free a shared memory buffer.
    /// </summary>
    [LibraryImport("tensor_gpu", EntryPoint = "metal_free_buffer")]
    internal static unsafe partial void FreeBuffer(nint ctx, void* ptr);

    /// <summary>
    /// Element-wise tensor addition: out[i] = a[i] + b[i]
    /// Dispatched as a Metal compute shader.
    /// </summary>
    [LibraryImport("tensor_gpu", EntryPoint = "metal_tensor_add")]
    internal static unsafe partial int TensorAdd(
        nint ctx, float* a, float* b, float* result, uint length);

    /// <summary>
    /// Matrix multiplication: C = A × B
    /// Uses Metal Performance Shaders (MPS) or custom compute kernel.
    /// </summary>
    [LibraryImport("tensor_gpu", EntryPoint = "metal_tensor_matmul")]
    internal static unsafe partial int TensorMatmul(
        nint ctx,
        float* a, float* b, float* c,
        uint m, uint n, uint k);

    /// <summary>
    /// Waits for all submitted GPU work to complete.
    /// </summary>
    [LibraryImport("tensor_gpu", EntryPoint = "metal_synchronize")]
    internal static partial void Synchronize(nint ctx);
}
