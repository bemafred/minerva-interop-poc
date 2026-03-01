using Minerva.Interop.Memory;
using Minerva.Interop.Platform;

namespace Minerva.Interop.Compute;

/// <summary>
/// Contract for a compute backend. Each implementation owns:
/// - Memory allocation (in the appropriate address space)
/// - Tensor operations (dispatched to the appropriate hardware)
/// - Synchronization (waiting for async compute to complete)
///
/// Disposed when the backend is no longer needed — releases device contexts,
/// handles, command queues, etc.
/// </summary>
public interface IComputeBackend : IDisposable
{
    /// <summary>
    /// Human-readable name for diagnostics: "Metal (Apple M2 Max)", "CUDA (RTX 4090)", "CPU BLAS (OpenBLAS)"
    /// </summary>
    string Name { get; }

    /// <summary>
    /// The memory residence model this backend uses for tensor allocations.
    /// </summary>
    MemoryResidence DefaultResidence { get; }

    /// <summary>
    /// Allocate a tensor buffer suitable for this backend's compute operations.
    /// </summary>
    TensorBuffer<float> Allocate(int rows, int cols);

    /// <summary>
    /// If the backend uses staged memory (discrete GPU), copies CPU→GPU.
    /// No-op for unified memory and CPU-only backends.
    /// </summary>
    void SyncToDevice(TensorBuffer<float> buffer);

    /// <summary>
    /// If the backend uses staged memory, copies GPU→CPU.
    /// No-op for unified memory and CPU-only backends.
    /// </summary>
    void SyncFromDevice(TensorBuffer<float> buffer);

    /// <summary>
    /// C = A × B (matrix multiplication)
    /// A is [m × k], B is [k × n], C is [m × n]. Row-major.
    /// </summary>
    void MatMul(TensorBuffer<float> a, TensorBuffer<float> b, TensorBuffer<float> c);

    /// <summary>
    /// C = A + B (element-wise addition)
    /// All buffers must have the same dimensions.
    /// </summary>
    void Add(TensorBuffer<float> a, TensorBuffer<float> b, TensorBuffer<float> c);

    /// <summary>
    /// Waits for all pending compute operations to complete.
    /// </summary>
    void Synchronize();
}

/// <summary>
/// Factory for creating the best available backend.
/// </summary>
public static class ComputeBackend
{
    /// <summary>
    /// Detect platform capabilities and create the best available backend.
    /// Falls back to CPU BLAS if no GPU is available.
    /// </summary>
    public static IComputeBackend Create()
    {
        var cap = PlatformDetector.Detect();
        NativeResolver.Register(cap);
        return Create(cap);
    }

    /// <summary>
    /// Create a backend from pre-detected capabilities. Useful for testing
    /// or forcing a specific backend.
    /// </summary>
    public static IComputeBackend Create(ComputeCapability capability)
    {
        // GPU path first
        if (capability.GpuProvider == GpuProvider.Metal)
            return new MetalBackend(capability);

        if (capability.GpuProvider == GpuProvider.Cuda)
            return new CudaBackend(capability);

        // CPU fallback
        if (capability.BlasProvider != CpuBlasProvider.None)
            return new CpuBackend(capability);

        throw new PlatformNotSupportedException(
            $"No compute backend available. Detected: {capability}. " +
            "Ensure Accelerate (macOS), OpenBLAS (Linux/Windows), or CUDA is installed.");
    }

    /// <summary>
    /// Force creation of the CPU BLAS backend, even if GPU is available.
    /// Useful for correctness verification (comparing GPU results against CPU).
    /// </summary>
    public static IComputeBackend CreateCpuOnly()
    {
        var cap = PlatformDetector.Detect();
        NativeResolver.Register(cap);

        if (cap.BlasProvider == CpuBlasProvider.None)
            throw new PlatformNotSupportedException("No CPU BLAS library found.");

        return new CpuBackend(cap);
    }
}
