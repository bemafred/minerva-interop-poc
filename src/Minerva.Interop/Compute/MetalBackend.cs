using Minerva.Interop.Bindings;
using Minerva.Interop.Memory;
using Minerva.Interop.Platform;

namespace Minerva.Interop.Compute;

/// <summary>
/// Metal compute backend for macOS Apple Silicon.
/// Uses unified memory — allocations are CPU and GPU visible simultaneously.
/// No explicit transfers needed.
/// </summary>
internal sealed class MetalBackend : IComputeBackend
{
    private nint _context;
    private bool _disposed;

    public MetalBackend(ComputeCapability capability)
    {
        _context = MetalBindings.CreateContext();
        if (_context == 0)
            throw new InvalidOperationException("Failed to create Metal context.");
    }

    public string Name => "Metal (Apple Silicon)";
    public MemoryResidence DefaultResidence => MemoryResidence.Unified;

    public unsafe TensorBuffer<float> Allocate(int rows, int cols)
    {
        ThrowIfDisposed();

        int length = rows * cols;
        nuint bytes = (nuint)(length * sizeof(float));

        void* ptr = MetalBindings.AllocShared(_context, bytes);
        if (ptr == null)
            throw new OutOfMemoryException($"Metal shared allocation failed for {bytes} bytes.");

        // UMA: the returned pointer IS both the CPU pointer and GPU pointer.
        // Same physical memory, zero copy.
        var ctx = _context;
        return new TensorBuffer<float>(
            rows, cols,
            cpuPtr: (float*)ptr,
            gpuPtr: ptr,
            residence: MemoryResidence.Unified,
            disposer: buffer =>
            {
                if (buffer.GpuPointer != null)
                    MetalBindings.FreeBuffer(ctx, buffer.GpuPointer);
            });
    }

    public void SyncToDevice(TensorBuffer<float> buffer)
    {
        // UMA: no-op. CPU writes are immediately visible to GPU.
    }

    public void SyncFromDevice(TensorBuffer<float> buffer)
    {
        // UMA: no-op. GPU writes are immediately visible to CPU
        // (after synchronization ensures compute has completed).
    }

    public unsafe void MatMul(TensorBuffer<float> a, TensorBuffer<float> b, TensorBuffer<float> c)
    {
        ThrowIfDisposed();

        int m = a.Rows;
        int k = a.Cols;
        int n = b.Cols;

        if (b.Rows != k || c.Rows != m || c.Cols != n)
            throw new ArgumentException(
                $"Dimension mismatch: A[{a.Rows}×{a.Cols}] × B[{b.Rows}×{b.Cols}] → C[{c.Rows}×{c.Cols}]");

        int result = MetalBindings.TensorMatmul(
            _context,
            (float*)a.GpuPointer, (float*)b.GpuPointer, (float*)c.GpuPointer,
            (uint)m, (uint)n, (uint)k);

        if (result != 0)
            throw new InvalidOperationException($"Metal matmul failed with code {result}");
    }

    public unsafe void Add(TensorBuffer<float> a, TensorBuffer<float> b, TensorBuffer<float> c)
    {
        ThrowIfDisposed();

        if (a.Length != b.Length || a.Length != c.Length)
            throw new ArgumentException("Dimension mismatch for element-wise add.");

        int result = MetalBindings.TensorAdd(
            _context,
            (float*)a.GpuPointer, (float*)b.GpuPointer, (float*)c.GpuPointer,
            (uint)a.Length);

        if (result != 0)
            throw new InvalidOperationException($"Metal tensor add failed with code {result}");
    }

    public void Synchronize()
    {
        ThrowIfDisposed();
        MetalBindings.Synchronize(_context);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_context != 0)
        {
            MetalBindings.DestroyContext(_context);
            _context = 0;
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(MetalBackend));
    }
}
