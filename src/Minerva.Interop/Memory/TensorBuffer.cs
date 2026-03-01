using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Minerva.Interop.Memory;

/// <summary>
/// Describes where a tensor buffer physically resides and how it can be accessed.
/// </summary>
public enum MemoryResidence
{
    /// <summary>CPU-only allocation (NativeMemory). Always CPU-readable/writable.</summary>
    Cpu,

    /// <summary>Unified memory (macOS UMA). CPU and GPU share the same physical pages.</summary>
    Unified,

    /// <summary>GPU-only allocation (CUDA device memory). Not directly CPU-accessible.</summary>
    Device,

    /// <summary>Paired allocation: CPU staging buffer + GPU device buffer.</summary>
    Staged
}

/// <summary>
/// A contiguous block of unmanaged memory holding a 2D tensor of <typeparamref name="T"/>.
/// Abstracts over CPU, unified (UMA), and discrete GPU memory.
///
/// This is the type whose viability the POC exists to test. The hypothesis:
/// a single type can span UMA zero-copy and discrete staged-copy without
/// leaking the memory topology into calling code.
/// </summary>
public sealed unsafe class TensorBuffer<T> : IDisposable
    where T : unmanaged
{
    private readonly MemoryResidence _residence;
    private readonly int _rows;
    private readonly int _cols;
    private readonly int _length;

    // CPU-visible pointer (always set for Cpu/Unified/Staged; null for Device-only)
    private T* _cpuPtr;

    // GPU-visible pointer (set for Unified/Device/Staged; null for Cpu-only)
    private void* _gpuPtr;

    // Disposal tracking
    private readonly Action<TensorBuffer<T>>? _disposer;
    private bool _disposed;

    internal TensorBuffer(
        int rows, int cols,
        T* cpuPtr, void* gpuPtr,
        MemoryResidence residence,
        Action<TensorBuffer<T>>? disposer)
    {
        _rows = rows;
        _cols = cols;
        _length = rows * cols;
        _cpuPtr = cpuPtr;
        _gpuPtr = gpuPtr;
        _residence = residence;
        _disposer = disposer;
    }

    public int Rows => _rows;
    public int Cols => _cols;
    public int Length => _length;
    public MemoryResidence Residence => _residence;

    /// <summary>
    /// Size in bytes of the tensor data.
    /// </summary>
    public nuint ByteLength => (nuint)(_length * Unsafe.SizeOf<T>());

    /// <summary>
    /// Returns a Span over the CPU-visible memory.
    ///
    /// For Unified: this IS the GPU memory (zero-copy).
    /// For Staged: this is the staging buffer (must sync after GPU compute).
    /// For Device-only: throws — no CPU-visible memory exists.
    /// </summary>
    public Span<T> AsSpan()
    {
        ThrowIfDisposed();

        if (_cpuPtr == null)
            throw new InvalidOperationException(
                $"TensorBuffer with residence {_residence} has no CPU-visible memory. " +
                "Use SyncFromDevice() on a Staged buffer, or allocate with CPU visibility.");

        return new Span<T>(_cpuPtr, _length);
    }

    /// <summary>
    /// Raw pointer to CPU-visible memory. Null if Device-only.
    /// </summary>
    public T* CpuPointer
    {
        get { ThrowIfDisposed(); return _cpuPtr; }
    }

    /// <summary>
    /// Raw pointer to GPU-visible memory. Null if CPU-only.
    /// For Unified: same physical memory as CpuPointer.
    /// For Device/Staged: GPU VRAM address (not dereferenceable from CPU).
    /// </summary>
    public void* GpuPointer
    {
        get { ThrowIfDisposed(); return _gpuPtr; }
    }

    /// <summary>
    /// Whether the CPU can directly read/write this buffer without a copy.
    /// True for Cpu and Unified. False for Device. True for Staged (via staging buffer).
    /// </summary>
    public bool IsCpuAccessible => _cpuPtr != null;

    /// <summary>
    /// Whether this buffer can be passed to GPU compute kernels.
    /// </summary>
    public bool IsGpuAccessible => _gpuPtr != null;

    public void Dispose()
    {
        if (_disposed) return;
        _disposer?.Invoke(this);
        _disposed = true;
        _cpuPtr = null;
        _gpuPtr = null;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(TensorBuffer<T>));
    }
}
