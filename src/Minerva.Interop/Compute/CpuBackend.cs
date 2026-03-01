using Minerva.Interop.Bindings;
using Minerva.Interop.Memory;
using Minerva.Interop.Platform;

namespace Minerva.Interop.Compute;

/// <summary>
/// CPU-only compute backend using CBLAS (Accelerate or OpenBLAS).
/// Available on all platforms. Used as both fallback and correctness reference.
/// </summary>
internal sealed class CpuBackend : IComputeBackend
{
    private readonly ComputeCapability _capability;

    public CpuBackend(ComputeCapability capability)
    {
        _capability = capability;
    }

    public string Name => $"CPU BLAS ({_capability.BlasProvider})";
    public MemoryResidence DefaultResidence => MemoryResidence.Cpu;

    public TensorBuffer<float> Allocate(int rows, int cols)
        => CpuMemoryAllocator.Allocate<float>(rows, cols);

    public void SyncToDevice(TensorBuffer<float> buffer) { /* no-op: CPU only */ }
    public void SyncFromDevice(TensorBuffer<float> buffer) { /* no-op: CPU only */ }
    public void Synchronize() { /* no-op: synchronous execution */ }

    public unsafe void MatMul(TensorBuffer<float> a, TensorBuffer<float> b, TensorBuffer<float> c)
    {
        // A [m×k] × B [k×n] = C [m×n], row-major
        int m = a.Rows;
        int k = a.Cols;
        int n = b.Cols;

        if (b.Rows != k || c.Rows != m || c.Cols != n)
            throw new ArgumentException(
                $"Dimension mismatch: A[{a.Rows}×{a.Cols}] × B[{b.Rows}×{b.Cols}] → C[{c.Rows}×{c.Cols}]");

        BlasBindings.Sgemm(
            BlasBindings.CblasOrder.RowMajor,
            BlasBindings.CblasTranspose.NoTrans,
            BlasBindings.CblasTranspose.NoTrans,
            m, n, k,
            alpha: 1.0f,
            a: a.CpuPointer, lda: k,
            b: b.CpuPointer, ldb: n,
            beta: 0.0f,
            c: c.CpuPointer, ldc: n);
    }

    public unsafe void Add(TensorBuffer<float> a, TensorBuffer<float> b, TensorBuffer<float> c)
    {
        if (a.Length != b.Length || a.Length != c.Length)
            throw new ArgumentException("Dimension mismatch for element-wise add.");

        // Copy b → c, then c = 1.0*a + c (saxpy)
        var srcSpan = b.AsSpan();
        var dstSpan = c.AsSpan();
        srcSpan.CopyTo(dstSpan);

        BlasBindings.Saxpy(
            a.Length,
            alpha: 1.0f,
            x: a.CpuPointer, incX: 1,
            y: c.CpuPointer, incY: 1);
    }

    public void Dispose() { /* nothing to dispose for CPU backend */ }
}
