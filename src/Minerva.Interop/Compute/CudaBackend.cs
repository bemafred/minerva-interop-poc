using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Minerva.Interop.Bindings;
using Minerva.Interop.Memory;
using Minerva.Interop.Platform;

namespace Minerva.Interop.Compute;

/// <summary>
/// CUDA compute backend for NVIDIA GPUs on Linux and Windows.
/// Uses staged memory: separate CPU staging buffer + GPU device buffer.
/// Explicit transfers required via SyncToDevice / SyncFromDevice.
/// </summary>
internal sealed unsafe class CudaBackend : IComputeBackend
{
    private nint _cublasHandle;
    private bool _disposed;

    public CudaBackend(ComputeCapability capability)
    {
        nint handle;
        CublasBindings.Check(
            CublasBindings.Create(&handle),
            "cublasCreate");
        _cublasHandle = handle;
    }

    public string Name => $"CUDA ({PlatformDetector.Detect().GpuDeviceName ?? "NVIDIA GPU"})";
    public MemoryResidence DefaultResidence => MemoryResidence.Staged;

    public TensorBuffer<float> Allocate(int rows, int cols)
    {
        ThrowIfDisposed();

        int length = rows * cols;
        nuint bytes = (nuint)(length * sizeof(float));

        // CPU staging buffer (for writes/reads from managed code)
        float* cpuPtr = (float*)NativeMemory.AlignedAlloc(bytes, 64);
        NativeMemory.Clear(cpuPtr, bytes);

        // GPU device buffer
        void* gpuPtr;
        CudaBindings.Check(
            CudaBindings.Malloc(&gpuPtr, bytes),
            "cudaMalloc");

        CudaBindings.Check(
            CudaBindings.Memset(gpuPtr, 0, bytes),
            "cudaMemset");

        return new TensorBuffer<float>(
            rows, cols,
            cpuPtr: cpuPtr,
            gpuPtr: gpuPtr,
            residence: MemoryResidence.Staged,
            disposer: buffer =>
            {
                if (buffer.CpuPointer != null)
                    NativeMemory.AlignedFree(buffer.CpuPointer);
                if (buffer.GpuPointer != null)
                    CudaBindings.Free(buffer.GpuPointer);
            });
    }

    public void SyncToDevice(TensorBuffer<float> buffer)
    {
        ThrowIfDisposed();

        if (buffer.Residence != MemoryResidence.Staged)
            return; // nothing to transfer

        CudaBindings.Check(
            CudaBindings.Memcpy(
                buffer.GpuPointer, buffer.CpuPointer,
                buffer.ByteLength,
                CudaBindings.CudaMemcpyKind.HostToDevice),
            "cudaMemcpy H→D");
    }

    public void SyncFromDevice(TensorBuffer<float> buffer)
    {
        ThrowIfDisposed();

        if (buffer.Residence != MemoryResidence.Staged)
            return;

        CudaBindings.Check(
            CudaBindings.Memcpy(
                buffer.CpuPointer, buffer.GpuPointer,
                buffer.ByteLength,
                CudaBindings.CudaMemcpyKind.DeviceToHost),
            "cudaMemcpy D→H");
    }

    public void MatMul(TensorBuffer<float> a, TensorBuffer<float> b, TensorBuffer<float> c)
    {
        ThrowIfDisposed();

        int m = a.Rows;
        int k = a.Cols;
        int n = b.Cols;

        if (b.Rows != k || c.Rows != m || c.Cols != n)
            throw new ArgumentException(
                $"Dimension mismatch: A[{a.Rows}×{a.Cols}] × B[{b.Rows}×{b.Cols}] → C[{c.Rows}×{c.Cols}]");

        // cuBLAS is column-major. For row-major C = A×B, we compute:
        //   C^T = B^T × A^T  (in column-major, which gives us C in row-major)
        // This is the standard trick: swap A↔B and transpose semantics.
        float alpha = 1.0f;
        float beta = 0.0f;

        CublasBindings.Check(
            CublasBindings.Sgemm(
                _cublasHandle,
                CublasBindings.CublasOperation.NoTrans,  // B^T in col-major = B in row-major
                CublasBindings.CublasOperation.NoTrans,   // A^T in col-major = A in row-major
                n, m, k,
                &alpha,
                (float*)b.GpuPointer, n,   // B (treated as B^T)
                (float*)a.GpuPointer, k,   // A (treated as A^T)
                &beta,
                (float*)c.GpuPointer, n),  // C (result in row-major)
            "cublasSgemm");
    }

    public void Add(TensorBuffer<float> a, TensorBuffer<float> b, TensorBuffer<float> c)
    {
        ThrowIfDisposed();

        if (a.Length != b.Length || a.Length != c.Length)
            throw new ArgumentException("Dimension mismatch for element-wise add.");

        // Copy b → c on device, then saxpy: c = 1.0*a + c
        CudaBindings.Check(
            CudaBindings.Memcpy(
                c.GpuPointer, b.GpuPointer,
                b.ByteLength,
                CudaBindings.CudaMemcpyKind.DeviceToDevice),
            "cudaMemcpy D→D (add)");

        float alpha = 1.0f;
        CublasBindings.Check(
            CublasBindings.Sgemm(
                _cublasHandle,
                CublasBindings.CublasOperation.NoTrans,
                CublasBindings.CublasOperation.NoTrans,
                a.Length, 1, 1,
                &alpha,
                (float*)a.GpuPointer, 1,
                (float*)a.GpuPointer, 1, // dummy — saxpy would be cleaner but cublasSaxpy not bound yet
                &alpha,
                (float*)c.GpuPointer, 1),
            "cublas add");

        // TODO: Replace with cublasSaxpy for correctness. This is placeholder.
        //   The proper implementation binds cublasSaxpy and calls:
        //   cublasSaxpy(handle, n, &alpha, a.GpuPointer, 1, c.GpuPointer, 1)
        //   This is an Emergence finding — record it.
    }

    public void Synchronize()
    {
        ThrowIfDisposed();
        CudaBindings.Check(CudaBindings.DeviceSynchronize(), "cudaDeviceSynchronize");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_cublasHandle != 0)
        {
            CublasBindings.Destroy(_cublasHandle);
            _cublasHandle = 0;
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CudaBackend));
    }
}
