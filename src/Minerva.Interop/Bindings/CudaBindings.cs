using System.Runtime.InteropServices;

namespace Minerva.Interop.Bindings;

/// <summary>
/// CUDA Runtime API bindings. Resolved via "tensor_gpu" → libcudart.so / cudart64_12.dll.
/// Only called on Linux/Windows when CUDA is detected.
/// </summary>
internal static partial class CudaBindings
{
    internal enum CudaError : int
    {
        Success = 0,
        InvalidValue = 1,
        MemoryAllocation = 2,
        NotInitialized = 3,
        InvalidDevice = 100,
        NoDevice = 38
    }

    internal enum CudaMemcpyKind : int
    {
        HostToHost = 0,
        HostToDevice = 1,
        DeviceToHost = 2,
        DeviceToDevice = 3
    }

    [LibraryImport("tensor_gpu", EntryPoint = "cudaMalloc")]
    internal static unsafe partial CudaError Malloc(void** devPtr, nuint size);

    [LibraryImport("tensor_gpu", EntryPoint = "cudaFree")]
    internal static unsafe partial CudaError Free(void* devPtr);

    [LibraryImport("tensor_gpu", EntryPoint = "cudaMemcpy")]
    internal static unsafe partial CudaError Memcpy(
        void* dst, void* src, nuint count, CudaMemcpyKind kind);

    [LibraryImport("tensor_gpu", EntryPoint = "cudaMemset")]
    internal static unsafe partial CudaError Memset(void* devPtr, int value, nuint count);

    [LibraryImport("tensor_gpu", EntryPoint = "cudaDeviceSynchronize")]
    internal static partial CudaError DeviceSynchronize();

    [LibraryImport("tensor_gpu", EntryPoint = "cudaGetDeviceCount")]
    internal static unsafe partial CudaError GetDeviceCount(int* count);

    /// <summary>
    /// Checks a CUDA error code and throws on failure.
    /// </summary>
    internal static void Check(CudaError error, string operation)
    {
        if (error != CudaError.Success)
            throw new InvalidOperationException(
                $"CUDA error in {operation}: {error} ({(int)error})");
    }
}
