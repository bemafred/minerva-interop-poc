using System.Runtime.InteropServices;

namespace Minerva.Interop.Bindings;

/// <summary>
/// cuBLAS bindings for GPU-accelerated matrix operations on NVIDIA hardware.
/// Resolved via "tensor_cublas" → libcublas.so / cublas64_12.dll.
/// </summary>
internal static partial class CublasBindings
{
    internal enum CublasStatus : int
    {
        Success = 0,
        NotInitialized = 1,
        AllocFailed = 3,
        InvalidValue = 7,
        MappingError = 11,
        ExecutionFailed = 13,
        InternalError = 14
    }

    internal enum CublasOperation : int
    {
        NoTrans = 0,
        Trans = 1,
        ConjTrans = 2
    }

    [LibraryImport("tensor_cublas", EntryPoint = "cublasCreate_v2")]
    internal static unsafe partial CublasStatus Create(nint* handle);

    [LibraryImport("tensor_cublas", EntryPoint = "cublasDestroy_v2")]
    internal static partial CublasStatus Destroy(nint handle);

    /// <summary>
    /// cuBLAS sgemm — note: cuBLAS uses column-major by default.
    /// We'll need to transpose or swap A/B to get row-major semantics.
    /// </summary>
    [LibraryImport("tensor_cublas", EntryPoint = "cublasSgemm_v2")]
    internal static unsafe partial CublasStatus Sgemm(
        nint handle,
        CublasOperation transa,
        CublasOperation transb,
        int m, int n, int k,
        float* alpha,
        float* a, int lda,
        float* b, int ldb,
        float* beta,
        float* c, int ldc);

    internal static void Check(CublasStatus status, string operation)
    {
        if (status != CublasStatus.Success)
            throw new InvalidOperationException(
                $"cuBLAS error in {operation}: {status} ({(int)status})");
    }
}
