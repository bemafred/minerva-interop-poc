using System.Runtime.InteropServices;

namespace Minerva.Interop.Bindings;

/// <summary>
/// CBLAS bindings via [LibraryImport]. The logical name "tensor_blas" is resolved
/// by NativeResolver to Accelerate.framework (macOS) or OpenBLAS (Linux/Windows).
///
/// Both implement the standard CBLAS interface — same function signatures.
/// </summary>
internal static partial class BlasBindings
{
    internal enum CblasOrder : int { RowMajor = 101, ColMajor = 102 }
    internal enum CblasTranspose : int { NoTrans = 111, Trans = 112, ConjTrans = 113 }

    /// <summary>
    /// Single-precision general matrix multiplication.
    /// C = alpha * op(A) * op(B) + beta * C
    ///
    /// This is the workhorse — validates that our P/Invoke chain works end-to-end.
    /// </summary>
    [LibraryImport("tensor_blas", EntryPoint = "cblas_sgemm")]
    internal static unsafe partial void Sgemm(
        CblasOrder order,
        CblasTranspose transA,
        CblasTranspose transB,
        int m, int n, int k,
        float alpha,
        float* a, int lda,
        float* b, int ldb,
        float beta,
        float* c, int ldc);

    /// <summary>
    /// Single-precision vector scaling: x = alpha * x
    /// </summary>
    [LibraryImport("tensor_blas", EntryPoint = "cblas_sscal")]
    internal static unsafe partial void Sscal(
        int n,
        float alpha,
        float* x,
        int incX);

    /// <summary>
    /// Single-precision vector addition: y = alpha * x + y
    /// </summary>
    [LibraryImport("tensor_blas", EntryPoint = "cblas_saxpy")]
    internal static unsafe partial void Saxpy(
        int n,
        float alpha,
        float* x, int incX,
        float* y, int incY);

    /// <summary>
    /// Single-precision dot product: result = x · y
    /// </summary>
    [LibraryImport("tensor_blas", EntryPoint = "cblas_sdot")]
    internal static unsafe partial float Sdot(
        int n,
        float* x, int incX,
        float* y, int incY);
}
