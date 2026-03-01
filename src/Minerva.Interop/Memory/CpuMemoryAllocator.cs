using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Minerva.Interop.Memory;

/// <summary>
/// Allocates CPU-resident tensor buffers using NativeMemory with alignment.
/// Available on all platforms. GC-invisible.
/// </summary>
public static class CpuMemoryAllocator
{
    /// <summary>
    /// Alignment for SIMD-friendly access. 64 bytes covers AVX-512 and NEON.
    /// </summary>
    private const nuint Alignment = 64;

    public static unsafe TensorBuffer<T> Allocate<T>(int rows, int cols)
        where T : unmanaged
    {
        var length = rows * cols;
        var byteLength = (nuint)(length * Unsafe.SizeOf<T>());

        var ptr = (T*)NativeMemory.AlignedAlloc(byteLength, Alignment);

        // Zero-initialize — predictable state, no garbage reads
        NativeMemory.Clear(ptr, byteLength);

        return new TensorBuffer<T>(
            rows, cols,
            cpuPtr: ptr,
            gpuPtr: null,
            residence: MemoryResidence.Cpu,
            disposer: static buffer =>
            {
                if (buffer.CpuPointer != null)
                    NativeMemory.AlignedFree(buffer.CpuPointer);
            });
    }
}
