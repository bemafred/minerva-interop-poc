using System.Reflection;
using System.Runtime.InteropServices;

namespace Minerva.Interop.Platform;

/// <summary>
/// Registers a DllImportResolver that maps logical library names to
/// platform-specific paths. Called once at startup before any P/Invoke.
///
/// Logical names used in [LibraryImport]:
///   "tensor_blas"    → Accelerate (macOS) / OpenBLAS (Linux/Windows)
///   "tensor_gpu"     → metal_bridge (macOS) / cudart (Linux/Windows)
///   "tensor_cublas"  → cublas (Linux/Windows only)
/// </summary>
public static class NativeResolver
{
    private static bool _registered;

    public static void Register(ComputeCapability capability)
    {
        if (_registered) return;
        _registered = true;

        var assembly = typeof(NativeResolver).Assembly;

        NativeLibrary.SetDllImportResolver(assembly, (name, asm, searchPath) =>
            name switch
            {
                "tensor_blas" => ResolveBlas(capability),
                "tensor_gpu" => ResolveGpu(capability),
                "tensor_cublas" => ResolveCuBlas(capability),
                _ => IntPtr.Zero
            });
    }

    private static IntPtr ResolveBlas(ComputeCapability cap)
    {
        return cap.BlasProvider switch
        {
            CpuBlasProvider.Accelerate =>
                NativeLibrary.Load("/System/Library/Frameworks/Accelerate.framework/Accelerate"),

            CpuBlasProvider.OpenBlas => cap.Os switch
            {
                OperatingSystem.Linux => TryLoadAny("libopenblas.so.0", "libopenblas.so"),
                OperatingSystem.Windows => TryLoadAny("openblas.dll", "libopenblas.dll"),
                _ => IntPtr.Zero
            },

            _ => IntPtr.Zero
        };
    }

    private static IntPtr ResolveGpu(ComputeCapability cap)
    {
        return cap.GpuProvider switch
        {
            GpuProvider.Metal =>
                TryLoadAny("libmetal_bridge.dylib", "metal_bridge"),

            GpuProvider.Cuda => cap.Os switch
            {
                OperatingSystem.Linux => TryLoadAny("libcudart.so.12", "libcudart.so.11.0", "libcudart.so"),
                OperatingSystem.Windows => TryLoadAny("cudart64_12.dll", "cudart64_11.dll"),
                _ => IntPtr.Zero
            },

            _ => IntPtr.Zero
        };
    }

    private static IntPtr ResolveCuBlas(ComputeCapability cap)
    {
        if (cap.GpuProvider != GpuProvider.Cuda)
            return IntPtr.Zero;

        return cap.Os switch
        {
            OperatingSystem.Linux => TryLoadAny("libcublas.so.12", "libcublas.so.11", "libcublas.so"),
            OperatingSystem.Windows => TryLoadAny("cublas64_12.dll", "cublas64_11.dll"),
            _ => IntPtr.Zero
        };
    }

    private static IntPtr TryLoadAny(params string[] names)
    {
        foreach (var name in names)
        {
            if (NativeLibrary.TryLoad(name, out var handle))
                return handle;
        }
        return IntPtr.Zero;
    }
}
