using System.Runtime.InteropServices;

namespace Minerva.Interop.Platform;

/// <summary>
/// Describes the compute capabilities discovered on the current platform.
/// Immutable after construction — detection runs once at startup.
/// </summary>
public sealed record ComputeCapability(
    OperatingSystem Os,
    CpuBlasProvider BlasProvider,
    GpuProvider GpuProvider,
    string? GpuDeviceName)
{
    public bool HasGpu => GpuProvider != GpuProvider.None;

    public override string ToString() =>
        $"OS={Os}, BLAS={BlasProvider}, GPU={GpuProvider}" +
        (GpuDeviceName is not null ? $" ({GpuDeviceName})" : "");
}

public enum OperatingSystem { MacOS, Linux, Windows, Unknown }
public enum CpuBlasProvider { Accelerate, OpenBlas, None }
public enum GpuProvider { Metal, Cuda, None }

/// <summary>
/// Probes the current platform for available native libraries.
/// No exceptions — returns a capability descriptor with graceful degradation.
/// </summary>
public static class PlatformDetector
{
    public static ComputeCapability Detect()
    {
        var os = DetectOs();
        var blas = DetectBlas(os);
        var (gpu, deviceName) = DetectGpu(os);

        return new ComputeCapability(os, blas, gpu, deviceName);
    }

    private static OperatingSystem DetectOs()
    {
        if (System.OperatingSystem.IsMacOS()) return OperatingSystem.MacOS;
        if (System.OperatingSystem.IsLinux()) return OperatingSystem.Linux;
        if (System.OperatingSystem.IsWindows()) return OperatingSystem.Windows;
        return OperatingSystem.Unknown;
    }

    private static CpuBlasProvider DetectBlas(OperatingSystem os)
    {
        if (os == OperatingSystem.MacOS)
        {
            // Accelerate.framework is always present on macOS
            return TryLoadLibrary("/System/Library/Frameworks/Accelerate.framework/Accelerate")
                ? CpuBlasProvider.Accelerate
                : CpuBlasProvider.None;
        }

        // Linux / Windows — try OpenBLAS
        string[] openBlasNames = os == OperatingSystem.Windows
            ? ["openblas.dll", "libopenblas.dll"]
            : ["libopenblas.so.0", "libopenblas.so"];

        foreach (var name in openBlasNames)
        {
            if (TryLoadLibrary(name))
                return CpuBlasProvider.OpenBlas;
        }

        return CpuBlasProvider.None;
    }

    private static (GpuProvider provider, string? deviceName) DetectGpu(OperatingSystem os)
    {
        if (os == OperatingSystem.MacOS)
            return DetectMetal();

        return DetectCuda(os);
    }

    private static (GpuProvider, string?) DetectMetal()
    {
        // Metal bridge is our compiled dylib — if it loads, Metal is available
        if (TryLoadLibrary("metal_bridge") || TryLoadLibrary("libmetal_bridge.dylib"))
            return (GpuProvider.Metal, "Apple Silicon GPU");

        return (GpuProvider.None, null);
    }

    private static (GpuProvider, string?) DetectCuda(OperatingSystem os)
    {
        foreach (var name in GetCudaRuntimeCandidates(os))
        {
            if (TryLoadLibrary(name))
                return (GpuProvider.Cuda, DetectCudaDeviceName());
        }

        return (GpuProvider.None, null);
    }

    private static string? DetectCudaDeviceName()
    {
        // Will be implemented when CUDA bindings are wired — returns null for now.
        // In practice: cudaGetDeviceProperties → prop.name
        return null;
    }

    /// <summary>
    /// Returns candidate library names/paths for the CUDA runtime.
    /// On Windows, scans CUDA_PATH\bin to discover the installed version dynamically.
    /// On Linux, uses versioned SONAMEs with an unversioned symlink as final fallback.
    /// </summary>
    internal static string[] GetCudaRuntimeCandidates(OperatingSystem os) => os switch
    {
        OperatingSystem.Windows => DiscoverWindowsCudaLibs("cudart64_*.dll",
            ["cudart64_13.dll", "cudart64_12.dll", "cudart64_11.dll"]),
        _ => ["libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so"]
    };

    /// <summary>
    /// Returns candidate library names/paths for cuBLAS.
    /// Same discovery strategy as <see cref="GetCudaRuntimeCandidates"/>.
    /// </summary>
    internal static string[] GetCuBlasCandidates(OperatingSystem os) => os switch
    {
        OperatingSystem.Windows => DiscoverWindowsCudaLibs("cublas64_*.dll",
            ["cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll"]),
        _ => ["libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so"]
    };

    /// <summary>
    /// Scans CUDA_PATH\bin for matching DLLs to avoid hardcoding CUDA version numbers.
    /// Returns full paths (loadable even if CUDA bin isn't on PATH yet).
    /// Falls back to well-known names if CUDA_PATH is unset or no match is found.
    /// </summary>
    private static string[] DiscoverWindowsCudaLibs(string glob, string[] fallback)
    {
        var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
        if (cudaPath is null) return fallback;

        var binDir = Path.Combine(cudaPath, "bin");
        if (!Directory.Exists(binDir)) return fallback;

        var discovered = Directory.GetFiles(binDir, glob);
        return discovered.Length > 0 ? discovered : fallback;
    }

    private static bool TryLoadLibrary(string name)
    {
        if (NativeLibrary.TryLoad(name, out var handle))
        {
            NativeLibrary.Free(handle);
            return true;
        }

        // Bare names don't search the app's output directory — try AppContext.BaseDirectory
        var fullPath = Path.Combine(AppContext.BaseDirectory, name);
        if (NativeLibrary.TryLoad(fullPath, out handle))
        {
            NativeLibrary.Free(handle);
            return true;
        }

        return false;
    }
}
