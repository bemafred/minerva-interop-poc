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
    /// On Windows: bare DLL names first (OS loader via PATH), then CUDA_PATH scan as fallback.
    /// On Linux: versioned SONAMEs with unversioned symlink as final fallback.
    /// </summary>
    internal static string[] GetCudaRuntimeCandidates(OperatingSystem os) => os switch
    {
        OperatingSystem.Windows => DiscoverWindowsCudaLibs("cudart64_*.dll"),
        _ => ["libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so"]
    };

    /// <summary>
    /// Returns candidate library names/paths for cuBLAS.
    /// Same discovery strategy as <see cref="GetCudaRuntimeCandidates"/>.
    /// </summary>
    internal static string[] GetCuBlasCandidates(OperatingSystem os) => os switch
    {
        OperatingSystem.Windows => DiscoverWindowsCudaLibs("cublas64_*.dll"),
        _ => ["libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so"]
    };

    /// <summary>
    /// Discovers CUDA DLLs on Windows using a two-stage strategy:
    /// 1. Bare DLL names — let the OS loader find them via PATH (works when CUDA bin is on PATH).
    /// 2. CUDA_PATH scan — glob both bin\x64 (v12+) and bin (v9–v11) for full paths as fallback.
    /// Bare names are tried first because the OS loader already knows the correct path,
    /// making this resilient to future directory structure changes.
    /// </summary>
    private static string[] DiscoverWindowsCudaLibs(string glob)
    {
        var candidates = new List<string>();

        // Stage 1: extract bare DLL names from CUDA_PATH scan (if available) so the
        // OS loader can find them via PATH. This handles any directory layout.
        var fullPaths = ScanCudaPath(glob);
        foreach (var path in fullPaths)
            candidates.Add(Path.GetFileName(path));

        // Stage 2: full paths from CUDA_PATH as fallback (works even if PATH isn't configured yet)
        candidates.AddRange(fullPaths);

        return candidates.Count > 0 ? candidates.ToArray() : [];
    }

    /// <summary>
    /// Scans CUDA_PATH for DLLs matching the glob pattern.
    /// Tries bin\x64 (CUDA 12+) then bin (CUDA 9–11) to handle layout differences.
    /// </summary>
    private static string[] ScanCudaPath(string glob)
    {
        var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
        if (cudaPath is null) return [];

        // CUDA 12+ uses bin\x64, older versions use bin directly
        string[] binDirs = [Path.Combine(cudaPath, "bin", "x64"), Path.Combine(cudaPath, "bin")];

        foreach (var binDir in binDirs)
        {
            if (!Directory.Exists(binDir)) continue;

            var discovered = Directory.GetFiles(binDir, glob);
            if (discovered.Length > 0) return discovered;
        }

        return [];
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
