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
        string[] cudaNames = os == OperatingSystem.Windows
            ? ["cudart64_12.dll", "cudart64_11.dll"]
            : ["libcudart.so.12", "libcudart.so.11.0", "libcudart.so"];

        foreach (var name in cudaNames)
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

    private static bool TryLoadLibrary(string name)
    {
        try
        {
            var handle = NativeLibrary.Load(name);
            NativeLibrary.Free(handle);
            return true;
        }
        catch
        {
            return false;
        }
    }
}
