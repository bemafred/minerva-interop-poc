# ADR-001: Cross-Platform Native Interop Strategy

## Status
Proposed

## Context
Minerva requires bare-metal tensor computation across macOS (Apple Silicon / Metal),
Linux (NVIDIA / CUDA), and Windows (NVIDIA / CUDA). The .NET runtime must call into
platform-specific native compute and BLAS libraries with minimal overhead and zero
managed allocations on hot paths.

This POC exists to validate (or falsify) the hypothesis that a single C# abstraction
can span three fundamentally different memory architectures:
- **Unified Memory (UMA)**: macOS Apple Silicon — CPU and GPU share physical memory
- **Discrete Memory**: Linux/Windows with NVIDIA GPU — separate CPU RAM and GPU VRAM
- **CPU-only**: Any platform without GPU — BLAS fallback

## Decision
We use the following .NET interop mechanisms:

1. **`[LibraryImport]`** (source-generated P/Invoke, .NET 7+) for all native calls.
   No legacy `[DllImport]`. Source generation eliminates runtime marshalling overhead.

2. **`NativeLibrary.SetDllImportResolver`** for platform-specific library resolution
   at startup. One C# binding surface, platform-dispatched at load time.

3. **`NativeMemory`** for CPU-side allocations (aligned, unmanaged, GC-invisible).

4. **`Span<T>` / pointer arithmetic** for zero-copy access to native buffers.

We build native code only where no C API exists:
- **Metal bridge** (`metal_bridge.dylib`): Thin Objective-C wrapper exposing Metal
  compute as C functions. ~200 lines. macOS only.
- **Metal shaders** (`kernels.metallib`): Compiled compute kernels.

We do NOT build:
- CUDA libraries (vendor-shipped)
- BLAS libraries (OS-shipped or package-managed)
- Custom CUDA kernels (cuBLAS suffices for POC)

## Consequences
- Linux and Windows POC requires zero native compilation (only CUDA toolkit install)
- macOS requires Xcode command line tools for Metal bridge build
- Graceful degradation: GPU unavailable → CPU BLAS fallback, always functional
- BCL-only on the C# side: no NuGet packages, no wrapper libraries

## Risks / Open Questions (Emergence Targets)
- Does `NativeLibrary` resolver compose cleanly with framework bundles on macOS?
- Can `AsSpan()` abstract over both UMA pointers and discrete-memory staging buffers?
- How do CUDA error codes, Metal NSError, and silent BLAS failures unify?
- Can GitHub Actions CI cover Metal (macOS runner) + CUDA (Linux runner)?

## References
- [LibraryImport source generation](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/pinvoke-source-generation)
- [NativeLibrary API](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.interopservices.nativelibrary)
- [Metal Compute](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
