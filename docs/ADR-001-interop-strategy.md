# ADR-001: Cross-Platform Native Interop Strategy

## Status
**Accepted** — validated on macOS Apple Silicon and Linux x86_64 (2026-03-02). CUDA path untested (available NVIDIA hardware too old for supported CUDA toolkit).

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

### Legacy GPU Handling
CUDA support requires a minimum compute capability (CC 3.5+ for current CUDA toolkits). Older GPUs
(e.g., Fermi CC 2.1, Kepler CC < 3.5) are present in the system but unsupported by any maintained
CUDA toolkit or driver branch. The POC handles this identically to "no GPU present" — the
`ComputeBackend.Create()` factory detects the absence of a usable CUDA runtime and falls back to
`CpuBackend` with OpenBLAS. No special code paths or legacy driver shims are needed.

Validated on: GeForce GTS 450 (Fermi, CC 2.1), Ubuntu 24.04, kernel 6.17. Last supported CUDA
toolkit for this GPU was CUDA 8.0 (2016); last supported driver was the 390.xx branch (EOL 2022),
which is incompatible with kernels 6.x+.

## Risks / Open Questions (Emergence Targets)
- ~~Does `NativeLibrary` resolver compose cleanly with framework bundles on macOS?~~
  **Resolved.** Yes. Accelerate resolves via framework path, `libmetal_bridge.dylib` via `AppContext.BaseDirectory`. No loader conflicts.
- ~~Does `NativeLibrary` resolver work with Linux shared libraries (OpenBLAS)?~~
  **Resolved.** Yes. `libopenblas.so` resolves via standard `ld.so` search paths. No resolver conflicts.
- ~~Can `AsSpan()` abstract over both UMA pointers and discrete-memory staging buffers?~~
  **Partially resolved.** Confirmed for UMA (`Unified` — CPU==GPU pointer, zero-copy). CUDA `Staged` path (separate CPU/GPU buffers) not yet validated on hardware.
- ~~Does graceful degradation work when a GPU is present but unsupported?~~
  **Resolved.** Validated on GeForce GTS 450 (Fermi, CC 2.1). No CUDA runtime is loadable, so the factory falls back to `CpuBackend` cleanly — same path as "no GPU at all".
- How do CUDA error codes, Metal NSError, and silent BLAS failures unify?
  **Open.** Metal + Accelerate (macOS) and OpenBLAS (Linux) paths exercised. Error unification across all three backends remains untested.
- Can GitHub Actions CI cover Metal (macOS runner) + CUDA (Linux runner)?
  **Open.** Not yet attempted.

## Emergence Findings

### macOS (Apple Silicon)
- Accelerate's AMX coprocessor outperforms Metal GPU at 1024×1024 matmul (~2329 vs ~1922 GFLOPS). GPU compute dispatch overhead means the GPU only wins at larger sizes or when the CPU is saturated.
- UMA is truly zero-copy: `CPU==GPU pointer: True`, `SyncToDevice: 0.00 ms` (genuine no-op). The `Unified` residence in `TensorBuffer<T>` eliminates staging-buffer complexity on Apple Silicon.
- Bit-exact agreement between GPU and CPU backends (max error: 0.00E+000), confirming that the `IComputeBackend` abstraction does not introduce numerical divergence for single-precision matmul.

### Linux x86_64 (Legacy NVIDIA — CPU fallback)
- OpenBLAS on x86_64 delivers ~27.4 GFLOPS vs Accelerate's ~2329 GFLOPS on Apple Silicon. The ~85× gap reflects the AMX coprocessor advantage; OpenBLAS uses AVX/SSE on consumer x86_64 hardware. The `IComputeBackend` abstraction handles this performance envelope transparently.
- Cross-platform numerical agreement: output values c[0]=246.0388, c[1048575]=250.6264 match between macOS Accelerate and Linux OpenBLAS, confirming consistent single-precision matmul results across BLAS implementations.
- Legacy GPU detection is a non-event: the GeForce GTS 450 (Fermi, CC 2.1) is invisible to the CUDA runtime (no compatible driver installed), so `ComputeBackend.Create()` falls back to `CpuBackend` without errors or special handling. This validates that graceful degradation covers not just "no GPU" but "GPU present, unusable".

## References
- [LibraryImport source generation](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/pinvoke-source-generation)
- [NativeLibrary API](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.interopservices.nativelibrary)
- [Metal Compute](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
