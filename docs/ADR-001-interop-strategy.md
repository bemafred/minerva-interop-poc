# ADR-001: Cross-Platform Native Interop Strategy

## Status
**Accepted** — validated on macOS Apple Silicon, Linux x86_64, and Windows x86_64 with CUDA (2026-03-02). All three memory models (Unified, Staged, Cpu) confirmed on hardware.

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
7 of 8 questions resolved. One remaining: error unification across backends.

- ~~Does `NativeLibrary` resolver compose cleanly with framework bundles on macOS?~~
  **Resolved.** Yes. Accelerate resolves via framework path, `libmetal_bridge.dylib` via `AppContext.BaseDirectory`. No loader conflicts.
- ~~Does `NativeLibrary` resolver work with Linux shared libraries (OpenBLAS)?~~
  **Resolved.** Yes. `libopenblas.so` resolves via standard `ld.so` search paths. No resolver conflicts.
- ~~Can `AsSpan()` abstract over both UMA pointers and discrete-memory staging buffers?~~
  **Resolved.** Confirmed for UMA (`Unified` — CPU==GPU pointer, zero-copy on macOS) and discrete (`Staged` — separate CPU/GPU buffers with explicit `cudaMemcpy` on Windows). Both models work through the same `TensorBuffer<T>` interface.
- ~~Does graceful degradation work when a GPU is present but unsupported?~~
  **Resolved.** Validated on GeForce GTS 450 (Fermi, CC 2.1). No CUDA runtime is loadable, so the factory falls back to `CpuBackend` cleanly — same path as "no GPU at all".
- ~~Does the resolver handle CUDA major version upgrades without code changes?~~
  **Resolved.** No — hardcoded DLL names (`cudart64_12.dll`) silently missed CUDA 13.1's `cudart64_13.dll`, falling back to CPU with no error. Fixed by scanning `%CUDA_PATH%\bin` at startup via glob pattern. On Linux, the unversioned `libcudart.so` symlink already handles version drift.
- ~~Does `cudaSetDevice` need to be called before `cublasCreate`?~~
  **Resolved.** Yes, on Windows. Loading `cudart` does not implicitly create a device context. Without `cudaSetDevice(0)`, `cublasCreate` returns `NotInitialized (1)`. Added explicit initialization before cuBLAS handle creation.
- How do CUDA error codes, Metal NSError, and silent BLAS failures unify?
  **Partially resolved.** CUDA errors surface via `CudaBindings.Check` / `CublasBindings.Check` (throw on non-success). Metal errors handled on macOS. OpenBLAS BLAS calls are silent (no error codes). A unified error strategy across all three backends is not yet designed.
- ~~Can GitHub Actions CI cover Metal (macOS runner) + CUDA (Linux runner)?~~
  **Resolved.** CI runs on all three platforms: macOS-14 (Apple Silicon, Metal + Accelerate), ubuntu-24.04 (OpenBLAS CPU), windows-latest (OpenBLAS CPU). CUDA requires a GPU runner not available in standard GitHub Actions, but the build and CPU fallback path is validated on all platforms.

## Emergence Findings

### macOS (Apple Silicon)
- Accelerate's AMX coprocessor outperforms Metal GPU at 1024×1024 matmul (~2329 vs ~1922 GFLOPS). GPU compute dispatch overhead means the GPU only wins at larger sizes or when the CPU is saturated.
- UMA is truly zero-copy: `CPU==GPU pointer: True`, `SyncToDevice: 0.00 ms` (genuine no-op). The `Unified` residence in `TensorBuffer<T>` eliminates staging-buffer complexity on Apple Silicon.
- Bit-exact agreement between GPU and CPU backends (max error: 0.00E+000), confirming that the `IComputeBackend` abstraction does not introduce numerical divergence for single-precision matmul.

### Linux x86_64 (Legacy NVIDIA — CPU fallback)
- OpenBLAS on x86_64 delivers ~27.4 GFLOPS vs Accelerate's ~2329 GFLOPS on Apple Silicon. The ~85× gap reflects the AMX coprocessor advantage; OpenBLAS uses AVX/SSE on consumer x86_64 hardware. The `IComputeBackend` abstraction handles this performance envelope transparently.
- Cross-platform numerical agreement: output values c[0]=246.0388, c[1048575]=250.6264 match between macOS Accelerate and Linux OpenBLAS, confirming consistent single-precision matmul results across BLAS implementations.
- Legacy GPU detection is a non-event: the GeForce GTS 450 (Fermi, CC 2.1) is invisible to the CUDA runtime (no compatible driver installed), so `ComputeBackend.Create()` falls back to `CpuBackend` without errors or special handling. This validates that graceful degradation covers not just "no GPU" but "GPU present, unusable".

### Windows x86_64 (NVIDIA Quadro M2000M — CUDA Staged)
- **Staged memory model validated**: `CPU==GPU pointer: False` confirms separate CPU (`NativeMemory`) and GPU (`cudaMalloc`) address spaces. `SyncToDevice: 2.00 ms` is real PCIe transfer cost for 4 MB. This completes validation of all three `TensorBuffer<T>` memory residences.
- Quadro M2000M (Maxwell, CC 5.0) delivers ~45.5 GFLOPS via cuBLAS — 7.5× faster than OpenBLAS on the same system (~6.1 GFLOPS). Even a 2015 mobile workstation GPU significantly outperforms the CPU for matrix math.
- GPU allocation overhead is substantial: `cudaMalloc` at 10.65 ms is ~7.5× slower than `NativeMemory.AlignedAlloc` at 1.41 ms. Pre-allocation is essential for real workloads.
- `cudaSetDevice(0)` must be called before `cublasCreate` on Windows — loading `cudart` alone does not implicitly initialize a device context. Without it, cuBLAS returns `NotInitialized (1)`.
- GPU vs CPU numerical divergence (c[1048575]: 250.6260 vs 250.6264) is within single-precision tolerance — expected from non-deterministic parallel reduction order in GPU matmul.
- Output value c[0]=246.0388 matches exactly across all four backends (Accelerate, Metal, OpenBLAS, cuBLAS), confirming the `IComputeBackend` abstraction produces consistent results.

### CUDA Version Discovery (Windows)
- Hardcoded CUDA DLL names are fragile across major toolkit versions. CUDA 13.1 ships `cudart64_13.dll` / `cublas64_13.dll`, but the original resolver only tried v12/v11 names — silently degrading to CPU. The failure mode is silent (no error, just slower), which makes it particularly insidious in production.
- Two-stage discovery fixes this: bare DLL names via OS loader (PATH) first, then `%CUDA_PATH%` scan of both `bin\x64` (v12+) and `bin` (v9–v11) as fallback. No hardcoded version lists needed.
- CUDA Toolkit directory structure changed between versions (v9: `bin\`, v12+: `bin\x64\`). Scanning both paths handles this transparently.
- Linux avoids this problem entirely: the unversioned `libcudart.so` symlink (maintained by the CUDA installer and `ldconfig`) provides forward compatibility across all major versions. Windows has no equivalent convention — DLL names are always versioned.

## References
- [LibraryImport source generation](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/pinvoke-source-generation)
- [NativeLibrary API](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.interopservices.nativelibrary)
- [Metal Compute](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
