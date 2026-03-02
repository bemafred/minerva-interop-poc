# Minerva Interop POC

**Cross-platform .NET native interop for bare-metal tensor computation.**

This is an Emergence-phase experiment — a dedicated POC to validate (or falsify) the hypothesis that a single C# abstraction can span three fundamentally different memory architectures for GPU and CPU tensor math.

> Part of the [Sky Omega](https://github.com/bemafred/sky-omega) ecosystem.
> Minerva is a planned LLM inference substrate. This POC explores its interop foundation.

## What It Proves

| # | Hypothesis | Status |
|---|---|--------|
| 1 | `[LibraryImport]` source-generated P/Invoke works for BLAS, CUDA, Metal | ✅ Confirmed (Accelerate + Metal on macOS, OpenBLAS on Linux/Windows, CUDA on Windows) |
| 2 | `NativeLibrary.SetDllImportResolver` dispatches correctly per platform | ✅ Confirmed (macOS framework bundles, Linux shared libs, Windows CUDA_PATH discovery) |
| 3 | `TensorBuffer<T>` abstraction spans UMA and discrete memory models | ✅ Confirmed (Unified + Cpu + Staged — all three memory models validated) |
| 4 | GPU results match CPU BLAS reference computation | ✅ Confirmed (Metal: 0.00E+000, CUDA: 3.54E-008 — within single-precision tolerance) |
| 5 | Graceful degradation: GPU absent → CPU BLAS fallback | ✅ Confirmed (macOS + Linux + Windows; legacy GPU correctly skipped) |

## Platform Coverage

| Platform | GPU Backend | CPU BLAS | Native Build Required |
|----------|------------|----------|----------------------|
| macOS (Apple Silicon) | Metal via C bridge | Accelerate.framework | `libmetal_bridge.dylib` |
| Linux (NVIDIA, CC 3.5+) | CUDA Runtime + cuBLAS | OpenBLAS | None |
| Linux (Legacy/No GPU) | None (CPU fallback) | OpenBLAS | None |
| Windows (NVIDIA, CC 5.0+) | CUDA Runtime + cuBLAS | OpenBLAS | None |

## Quick Start

### macOS

```bash
# Prerequisites
xcode-select --install   # Xcode command line tools (for Metal bridge)

# Build Metal bridge
cd native/metal && make && make install && cd ../..

# Run
dotnet run --project src/Minerva.Interop.Poc
```

### Linux

```bash
# Prerequisites
sudo apt install libopenblas-dev   # CPU BLAS
# Optional: CUDA Toolkit for GPU support (requires NVIDIA GPU with compute capability 3.5+)
# Legacy GPUs (Fermi/Kepler <3.5) are not supported by current CUDA — CPU fallback is automatic.
# CUDA detection uses the unversioned libcudart.so symlink — works across all toolkit versions.

# Run (no native build needed)
dotnet run --project src/Minerva.Interop.Poc
```

### Windows

```powershell
# Prerequisites: .NET 10 SDK, OpenBLAS (download .dll), optional CUDA Toolkit
# CUDA detection scans %CUDA_PATH%\bin for cudart/cublas DLLs — version-agnostic,
# no code changes needed when upgrading CUDA toolkit. Ensure CUDA_PATH is set (the
# CUDA installer does this by default).

# Run
dotnet run --project src/Minerva.Interop.Poc
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Program.cs (POC runner)                                │
│  Detect → Verify → Benchmark → Observe                  │
├─────────────────────────────────────────────────────────┤
│  IComputeBackend                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │ Metal    │  │ CUDA     │  │ CPU BLAS             │  │
│  │ Backend  │  │ Backend  │  │ (Accelerate/OpenBLAS)│  │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────────┘  │
├───────┼──────────────┼─────────────────┼────────────────┤
│  TensorBuffer<T>  — unified memory abstraction          │
│  ┌─────────┐  ┌──────────┐  ┌──────────────┐           │
│  │ Unified │  │ Staged   │  │ CPU          │           │
│  │ (UMA)   │  │ (CPU+GPU)│  │              │           │
│  └─────────┘  └──────────┘  └──────────────┘           │
├─────────────────────────────────────────────────────────┤
│  P/Invoke Bindings ([LibraryImport], source-generated)  │
│  BlasBindings · CudaBindings · CublasBindings · Metal   │
├─────────────────────────────────────────────────────────┤
│  NativeResolver (platform-specific library dispatch)    │
└─────────────────────────────────────────────────────────┘
          │                │                │
    Accelerate.fw    libcudart.so    libmetal_bridge.dylib
    / OpenBLAS       libcublas.so    (our C bridge)
```

## Design Principles

- **BCL-only**: No NuGet packages. `NativeMemory`, `Span<T>`, `NativeLibrary`, `LibraryImport`.
- **Zero-GC hot paths**: No managed allocations in compute dispatch.
- **Graceful degradation**: Always runs, even without GPU.
- **Semantic naming**: No `Handler`, `Manager`, `Utility` — every type name carries domain meaning.

## POC Results (macOS, Apple Silicon)

Run date: 2026-03-02

### Correctness (Phase 3 — 4×4 matmul)

| Test | Result |
|------|--------|
| CPU: A × I = A | PASS (max error: 0.00E+000) |
| GPU: A × I = A | PASS (max error: 0.00E+000) |
| GPU vs CPU reference | PASS (max error: 0.00E+000) |

### Performance (Phase 4 — 1024×1024 matmul, 5 runs)

| Backend | Avg | Min | Max | GFLOPS |
|---------|-----|-----|-----|--------|
| CPU BLAS (Accelerate) | 0.92 ms | 0.89 ms | 1.02 ms | ~2329 |
| GPU (Metal) | 1.12 ms | 1.04 ms | 1.27 ms | ~1922 |

Both backends produce identical output values (c[0]=246.0388, c[1048575]=250.6260).

### Memory Model (Phase 5)

| Property | Metal | CPU BLAS |
|----------|-------|----------|
| Residence | Unified | Cpu |
| CPU==GPU pointer | True (UMA zero-copy) | N/A |
| Alloc 1024×1024 | 0.01 ms | 0.05 ms |
| Fill (CPU) | 0.69 ms | 0.07 ms |
| Sync→Device | 0.00 ms (no-op, UMA) | 0.00 ms |

## POC Results (Linux x86_64, Legacy NVIDIA GPU)

Run date: 2026-03-02
System: Ubuntu 24.04 LTS, kernel 6.17, NVIDIA GeForce GTS 450 (Fermi, CC 2.1 — no CUDA support), OpenBLAS

The GTS 450 is a Fermi-generation GPU (2010). CUDA 8.0 was the last toolkit to support compute capability 2.x, and the legacy 390.xx driver branch is incompatible with modern kernels. The POC correctly detects no usable GPU and falls back to CPU BLAS.

### Correctness (Phase 3 — 4×4 matmul)

| Test | Result |
|------|--------|
| CPU: A × I = A | PASS (max error: 0.00E+000) |

### Performance (Phase 4 — 1024×1024 matmul, 5 runs)

| Backend | Avg | Min | Max | GFLOPS |
|---------|-----|-----|-----|--------|
| CPU BLAS (OpenBLAS) | 78.30 ms | 36.20 ms | 118.16 ms | ~27.4 |

Output values: c[0]=246.0388, c[1048575]=250.6264 (matches macOS reference).

### Memory Model (Phase 5)

| Property | CPU BLAS |
|----------|----------|
| Residence | Cpu |
| CPU accessible | True |
| GPU accessible | False |
| Alloc 1024×1024 | 3.52 ms |
| Fill (CPU) | 1.74 ms |
| Sync→Device | 0.01 ms (no-op) |

## POC Results (Windows 10 x86_64, NVIDIA Quadro M2000M)

Run date: 2026-03-02
System: Windows 10, NVIDIA Quadro M2000M (Maxwell, CC 5.0, 4 GB GDDR5), CUDA 12.6, driver 566.24, OpenBLAS

First validation of the CUDA `Staged` memory path — separate CPU and GPU buffers with explicit `cudaMemcpy` transfers.

### Correctness (Phase 3 — 4×4 matmul)

| Test | Result |
|------|--------|
| CPU: A × I = A | PASS (max error: 0.00E+000) |
| GPU: A × I = A | PASS (max error: 0.00E+000) |
| GPU vs CPU reference | PASS (max error: 3.54E-008) |

### Performance (Phase 4 — 1024×1024 matmul, 5 runs)

| Backend | Avg | Min | Max | GFLOPS |
|---------|-----|-----|-----|--------|
| CPU BLAS (OpenBLAS) | 353.92 ms | 33.63 ms | 895.22 ms | ~6.1 |
| GPU (CUDA, Quadro M2000M) | 47.24 ms | 15.58 ms | 117.54 ms | ~45.5 |

GPU is ~7.5× faster on average. Output values: c[0]=246.0388 (both), c[1048575]=250.6260 (GPU) vs 250.6264 (CPU) — within single-precision tolerance.

### Memory Model (Phase 5)

| Property | CUDA (Staged) | CPU BLAS |
|----------|--------------|----------|
| Residence | Staged | Cpu |
| CPU accessible | True | True |
| GPU accessible | True | False |
| CPU==GPU pointer | False (discrete) | False |
| Alloc 1024×1024 | 10.65 ms | 1.41 ms |
| Fill (CPU) | 3.11 ms | 1.28 ms |
| Sync→Device | 2.00 ms (cudaMemcpy) | 0.01 ms (no-op) |

## Emergence Observations

1. **Accelerate's AMX coprocessor outperforms Metal compute at 1024×1024.** CPU BLAS (~2329 GFLOPS) beat the Metal GPU (~1922 GFLOPS) by ~21%. Apple's AMX units are specialized matrix engines invoked transparently by Accelerate — they avoid GPU dispatch overhead entirely. Implication: on Apple Silicon, GPU compute only wins at larger matrix sizes or when the CPU is saturated with other work.

2. **UMA is truly zero-copy.** `CPU==GPU pointer: True` confirms that Metal's `MTLStorageModeShared` gives both sides the same virtual address. `SyncToDevice` is a genuine no-op (0.00 ms), not just a fast copy. This validates the `TensorBuffer<T>` design — the `Unified` residence eliminates an entire class of staging-buffer bugs.

3. **Metal allocation is 5× faster than NativeMemory.AlignedAlloc** for the same 4 MB buffer (0.01 ms vs 0.05 ms). The Metal allocator likely returns from a pre-warmed page pool, while `NativeMemory.AlignedAlloc` goes through the system allocator with 64-byte alignment overhead.

4. **Metal fill is 10× slower than CPU fill** (0.69 ms vs 0.07 ms). Writing to shared UMA pages that are also GPU-mapped may incur cache coherency traffic on Apple Silicon, or the benchmark captures first-touch page fault latency on the Metal-allocated buffer.

5. **`NativeLibrary.SetDllImportResolver` composes cleanly with macOS framework bundles.** Accelerate resolves via framework path; the custom `libmetal_bridge.dylib` resolves via `AppContext.BaseDirectory`. No loader conflicts.

6. **Graceful degradation validated on legacy NVIDIA hardware.** A system with a GeForce GTS 450 (Fermi, compute capability 2.1) — unsupported by any current CUDA toolkit — correctly falls back to CPU BLAS without errors. The detection path handles the "GPU present but unusable" scenario identically to "no GPU present".

7. **OpenBLAS on x86_64 delivers ~27 GFLOPS vs Accelerate's ~2329 GFLOPS on Apple Silicon.** The ~85× gap reflects the AMX coprocessor advantage: Apple's matrix engine is purpose-built for linear algebra, while OpenBLAS on an older Xeon/consumer CPU uses AVX/SSE. This underscores that the `IComputeBackend` abstraction correctly adapts to wildly different performance envelopes without code changes.

8. **Cross-platform numerical agreement confirmed.** Output values c[0]=246.0388, c[1048575]=250.6264 match across macOS Accelerate and Linux OpenBLAS, confirming that the `TensorBuffer<T>` + `IComputeBackend` abstraction produces consistent single-precision results across platforms and BLAS implementations.

9. **Hardcoded CUDA DLL versions break on major toolkit upgrades.** Installing CUDA 13.1 on Windows produced `cudart64_13.dll` and `cublas64_13.dll`, but the resolver only tried v12 and v11 names — silently falling back to CPU with no error. Fixed by two-stage discovery: bare DLL names via OS loader (PATH) first, then CUDA_PATH glob scan of both `bin\x64` (v12+) and `bin` (v9–v11) as fallback. On Linux, the unversioned `libcudart.so` symlink already provides forward compatibility.

10. **CUDA `Staged` memory model validated.** `CPU==GPU pointer: False` confirms separate address spaces. `SyncToDevice: 2.00 ms` is the real cost of moving 4 MB across PCIe. This validates the third and final `TensorBuffer<T>` memory model — all three residences (Unified, Staged, Cpu) are now confirmed on hardware.

11. **CUDA GPU allocation is ~7.5× slower than CPU.** `cudaMalloc` takes 10.65 ms vs `NativeMemory.AlignedAlloc` at 1.41 ms for a 4 MB buffer. GPU memory allocation involves driver-level page table setup and device memory management, making pre-allocation essential for real workloads.

12. **Quadro M2000M (Maxwell) delivers ~45.5 GFLOPS via cuBLAS** — 7.5× faster than the same system's OpenBLAS CPU path (~6.1 GFLOPS). Even a 2015 mobile workstation GPU significantly outperforms the CPU for matrix math.

13. **`cudaSetDevice(0)` required before `cublasCreate` on Windows.** Loading `cudart64_12.dll` does not implicitly initialize a CUDA device context on Windows. Without the explicit `cudaSetDevice` call, `cublasCreate` fails with `NotInitialized (1)`. This differs from typical Linux CUDA behaviour where the runtime lazily initializes on first API call.

14. **GPU vs CPU numerical divergence is within single-precision tolerance.** c[1048575]=250.6260 (CUDA) vs 250.6264 (CPU/Metal/OpenBLAS). The ~0.0004 difference reflects non-deterministic floating-point reduction order in parallel GPU matmul — expected and acceptable for f32.

## License

MIT — see [Sky Omega](https://github.com/bemafred/sky-omega)
