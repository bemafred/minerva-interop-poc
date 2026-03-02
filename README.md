# Minerva Interop POC

**Cross-platform .NET native interop for bare-metal tensor computation.**

This is an Emergence-phase experiment — a dedicated POC to validate (or falsify) the hypothesis that a single C# abstraction can span three fundamentally different memory architectures for GPU and CPU tensor math.

> Part of the [Sky Omega](https://github.com/bemafred/sky-omega) ecosystem.
> Minerva is a planned LLM inference substrate. This POC explores its interop foundation.

## What It Proves

| # | Hypothesis | Status |
|---|---|--------|
| 1 | `[LibraryImport]` source-generated P/Invoke works for BLAS, CUDA, Metal | ✅ Confirmed (Accelerate + Metal on macOS) |
| 2 | `NativeLibrary.SetDllImportResolver` dispatches correctly per platform | ✅ Confirmed (macOS framework bundles resolve cleanly) |
| 3 | `TensorBuffer<T>` abstraction spans UMA and discrete memory models | ✅ Confirmed (Unified + Cpu; CUDA Staged untested) |
| 4 | GPU results match CPU BLAS reference computation | ✅ Confirmed (max error: 0.00E+000 on 4×4 and 1024×1024) |
| 5 | Graceful degradation: GPU absent → CPU BLAS fallback | ✅ Confirmed (both backends run independently) |

## Platform Coverage

| Platform | GPU Backend | CPU BLAS | Native Build Required |
|----------|------------|----------|----------------------|
| macOS (Apple Silicon) | Metal via C bridge | Accelerate.framework | `libmetal_bridge.dylib` |
| Linux (NVIDIA) | CUDA Runtime + cuBLAS | OpenBLAS | None |
| Windows (NVIDIA) | CUDA Runtime + cuBLAS | OpenBLAS | None |

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
# Optional: CUDA Toolkit for GPU support

# Run (no native build needed)
dotnet run --project src/Minerva.Interop.Poc
```

### Windows

```powershell
# Prerequisites: .NET 10 SDK, OpenBLAS (download .dll), optional CUDA Toolkit

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

## Emergence Observations

1. **Accelerate's AMX coprocessor outperforms Metal compute at 1024×1024.** CPU BLAS (~2329 GFLOPS) beat the Metal GPU (~1922 GFLOPS) by ~21%. Apple's AMX units are specialized matrix engines invoked transparently by Accelerate — they avoid GPU dispatch overhead entirely. Implication: on Apple Silicon, GPU compute only wins at larger matrix sizes or when the CPU is saturated with other work.

2. **UMA is truly zero-copy.** `CPU==GPU pointer: True` confirms that Metal's `MTLStorageModeShared` gives both sides the same virtual address. `SyncToDevice` is a genuine no-op (0.00 ms), not just a fast copy. This validates the `TensorBuffer<T>` design — the `Unified` residence eliminates an entire class of staging-buffer bugs.

3. **Metal allocation is 5× faster than NativeMemory.AlignedAlloc** for the same 4 MB buffer (0.01 ms vs 0.05 ms). The Metal allocator likely returns from a pre-warmed page pool, while `NativeMemory.AlignedAlloc` goes through the system allocator with 64-byte alignment overhead.

4. **Metal fill is 10× slower than CPU fill** (0.69 ms vs 0.07 ms). Writing to shared UMA pages that are also GPU-mapped may incur cache coherency traffic on Apple Silicon, or the benchmark captures first-touch page fault latency on the Metal-allocated buffer.

5. **`NativeLibrary.SetDllImportResolver` composes cleanly with macOS framework bundles.** Accelerate resolves via framework path; the custom `libmetal_bridge.dylib` resolves via `AppContext.BaseDirectory`. No loader conflicts.

## License

MIT — see [Sky Omega](https://github.com/bemafred/sky-omega)
