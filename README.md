# Minerva Interop POC

**Cross-platform .NET native interop for bare-metal tensor computation.**

This is an Emergence-phase experiment — a dedicated POC to validate (or falsify) the hypothesis that a single C# abstraction can span three fundamentally different memory architectures for GPU and CPU tensor math.

> Part of the [Sky Omega](https://github.com/bemafred/sky-omega) ecosystem.
> Minerva is a planned LLM inference substrate. This POC explores its interop foundation.

## What It Proves

| # | Hypothesis | Status |
|---|---|--------|
| 1 | `[LibraryImport]` source-generated P/Invoke works for BLAS, CUDA, Metal | 🔬 Testing |
| 2 | `NativeLibrary.SetDllImportResolver` dispatches correctly per platform | 🔬 Testing |
| 3 | `TensorBuffer<T>` abstraction spans UMA and discrete memory models | 🔬 Testing |
| 4 | GPU results match CPU BLAS reference computation | 🔬 Testing |
| 5 | Graceful degradation: GPU absent → CPU BLAS fallback | 🔬 Testing |

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
# Prerequisites: .NET 9 SDK, OpenBLAS (download .dll), optional CUDA Toolkit

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

## Emergence Observations

Discoveries made during POC development:

_To be filled as the POC runs on real hardware and surfaces unknown unknowns._

## License

MIT — see [Sky Omega](https://github.com/bemafred/sky-omega)
