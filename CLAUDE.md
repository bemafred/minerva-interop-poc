# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Build the solution
dotnet build

# Run the POC (5 phases: detect → create → verify → benchmark → observe)
dotnet run --project src/Minerva.Interop.Poc

# macOS only: build the Metal bridge first
cd native/metal && make && make install && cd ../..

# Linux: install OpenBLAS before running
sudo apt install libopenblas-dev
```

There are no test projects. The POC runner (`Program.cs`) serves as validation: Phase 3 runs correctness checks (matmul against identity, cross-backend comparison with CPU reference) and Phase 4 runs benchmarks.

## Architecture

This is a .NET 10 POC validating zero-overhead native GPU/CPU interop via `[LibraryImport]` source-generated P/Invoke. BCL-only — zero NuGet packages.

**Layered design (top → bottom):**

1. **POC Runner** (`Minerva.Interop.Poc/Program.cs`) — orchestrates detect/create/verify/benchmark/observe phases via `IComputeBackend`.
2. **`IComputeBackend`** (`Compute/IComputeBackend.cs`) — uniform interface (`Allocate`, `SyncToDevice`, `SyncFromDevice`, `MatMul`, `Add`, `Synchronize`) with a `ComputeBackend.Create()` factory.
3. **Backend implementations** — `MetalBackend`, `CudaBackend`, `CpuBackend`. GPU failure gracefully degrades to CPU.
4. **`TensorBuffer<T>`** (`Memory/TensorBuffer.cs`) — core abstraction: dual `CpuPointer`/`GpuPointer`, `MemoryResidence` enum (`Cpu`, `Unified`, `Device`, `Staged`), callback disposer, `AsSpan()`.
5. **P/Invoke Bindings** (`Bindings/`) — `[LibraryImport]` on `partial` methods; three logical library names: `"tensor_blas"`, `"tensor_gpu"`, `"tensor_cublas"`.
6. **`NativeResolver`** (`Platform/NativeResolver.cs`) — `NativeLibrary.SetDllImportResolver` mapping logical names to OS-specific paths at startup.

**Memory models per backend:**
- **Metal:** `Unified` — single pointer, `SyncToDevice`/`SyncFromDevice` are no-ops (UMA/`MTLStorageModeShared`).
- **CUDA:** `Staged` — separate `NativeMemory` CPU buffer + `cudaMalloc` GPU buffer; explicit `cudaMemcpy`.
- **CPU:** `Cpu` — `NativeMemory.AlignedAlloc` at 64-byte alignment (AVX-512/NEON); no GPU pointer.

**Native code** lives in `native/metal/` (Obj-C Metal bridge + MSL shaders, built via Makefile) and `native/cuda/` (CUDA kernels, not needed for POC — ops go through `libcudart`/`libcublas` directly).

## Conventions

- `AllowUnsafeBlocks` and nullable reference types enabled in both projects
- No managed allocations on hot paths — use `NativeMemory`, `Span<T>`, pointers
- `[LibraryImport]` (source-generated) only, never legacy `[DllImport]`
- Domain-specific naming — no `Handler`, `Manager`, `Utility` suffixes
- Architecture decision records in `docs/` (see `ADR-001-interop-strategy.md`)
