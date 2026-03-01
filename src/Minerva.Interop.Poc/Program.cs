using System.Diagnostics;
using Minerva.Interop.Compute;
using Minerva.Interop.Memory;
using Minerva.Interop.Platform;

namespace Minerva.Interop.Poc;

/// <summary>
/// Minerva Interop POC — Cross-Platform Tensor Compute Validation
///
/// This program validates that .NET can call into platform-specific native
/// compute libraries (Metal, CUDA, CPU BLAS) through a single abstraction.
///
/// What it proves:
///   1. P/Invoke via [LibraryImport] works for BLAS, CUDA, and Metal bridge
///   2. NativeLibrary resolver dispatches correctly per platform
///   3. TensorBuffer abstraction spans UMA and discrete memory models
///   4. Correctness: GPU results match CPU reference computation
///   5. Basic timing characteristics of each memory/compute path
///
/// What it does NOT prove (yet — Emergence will surface these):
///   - Production performance characteristics
///   - Error recovery and edge cases
///   - Concurrent dispatch / async compute
///   - Memory pressure behavior
/// </summary>
internal static class Program
{
    // Matrix dimensions for the benchmark
    private const int M = 1024;
    private const int N = 1024;
    private const int K = 1024;

    // Small dimensions for correctness verification
    private const int VerifyDim = 4;

    // Number of warmup + timed iterations
    private const int WarmupRuns = 2;
    private const int TimedRuns = 5;

    static int Main(string[] args)
    {
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine("  Minerva Interop POC — Cross-Platform Tensor Compute");
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine();

        // ── Phase 1: Platform Detection ──────────────────────────────────
        Console.WriteLine("Phase 1: Platform Detection");
        Console.WriteLine("───────────────────────────────────────────────────────────────");

        var capability = PlatformDetector.Detect();
        NativeResolver.Register(capability);

        Console.WriteLine($"  OS:   {capability.Os}");
        Console.WriteLine($"  BLAS: {capability.BlasProvider}");
        Console.WriteLine($"  GPU:  {capability.GpuProvider}" +
            (capability.GpuDeviceName is not null ? $" ({capability.GpuDeviceName})" : ""));
        Console.WriteLine();

        // ── Phase 2: Backend Creation ────────────────────────────────────
        Console.WriteLine("Phase 2: Backend Creation");
        Console.WriteLine("───────────────────────────────────────────────────────────────");

        IComputeBackend? gpuBackend = null;
        IComputeBackend? cpuBackend = null;

        try
        {
            if (capability.HasGpu)
            {
                gpuBackend = ComputeBackend.Create(capability);
                Console.WriteLine($"  GPU backend: {gpuBackend.Name}");
                Console.WriteLine($"  Memory model: {gpuBackend.DefaultResidence}");
            }
            else
            {
                Console.WriteLine("  No GPU detected — GPU tests will be skipped.");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  GPU backend creation failed: {ex.Message}");
            Console.WriteLine("  Continuing with CPU only.");
        }

        try
        {
            cpuBackend = ComputeBackend.CreateCpuOnly();
            Console.WriteLine($"  CPU backend: {cpuBackend.Name}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  CPU backend creation failed: {ex.Message}");
            if (gpuBackend is null)
            {
                Console.Error.WriteLine("FATAL: No compute backend available.");
                return 1;
            }
        }

        Console.WriteLine();

        // ── Phase 3: Correctness Verification ────────────────────────────
        Console.WriteLine("Phase 3: Correctness Verification (4×4 matmul)");
        Console.WriteLine("───────────────────────────────────────────────────────────────");

        if (cpuBackend is not null)
            VerifyCorrectness(cpuBackend, "CPU");

        if (gpuBackend is not null)
            VerifyCorrectness(gpuBackend, "GPU", cpuBackend);

        Console.WriteLine();

        // ── Phase 4: Performance Measurement ─────────────────────────────
        Console.WriteLine($"Phase 4: Performance ({M}×{K} × {K}×{N} matmul, {TimedRuns} runs)");
        Console.WriteLine("───────────────────────────────────────────────────────────────");

        if (cpuBackend is not null)
            BenchmarkMatMul(cpuBackend, "CPU BLAS");

        if (gpuBackend is not null)
            BenchmarkMatMul(gpuBackend, "GPU");

        Console.WriteLine();

        // ── Phase 5: Memory Model Observations ───────────────────────────
        Console.WriteLine("Phase 5: Memory Model Observations");
        Console.WriteLine("───────────────────────────────────────────────────────────────");

        if (gpuBackend is not null)
            ObserveMemoryModel(gpuBackend);

        if (cpuBackend is not null)
            ObserveMemoryModel(cpuBackend);

        Console.WriteLine();

        // ── Cleanup ──────────────────────────────────────────────────────
        gpuBackend?.Dispose();
        cpuBackend?.Dispose();

        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine("  POC complete. Emergence observations logged above.");
        Console.WriteLine("═══════════════════════════════════════════════════════════════");

        return 0;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Verification: small matrix with known results
    // ─────────────────────────────────────────────────────────────────────────

    static void VerifyCorrectness(IComputeBackend backend, string label,
                                   IComputeBackend? referenceBackend = null)
    {
        const int dim = VerifyDim;

        using var a = backend.Allocate(dim, dim);
        using var b = backend.Allocate(dim, dim);
        using var c = backend.Allocate(dim, dim);

        // Fill A with row index + 1, B with identity
        var spanA = a.AsSpan();
        var spanB = b.AsSpan();

        for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
        {
            spanA[i * dim + j] = i + 1;          // A = [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
            spanB[i * dim + j] = (i == j) ? 1 : 0; // B = identity
        }

        // Transfer to device if needed
        backend.SyncToDevice(a);
        backend.SyncToDevice(b);

        // Compute C = A × I = A
        backend.MatMul(a, b, c);
        backend.Synchronize();
        backend.SyncFromDevice(c);

        var spanC = c.AsSpan();

        // Verify: C should equal A (since B is identity)
        bool correct = true;
        float maxError = 0;

        for (int i = 0; i < dim * dim; i++)
        {
            float error = MathF.Abs(spanC[i] - spanA[i]);
            maxError = MathF.Max(maxError, error);
            if (error > 1e-5f)
                correct = false;
        }

        Console.WriteLine($"  {label}: C = A × I → " +
            (correct ? "PASS" : "FAIL") +
            $" (max error: {maxError:E2})");

        // If we have a reference backend, also verify a non-trivial multiplication
        if (referenceBackend is not null)
            VerifyCrossBackend(backend, referenceBackend, label);
    }

    static void VerifyCrossBackend(IComputeBackend testBackend, IComputeBackend refBackend, string label)
    {
        const int dim = VerifyDim;
        var rng = new Random(42);

        // Reference computation on CPU
        using var refA = refBackend.Allocate(dim, dim);
        using var refB = refBackend.Allocate(dim, dim);
        using var refC = refBackend.Allocate(dim, dim);

        var refSpanA = refA.AsSpan();
        var refSpanB = refB.AsSpan();

        for (int i = 0; i < dim * dim; i++)
        {
            refSpanA[i] = (float)(rng.NextDouble() * 2 - 1);
            refSpanB[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        refBackend.MatMul(refA, refB, refC);
        refBackend.Synchronize();
        var refSpanC = refC.AsSpan();

        // Same computation on test backend
        using var testA = testBackend.Allocate(dim, dim);
        using var testB = testBackend.Allocate(dim, dim);
        using var testC = testBackend.Allocate(dim, dim);

        // Copy same input data
        rng = new Random(42); // reset seed
        var testSpanA = testA.AsSpan();
        var testSpanB = testB.AsSpan();

        for (int i = 0; i < dim * dim; i++)
        {
            testSpanA[i] = (float)(rng.NextDouble() * 2 - 1);
            testSpanB[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        testBackend.SyncToDevice(testA);
        testBackend.SyncToDevice(testB);

        testBackend.MatMul(testA, testB, testC);
        testBackend.Synchronize();
        testBackend.SyncFromDevice(testC);

        var testSpanC = testC.AsSpan();

        // Compare results
        float maxError = 0;
        for (int i = 0; i < dim * dim; i++)
        {
            float error = MathF.Abs(testSpanC[i] - refSpanC[i]);
            maxError = MathF.Max(maxError, error);
        }

        bool correct = maxError < 1e-4f; // float32 tolerance
        Console.WriteLine($"  {label} vs CPU reference: " +
            (correct ? "PASS" : "FAIL") +
            $" (max error: {maxError:E2})");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Benchmark: large matmul timing
    // ─────────────────────────────────────────────────────────────────────────

    static void BenchmarkMatMul(IComputeBackend backend, string label)
    {
        using var a = backend.Allocate(M, K);
        using var b = backend.Allocate(K, N);
        using var c = backend.Allocate(M, N);

        // Fill with non-trivial data
        var spanA = a.AsSpan();
        var spanB = b.AsSpan();
        for (int i = 0; i < spanA.Length; i++) spanA[i] = 0.01f * (i % 100);
        for (int i = 0; i < spanB.Length; i++) spanB[i] = 0.01f * ((i + 37) % 100);

        backend.SyncToDevice(a);
        backend.SyncToDevice(b);

        // Warmup
        for (int i = 0; i < WarmupRuns; i++)
        {
            backend.MatMul(a, b, c);
            backend.Synchronize();
        }

        // Timed runs
        var sw = Stopwatch.StartNew();
        var times = new double[TimedRuns];

        for (int i = 0; i < TimedRuns; i++)
        {
            sw.Restart();
            backend.MatMul(a, b, c);
            backend.Synchronize();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }

        backend.SyncFromDevice(c);

        double avg = times.Average();
        double min = times.Min();
        double max = times.Max();

        // GFLOPS: matmul is 2*M*N*K floating point operations
        double gflops = (2.0 * M * N * K) / (avg * 1e-3) / 1e9;

        Console.WriteLine($"  {label,-16} avg={avg,8:F2}ms  min={min,8:F2}ms  max={max,8:F2}ms  ~{gflops:F1} GFLOPS");
        Console.WriteLine($"  {"",-16} c[0]={c.AsSpan()[0]:F4}  c[{M * N - 1}]={c.AsSpan()[M * N - 1]:F4}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Memory model observation
    // ─────────────────────────────────────────────────────────────────────────

    static void ObserveMemoryModel(IComputeBackend backend)
    {
        const int testSize = 256;

        using var buf = backend.Allocate(testSize, testSize);

        Console.WriteLine($"  {backend.Name}:");
        Console.WriteLine($"    Residence: {buf.Residence}");
        Console.WriteLine($"    CPU accessible: {buf.IsCpuAccessible}");
        Console.WriteLine($"    GPU accessible: {buf.IsGpuAccessible}");

        unsafe
        {
            bool samePointer = buf.CpuPointer != null && buf.GpuPointer != null &&
                               (nint)buf.CpuPointer == (nint)buf.GpuPointer;
            Console.WriteLine($"    CPU==GPU pointer: {samePointer}" +
                (samePointer ? " (UMA confirmed — zero-copy)" : ""));
        }

        // Measure allocation + fill cost
        var sw = Stopwatch.StartNew();
        using var bench = backend.Allocate(M, N);
        var allocMs = sw.Elapsed.TotalMilliseconds;

        sw.Restart();
        bench.AsSpan().Fill(1.0f);
        var fillMs = sw.Elapsed.TotalMilliseconds;

        sw.Restart();
        backend.SyncToDevice(bench);
        var syncMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine($"    Alloc {M}×{N}: {allocMs:F2}ms");
        Console.WriteLine($"    Fill (CPU):  {fillMs:F2}ms");
        Console.WriteLine($"    Sync→Device: {syncMs:F2}ms" +
            (buf.Residence == MemoryResidence.Unified ? " (no-op, UMA)" : ""));
    }
}
