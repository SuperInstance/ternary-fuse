# ternary-fuse

**Operator fusion for ternary networks. Matmul → bias → activate in a single memory pass — because the computation is free, only the data movement costs.**

## Why This Exists

Here's the thing about ternary neural networks that most papers gloss over: the individual operations are *embarrassingly* simple. Ternary matmul is XNOR+popcount — no multipliers needed. Ternary bias is integer addition. Ternary activation is `sign()` or a clamp. These aren't the bottleneck.

The bottleneck is memory. Every time you write an intermediate result to DRAM and read it back, you've spent 100-1000× more energy than the computation itself. In float networks, fusion is a nice optimization. In ternary networks, it's the *entire point* — if you're not fusing, you're doing it wrong.

This crate takes the most common ternary operation chains and collapses them into single-pass kernels that never write intermediates.

```
UNFUSED (3 memory round-trips):
  DRAM → registers: load weights          [~100 cycles]
  registers → DRAM: store matmul result   [~100 cycles]
  DRAM → registers: reload for bias       [~100 cycles]
  registers → DRAM: store biased result   [~100 cycles]
  DRAM → registers: reload for activation [~100 cycles]
  registers → DRAM: store final output    [~100 cycles]
  
  Computation: ~10 cycles. Data movement: ~600 cycles.

FUSED (1 memory round-trip):
  DRAM → registers: load weights
  registers: matmul → bias → activate     [~10 cycles]
  registers → DRAM: store final output    [~100 cycles]
  
  Computation: ~10 cycles. Data movement: ~200 cycles.
```

Three passes become one. Not 3× faster — more like 3× less memory traffic, which on real hardware is what actually matters.

## The Key Insight

Fusion rules for ternary operations are *different* from float operations. In float land, you fuse matmul+bias because it saves one allocation. In ternary land, the fusion is *semantically simpler* — the entire chain operates in Z₃ (integers mod 3, mapped to {-1, 0, +1}). There are no numerical precision concerns from fusing. No accumulation errors. The math is exact.

This means we can fuse more aggressively:
- **Matmul + BiasAdd** → bias gets absorbed into the final accumulation
- **Matmul + Activation** → sign/clamp happens inside the inner loop
- **BiasAdd + Activation** → one pass, no intermediate
- **Scale + Activation** → multiply-then-ternarize in one step
- **Add + Activation** → element-wise Z₃ addition with immediate ternarization

The `FusedKernel` type models these rules and tells you exactly how many fusions are possible for any given compute graph.

## Quick Start

```rust
use ternary_fuse::*;

// The workhorse: fused matmul + bias + ReLU in one pass
let weights = vec![1i8, 0, -1, 1, -1, 0, 1, -1]; // 2×4 ternary matrix
let input   = vec![1i8, -1, 0, 1];                 // 4-element ternary vector
let bias    = vec![0i8, 1];                         // 2-element bias

let output = fused_matmul_bias_relu(&weights, &input, &bias, 2, 4, 2);
// output.len() == 2, all values in {0, 1} (ReLU clamped)
// Zero intermediate allocations. One pass through the data.

// Fused Z₃ addition with activation
let a = vec![-1i8, 0, 1, -1];
let b = vec![1i8, 0, -1, 1];
let sum = fused_add_activate(&a, &b, true);
// Z₃ arithmetic: -1+1=0, 0+0=0, 1+(-1)=0, -1+1=0
// With ReLU: negatives become 0 → all zeros
```

## Architecture

### Operation Model

Every operation in the fused pipeline is an `Op`:

```rust
pub enum Op {
    Add,                                    // Element-wise Z₃ addition
    Matmul { rows, inner, cols },           // Ternary matrix multiply
    BiasAdd { size },                       // Broadcast bias addition
    ReLU,                                   // Clamp negatives to 0
    Sign,                                   // Sign function (identity for {-1,0,1})
    Tanh,                                   // Continuous → nearest trit
    Scale { factor: f64 },                  // Float scale + re-ternarize
    Reshape { new_shape: Vec<usize> },      // Metadata only, no data movement
    Transpose { dim_a, dim_b },             // Dimension swap
}
```

Each `Op` can estimate its memory reads/writes via `memory_ops()`, which makes it possible to *quantify* how much bandwidth fusion saves — not guess.

### Fusion Engine

```rust
// Build a compute graph
let mut kernel = FusedKernel::new(vec![
    Op::Matmul { rows: 128, inner: 256, cols: 64 },
    Op::BiasAdd { size: 64 },
    Op::ReLU,
]);

// Fuse it
let fusions = kernel.fuse();  // Returns 2 (matmul+bias, then result+relu)

// Analyze the savings
let analysis = FusionAnalysis::analyze(&mut kernel, 128 * 256);
println!("Memory saved: {:.1}%", analysis.memory_saved_fraction * 100.0);
println!("Ops reduced: {} → {}", analysis.original_ops, analysis.fused_ops);
```

### Fusion Rules

Not everything can be fused. The rules are explicit and conservative:

| Producer | Consumer | Fuses? | Why |
|----------|----------|--------|-----|
| Matmul | BiasAdd | ✅ | Bias absorbed into accumulator |
| Matmul | ReLU | ✅ | Clamp inside inner loop |
| Matmul | Sign | ✅ | Sign inside inner loop |
| BiasAdd | ReLU | ✅ | Sequential element-wise |
| BiasAdd | Sign | ✅ | Sequential element-wise |
| Scale | ReLU/Sign/Tanh | ✅ | Scale+ternarize in one pass |
| Add | ReLU/Sign | ✅ | Add+clamp together |
| Reshape | Transpose | ❌ | Metadata ops, nothing to fuse |

When two adjacent operations can't be fused, they remain separate — no magic, no hidden costs.

### `FusionAnalysis`

This type does something quietly useful: it computes the *actual* memory savings as a fraction. Not a theoretical upper bound — the real number based on your specific graph.

```rust
pub struct FusionAnalysis {
    pub original_ops: usize,          // Ops before fusion
    pub fused_ops: usize,             // Ops after fusion
    pub fusions_performed: usize,     // Number of fusions
    pub memory_saved_fraction: f64,   // 0.0 to ~0.67 (at best, 2/3 saved)
}
```

For a typical ternary layer (matmul → bias → ReLU), you'll see `memory_saved_fraction` around 0.5-0.67 — half to two-thirds of memory traffic eliminated.

## API Reference

### Core Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `fused_matmul_bias_relu` | `(a: &[i8], b: &[i8], bias: &[i8], rows: usize, inner: usize, cols: usize) → Vec<i8>` | Single-pass matmul + bias + ReLU |
| `fused_add_activate` | `(a: &[i8], b: &[i8], activate: bool) → Vec<i8>` | Z₃ addition, optionally followed by ReLU |

### FusedKernel

| Method | Description |
|--------|-------------|
| `new(ops: Vec<Op>)` | Build a kernel from an op sequence |
| `fuse() → usize` | Fuse adjacent ops, returns count of fusions |
| `memory_ops_unfused(input_size) → (reads, writes)` | Total memory traffic (unfused baseline) |
| `len() → usize` | Number of ops |
| `is_empty() → bool` | Empty check |

### FusionAnalysis

| Method | Description |
|--------|-------------|
| `analyze(kernel: &mut FusedKernel, input_size: usize) → FusionAnalysis` | Fuse + measure savings |

### Op

| Method | Description |
|--------|-------------|
| `memory_ops(input_size) → (reads, writes)` | Per-op memory traffic estimate |

## Real-World Example: Quantifying Fusion for a Ternary Layer

```rust
use ternary_fuse::*;

// A typical ternary linear layer: 784 → 256 → 10
// Layer 1: matmul(784×256) + bias(256) + ReLU
// Layer 2: matmul(256×10) + bias(10) + sign

for (name, rows, inner, cols) in [("hidden", 784, 256, 256), ("output", 256, 10, 10)] {
    let mut kernel = FusedKernel::new(vec![
        Op::Matmul { rows, inner: *inner, cols },
        Op::BiasAdd { size: *cols },
        Op::ReLU,
    ]);
    
    let analysis = FusionAnalysis::analyze(&mut kernel, rows * inner);
    println!(
        "{}: {} ops → {} ops, {:.1}% memory saved",
        name, analysis.original_ops, analysis.fused_ops,
        analysis.memory_saved_fraction * 100.0
    );
}
```

## Performance Characteristics

- **Zero intermediate allocations** — fused kernels produce output directly
- **Exact arithmetic** — no floating-point accumulation errors in the fusion
- **Cache-friendly** — single-pass means each cache line is loaded once
- **No unsafe code** — pure safe Rust
- **No dependencies** — no external crates, no build complexity

On hardware with XNOR+popcount support (which is effectively all modern CPUs), the fused kernel runs at memory bandwidth speed. The computation genuinely is free — the only cost is moving data, and fusion cuts that by 50-67%.

## Ecosystem Connections

- **`ternary-matmul`** — Unfused ternary matrix multiply (the building block)
- **`ternary-activation`** — Standalone activation functions (ReLU, sign, tanh)
- **`ternary-kernel-launch`** — GPU kernel launch infrastructure
- **`ternary-accumulator`** — Gradient accumulation (training-time counterpart)
- **`ternary-compiler`** — Higher-level graph compiler that emits fused kernels

## Open Questions

- **GPU kernels**: The current implementation is CPU-only. The fusion analysis is architecture-independent, but the actual fused matmul should use CUDA/HIP shared memory for GPU targets.
- **Larger fusions**: Currently limited to pairwise fusion. Could a three-way fusion (matmul + bias + activate as one unit) be modeled more efficiently?
- **Quantized scale**: The `Scale` op involves floating-point multiplication followed by re-ternarization. Is there a purely-integer path that avoids the float?
- **Streaming**: For very large matrices that don't fit in cache, can the fused kernel be applied in tiles?

## License

Apache-2.0
