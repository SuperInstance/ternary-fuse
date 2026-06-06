# ternary-fuse

*Operator fusion for ternary networks. Matmul → bias → activate in a single memory pass.*

## Why This Exists

In float networks, operator fusion is an optimization — you save memory bandwidth by combining operations. In ternary networks, it's the entire point. Ternary matmul uses XNOR+popcount (no multipliers). Ternary bias is integer addition. Ternary activation is sign(). These three operations are so simple that running them separately means you're spending more time loading/storing data than computing.

This crate fuses the most common ternary operation chains into single-pass kernels that never write intermediates to memory.

## Architecture

```
UNFUSED (3 memory passes):
  Load weights → matmul → Store intermediates
  Load intermediates → add bias → Store again
  Load again → sign() → Store output

FUSED (1 memory pass):
  Load weights → matmul → add bias → sign() → Store output
```

### Key Operations

- **`fused_matmul_bias_relu`** — The workhorse: matrix multiply, add bias, apply ReLU (sign threshold) in one pass
- **`fused_add_activate`** — Element-wise add followed by ternary activation
- **`FusionAnalysis`** — Analyze which operations can be fused for a given model architecture

## Usage

```rust
use ternary_fuse::*;

let weights: &[i8] = &[-1, 1, 0, -1];  // 2x2 ternary matrix
let input: &[i8] = &[1, -1];
let bias: &[i8] = &[0, 1];

let output = fused_matmul_bias_relu(weights, input, bias, 2, 2);
// Output is ternary: {-1, 0, +1} with no intermediate allocations

// Analyze what can be fused
let analysis = FusionAnalysis::analyze(&["matmul", "bias", "relu"]);
assert!(analysis.fusable);
```

## The Deeper Idea

Fusion is where ternary networks stop being "interesting research" and start being "practical advantage." A fused ternary kernel has:
- 0 floating-point operations
- 0 multiplier units needed  
- 1 memory pass instead of 3

On hardware with XNOR+popcount support (which is... everything), the fused kernel runs at essentially memory bandwidth speed. The computation is free; only the data movement costs.

## Related Crates

- `ternary-matmul` — Unfused ternary matrix multiply
- `ternary-activation` — Standalone activation functions
- `ternary-kernel-launch` — GPU kernel launch infrastructure
- `ternary-accumulator` — Gradient accumulation (the training counterpart)
