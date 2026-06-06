# ternary-fuse

*One pass over memory instead of three. The operation disappears into the one next to it.*

---

Operator fusion for ternary networks. Instead of running matmul → add bias → activate as three separate kernel launches with three memory round-trips, fuse them into a single pass. For ternary weights, the fused kernel is dramatically simpler than float kernels — no multiply, just sign matching.

The crate provides: a `FusedKernel` builder with fuse analysis, fusion rules for common op pairs (matmul+bias, bias+relu, scale+activation), a fused_matmul_bias_relu single-pass implementation, fused_add_activate with Z₃ arithmetic, and a FusionAnalysis that measures memory bandwidth savings.

9 tests: fused matmul+bias+relu, fused add+activate, Z₃ addition, kernel fusion (matmul+bias, bias+relu, multi-fuse), no-fuse for unrelated ops, fusion analysis, memory estimates.

Part of [SuperInstance](https://github.com/SuperInstance/SuperInstance).

License: MIT
