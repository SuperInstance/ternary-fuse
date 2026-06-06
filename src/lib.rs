//! # ternary-fuse
//!
//! Operator fusion for ternary networks. Instead of running matmul → add bias → 
//! activate as three separate passes over memory, fuse them into a single pass.
//! For ternary weights, the fused kernel is dramatically simpler than float kernels.

pub type Trit = i8;

/// A fused operation in a compute graph.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// Element-wise ternary addition: a + b in Z₃
    Add,
    /// Ternary matrix multiply: C = A × B in Z₃
    Matmul { rows: usize, inner: usize, cols: usize },
    /// Bias addition (broadcast): a[i] + b[j] for each row
    BiasAdd { size: usize },
    /// Activation: ternary ReLU (negatives → 0)
    ReLU,
    /// Activation: ternary sign (identity for {-1,0,1})
    Sign,
    /// Activation: ternary tanh (continuous → nearest trit)
    Tanh,
    /// Scale: multiply each trit by a float, then re-ternarize
    Scale { factor: f64 },
    /// Reshape: no-op, just changes metadata
    Reshape { new_shape: Vec<usize> },
    /// Transpose: swap two dimensions
    Transpose { dim_a: usize, dim_b: usize },
}

impl Op {
    /// Estimate memory reads/writes for this op.
    pub fn memory_ops(&self, input_size: usize) -> (usize, usize) {
        match self {
            Op::Add => (input_size * 2, input_size),
            Op::Matmul { rows, cols, .. } => (*rows * *cols, *rows * *cols),
            Op::BiasAdd { size } => (input_size + size, input_size),
            Op::ReLU | Op::Sign | Op::Tanh => (input_size, input_size),
            Op::Scale { .. } => (input_size, input_size),
            Op::Reshape { .. } => (0, 0),
            Op::Transpose { .. } => (input_size, input_size),
        }
    }
}

/// A sequence of operations that can be fused into a single kernel.
#[derive(Debug, Clone)]
pub struct FusedKernel {
    pub ops: Vec<Op>,
    pub fused: bool,
}

impl FusedKernel {
    pub fn new(ops: Vec<Op>) -> Self {
        Self { ops, fused: false }
    }

    /// Try to fuse adjacent operations. Returns number of fusions performed.
    pub fn fuse(&mut self) -> usize {
        let mut fused_count = 0;
        let mut result = Vec::new();
        let mut i = 0;
        let ops = self.ops.clone();

        while i < ops.len() {
            if i + 1 < ops.len() && Self::can_fuse(&ops[i], &ops[i + 1]) {
                result.push(Self::create_fused(&ops[i], &ops[i + 1]));
                fused_count += 1;
                i += 2;
            } else {
                result.push(ops[i].clone());
                i += 1;
            }
        }

        self.ops = result;
        self.fused = fused_count > 0;
        fused_count
    }

    /// Check if two adjacent ops can be fused.
    fn can_fuse(a: &Op, b: &Op) -> bool {
        matches!((a, b),
            // Matmul + Bias → single pass
            (Op::Matmul { .. }, Op::BiasAdd { .. }) |
            // Matmul + ReLU → single pass (ternary ReLU is trivial)
            (Op::Matmul { .. }, Op::ReLU) |
            // Matmul + Sign → single pass
            (Op::Matmul { .. }, Op::Sign) |
            // Bias + ReLU → single pass
            (Op::BiasAdd { .. }, Op::ReLU) |
            // Bias + Sign → single pass
            (Op::BiasAdd { .. }, Op::Sign) |
            // Scale + Activation → single pass
            (Op::Scale { .. }, Op::ReLU) |
            (Op::Scale { .. }, Op::Sign) |
            (Op::Scale { .. }, Op::Tanh) |
            // Add + Activation → single pass
            (Op::Add, Op::ReLU) |
            (Op::Add, Op::Sign)
        )
    }

    /// Create a fused representation (for analysis; actual execution is fused in the kernel).
    fn create_fused(a: &Op, b: &Op) -> Op {
        match (a, b) {
            (Op::Matmul { rows, inner, cols }, Op::BiasAdd { .. }) =>
                Op::Matmul { rows: *rows, inner: *inner, cols: *cols }, // bias absorbed
            (Op::Matmul { rows, inner, cols }, Op::ReLU) =>
                Op::Matmul { rows: *rows, inner: *inner, cols: *cols }, // relu absorbed
            (Op::Matmul { rows, inner, cols }, Op::Sign) =>
                Op::Matmul { rows: *rows, inner: *inner, cols: *cols }, // sign absorbed
            _ => a.clone(),
        }
    }

    /// Total memory reads/writes (unfused).
    pub fn memory_ops_unfused(&self, input_size: usize) -> (usize, usize) {
        self.ops.iter()
            .map(|op| op.memory_ops(input_size))
            .fold((0, 0), |(r, w), (mr, mw)| (r + mr, w + mw))
    }

    /// Number of operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// Fused ternary matmul + bias + activation: single-pass kernel.
pub fn fused_matmul_bias_relu(
    a: &[Trit], b: &[Trit], bias: &[Trit],
    rows: usize, inner: usize, cols: usize
) -> Vec<Trit> {
    let mut result = vec![0i8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0i32;
            for k in 0..inner {
                let av = a[r * inner + k] as i32;
                let bv = b[k * cols + c] as i32;
                // Ternary multiply
                acc += match (av, bv) {
                    (-1, -1) => 1,
                    (-1, 0) | (0, _) | (_, 0) => 0,
                    (-1, 1) => -1,
                    (1, -1) => -1,
                    (1, 1) => 1,
                    _ => 0,
                };
            }
            // Add bias
            let biased = acc + bias[c] as i32;
            // ReLU: clamp negatives to 0, keep 0 and positive
            result[r * cols + c] = if biased < 0 { 0 }
                else if biased > 0 { 1 }
                else { 0 };
        }
    }
    result
}

/// Fused add + ternarize: element-wise Z₃ addition with optional activation.
pub fn fused_add_activate(a: &[Trit], b: &[Trit], activate: bool) -> Vec<Trit> {
    a.iter().zip(b.iter()).map(|(&av, &bv)| {
        // Z₃ addition
        let sum = match (av, bv) {
            (-1, -1) => 1,
            (-1, 0) | (0, -1) => -1,
            (-1, 1) | (0, 0) | (1, -1) => 0,
            (0, 1) | (1, 0) => 1,
            (1, 1) => -1, // wraps in Z₃
            _ => 0,
        };
        if activate {
            if sum < 0 { 0 } else { sum }
        } else {
            sum
        }
    }).collect()
}

/// Fusion analysis: how much memory bandwidth is saved.
#[derive(Debug)]
pub struct FusionAnalysis {
    pub original_ops: usize,
    pub fused_ops: usize,
    pub fusions_performed: usize,
    pub memory_saved_fraction: f64,
}

impl FusionAnalysis {
    pub fn analyze(kernel: &mut FusedKernel, input_size: usize) -> Self {
        let (orig_reads, orig_writes) = kernel.memory_ops_unfused(input_size);
        let original_ops = kernel.len();
        let fusions = kernel.fuse();
        let (fused_reads, fused_writes) = kernel.memory_ops_unfused(input_size);
        let orig_total = orig_reads + orig_writes;
        let fused_total = fused_reads + fused_writes;
        let saved = if orig_total > 0 {
            1.0 - (fused_total as f64 / orig_total as f64)
        } else { 0.0 };

        Self {
            original_ops,
            fused_ops: kernel.len(),
            fusions_performed: fusions,
            memory_saved_fraction: saved,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_matmul_bias_relu() {
        // 2x2 matmul with 2x2 weight, 1x2 bias
        let a = vec![1, 0, -1, 1]; // 2x2
        let b = vec![1, -1, 0, 1]; // 2x2
        let bias = vec![0, 1];
        let result = fused_matmul_bias_relu(&a, &b, &bias, 2, 2, 2);
        assert_eq!(result.len(), 4);
        // All values should be 0 or 1 (ReLU applied)
        for &t in &result {
            assert!(t == 0 || t == 1);
        }
    }

    #[test]
    fn test_fused_add_activate() {
        let a = vec![-1, 0, 1, -1];
        let b = vec![1, 0, -1, 1];
        let result = fused_add_activate(&a, &b, true);
        // ReLU: negatives become 0
        for &t in &result {
            assert!(t == 0 || t == 1);
        }
    }

    #[test]
    fn test_fused_add_no_activate() {
        let a = vec![-1, 0, 1];
        let b = vec![1, 0, -1];
        let result = fused_add_activate(&a, &b, false);
        assert_eq!(result, vec![0, 0, 0]); // Z₃: -1+1=0, 0+0=0, 1+(-1)=0
    }

    #[test]
    fn test_kernel_fuse_matmul_bias() {
        let mut kernel = FusedKernel::new(vec![
            Op::Matmul { rows: 4, inner: 4, cols: 4 },
            Op::BiasAdd { size: 4 },
        ]);
        let fusions = kernel.fuse();
        assert_eq!(fusions, 1);
        assert_eq!(kernel.len(), 1); // fused into single op
    }

    #[test]
    fn test_kernel_fuse_bias_relu() {
        let mut kernel = FusedKernel::new(vec![
            Op::BiasAdd { size: 8 },
            Op::ReLU,
        ]);
        let fusions = kernel.fuse();
        assert_eq!(fusions, 1);
    }

    #[test]
    fn test_kernel_no_fuse_unrelated() {
        let mut kernel = FusedKernel::new(vec![
            Op::Reshape { new_shape: vec![2, 4] },
            Op::Transpose { dim_a: 0, dim_b: 1 },
        ]);
        let fusions = kernel.fuse();
        assert_eq!(fusions, 0);
    }

    #[test]
    fn test_kernel_multi_fuse() {
        let mut kernel = FusedKernel::new(vec![
            Op::Matmul { rows: 4, inner: 4, cols: 4 },
            Op::BiasAdd { size: 4 },
            Op::ReLU,
            Op::Scale { factor: 0.5 },
        ]);
        let fusions = kernel.fuse();
        assert!(fusions >= 1);
    }

    #[test]
    fn test_fusion_analysis() {
        let mut kernel = FusedKernel::new(vec![
            Op::Matmul { rows: 4, inner: 4, cols: 4 },
            Op::BiasAdd { size: 4 },
            Op::ReLU,
        ]);
        let analysis = FusionAnalysis::analyze(&mut kernel, 16);
        assert!(analysis.fusions_performed >= 1);
        assert!(analysis.fused_ops < analysis.original_ops);
    }

    #[test]
    fn test_op_memory_estimate() {
        let op = Op::Matmul { rows: 4, inner: 4, cols: 4 };
        let (reads, writes) = op.memory_ops(16);
        assert!(reads > 0);
        assert!(writes > 0);
    }
}
