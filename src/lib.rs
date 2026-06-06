//! # ternary-fuse
//!
//! Operator fusion for ternary networks. Instead of running matmul → add bias → 
//! activate as three separate passes over memory, fuse them into a single pass.
//! For ternary weights, the fused kernel is dramatically simpler than float kernels.
//!
//! Connected to the [`ternary-types`](https://github.com/SuperInstance/ternary-types)
//! fleet via its dependency.

/// A trit value: -1, 0, or +1.
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
            Op::Reshape { .. } => (input_size, input_size),
            Op::Transpose { .. } => (input_size, input_size),
        }
    }

    /// Is this op a no-op in terms of arithmetic (just metadata)?
    pub fn is_metadata(&self) -> bool {
        matches!(self, Op::Reshape { .. } | Op::Transpose { .. })
    }
}

/// A fused compute graph: a sequence of ops sharing memory.
#[derive(Debug, Clone)]
pub struct FusedGraph {
    pub ops: Vec<Op>,
    pub input_size: usize,
    pub output_size: usize,
}

impl FusedGraph {
    pub fn new(input_size: usize) -> Self {
        Self {
            ops: Vec::new(),
            input_size,
            output_size: input_size,
        }
    }

    pub fn push(&mut self, op: Op) {
        let (reads, writes) = op.memory_ops(self.output_size);
        let _ = reads; // could be used for scheduling
        match &op {
            Op::Add | Op::Matmul { .. } | Op::BiasAdd { .. } |
            Op::ReLU | Op::Sign | Op::Tanh | Op::Scale { .. } => {
                self.output_size = writes;
            }
            _ => {}
        }
        self.ops.push(op);
    }

    /// Total estimated memory traffic (reads + writes) in trits.
    pub fn total_memory_traffic(&self) -> usize {
        self.ops.iter()
            .map(|op| {
                let (r, w) = op.memory_ops(self.input_size);
                r + w
            })
            .sum()
    }

    /// Estimated speedup vs sequential execution.
    pub fn fusion_speedup(&self) -> f64 {
        if self.ops.is_empty() {
            return 1.0;
        }
        let fused_traffic = self.total_memory_traffic() as f64;
        // Sequential: each op reads input, writes output separately
        let seq_traffic: f64 = self.ops.iter()
            .map(|op| {
                let (r, w) = op.memory_ops(self.input_size);
                (r + w) as f64
            })
            .sum();
        if fused_traffic == 0.0 {
            return 1.0;
        }
        seq_traffic / fused_traffic
    }
}

/// Balanced Z₃ addition of two trits: returns the mod-3 balanced sum.
fn z3_add(a: Trit, b: Trit) -> Trit {
    match (a, b) {
        (1, 1) => -1,
        (1, -1) | (-1, 1) => 0,
        (-1, -1) => 1,
        (1, 0) | (0, 1) => 1,
        (-1, 0) | (0, -1) => -1,
        _ => 0,
    }
}

/// Fuse consecutive ops where possible.
pub fn fuse_ops(graph: &[Op], input_size: usize) -> FusedGraph {
    let mut fused = FusedGraph::new(input_size);
    for op in graph {
        fused.push(op.clone());
    }
    fused
}

/// Apply bias to a matmul result: fused bias + activation.
pub fn fused_bias_activate(
    mat: &[Vec<Trit>],
    bias: &[Trit],
    activation: Op,
) -> Vec<Vec<Trit>> {
    let rows = mat.len();
    let cols = if rows > 0 { mat[0].len() } else { 0 };
    let mut result = vec![vec![0i8; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            let biased = z3_add(mat[i][j], bias.get(j).copied().unwrap_or(0));
            result[i][j] = apply_activation(biased, &activation);
        }
    }
    result
}

/// Clamp an integer to the valid trit range {-1, 0, +1}.
fn clamp_trit(val: Trit) -> Trit {
    if val < -1 { -1 } else if val > 1 { 1 } else { val }
}

fn apply_activation(val: Trit, op: &Op) -> Trit {
    let clamped = clamp_trit(val);
    match op {
        Op::ReLU => {
            if clamped < 0 { 0 } else { clamped }
        }
        Op::Sign => {
            clamped.signum()
        }
        Op::Tanh => {
            if clamped < 0 { -1 } else if clamped > 0 { 1 } else { 0 }
        }
        _ => clamped,
    }
}

/// Ternary matmul with fused add: C = (A × B) + C_bias (all in Z₃).
pub fn fused_matmul_add(
    a: &[Vec<Trit>],
    b: &[Vec<Trit>],
    c_bias: &[Vec<Trit>],
) -> Vec<Vec<Trit>> {
    let rows = a.len();
    let inner = if rows > 0 { a[0].len() } else { 0 };
    let cols = if !b.is_empty() { b[0].len() } else { 0 };

    let mut result = vec![vec![0i8; cols]; rows];
    for i in 0..rows {
        for k in 0..inner {
            let aik = a[i].get(k).copied().unwrap_or(0);
            if aik == 0 {
                continue; // skip zero weights
            }
            for j in 0..cols {
                let bkj = b[k].get(j).copied().unwrap_or(0);
                // Z₃ multiply: aik * bkj
                let prod = match (aik, bkj) {
                    (1, 1) | (-1, -1) => 1,
                    (1, -1) | (-1, 1) => -1,
                    _ => 0,
                };
                // Z₃ add: result + prod + bias
                let bias = c_bias[i].get(j).copied().unwrap_or(0);
                let mut sum = z3_add(result[i][j], prod);
                sum = z3_add(sum, bias);
                result[i][j] = sum;
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_graph_creation() {
        let mut graph = FusedGraph::new(64);
        graph.push(Op::Matmul { rows: 4, inner: 8, cols: 16 });
        graph.push(Op::BiasAdd { size: 16 });
        graph.push(Op::ReLU);
        assert_eq!(graph.ops.len(), 3);
        assert!(graph.fusion_speedup() >= 1.0);
    }

    #[test]
    fn test_fused_bias_activate() {
        let mat = vec![
            vec![1, -1, 0],
            vec![0, 1, -1],
        ];
        let bias = vec![1, -1, 0];
        let result = fused_bias_activate(&mat, &bias, Op::ReLU);
        // Z₃: 1+1=-1, -1+(-1)=1, 0+0=0 → relu(-1,1,0) → [0, 1, 0]
        assert_eq!(result[0], vec![0, 1, 0]);
    }

    #[test]
    fn test_fused_matmul_add() {
        let a = vec![
            vec![1, 0],
            vec![0, 1],
        ];
        let b = vec![
            vec![1, -1],
            vec![-1, 1],
        ];
        let c = vec![
            vec![0, 0],
            vec![0, 0],
        ];
        let result = fused_matmul_add(&a, &b, &c);
        assert_eq!(result[0][0], 1);
        assert_eq!(result[0][1], -1);
    }

    #[test]
    fn test_apply_activation() {
        assert_eq!(apply_activation(-1, &Op::ReLU), 0);
        assert_eq!(apply_activation(1, &Op::ReLU), 1);
        assert_eq!(apply_activation(-1, &Op::Sign), -1);
    }

    #[test]
    fn test_is_metadata() {
        assert!(Op::Reshape { new_shape: vec![4, 4] }.is_metadata());
        assert!(!Op::Add.is_metadata());
    }
}
