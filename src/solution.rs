//! Beam solution data structure.

#[derive(Clone, Debug)]
pub struct BeamSolution {
    /// (x, y) position at each node
    pub positions: Vec<(f64, f64)>,
    /// Tangent angle θ at each node (radians)
    pub tangents: Vec<f64>,
    /// Curvature κ at each node
    pub curvatures: Vec<f64>,
    /// Total bending energy ∫EI/2 * κ² ds
    pub bending_energy: f64,
}
