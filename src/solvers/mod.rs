//! Solver trait and implementations.

pub mod bezier;
pub mod analytical;

pub use bezier::BezierSolver;
pub use analytical::AnalyticalSolver;

pub trait Solver {
    fn solve(&self, config: &crate::beam::BeamConfig) -> Result<crate::solution::BeamSolution, String>;
    fn name(&self) -> &str;
}
