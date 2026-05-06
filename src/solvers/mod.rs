//! Solver trait and implementations.

pub mod bezier;
pub mod analytical;
pub mod energy_minimization;
pub mod shooting_method;

pub use bezier::BezierSolver;
pub use analytical::AnalyticalSolver;
pub use energy_minimization::EnergyMinimizationSolver;
pub use shooting_method::ShootingMethodSolver;

pub trait Solver {
    fn solve(&self, config: &crate::beam::BeamConfig) -> Result<crate::solution::BeamSolution, String>;
    fn name(&self) -> &str;
}
