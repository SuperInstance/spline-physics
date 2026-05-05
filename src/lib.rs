//! Spline Physics Simulation Environment
//!
//! Validates ANALOG_SPLINE (quadratic Bézier) against true elastic beam physics.
//! Follows shipwright methodology: loft at 1:1 digitally before physical prototype.

pub mod beam;
pub mod material;
pub mod cross_section;
pub mod solution;
pub mod solvers;
pub mod comparison;
