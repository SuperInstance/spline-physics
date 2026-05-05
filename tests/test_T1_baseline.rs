//! Test T1: Flat baseline — verify solver returns straight line with zero error

use spline_physics::beam::BeamConfig;
use spline_physics::cross_section::{CrossSection, Rectangular};
use spline_physics::material::Steel;
use spline_physics::solvers::{BezierSolver, Solver};

#[test]
fn test_flat_baseline_bezier() {
    // Pin A at (0, 0), Pin B at (0.2, 0) — 200mm flat beam
    // Expected: zero curvature everywhere
    // Pass criteria: max_curvature_error < 1e-6

    let config = BeamConfig {
        length: 0.2,
        pin_positions: vec![(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)],
        material: Box::new(Steel),
        cross_section: Box::new(Rectangular {
            width: 0.02,
            height: 0.002,
        }),
        num_nodes: 50,
    };

    let solver = BezierSolver;
    let solution = solver.solve(&config).expect("solver should succeed");

    // For a flat line, curvature should be essentially zero
    let max_curvature = solution
        .curvatures
        .iter()
        .map(|k| k.abs())
        .fold(0.0f64, |a, b| a.max(b));

    assert!(
        max_curvature < 1e-6,
        "Flat beam should have near-zero curvature, got {}",
        max_curvature
    );

    // All y positions should be essentially zero
    let max_y = solution
        .positions
        .iter()
        .map(|(_, y)| y.abs())
        .fold(0.0f64, |a, b| a.max(b));

    assert!(
        max_y < 1e-10,
        "Flat beam y positions should all be zero, max deviation was {}",
        max_y
    );
}

#[test]
fn test_flat_baseline_analytical() {
    use spline_physics::solvers::analytical::AnalyticalSolver;

    let config = BeamConfig {
        length: 0.2,
        pin_positions: vec![(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)],
        material: Box::new(Steel),
        cross_section: Box::new(Rectangular {
            width: 0.02,
            height: 0.002,
        }),
        num_nodes: 50,
    };

    let solver = AnalyticalSolver::new("T1");
    let solution = solver.solve(&config).expect("solver should succeed");

    let max_curvature = solution
        .curvatures
        .iter()
        .map(|k| k.abs())
        .fold(0.0f64, |a, b| a.max(b));

    assert!(
        max_curvature < 1e-12,
        "Analytical T1 should have exactly zero curvature, got {}",
        max_curvature
    );
}
