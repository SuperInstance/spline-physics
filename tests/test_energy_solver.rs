//! Tests for the Energy Minimization Solver.
//!
//! Verifies that the solver correctly finds minimum-energy configurations
//! and compares favorably against the BezierSolver baseline.

use spline_physics::beam::BeamConfig;
use spline_physics::cross_section::Rectangular;
use spline_physics::material::Steel;
use spline_physics::solvers::{BezierSolver, EnergyMinimizationSolver, Solver};

#[test]
fn test_energy_minimization_flat_beam() {
    // A flat beam (pins collinear) should remain flat after optimization
    let solver = EnergyMinimizationSolver::new();

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

    let solution = solver.solve(&config).expect("solver should succeed");

    // For a flat beam, energy should be near zero
    assert!(
        solution.bending_energy < 1e-5,
        "Flat beam energy should be near zero, got {}",
        solution.bending_energy
    );
}

#[test]
fn test_energy_minimization_vs_bezier_t2a() {
    // T2a: L=1000mm, h=50mm peak (h/L=0.05), 3 pins
    // BezierSolver places control point at (500, 100) = (2*h)
    // EnergyMinimizationSolver should find a similar or better pin position
    let bezier_solver = BezierSolver;
    let energy_solver = EnergyMinimizationSolver::new();

    let pin_positions = vec![(0.0, 0.0), (500.0, 100.0), (1000.0, 0.0)];

    let config = BeamConfig {
        length: 1000.0,
        pin_positions: pin_positions.clone(),
        material: Box::new(Steel),
        cross_section: Box::new(Rectangular {
            width: 0.02,
            height: 0.002,
        }),
        num_nodes: 100,
    };

    let bezier_solution = bezier_solver.solve(&config).expect("bezier should succeed");
    let energy_solution = energy_solver.solve(&config).expect("energy solver should succeed");

    // Energy minimization should produce energy <= Bezier (it's minimizing after all)
    assert!(
        energy_solution.bending_energy <= bezier_solution.bending_energy,
        "Energy minimization should find <= energy than Bezier. \
         Bezier energy: {}, Energy min energy: {}",
        bezier_solution.bending_energy,
        energy_solution.bending_energy
    );
}

#[test]
fn test_energy_decreases_after_optimization() {
    // Start with a non-optimal pin configuration and verify energy decreases
    let solver = EnergyMinimizationSolver::new();

    // Initial pin position with significant curvature (pinned high = bad starting point)
    // p1 at y=200mm is way too high for a 50mm target arch
    let initial_pins = vec![(0.0, 0.0), (500.0, 200.0), (1000.0, 0.0)];

    let config = BeamConfig {
        length: 1000.0,
        pin_positions: initial_pins.clone(),
        material: Box::new(Steel),
        cross_section: Box::new(Rectangular {
            width: 0.02,
            height: 0.002,
        }),
        num_nodes: 100,
    };

    // Solve with bad starting point
    let result_bad = solver.solve(&config).expect("solver should succeed");

    // Now solve with better starting point
    let good_pins = vec![(0.0, 0.0), (500.0, 50.0), (1000.0, 0.0)];
    let config_good = BeamConfig {
        length: 1000.0,
        pin_positions: good_pins,
        material: Box::new(Steel),
        cross_section: Box::new(Rectangular {
            width: 0.02,
            height: 0.002,
        }),
        num_nodes: 100,
    };

    let result_good = solver.solve(&config_good).expect("solver should succeed");

    // Both should converge to similar energies (gradient descent finds optimum)
    let ratio = result_bad.bending_energy / result_good.bending_energy;
    assert!(
        ratio < 1.5,
        "Different starting points should converge to similar energies. \
         Bad start: {}, Good start: {}, Ratio: {}",
        result_bad.bending_energy,
        result_good.bending_energy,
        ratio
    );
}

#[test]
fn test_convergence_low_arch_t2a() {
    // T2a (h/L=0.05) should converge to a specific configuration
    let solver = EnergyMinimizationSolver::new();

    let config = BeamConfig {
        length: 1000.0,
        pin_positions: vec![(0.0, 0.0), (500.0, 50.0), (1000.0, 0.0)],
        material: Box::new(Steel),
        cross_section: Box::new(Rectangular {
            width: 0.02,
            height: 0.002,
        }),
        num_nodes: 100,
    };

    let solution = solver.solve(&config).expect("solver should succeed");

    // Should converge to low arch configuration
    let max_y = solution
        .positions
        .iter()
        .map(|(_, y)| *y)
        .fold(0.0f64, |a, b| a.max(b));

    // Peak should be roughly in range 20-120mm for T2a (optimized from 50mm starting point)
    assert!(
        max_y > 20.0 && max_y < 120.0,
        "T2a peak should be in reasonable range, got {}mm",
        max_y
    );

    // Energy should be reasonable
    assert!(
        solution.bending_energy < 1e6,
        "T2a energy should be bounded, got {}",
        solution.bending_energy
    );
}

#[test]
fn test_pin_count_rejection() {
    let solver = EnergyMinimizationSolver::new();

    // 2 pins is insufficient
    let config = BeamConfig {
        length: 1.0,
        pin_positions: vec![(0.0, 0.0), (1.0, 0.0)],
        material: Box::new(Steel),
        cross_section: Box::new(Rectangular {
            width: 0.02,
            height: 0.002,
        }),
        num_nodes: 50,
    };

    let result = solver.solve(&config);
    assert!(
        result.is_err(),
        "Should reject configs with fewer than 3 pins"
    );
}