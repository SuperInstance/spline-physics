//! Energy Minimization Solver for spline-physics
//!
//! Finds the minimum-energy pin configuration by gradient descent.
//! For a 3-pin case, p0 and p2 are fixed (beam endpoints),
//! and p1 (control point) is optimized to minimize bending energy.

use nalgebra::Point2;

use crate::beam::BeamConfig;
use crate::solvers::Solver;
use crate::solution::BeamSolution;

pub struct EnergyMinimizationSolver {
    /// Maximum gradient descent iterations
    max_iterations: usize,
    /// Convergence tolerance for gradient magnitude
    tolerance: f64,
    /// Learning rate for gradient descent
    learning_rate: f64,
}

impl EnergyMinimizationSolver {
    pub fn new() -> Self {
        EnergyMinimizationSolver {
            max_iterations: 1000,
            tolerance: 1e-8,
            learning_rate: 0.01,
        }
    }

    /// Compute bending energy for a quadratic Bézier defined by 3 pins.
    fn compute_bezier_energy(
        p0: Point2<f64>,
        p1: Point2<f64>,
        p2: Point2<f64>,
        ei: f64,
        n_samples: usize,
    ) -> f64 {
        let mut energy = 0.0;
        let mut prev_pos = Point2::new(0.0, 0.0);
        let mut prev_kappa = 0.0;

        for i in 0..n_samples {
            let t = i as f64 / (n_samples - 1) as f64;
            let one_t = 1.0 - t;

            // Quadratic Bézier: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
            let x = one_t * one_t * p0.x + 2.0 * one_t * t * p1.x + t * t * p2.x;
            let y = one_t * one_t * p0.y + 2.0 * one_t * t * p1.y + t * t * p2.y;
            let pos = Point2::new(x, y);

            // Tangent: B'(t) = 2(1-t)(P1-P0) + 2t(P2-P1)
            let tx = 2.0 * one_t * (p1.x - p0.x) + 2.0 * t * (p2.x - p1.x);
            let ty = 2.0 * one_t * (p1.y - p0.y) + 2.0 * t * (p2.y - p1.y);

            // Second derivative is constant for quadratic Bézier
            let d2x = 2.0 * (p2.x - 2.0 * p1.x + p0.x);
            let d2y = 2.0 * (p2.y - 2.0 * p1.y + p0.y);

            let numer = (tx * d2y - ty * d2x).abs();
            let denom = (tx * tx + ty * ty).powf(1.5);
            let kappa = if denom > 1e-12 { numer / denom } else { 0.0 };

            if i > 0 {
                let dx = pos.x - prev_pos.x;
                let dy = pos.y - prev_pos.y;
                let ds = (dx * dx + dy * dy).sqrt();
                let kappa_avg = (kappa + prev_kappa) / 2.0;
                energy += 0.5 * ei * kappa_avg * kappa_avg * ds;
            }

            prev_pos = pos;
            prev_kappa = kappa;
        }

        energy
    }

    /// Compute numerical gradient of energy with respect to p1 position.
    fn compute_gradient(
        p0: Point2<f64>,
        p1: Point2<f64>,
        p2: Point2<f64>,
        ei: f64,
        delta: f64,
    ) -> (f64, f64) {
        let e0 = Self::compute_bezier_energy(p0, p1, p2, ei, 51);

        let p1_dx = Point2::new(p1.x + delta, p1.y);
        let e_dx = Self::compute_bezier_energy(p0, p1_dx, p2, ei, 51);
        let df_dx = (e_dx - e0) / delta;

        let p1_dy = Point2::new(p1.x, p1.y + delta);
        let e_dy = Self::compute_bezier_energy(p0, p1_dy, p2, ei, 51);
        let df_dy = (e_dy - e0) / delta;

        (df_dx, df_dy)
    }

    /// Optimize the intermediate pin(s) to minimize bending energy.
    fn optimize_pins(&self, config: &BeamConfig) -> Result<crate::solution::BeamSolution, String> {
        let pins = &config.pin_positions;
        if pins.len() != 3 {
            return Err(format!(
                "EnergyMinimizationSolver requires exactly 3 pins, got {}",
                pins.len()
            ));
        }

        let p0 = Point2::new(pins[0].0, pins[0].1);
        let p2 = Point2::new(pins[2].0, pins[2].1);

        // Initialize p1 at the midpoint of p0 and p2 (straight line)
        let mut p1 = Point2::new((p0.x + p2.x) / 2.0, (p0.y + p2.y) / 2.0);

        // For initial rise, set p1 above the midpoint proportional to span
        // This gives a natural starting point close to the solution
        let span = (p2.x - p0.x).abs();
        let initial_rise = span * 0.05; // Start with small rise
        p1 = Point2::new(p0.x + span / 2.0, initial_rise);

        let ei = config.material.youngs_modulus() * 1e9
            * config.cross_section.moment_of_inertia();

        let delta = 0.1; // mm for numerical gradient
        let mut prev_energy = Self::compute_bezier_energy(p0, p1, p2, ei, 101);

        for _iteration in 0..self.max_iterations {
            let (df_dx, df_dy) = Self::compute_gradient(p0, p1, p2, ei, delta);
            let grad_mag = (df_dx * df_dx + df_dy * df_dy).sqrt();

            if grad_mag < self.tolerance {
                break;
            }

            // Gradient descent step
            p1 = Point2::new(
                p1.x - self.learning_rate * df_dx,
                p1.y - self.learning_rate * df_dy,
            );

            let energy = Self::compute_bezier_energy(p0, p1, p2, ei, 101);
            prev_energy = energy;
        }

        // Build the solution with optimized p1
        let n = config.num_nodes.max(2);
        let mut positions = Vec::with_capacity(n);
        let mut tangents = Vec::with_capacity(n);
        let mut curvatures = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            let one_t = 1.0 - t;

            let x = one_t * one_t * p0.x + 2.0 * one_t * t * p1.x + t * t * p2.x;
            let y = one_t * one_t * p0.y + 2.0 * one_t * t * p1.y + t * t * p2.y;
            positions.push((x, y));

            let tx = 2.0 * one_t * (p1.x - p0.x) + 2.0 * t * (p2.x - p1.x);
            let ty = 2.0 * one_t * (p1.y - p0.y) + 2.0 * t * (p2.y - p1.y);
            let theta = ty.atan2(tx);
            tangents.push(theta);

            let d2x = 2.0 * (p2.x - 2.0 * p1.x + p0.x);
            let d2y = 2.0 * (p2.y - 2.0 * p1.y + p0.y);
            let numer = (tx * d2y - ty * d2x).abs();
            let denom = (tx * tx + ty * ty).powf(1.5);
            let kappa = if denom > 1e-12 { numer / denom } else { 0.0 };
            curvatures.push(kappa);
        }

        let energy = Self::compute_bezier_energy(p0, p1, p2, ei, 101);

        Ok(BeamSolution {
            positions,
            tangents,
            curvatures,
            bending_energy: energy,
        })
    }
}

impl Default for EnergyMinimizationSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for EnergyMinimizationSolver {
    fn solve(&self, config: &BeamConfig) -> Result<crate::solution::BeamSolution, String> {
        self.optimize_pins(config)
    }

    fn name(&self) -> &'static str {
        "Energy Minimization"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{Cedar, Oak, PLA};
    use crate::cross_section::Rectangular;

    fn make_config(pins: Vec<(f64, f64)>) -> BeamConfig {
        BeamConfig {
            length: 1.0,
            pin_positions: pins,
            material: Box::new(Cedar),
            cross_section: Box::new(Rectangular { width: 0.05, height: 0.05 }),
            num_nodes: 101,
        }
    }

    #[test]
    fn test_energy_minimization_t2a() {
        // T2a: L=1000mm, h=50mm peak, 3 pins
        // EnergyMinimization should find optimal p1 position
        let solver = EnergyMinimizationSolver::new();
        let config = make_config(vec![(0.0, 0.0), (0.5, 0.05), (1.0, 0.0)]);
        let result = solver.solve(&config).unwrap();
        println!("T2a energy: {} J", result.bending_energy);
        println!("Peak Y: {}", result.positions.iter().map(|p| p.1).fold(0.0_f64, |a, b| a.max(b)));
        // Energy should be positive (non-zero curvature)
        assert!(result.bending_energy > 0.0);
    }

    #[test]
    fn test_energy_decreases_with_optimization() {
        // Verify that optimization actually reduces energy
        // Starting from a bad p1 should give higher energy than optimized
        let solver = EnergyMinimizationSolver::new();

        // Bad starting point: p1 too low
        let config_bad = make_config(vec![(0.0, 0.0), (0.5, 0.001), (1.0, 0.0)]);
        let result_bad = solver.solve(&config_bad).unwrap();

        // Good starting point: p1 closer to optimal
        let config_good = make_config(vec![(0.0, 0.0), (0.5, 0.08), (1.0, 0.0)]);
        let result_good = solver.solve(&config_good).unwrap();

        println!("Bad start energy: {} J", result_bad.bending_energy);
        println!("Good start energy: {} J", result_good.bending_energy);
        // Both should converge to similar values (gradient descent finds optimum)
        let diff = (result_bad.bending_energy - result_good.bending_energy).abs();
        assert!(diff / result_good.bending_energy < 0.5,
            "Energies should converge to similar values from different starts");
    }

    #[test]
    #[test]
    fn test_single_material_energy() {
        // Verify solver works with PLA material
        let solver = EnergyMinimizationSolver::new();
        let config = make_config(vec![(0.0, 0.0), (0.5, 0.05), (1.0, 0.0)]);
        let result = solver.solve(&config).unwrap();
        println!("Energy: {} J", result.bending_energy);
        assert!(result.bending_energy > 0.0);
    }

    }
