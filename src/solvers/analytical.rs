//! Analytical reference solutions for validation.
//!
//! - T1: straight line (zero curvature)
//! - T2: circular arc for symmetric arch
//! - T6: parabola for uniform load (simply supported beam)

use crate::beam::BeamConfig;
use crate::solvers::Solver;
use crate::solution::BeamSolution;

pub struct AnalyticalSolver {
    test_case: &'static str,
}

impl AnalyticalSolver {
    pub fn new(test_case: &'static str) -> Self {
        Self { test_case }
    }
}

impl Solver for AnalyticalSolver {
    fn solve(&self, config: &BeamConfig) -> Result<BeamSolution, String> {
        let pins = &config.pin_positions;
        let n = config.num_nodes.max(2);
        let mut positions = Vec::with_capacity(n);
        let mut tangents = Vec::with_capacity(n);
        let mut curvatures = Vec::with_capacity(n);

        match self.test_case {
            "T1" => {
                // Straight line — zero curvature everywhere
                let x0 = pins.first().map(|p| p.0).unwrap_or(0.0);
                let y0 = pins.first().map(|p| p.1).unwrap_or(0.0);
                let x1 = pins.last().map(|p| p.0).unwrap_or(config.length);

                for i in 0..n {
                    let t = i as f64 / (n - 1) as f64;
                    positions.push((x0 + t * (x1 - x0), y0));
                    tangents.push(0.0);
                    curvatures.push(0.0);
                }
            }
            "T2" => {
                // Circular arc for symmetric arch (3 pins)
                if pins.len() != 3 {
                    return Err("T2 requires exactly 3 pins".to_string());
                }
                let (x0, y0) = pins[0];
                let (x1, y1) = pins[1];
                let (x2, y2) = pins[2];

                // Compute circular arc through 3 points
                let mx = (x0 + x2) / 2.0;
                let my = (y0 + y2) / 2.0;
                let dx1 = x1 - x0;
                let dy1 = y1 - y0;
                let dx2 = x2 - x0;
                let dy2 = y2 - y0;
                let cross = dx1 * dy2 - dy1 * dx2;

                if cross.abs() < 1e-12 {
                    return Err("T2 pins are collinear".to_string());
                }

                let radius_sq = (x0 - mx).powi(2) + (y0 - my).powi(2);
                let radius = radius_sq.sqrt();

                for i in 0..n {
                    let t = i as f64 / (n - 1) as f64;
                    let theta = std::f64::consts::PI * t;
                    positions.push((mx + radius * theta.cos(), my + radius * theta.sin()));
                    tangents.push(theta + std::f64::consts::FRAC_PI_2);
                    curvatures.push(1.0 / radius);
                }
            }
            "T6" => {
                // Parabola for uniform load on simply supported beam
                let x0 = pins.first().map(|p| p.0).unwrap_or(0.0);
                let x1 = pins.last().map(|p| p.0).unwrap_or(config.length);
                let y0 = pins.first().map(|p| p.1).unwrap_or(0.0);
                let y1 = pins.last().map(|p| p.1).unwrap_or(0.0);
                let span = x1 - x0;
                let sag = y1 - y0;

                for i in 0..n {
                    let t = i as f64 / (n - 1) as f64;
                    let x = x0 + t * span;
                    // Parabola: y = 4d(t - t²) where d is max deflection
                    let y = y0 + 4.0 * sag * t * (1.0 - t);
                    positions.push((x, y));

                    // dy/dx = 4d(1 - 2t) / span
                    let dydx = 4.0 * sag * (1.0 - 2.0 * t) / span;
                    tangents.push(dydx.atan());

                    // d²y/dx² = -8d / span² (constant curvature for parabola)
                    let d2ydx2 = -8.0 * sag / (span * span);
                    // Curvature κ ≈ y'' / (1 + y'²)^(3/2)
                    let kappa = d2ydx2 / (1.0 + dydx * dydx).powf(1.5);
                    curvatures.push(kappa);
                }
            }
            _ => return Err(format!("Unknown test case: {}", self.test_case)),
        }

        // Compute bending energy
        let ei = config.material.youngs_modulus() * 1e9 * config.cross_section.moment_of_inertia();
        let mut energy = 0.0;
        for i in 0..n.saturating_sub(1) {
            let dx = positions[i + 1].0 - positions[i].0;
            let dy = positions[i + 1].1 - positions[i].1;
            let ds = (dx * dx + dy * dy).sqrt();
            let kappa_avg = (curvatures[i] + curvatures[i + 1]) / 2.0;
            energy += 0.5 * ei * kappa_avg * kappa_avg * ds;
        }

        Ok(BeamSolution {
            positions,
            tangents,
            curvatures,
            bending_energy: energy,
        })
    }

    fn name(&self) -> &str {
        match self.test_case {
            "T1" => "Analytical T1 (straight line)",
            "T2" => "Analytical T2 (circular arc)",
            "T6" => "Analytical T6 (parabola)",
            _ => "Analytical unknown",
        }
    }
}
