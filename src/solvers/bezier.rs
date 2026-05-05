//! Reference quadratic Bézier solver.
//!
//! Mirrors the ANALOG_SPLINE implementation in constraint-theory-llvm.

use nalgebra::{Vector2, Point2};

use crate::beam::BeamConfig;
use crate::solvers::Solver;
use crate::solution::BeamSolution;

pub struct BezierSolver;

impl Solver for BezierSolver {
    fn solve(&self, config: &BeamConfig) -> Result<BeamSolution, String> {
        let pins = &config.pin_positions;
        if pins.len() != 3 {
            return Err(format!("BezierSolver requires exactly 3 pins, got {}", pins.len()));
        }

        let p0 = Point2::new(pins[0].0, pins[0].1);
        let p1 = Point2::new(pins[1].0, pins[1].1);
        let p2 = Point2::new(pins[2].0, pins[2].1);

        let n = config.num_nodes.max(2);
        let mut positions = Vec::with_capacity(n);
        let mut tangents = Vec::with_capacity(n);
        let mut curvatures = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;

            // Quadratic Bézier: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
            let one_t = 1.0 - t;
            let x = one_t * one_t * p0.x + 2.0 * one_t * t * p1.x + t * t * p2.x;
            let y = one_t * one_t * p0.y + 2.0 * one_t * t * p1.y + t * t * p2.y;
            positions.push((x, y));

            // Tangent: B'(t) = 2(1-t)(P1-P0) + 2t(P2-P1)
            let tx = 2.0 * one_t * (p1.x - p0.x) + 2.0 * t * (p2.x - p1.x);
            let ty = 2.0 * one_t * (p1.y - p0.y) + 2.0 * t * (p2.y - p1.y);
            let theta = ty.atan2(tx);
            tangents.push(theta);

            // Curvature: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
            // Second derivative of quadratic Bézier is constant: B''(t) = 2(P2 - 2P1 + P0)
            let dx_dy = p1.y - p0.y;
            let dy_dx = p1.x - p0.x;
            let d2x = 2.0 * (p2.x - 2.0 * p1.x + p0.x);
            let d2y = 2.0 * (p2.y - 2.0 * p1.y + p0.y);

            let numer = (tx * d2y - ty * d2x).abs();
            let denom = (tx * tx + ty * ty).powf(1.5);
            let kappa = if denom > 1e-12 { numer / denom } else { 0.0 };
            curvatures.push(kappa);
        }

        // Compute bending energy: ∫ EI/2 * κ² ds
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
        "Bezier (reference)"
    }
}
