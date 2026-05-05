//! Metrics for comparing two beam solutions.

use crate::solution::BeamSolution;

/// Comparison metrics between a reference and candidate beam solution.
pub struct ComparisonMetrics {
    /// Maximum position error (Euclidean distance)
    pub max_position_error: f64,
    /// Root-mean-square position error
    pub rms_position_error: f64,
    /// Maximum curvature error
    pub max_curvature_error: f64,
    /// Peak height error (maximum y difference)
    pub peak_height_error: f64,
    /// Energy ratio (candidate / reference)
    pub energy_ratio: f64,
}

/// Compare two beam solutions, resampling to a common arc-length grid.
pub fn compare(reference: &BeamSolution, candidate: &BeamSolution) -> ComparisonMetrics {
    // Resample both curves to common arc-length parameterization
    let n = reference.positions.len().max(candidate.positions.len());
    let n = n.max(10);

    let mut ref_arc = Vec::with_capacity(n);
    let mut cand_arc = Vec::with_capacity(n);

    // Compute cumulative arc length for reference
    let mut ref_arc_len = vec![0.0; reference.positions.len()];
    for i in 1..reference.positions.len() {
        let dx = reference.positions[i].0 - reference.positions[i - 1].0;
        let dy = reference.positions[i].1 - reference.positions[i - 1].1;
        ref_arc_len[i] = ref_arc_len[i - 1] + (dx * dx + dy * dy).sqrt();
    }
    let ref_total = ref_arc_len.last().copied().unwrap_or(1.0);

    // Compute cumulative arc length for candidate
    let mut cand_arc_len = vec![0.0; candidate.positions.len()];
    for i in 1..candidate.positions.len() {
        let dx = candidate.positions[i].0 - candidate.positions[i - 1].0;
        let dy = candidate.positions[i].1 - candidate.positions[i - 1].1;
        cand_arc_len[i] = cand_arc_len[i - 1] + (dx * dx + dy * dy).sqrt();
    }
    let cand_total = cand_arc_len.last().copied().unwrap_or(1.0);

    // Resample at common arc-length positions
    let total_len = ref_total.max(cand_total).max(1.0);
    for i in 0..n {
        let s = (i as f64 / (n - 1) as f64) * total_len;

        // Sample reference
        let (rx, ry) = sample_at_arc(&reference.positions, &ref_arc_len, s);
        // Sample candidate
        let (cx, cy) = sample_at_arc(&candidate.positions, &cand_arc_len, s);

        ref_arc.push((rx, ry));
        cand_arc.push((cx, cy));
    }

    // Compute metrics
    let mut max_pos_err: f64 = 0.0;
    let mut sum_pos_err2 = 0.0;
    let mut max_kappa_err: f64 = 0.0;
    let mut peak_height_err: f64 = 0.0;

    let n_kappa = reference.curvatures.len().min(candidate.curvatures.len());
    for i in 0..n {
        let dx = cand_arc[i].0 - ref_arc[i].0;
        let dy = cand_arc[i].1 - ref_arc[i].1;
        let pos_err = (dx * dx + dy * dy).sqrt();
        max_pos_err = max_pos_err.max(pos_err);
        sum_pos_err2 += pos_err * pos_err;
        peak_height_err = peak_height_err.max((dy).abs());
    }

    for i in 0..n_kappa {
        let kappa_err = (candidate.curvatures[i] - reference.curvatures[i]).abs();
        max_kappa_err = max_kappa_err.max(kappa_err);
    }

    let rms_pos_err = (sum_pos_err2 / n as f64).sqrt();
    let energy_ratio = if reference.bending_energy > 1e-12 {
        candidate.bending_energy / reference.bending_energy
    } else {
        0.0
    };

    ComparisonMetrics {
        max_position_error: max_pos_err,
        rms_position_error: rms_pos_err,
        max_curvature_error: max_kappa_err,
        peak_height_error: peak_height_err,
        energy_ratio,
    }
}

fn sample_at_arc(positions: &[(f64, f64)], arc_len: &[f64], s: f64) -> (f64, f64) {
    if positions.is_empty() {
        return (0.0, 0.0);
    }
    if positions.len() == 1 {
        return positions[0];
    }

    let s_clamped = s.min(*arc_len.last().unwrap_or(&0.0));

    // Find segment containing s
    for i in 1..positions.len() {
        if arc_len[i] >= s_clamped {
            let seg_len = arc_len[i] - arc_len[i - 1];
            let t = if seg_len > 1e-12 {
                (s_clamped - arc_len[i - 1]) / seg_len
            } else {
                0.0
            };
            let x = positions[i - 1].0 + t * (positions[i].0 - positions[i - 1].0);
            let y = positions[i - 1].1 + t * (positions[i].1 - positions[i - 1].1);
            return (x, y);
        }
    }

    *positions.last().unwrap()
}
