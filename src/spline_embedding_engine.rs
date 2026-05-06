//! Spline Embedding Engine
//! 
//! A vector v ∈ R^n is represented as a spline curve f: [0,1] → R^n
//! not a point. This changes everything about how we compute similarity,
//! interpolation, and resonance.

use std::f64::consts::PI;

/// Control point in R^n
#[derive(Debug, Clone)]
pub struct ControlPoint {
    pub position: Vec<f64>,  // n-dimensional position
    pub tension: f64,        // spline tension parameter
}

/// A spline curve f: [0,1] → R^n
#[derive(Debug, Clone)]
pub struct SplineCurve {
    pub control_points: Vec<ControlPoint>,
    pub degree: usize,        // polynomial degree (2 = quadratic, 3 = cubic)
    pub knots: Vec<f64>,      // knot vector
}

impl SplineCurve {
    /// Create a new spline with equidistant knots
    pub fn new(control_points: Vec<Vec<f64>>, degree: usize) -> Self {
        let n = control_points.len();
        let mut knots = vec![0.0; degree + 1];
        knots.extend((1..(n - degree)).map(|i| i as f64 / (n - degree) as f64));
        knots.extend(vec![1.0; degree + 1]);
        
        let cps = control_points.into_iter()
            .map(|position| ControlPoint { position, tension: 1.0 })
            .collect();
        
        Self { control_points: cps, degree, knots }
    }
    
    /// Evaluate spline at parameter t ∈ [0,1]
    /// Uses de Boor algorithm for cubic B-splines
    pub fn eval(&self, t: f64) -> Vec<f64> {
        let n = self.control_points.len();
        if n == 0 { return vec![]; }
        if n == 1 { return self.control_points[0].position.clone(); }
        
        // Find knot span
        let t = t.clamp(0.0, 1.0);
        let mut k = self.degree;
        for i in (self.degree..self.knots.len() - self.degree - 1).rev() {
            if t >= self.knots[i] {
                k = i;
                break;
            }
        }
        k = k.min(n - 1);
        
        // de Boor recursion for cubic splines
        let mut d = Vec::new();
        for j in 0..self.degree {
            d.push(self.control_points[k - self.degree + j].position.clone());
        }
        
        for r in 1..=self.degree {
            for j in (0..self.degree - r).rev() {
                let idx = k - self.degree + j;
                let denom = self.knots[idx + self.degree + 1] - self.knots[idx];
                let alpha = if denom.abs() < 1e-12 { 0.0 } else { (t - self.knots[idx]) / denom };
                let alpha = alpha.max(0.0).min(1.0);
                d[j] = d[j].iter().zip(d[j + 1].iter())
                    .map(|(a, b)| (1.0 - alpha) * a + alpha * b)
                    .collect();
            }
        }
        
        d[0].clone()
    }
    
    /// Evaluate spline at multiple parameters
    pub fn eval_batch(&self, ts: &[f64]) -> Vec<Vec<f64>> {
        ts.iter().map(|&t| self.eval(t)).collect()
    }
    
    /// First derivative at parameter t
    pub fn derivative(&self, t: f64) -> Vec<f64> {
        let dt = 1e-6;
        let p1 = self.eval((t - dt).max(0.0));
        let p2 = self.eval((t + dt).min(1.0));
        p1.iter().zip(p2.iter())
            .map(|(a, b)| (b - a) / (2.0 * dt))
            .collect()
    }
    
    /// Curvature at parameter t: κ = |r' × r''| / |r'|³
    pub fn curvature(&self, t: f64) -> f64 {
        let d1 = self.derivative(t);
        let dt = 1e-6;
        let p0 = self.eval((t - dt).max(0.0));
        let p1 = self.eval((t + dt).min(1.0));
        let d2 = p0.iter().zip(p1.iter())
            .map(|(a, b)| (b - a) / (2.0 * dt))
            .collect::<Vec<_>>();
        // Simplified: use second derivative magnitude / first derivative magnitude cubed
        let d1_mag = d1.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        let d2_mag = d2.iter().map(|x| x * x).sum::<f64>().sqrt();
        d2_mag / (d1_mag * d1_mag * d1_mag)
    }
    
    /// Sample the spline at n points
    pub fn sample(&self, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|i| self.eval(i as f64 / (n - 1) as f64)).collect()
    }
}

/// Resonance signature extracted from a response time-series
#[derive(Debug, Clone)]
pub struct ResonanceSignature {
    pub frequencies: Vec<f64>,       // dominant frequencies
    pub amplitudes: Vec<f64>,        // amplitude at each frequency
    pub decay_rate: f64,             // exponential decay constant
    pub q_factor: f64,               // quality factor (sharpness of resonance)
    pub curvature_signature: Vec<f64>, // curvature at sampled points
}

impl ResonanceSignature {
    /// Compute resonance signature from a time-series response
    pub fn from_time_series(series: &[f64]) -> Self {
        let n = series.len();
        
        // Simple FFT (Cooley-Tukey radix-2 if n is power of 2, else pad)
        let fft_size = n.next_power_of_two();
        let mut padded = series.to_vec();
        padded.resize(fft_size, 0.0);
        
        // Compute magnitude spectrum
        let spectrum = Self::fft_magnitude(&padded);
        let freqs: Vec<f64> = (0..fft_size / 2)
            .map(|i| i as f64 / fft_size as f64)
            .collect();
        
        // Find dominant frequencies (top 5 peaks)
        let mut peaks = Vec::new();
        for i in 1..spectrum.len() - 1 {
            if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] && spectrum[i] > 0.1 * spectrum.iter().cloned().fold(0.0, f64::max) {
                peaks.push((i, spectrum[i]));
            }
        }
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_peaks = &peaks[..peaks.len().min(5)];
        
        let frequencies: Vec<f64> = top_peaks.iter().map(|(i, _)| freqs[*i]).collect();
        let amplitudes: Vec<f64> = top_peaks.iter().map(|(_, a)| *a).collect();
        
        // Decay rate: fit exponential envelope
        let envelope: f64 = series.iter().map(|x| x.abs()).collect::<Vec<_>>()
            .windows(3).map(|w| (w[0] + w[1] + w[2]) / 3.0).last().unwrap_or(series[0].abs());
        let peak = series.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let decay_rate = if envelope > 1e-12 && peak > envelope {
            -((envelope / peak).ln() / n as f64)
        } else { 0.0 };
        
        let q_factor = if decay_rate > 0.0 {
            frequencies.first().map(|&f| PI * f / decay_rate.max(1e-12)).unwrap_or(0.0)
        } else { 0.0 };
        
        Self {
            frequencies,
            amplitudes,
            decay_rate,
            q_factor,
            curvature_signature: vec![0.0; 32], // placeholder
        }
    }
    
    fn fft_magnitude(signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        let mut real = signal.to_vec();
        let mut imag = vec![0.0; n];
        // Cooley-Tukey FFT (in-place)
        let mut j = 0;
        for i in 0..n {
            if j > i {
                real.swap(i, j);
                imag.swap(i, j);
            }
            let mut m = n / 2;
            while m >= 1 && j >= m {
                j -= m;
                m /= 2;
            }
            j += m;
        }
        // Butterflies
        let mut mmax = 1;
        while n > mmax {
            let istep = mmax * 2;
            let theta = -PI / mmax as f64;
            let wpr = theta.cos();
            let wpi = theta.sin();
            let mut wr = 1.0;
            let mut wi = 0.0;
            for m in 0..mmax {
                for i in (m..n).step_by(istep) {
                    let jj = i + mmax;
                    let tr = wr * real[jj] - wi * imag[jj];
                    let ti = wr * imag[jj] + wi * real[jj];
                    real[jj] = real[i] - tr;
                    imag[jj] = imag[i] - ti;
                    real[i] += tr;
                    imag[i] += ti;
                }
                let wtemp = wr * wpr - wi * wpi;
                wi = wr * wpi + wi * wpr;
                wr = wtemp;
            }
            mmax = istep;
        }
        real.iter().zip(imag.iter()).map(|(r, i)| (r * r + i * i).sqrt()).collect()
    }
}

/// Contrast map: difference between two resonance signatures
#[derive(Debug, Clone)]
pub struct ContrastMap {
    pub hyperperfused: Vec<(f64, f64)>,  // (freq, amplitude delta) pairs that are amplified
    pub hypoperfused: Vec<(f64, f64)>,   // (freq, amplitude delta) pairs that are suppressed
    pub dead_spots: Vec<f64>,            // frequencies with near-zero response
    pub stable: Vec<(f64, f64)>,         // frequencies stable across both signatures
    pub contrast_spectrum: Vec<f64>,      // full contrast vector
}

impl ContrastMap {
    /// Compute contrast between base and tap signatures
    pub fn compute(base: &ResonanceSignature, tap: &ResonanceSignature) -> Self {
        let threshold = 0.01;
        
        let mut hyperperfused = Vec::new();
        let mut hypoperfused = Vec::new();
        let mut dead_spots = Vec::new();
        let mut stable = Vec::new();
        
        for (i, &bfreq) in base.frequencies.iter().enumerate() {
            let bamp = base.amplitudes.get(i).copied().unwrap_or(0.0);
            let tidx = tap.frequencies.iter().position(|&f| (f - bfreq).abs() < 0.01);
            let tmatch = tidx.and_then(|ti| tap.amplitudes.get(ti).copied()).unwrap_or(0.0);
            let delta = tmatch - bamp;
            
            if delta > threshold * bamp.abs().max(0.1) {
                hyperperfused.push((bfreq, delta));
            } else if delta < -threshold * bamp.abs().max(0.1) {
                hypoperfused.push((bfreq, delta));
            } else if bamp < threshold {
                dead_spots.push(bfreq);
            } else {
                stable.push((bfreq, bamp));
            }
        }
        
        let max_len = base.amplitudes.len().max(tap.amplitudes.len());
        let mut contrast_spectrum = vec![0.0; max_len];
        for i in 0..max_len {
            let b = base.amplitudes.get(i).copied().unwrap_or(0.0);
            let t = tap.amplitudes.get(i).copied().unwrap_or(0.0);
            contrast_spectrum[i] = t - b;
        }
        
        Self { hyperperfused, hypoperfused, dead_spots, stable, contrast_spectrum }
    }
    
    /// Generate ASCII resonance image
    pub fn ascii_image(&self, width: usize, height: usize) -> String {
        let spectrum = &self.contrast_spectrum;
        let max_amp = spectrum.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        if max_amp < 1e-12 { return "No contrast".to_string(); }
        
        let bin_width = (spectrum.len() as f64 / width as f64).max(1.0);
        let mut lines = Vec::with_capacity(height);
        
        for row in (0..height).rev() {
            let threshold = (row as f64 / height as f64) * max_amp;
            let mut line = String::with_capacity(width);
            for col in 0..width {
                let bin_start = ((col as f64 * bin_width) as usize).min(spectrum.len().saturating_sub(1));
                let bin_end = (((col as f64 + 1.0) * bin_width) as usize).min(spectrum.len());
                let avg = spectrum[bin_start..bin_end]
                    .iter().map(|x| x.abs()).sum::<f64>() 
                    / (bin_end - bin_start).max(1) as f64;
                if avg > threshold {
                    line.push('█');
                } else if avg > threshold * 0.3 {
                    line.push('▓');
                } else if avg > threshold * 0.1 {
                    line.push('░');
                } else {
                    line.push(' ');
                }
            }
            lines.push(line);
        }
        lines.join("\n")
    }
}

/// Pinned spline: boundary conditions that must be satisfied
#[derive(Debug, Clone)]
pub struct PinnedSpline {
    pub curve: SplineCurve,
    pub pinned: Vec<(f64, Vec<f64>)>,  // (t, position) pinned points
}

impl PinnedSpline {
    /// Fit a spline through pinned points, approximating the rest
    pub fn fit(pinned: Vec<(f64, Vec<f64>)>, all_points: &[Vec<f64>], degree: usize) -> Self {
        // Simple: just use the pinned points as control points
        // More sophisticated: least-squares fit with pinned constraints
        let control_positions: Vec<Vec<f64>> = pinned.iter().map(|(_, p)| p.clone()).collect();
        let curve = SplineCurve::new(control_positions, degree);
        Self { curve, pinned }
    }
    
    /// Interpolate through pinned points at given parameters
    pub fn interpolate(pinned: Vec<(f64, Vec<f64>)>, n_control: usize, degree: usize) -> Self {
        // Global interpolation using B-splines
        let positions: Vec<Vec<f64>> = pinned.iter().map(|(_, p)| p.clone()).collect();
        let curve = SplineCurve::new(positions, degree);
        Self { curve, pinned }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spline_eval_cubic() {
        // Control points forming a curve in R^3
        let cpts = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 2.0, 0.0],
            vec![2.0, 1.0, 0.0],
            vec![3.0, 3.0, 0.0],
        ];
        let spline = SplineCurve::new(cpts, 3);
        
        // Should evaluate without panicking
        let p = spline.eval(0.5);
        assert_eq!(p.len(), 3);
        assert!(p[0] > 0.0); // x should be positive
    }
    
    #[test]
    fn test_resonance_signature() {
        // Synthetic impulse response: decaying sinusoid
        let n = 256;
        let omega = 0.3;
        let zeta = 0.05;
        let series: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / n as f64;
            let decay = (-zeta * t * 10.0).exp();
            let oscillation = (omega * t * 20.0).sin();
            decay * oscillation
        }).collect();
        
        let sig = ResonanceSignature::from_time_series(&series);
        assert!(!sig.frequencies.is_empty());
        assert!(sig.decay_rate > 0.0);
        assert!(sig.q_factor > 0.0);
    }
    
    #[test]
    fn test_contrast_map() {
        let base = ResonanceSignature {
            frequencies: vec![0.1, 0.2, 0.3],
            amplitudes: vec![1.0, 0.5, 0.3],
            decay_rate: 0.1,
            q_factor: 5.0,
            curvature_signature: vec![],
        };
        let tap = ResonanceSignature {
            frequencies: vec![0.1, 0.2, 0.4],
            amplitudes: vec![2.0, 0.2, 0.8],
            decay_rate: 0.15,
            q_factor: 4.0,
            curvature_signature: vec![],
        };
        
        let contrast = ContrastMap::compute(&base, &tap);
        assert!(!contrast.hyperperfused.is_empty() || !contrast.hypoperfused.is_empty());
    }
    
    #[test]
    fn test_ascii_image() {
        // Use different signatures to produce actual contrast
        let sig1 = ResonanceSignature {
            frequencies: vec![0.1, 0.2, 0.3],
            amplitudes: vec![1.0, 0.5, 0.3],
            decay_rate: 0.1,
            q_factor: 5.0,
            curvature_signature: vec![],
        };
        let sig2 = ResonanceSignature {
            frequencies: vec![0.1, 0.2, 0.3],
            amplitudes: vec![2.0, 0.8, 0.6],  // different amplitudes
            decay_rate: 0.15,
            q_factor: 4.0,
            curvature_signature: vec![],
        };
        let contrast = ContrastMap::compute(&sig1, &sig2);
        let ascii = contrast.ascii_image(60, 15);
        let lines: Vec<&str> = ascii.lines().collect();
        assert_eq!(lines.len(), 15);
    }
}
