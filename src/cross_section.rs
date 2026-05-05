//! Cross-sectional geometry for beams.

pub trait CrossSection {
    fn moment_of_inertia(&self) -> f64;  // I in m⁴
    fn area(&self) -> f64;              // A in m²
}

pub struct Rectangular {
    pub width: f64,
    pub height: f64,
}

impl CrossSection for Rectangular {
    fn moment_of_inertia(&self) -> f64 {
        self.width * self.height.powi(3) / 12.0
    }
    fn area(&self) -> f64 {
        self.width * self.height
    }
}
