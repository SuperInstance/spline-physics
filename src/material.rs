//! Material properties for elastic beam simulation.

pub trait Material {
    fn youngs_modulus(&self) -> f64;  // E in GPa
    fn shear_modulus(&self) -> f64;   // G in GPa
    fn density(&self) -> f64;        // ρ in g/cm³
}

pub struct PLA;
pub struct Cedar;
pub struct Oak;
pub struct Steel;

impl Material for PLA {
    fn youngs_modulus(&self) -> f64 { 3.5 }
    fn shear_modulus(&self) -> f64 { 1.3 }
    fn density(&self) -> f64 { 1.25 }
}

impl Material for Cedar {
    fn youngs_modulus(&self) -> f64 { 6.0 }
    fn shear_modulus(&self) -> f64 { 2.2 }
    fn density(&self) -> f64 { 0.38 }
}

impl Material for Oak {
    fn youngs_modulus(&self) -> f64 { 12.0 }
    fn shear_modulus(&self) -> f64 { 4.5 }
    fn density(&self) -> f64 { 0.75 }
}

impl Material for Steel {
    fn youngs_modulus(&self) -> f64 { 200.0 }
    fn shear_modulus(&self) -> f64 { 77.0 }
    fn density(&self) -> f64 { 7.85 }
}
