//! Beam configuration for elastic beam simulation.

use crate::cross_section::CrossSection;
use crate::material::Material;

pub struct BeamConfig {
    /// Beam length in meters
    pub length: f64,
    /// Pin/pin positions: [(x0,y0), (x1,y1), ...]
    pub pin_positions: Vec<(f64, f64)>,
    /// Material properties
    pub material: Box<dyn Material>,
    /// Cross-sectional geometry
    pub cross_section: Box<dyn CrossSection>,
    /// Number of discretization points
    pub num_nodes: usize,
}
