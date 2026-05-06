//! Multi-segment beam debate — N-pin case with per-segment solver specialization

/// Pin agent — owns one pin position
#[derive(Clone, Debug)]
pub struct PinAgent {
    pub pin_index: usize,
    pub x_position: f64,
    pub belief_y: f64,
    pub confidence: f64,
    pub rounds: usize,
}

impl PinAgent {
    pub fn new(pin_index: usize, x: f64, y: f64) -> Self {
        PinAgent {
            pin_index,
            x_position: x,
            belief_y: y,
            confidence: 0.5,
            rounds: 0,
        }
    }
    
    pub fn update(&mut self, neighboring_beliefs: &[(usize, f64, f64)]) {
        self.rounds += 1;
        let k = 0.3_f64;
        let c = 0.5_f64;
        
        if neighboring_beliefs.is_empty() { return; }
        
        let mut force = 0.0_f64;
        let mut total_trust = 0.0_f64;
        
        for (_, neighbor_y, trust) in neighboring_beliefs {
            let dy = neighbor_y - self.belief_y;
            force += k * trust * dy;
            total_trust += *trust;
        }
        
        let damping = -c * force;
        self.belief_y += force + damping;
        
        if total_trust > 0.0 {
            let avg_trust = total_trust / neighboring_beliefs.len() as f64;
            self.confidence = (self.confidence + avg_trust.min(1.0)) / 2.0;
        }
    }
}

/// Segment agent — owns a segment's shape (via control point)
#[derive(Clone, Debug)]
pub struct SegmentAgent {
    pub segment_index: usize,
    pub belief_control_y: f64,
    pub confidence: f64,
    pub rounds: usize,
}

impl SegmentAgent {
    pub fn new(segment_index: usize, control_y: f64) -> Self {
        SegmentAgent {
            segment_index,
            belief_control_y: control_y,
            confidence: 0.5,
            rounds: 0,
        }
    }
}

/// N-pin beam debate
#[derive(Clone, Debug)]
pub struct BeamDebateN {
    pin_agents: Vec<PinAgent>,
    segment_agents: Vec<SegmentAgent>,
    min_agreement: f64,
    max_spread: f64,
    max_rounds: usize,
    history: Vec<DebateRoundN>,
}

#[derive(Clone, Debug)]
pub struct DebateRoundN {
    pub round: usize,
    pub pin_beliefs: Vec<(usize, f64, f64)>,
    pub segment_beliefs: Vec<(usize, f64, f64)>,
    pub agreement: f64,
    pub spread: f64,
    pub consensus: bool,
}

#[derive(Clone, Debug)]
pub struct ConsensusResultN {
    pub pin_positions: Vec<f64>,
    pub agreement_index: f64,
    pub belief_spread: f64,
    pub rounds: usize,
    pub consensus_reached: bool,
}

impl BeamDebateN {
    pub fn new(pin_xs: Vec<(f64, f64)>, initial_arch: f64) -> Self {
        let n = pin_xs.len();
        assert!(n >= 2, "Need at least 2 pins");
        
        let pin_agents: Vec<PinAgent> = pin_xs.iter()
            .enumerate()
            .map(|(i, &(x, y))| {
                let init_y = if i == 0 || i == n - 1 { y } else { initial_arch };
                PinAgent::new(i, x, init_y)
            })
            .collect();
        
        let segment_agents: Vec<SegmentAgent> = (0..n.saturating_sub(1))
            .map(|i| {
                // For a symmetric arch, control point of quadratic Bézier is at arch peak
                // If all pins are at y≈0, the arch height determines the control point
                let arch = if pin_xs.iter().all(|&(_, y)| y.abs() < 1e-6) {
                    initial_arch
                } else {
                    initial_arch.max((pin_xs[i].1 + pin_xs[i+1].1) / 2.0)
                };
                SegmentAgent::new(i, arch)
            })
            .collect();
        
        BeamDebateN {
            pin_agents,
            segment_agents,
            min_agreement: 0.92,
            max_spread: 0.005,
            max_rounds: 20,
            history: vec![],
        }
    }
    
    pub fn run(&mut self) -> ConsensusResultN {
        let n = self.pin_agents.len();
        
        for round in 0..self.max_rounds {
            self.update_pin_agents();
            self.update_segment_agents();
            self.update_pins_from_segments();
            
            let (agreement, spread) = self.compute_agreement();
            let consensus = agreement >= self.min_agreement && spread <= self.max_spread;
            
            self.history.push(DebateRoundN {
                round,
                pin_beliefs: self.pin_agents.iter()
                    .map(|a| (a.pin_index, a.belief_y, a.confidence))
                    .collect(),
                segment_beliefs: self.segment_agents.iter()
                    .map(|a| (a.segment_index, a.belief_control_y, a.confidence))
                    .collect(),
                agreement,
                spread,
                consensus,
            });
            
            if consensus { break; }
        }
        
        self.final_result()
    }
    
    fn update_pin_agents(&mut self) {
        let n = self.pin_agents.len();
        for i in 0..n {
            if i == 0 || i == n - 1 { continue; }
            
            let mut neighbors = vec![];
            if i > 0 {
                neighbors.push((i - 1, self.pin_agents[i - 1].belief_y, 0.8));
            }
            if i < n - 1 {
                neighbors.push((i + 1, self.pin_agents[i + 1].belief_y, 0.8));
            }
            
            self.pin_agents[i].update(&neighbors);
        }
    }
    
    fn update_segment_agents(&mut self) {
        // Each segment agent's control point encodes the arch shape.
        // It maintains a "target arch" via a soft spring: small restoring force
        // toward the initial_arch. This prevents collapse to flat.
        let initial_arch = 0.05_f64; // Shared baseline arch height
        for i in 0..self.segment_agents.len() {
            let y0 = self.pin_agents[i].belief_y;
            let y1 = self.pin_agents[i + 1].belief_y;
            
            // Segment springs to midpoint of its endpoint pins
            let target = (y0 + y1) / 2.0;
            let pull = 0.2 * (target - self.segment_agents[i].belief_control_y);
            
            // Arch restoration: push control point toward arch height
            // This is what prevents the beam from going flat
            let arch_restoration = 0.15 * (initial_arch - self.segment_agents[i].belief_control_y);
            
            self.segment_agents[i].belief_control_y += pull + arch_restoration;
            self.segment_agents[i].rounds += 1;
            
            // Neighbor smoothing: adjacent segments influence each other
            if i > 0 {
                let left = self.segment_agents[i - 1].belief_control_y;
                self.segment_agents[i].belief_control_y +=
                    0.1 * (left - self.segment_agents[i].belief_control_y);
            }
            if i < self.segment_agents.len() - 1 {
                let right = self.segment_agents[i + 1].belief_control_y;
                self.segment_agents[i].belief_control_y +=
                    0.1 * (right - self.segment_agents[i].belief_control_y);
            }
        }
    }
    
    fn update_pins_from_segments(&mut self) {
        let n = self.pin_agents.len();
        
        for i in 1..n.saturating_sub(1) {
            let left_ctrl = self.segment_agents[i - 1].belief_control_y;
            let right_ctrl = self.segment_agents[i].belief_control_y;
            
            let desired = (left_ctrl + right_ctrl) / 2.0;
            let pull = 0.15 * (desired - self.pin_agents[i].belief_y);
            self.pin_agents[i].belief_y += pull;
        }
    }
    
    fn compute_agreement(&self) -> (f64, f64) {
        let n = self.pin_agents.len();
        let mut total_error = 0.0_f64;
        let mut count = 0;
        
        for i in 1..n.saturating_sub(1) {
            let left_ctrl = self.segment_agents[i - 1].belief_control_y;
            let right_ctrl = self.segment_agents[i].belief_control_y;
            let predicted = (left_ctrl + right_ctrl) / 2.0;
            
            let error = (self.pin_agents[i].belief_y - predicted).abs();
            total_error += error;
            count += 1;
        }
        
        let avg_error = if count > 0 { total_error / count as f64 } else { 0.0 };
        
        let mean_y: f64 = self.pin_agents.iter().map(|a| a.belief_y).sum::<f64>() / n as f64;
        let variance: f64 = self.pin_agents.iter().map(|a| (a.belief_y - mean_y).powi(2)).sum::<f64>() / n as f64;
        let spread = variance.sqrt();
        
        let agreement = (-avg_error * 100.0).exp().min(1.0);
        (agreement, spread)
    }
    
    fn final_result(&self) -> ConsensusResultN {
        let last = self.history.last();
        
        ConsensusResultN {
            pin_positions: self.pin_agents.iter().map(|a| a.belief_y).collect(),
            agreement_index: last.map(|r| r.agreement).unwrap_or(0.0),
            belief_spread: last.map(|r| r.spread).unwrap_or(0.0),
            rounds: self.history.len(),
            consensus_reached: last.map(|r| r.consensus).unwrap_or(false),
        }
    }
    
    /// Print summary of the debate
    pub fn print_summary(&self) {
        println!("N-pin beam debate: {} pins, {} segments, {} rounds",
            self.pin_agents.len(), self.segment_agents.len(), self.history.len());
        let r = self.final_result();
        println!("  pin heights: {:?}", r.pin_positions);
        println!("  agreement: {:.1}, spread: {:.4}, consensus: {}",
            r.agreement_index * 100.0, r.belief_spread,
            if r.consensus_reached { "YES" } else { "NO" });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_3_pin() {
        let pins = vec![(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)];
        let mut d = BeamDebateN::new(pins, 0.05);
        let r = d.run();
        
        assert!(r.pin_positions[0].abs() < 1e-6);
        assert!(r.pin_positions[2].abs() < 1e-6);
        assert!(r.pin_positions[1] > 0.01);
        assert!(r.rounds <= 20);
    }
    
    #[test]
    fn test_5_pin() {
        let pins = vec![
            (0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0)
        ];
        let mut d = BeamDebateN::new(pins, 0.05);
        let r = d.run();
        
        assert!(r.pin_positions[0].abs() < 1e-6);
        assert!(r.pin_positions[4].abs() < 1e-6);
        assert!(r.pin_positions[2] >= r.pin_positions[1]);
        assert!(r.pin_positions[2] >= r.pin_positions[3]);
    }
    
    #[test]
    fn test_sloped_boundary() {
        let pins = vec![(0.0, 0.0), (0.5, 0.05), (1.0, 0.10)];
        let mut d = BeamDebateN::new(pins, 0.05);
        let r = d.run();
        
        assert!(r.pin_positions[2] > r.pin_positions[0]);
    }
}
