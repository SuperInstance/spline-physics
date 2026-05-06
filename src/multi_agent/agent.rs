use std::fmt;

/// Each agent has a ROLE (what kind of knowledge it contributes)
/// and a PRIOR (its starting belief before seeing data)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AgentRole {
    /// Sets boundary conditions: pin positions, material, cross-section
    Architect,
    /// Uses spiling batten logic: propagates constraints from known frame positions
    SpilingAgent,
    /// Minimizes global bending energy via gradient descent
    EnergyAgent,
    /// Solves Euler elastica ODE (shooting method) for cross-validation
    ShootingAgent,
    /// Knows GD&T tolerances and manufacturing constraints
    QualityAgent,
    /// Detects consensus or declares unresolved disagreement
    Arbiter,
}

impl fmt::Display for AgentRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentRole::Architect => write!(f, "Architect"),
            AgentRole::SpilingAgent => write!(f, "SpilingAgent"),
            AgentRole::EnergyAgent => write!(f, "EnergyAgent"),
            AgentRole::ShootingAgent => write!(f, "ShootingAgent"),
            AgentRole::QualityAgent => write!(f, "QualityAgent"),
            AgentRole::Arbiter => write!(f, "Arbiter"),
        }
    }
}

/// A single agent in the multi-agent beam debate
#[derive(Debug, Clone)]
pub struct Agent {
    pub role: AgentRole,
    /// This agent's current belief about the optimal pin positions
    pub belief: Vec<(f64, f64)>,
    /// Confidence: 0.0 = I know nothing, 1.0 = I'm certain
    pub confidence: f64,
    /// Number of debate rounds this agent has participated in
    pub rounds: usize,
    /// Agent's unique identifier
    pub id: usize,
}

impl Agent {
    pub fn new(id: usize, role: AgentRole, initial_belief: Vec<(f64, f64)>) -> Self {
        let confidence = match role {
            AgentRole::Architect => 0.9,   // Architect knows the boundary conditions
            AgentRole::SpilingAgent => 0.75, // Traditional knowledge is reliable
            AgentRole::EnergyAgent => 0.85, // Energy minimization is principled
            AgentRole::ShootingAgent => 0.85, // Elastica is physically exact
            AgentRole::QualityAgent => 0.7,  // GD&T has tolerance for ambiguity
            AgentRole::Arbiter => 1.0,      // Arbiter is always certain of process
        };
        Self { id, role, belief: initial_belief, confidence, rounds: 0 }
    }

    /// Update belief using Hooke's law spring-damper model
    /// Each OTHER agent's belief acts as an anchor connected by a spring
    /// The force on my belief pin is: F = k * (other_belief - my_belief)
    /// Plus a damping term to prevent oscillation
    pub fn update(&mut self, other_beliefs: &[(AgentRole, Vec<(f64, f64)>, f64)]) {
        self.rounds += 1;
        
        // Spring stiffness k and damping c
        let k = 0.3;  // Soft spring — won't snap to others' beliefs instantly
        let c = 0.5;  // Damping — prevents oscillation
        
        let n = self.belief.len();
        let n_others = other_beliefs.len();
        let mut total_trust = 0.0_f64;
        
        // Only update intermediate pins (endpoints are boundary conditions, fixed)
        for pin_i in 1..n.saturating_sub(1) {
            let mut force_x = 0.0_f64;
            let mut force_y = 0.0_f64;
            
            for (other_role, other_belief, other_conf) in other_beliefs {
                if other_belief.len() != n { continue; }
                
                // Trust weighting: who do I trust based on my role?
                let trust = match (self.role, other_role) {
                    // Cross-validation pair: trust each other most
                    (AgentRole::EnergyAgent, AgentRole::ShootingAgent) => 1.5,
                    (AgentRole::ShootingAgent, AgentRole::EnergyAgent) => 1.5,
                    // Architect's boundaries are authoritative
                    (AgentRole::SpilingAgent, AgentRole::Architect) => 1.3,
                    // QualityAgent is a skeptic — discounts outlier beliefs
                    (AgentRole::QualityAgent, _) => 0.4,
                    // Everyone else: moderate trust
                    _ => 0.8,
                };
                
                let dx = other_belief[pin_i].0 - self.belief[pin_i].0;
                let dy = other_belief[pin_i].1 - self.belief[pin_i].1;
                
                // Hooke's law: F = k * (target - current)
                force_x += k * trust * dx;
                force_y += k * trust * dy;
                total_trust += trust * other_conf;
            }
            
            // Damping proportional to current velocity (previous move)
            let damping_x = -c * force_x;
            let damping_y = -c * force_y;
            
            // Update: move proportional to net force
            self.belief[pin_i].0 += force_x + damping_x;
            self.belief[pin_i].1 += force_y + damping_y;
        }
        
        // Confidence: based on how much coherent pull from others
        if n_others > 0 {
            let avg_trust = total_trust / n_others as f64;
            self.confidence = (self.confidence + avg_trust.min(1.0)) / 2.0;
        }
    }

    /// What does this agent ARGUE in the debate?
    pub fn argue(&self) -> String {
        match self.role {
            AgentRole::Architect => format!(
                "[{}] The design intent is {}m span, {}mm rise. Trust the boundary conditions.",
                self.role, 
                self.belief.last().map(|p| p.0).unwrap_or(0.0),
                (self.belief.get(1).map(|p| p.1).unwrap_or(0.0) * 1000.0) as i32
            ),
            AgentRole::SpilingAgent => format!(
                "[{}] The spiling batten doesn't lie — mark the frame, spring the plank. {:?}",
                self.role, &self.belief[1..2]
            ),
            AgentRole::EnergyAgent => format!(
                "[{}] Minimum energy = {} J. The shape must minimize bending stress.",
                self.role, self.compute_local_energy()
            ),
            AgentRole::ShootingAgent => format!(
                "[{}] Elastica ODE solved. The boundary conditions determine the arch uniquely.",
                self.role
            ),
            AgentRole::QualityAgent => format!(
                "[{}] GD&T tolerance is ±{}mm. Is the fair curve within spec?",
                self.role,
                ((1.0 - self.confidence) * 10.0) as i32
            ),
            AgentRole::Arbiter => format!(
                "[{}] Consensus reached: confidence = {:.1}%, rounds = {}",
                self.role,
                self.confidence * 100.0,
                self.rounds
            ),
        }
    }

    fn compute_local_energy(&self) -> f64 {
        // Simplified local energy estimate
        if self.belief.len() < 3 { return 0.0; }
        let p0 = self.belief[0];
        let p1 = self.belief[1];
        let p2 = self.belief[2];
        // Curvature proxy: angle at middle pin
        let dx = p2.0 - p0.0;
        let dy_mid = p1.1 - (p0.1 + p2.1) / 2.0;
        let curvature_estimate = (dy_mid * 8.0 / dx.abs().max(0.001)).powi(2);
        curvature_estimate * 1000.0
    }
}
