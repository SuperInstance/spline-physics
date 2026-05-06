use super::agent::Agent;

/// Has the group reached consensus?
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    /// true if consensus has been reached
    pub reached: bool,
    /// How similar are all beliefs? (0.0 = total disagreement, 1.0 = perfect agreement)
    pub agreement_index: f64,
    /// The agreed-upon shape (average of all agent beliefs)
    pub consensus_shape: Vec<(f64, f64)>,
    /// Standard deviation of beliefs (lower = more agreement)
    pub belief_spread: f64,
    /// Number of rounds to reach consensus
    pub rounds: usize,
}

pub enum ConsensusState {
    Undecided,
    ConsensusReached,
    DisagreementDeclared { conflicting_agents: Vec<usize> },
}

/// Measure how much agents disagree about each pin position
pub fn measure_agreement(agents: &[Agent]) -> (f64, Vec<f64>) {
    if agents.is_empty() || agents[0].belief.is_empty() {
        return (1.0, vec![]);
    }
    
    let n_pins = agents[0].belief.len();
    let n_agents = agents.len();
    
    // Per-pin: standard deviation of x and y across agents
    let mut pin_spreads = Vec::with_capacity(n_pins);
    let mut total_variance = 0.0_f64;
    
    for pin_i in 0..n_pins {
        let mean_x: f64 = agents.iter().map(|a| a.belief[pin_i].0).sum::<f64>() / n_agents as f64;
        let mean_y: f64 = agents.iter().map(|a| a.belief[pin_i].1).sum::<f64>() / n_agents as f64;
        
        let var_x: f64 = agents.iter()
            .map(|a| (a.belief[pin_i].0 - mean_x).powi(2))
            .sum::<f64>() / n_agents as f64;
        let var_y: f64 = agents.iter()
            .map(|a| (a.belief[pin_i].1 - mean_y).powi(2))
            .sum::<f64>() / n_agents as f64;
        
        let spread = (var_x + var_y).sqrt();
        pin_spreads.push(spread);
        total_variance += var_x + var_y;
    }
    
    // Normalize: spread of 0 = perfect agreement = 1.0
    let avg_variance = total_variance / n_pins as f64;
    let agreement = if avg_variance < 1e-12 { 1.0 } else { (1.0 / (1.0 + avg_variance.sqrt() * 100.0)).min(1.0) };
    
    (agreement, pin_spreads)
}

/// Consensus is reached when agreement_index > threshold AND belief_spread is small
pub fn check_consensus(
    agents: &[Agent], 
    min_agreement: f64,
    max_spread: f64,
    _max_rounds: usize,
) -> ConsensusResult {
    let n = agents.len();
    if n == 0 {
        return ConsensusResult { reached: false, agreement_index: 0.0, consensus_shape: vec![], belief_spread: f64::INFINITY, rounds: 0 };
    }
    
    let (agreement, spreads) = measure_agreement(agents);
    let avg_spread = if spreads.is_empty() { 0.0 } else { spreads.iter().sum::<f64>() / spreads.len() as f64 };
    let max_round = agents.iter().map(|a| a.rounds).max().unwrap_or(0);
    
    // Consensus: high agreement + low spread + enough rounds for discussion
    let reached = agreement > min_agreement 
        && avg_spread < max_spread 
        && max_round >= 2; // At least 2 rounds of debate
    
    // Compute consensus shape (mean of all beliefs)
    let n_pins = agents[0].belief.len();
    let consensus_shape: Vec<(f64, f64)> = (0..n_pins).map(|pin_i| {
        let mean_x: f64 = agents.iter().map(|a| a.belief[pin_i].0).sum::<f64>() / n as f64;
        let mean_y: f64 = agents.iter().map(|a| a.belief[pin_i].1).sum::<f64>() / n as f64;
        (mean_x, mean_y)
    }).collect();
    
    ConsensusResult {
        reached,
        agreement_index: agreement,
        consensus_shape,
        belief_spread: avg_spread,
        rounds: max_round,
    }
}
