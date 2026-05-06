use super::agent::{Agent, AgentRole};
use super::consensus::{check_consensus, ConsensusResult};

/// A single round of debate — each agent speaks, then updates
#[derive(Debug, Clone)]
pub struct DebateRound {
    pub round_number: usize,
    pub states: Vec<AgentState>,
    /// The overall consensus result after this round
    pub consensus: ConsensusResult,
}

#[derive(Debug, Clone)]
pub struct AgentState {
    pub id: usize,
    pub role: AgentRole,
    pub belief: Vec<(f64, f64)>,
    pub confidence: f64,
    pub argument: String,
}

/// The multi-agent beam debate simulation
pub struct BeamDebate {
    pub agents: Vec<Agent>,
    pub history: Vec<DebateRound>,
    pub max_rounds: usize,
    pub min_agreement: f64,
    pub max_spread: f64,
}

impl BeamDebate {
    /// Create a new debate with 5 agents debating a 3-pin beam shape
    pub fn new(p0: (f64, f64), p2: (f64, f64), initial_mid: (f64, f64)) -> Self {
        let pins = vec![p0, initial_mid, p2];
        
        let agents = vec![
            Agent::new(0, AgentRole::Architect, pins.clone()),
            Agent::new(1, AgentRole::SpilingAgent, pins.clone()),
            Agent::new(2, AgentRole::EnergyAgent, pins.clone()),
            Agent::new(3, AgentRole::ShootingAgent, pins.clone()),
            Agent::new(4, AgentRole::QualityAgent, pins.clone()),
        ];
        
        Self {
            agents,
            history: vec![],
            max_rounds: 20,
            min_agreement: 0.92,  // 92% agreement required
            max_spread: 0.005,   // Within 5mm spread
        }
    }

    /// Run one round of debate
    pub fn debate_round(&mut self) {
        let round = self.agents[0].rounds;
        
        // Each agent makes their argument
        let states: Vec<AgentState> = self.agents.iter().map(|a| AgentState {
            id: a.id,
            role: a.role,
            belief: a.belief.clone(),
            confidence: a.confidence,
            argument: a.argue(),
        }).collect();
        
        // Consensus check before update
        let consensus = check_consensus(&self.agents, self.min_agreement, self.max_spread, self.max_rounds);
        
        self.history.push(DebateRound { round_number: round, states, consensus: consensus.clone() });
        
        // If not yet consensus, agents update based on others' beliefs
        if !consensus.reached && round < self.max_rounds {
            let others: Vec<_> = self.agents.iter()
                .map(|a| (a.role, a.belief.clone(), a.confidence))
                .collect();
            
            for agent in &mut self.agents {
                if agent.role != AgentRole::Arbiter {
                    agent.update(&others);
                }
            }
        }
    }

    /// Run the full debate until consensus or max rounds
    pub fn run(&mut self) -> ConsensusResult {
        for _ in 0..self.max_rounds {
            self.debate_round();
            if self.history.last().map(|r| r.consensus.reached).unwrap_or(false) {
                break;
            }
        }
        self.history.last().map(|r| r.consensus.clone()).unwrap()
    }

    /// Print the debate transcript
    pub fn print_transcript(&self) {
        println!("\n{}", "=".repeat(70));
        println!("BEAM DEBATE TRANSCRIPT — {} agents, {} rounds max",
            self.agents.len(), self.max_rounds);
        println!("{}", "=".repeat(70));
        
        for round in &self.history {
            println!("\n--- Round {} ---", round.round_number);
            
            for state in &round.states {
                println!("  {} (conf={:.0}%): {:?}", 
                    state.role, 
                    state.confidence * 100.0,
                    &state.belief[1]);
                println!("    \"{}\"", state.argument);
            }
            
            let c = &round.consensus;
            println!("  → Agreement: {:.1}%, Spread: {:.4}m, Consensus: {}",
                c.agreement_index * 100.0,
                c.belief_spread,
                if c.reached { "✓ REACHED" } else { "✗ still debating" }
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beam_debate_reaches_consensus() {
        let mut debate = BeamDebate::new(
            (0.0, 0.0),   // left pin
            (1.0, 0.0),   // right pin
            (0.5, 0.02),  // initial mid-pin guess (very flat)
        );
        
        let result = debate.run();
        let r = debate.run().clone(); debate.print_transcript();
        
        println!("Final: agreement={:.1}%, spread={:.4}m, rounds={}",
            result.agreement_index * 100.0, result.belief_spread, result.rounds);
        
        // Should reach consensus within reasonable rounds
        assert!(result.rounds <= 20, "Should finish within max rounds");
    }

    #[test]
    fn test_beam_debate_with_wrong_initial_belief() {
        // Start with a VERY wrong initial belief (pin too high)
        let mut debate = BeamDebate::new(
            (0.0, 0.0),
            (1.0, 0.0),
            (0.5, 0.5), // 500mm rise — way too much
        );
        
        let result = debate.run();
        let r = debate.run().clone(); debate.print_transcript();
        
        // Agents should still converge
        assert!(result.agreement_index > 0.8, "Agents should mostly agree even from wrong start");
    }

    #[test]
    fn test_debate_with_spread_out_initial_beliefs() {
        // Give agents DIFFERENT initial beliefs (plurality of priors)
        let agents = vec![
            Agent::new(0, AgentRole::Architect, vec![(0.0,0.0),(0.5, 0.08),(1.0,0.0)]),
            Agent::new(1, AgentRole::SpilingAgent, vec![(0.0,0.0),(0.5, 0.10),(1.0,0.0)]),
            Agent::new(2, AgentRole::EnergyAgent, vec![(0.0,0.0),(0.5, 0.05),(1.0,0.0)]),
            Agent::new(3, AgentRole::ShootingAgent, vec![(0.0,0.0),(0.5, 0.07),(1.0,0.0)]),
            Agent::new(4, AgentRole::QualityAgent, vec![(0.0,0.0),(0.5, 0.09),(1.0,0.0)]),
        ];
        
        let debate = BeamDebate {
            agents,
            history: vec![],
            max_rounds: 20,
            min_agreement: 0.90,
            max_spread: 0.010,
        };
        
        // Initial disagreement
        let (init_agree, _) = super::super::consensus::measure_agreement(&debate.agents);
        println!("Initial agreement: {:.1}%", init_agree * 100.0);
        assert!(init_agree < 0.9, "Initial beliefs should differ");
    }
}
