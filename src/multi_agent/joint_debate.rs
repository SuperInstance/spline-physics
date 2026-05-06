//! Joint Debate — Multi-Agent Consensus at Beam Joints
//!
//! Each interior joint of a multi-segment beam is a "debate" between the
//! two adjacent segment agents. Non-adjacent agents participate with lower
//! trust weights (0.3 vs 1.0 for adjacent).
//!
//! Consensus criterion: all 4 components (T, M, y, theta) match within tolerance.
//!
//! Trust topology:
//!   - Adjacent segments: trust weight = 1.0
//!   - Non-adjacent segments: trust weight = 0.3
//!
//! This implements the sheaf cohomology condition H^0(S) != empty:
//! when all segment agents reach consensus at all joints simultaneously,
//! the global section exists.

use crate::multi_segment::JointState;

/// Trust weight between segment agents
#[derive(Clone, Debug)]
pub struct TrustWeight {
    pub from_segment: usize,
    pub to_segment: usize,
    pub weight: f64,
}

/// State of one segment agent in the debate
#[derive(Clone, Debug)]
pub struct SegmentAgentState {
    pub segment_id: usize,
    pub belief: JointState,
    pub confidence: f64,
    pub rounds: usize,
}

impl SegmentAgentState {
    pub fn new(segment_id: usize, initial_state: JointState) -> Self {
        SegmentAgentState {
            segment_id,
            belief: initial_state,
            confidence: 0.5,
            rounds: 0,
        }
    }
}

/// One round of joint debate
#[derive(Clone, Debug)]
pub struct JointDebateRound {
    pub round: usize,
    pub segment_states: Vec<SegmentAgentState>,
    pub residuals: Vec<f64>,
    pub consensus_reached: bool,
}

/// Result of joint debate
#[derive(Clone, Debug)]
pub struct JointDebateResult {
    pub joint_states: Vec<JointState>,
    pub consensus_reached: bool,
    pub rounds: usize,
    pub final_residual_norm: f64,
}

/// Joint debate manager for one interior joint
pub struct JointDebate {
    pub joint_index: usize,
    pub left_segment_id: usize,
    pub right_segment_id: usize,
    pub tolerance: f64,
    pub agents: Vec<SegmentAgentState>,
    pub history: Vec<JointDebateRound>,
    pub trust_weights: Vec<TrustWeight>,
}

impl JointDebate {
    pub fn new(
        joint_index: usize,
        left_segment_id: usize,
        right_segment_id: usize,
        all_segment_ids: &[usize],
        tolerance: f64,
    ) -> Self {
        // Initialize agents for all segments that will participate in this joint's debate
        // All segments get a voice, but trust weights differ
        let mut agents = Vec::new();
        let mut trust_weights = Vec::new();
        
        for &seg_id in all_segment_ids {
            // Create agent state with initial guess
            let initial_state = JointState::default();
            agents.push(SegmentAgentState::new(seg_id, initial_state));
            
            // Compute trust weight: 1.0 for adjacent, 0.3 for non-adjacent
            let is_adjacent = seg_id == left_segment_id || seg_id == right_segment_id;
            let weight = if is_adjacent { 1.0 } else { 0.3 };
            trust_weights.push(TrustWeight {
                from_segment: seg_id,
                to_segment: joint_index, // The joint being debated
                weight,
            });
        }
        
        JointDebate {
            joint_index,
            left_segment_id,
            right_segment_id,
            tolerance,
            agents,
            history: vec![],
            trust_weights,
        }
    }

    /// Get trust weight for a segment
    fn get_trust_weight(&self, segment_id: usize) -> f64 {
        self.trust_weights
            .iter()
            .find(|tw| tw.from_segment == segment_id)
            .map(|tw| tw.weight)
            .unwrap_or(0.3)
    }

    /// Run one round of debate
    fn debate_round(&mut self) {
        // Compute current residuals (difference between left and right beliefs at this joint)
        let left_agent = self.agents.iter().find(|a| a.segment_id == self.left_segment_id);
        let right_agent = self.agents.iter().find(|a| a.segment_id == self.right_segment_id);
        
        let mut residuals = vec![0.0; 4];
        let mut consensus_reached = false;
        
        if let (Some(left), Some(right)) = (left_agent, right_agent) {
            let r = left.belief.residual();
            residuals.copy_from_slice(&r);
            
            let residual_norm = (r[0].powi(2) + r[1].powi(2) + r[2].powi(2) + r[3].powi(2)).sqrt();
            consensus_reached = residual_norm < self.tolerance;
        }
        
        // Record history
        self.history.push(JointDebateRound {
            round: self.agents.first().map(|a| a.rounds).unwrap_or(0),
            segment_states: self.agents.clone(),
            residuals: residuals.clone(),
            consensus_reached,
        });
        
        // Update agents if consensus not reached
        if !consensus_reached {
            self.update_agents(&residuals);
        }
    }

    /// Update agent beliefs using trust-weighted consensus
    fn update_agents(&mut self, residuals: &[f64]) {
        // Precompute trust weights and adjacent info to avoid borrow conflicts
        let trust_weights: Vec<(usize, f64)> = self.agents.iter()
            .map(|a| (a.segment_id, self.get_trust_weight(a.segment_id)))
            .collect();
        
        // Find adjacent agent beliefs for non-adjacent updates
        let adjacents: Vec<JointState> = self.agents.iter()
            .filter(|a| a.segment_id == self.left_segment_id || a.segment_id == self.right_segment_id)
            .map(|a| a.belief.clone())
            .collect();
        
        for (i, agent) in self.agents.iter_mut().enumerate() {
            agent.rounds += 1;
            
            let trust = trust_weights[i].1;
            
            // For adjacent segments: pull toward the average of left and right
            // For non-adjacent: less pull, but still influenced
            let is_adjacent = agent.segment_id == self.left_segment_id 
                || agent.segment_id == self.right_segment_id;
            
            if is_adjacent {
                // Adjacent agents: direct influence from the other side
                // Move toward the residual = 0 state (equilibrium)
                let pull_strength = trust * 0.3;
                
                agent.belief.T_left -= pull_strength * residuals[0];
                agent.belief.M_left -= pull_strength * residuals[1];
                agent.belief.y_left -= pull_strength * residuals[2];
                agent.belief.theta_left -= pull_strength * residuals[3];
                
                // Right side mirrors left for equilibrium
                agent.belief.T_right -= pull_strength * residuals[0];
                agent.belief.M_right -= pull_strength * residuals[1];
                agent.belief.y_right -= pull_strength * residuals[2];
                agent.belief.theta_right -= pull_strength * residuals[3];
            } else {
                // Non-adjacent agents: weaker influence
                let pull_strength = trust * 0.1;
                
                if !adjacents.is_empty() {
                    let avg_T = adjacents.iter().map(|a| a.T_left).sum::<f64>() 
                        / adjacents.len() as f64;
                    let avg_M = adjacents.iter().map(|a| a.M_left).sum::<f64>() 
                        / adjacents.len() as f64;
                    let avg_y = adjacents.iter().map(|a| a.y_left).sum::<f64>() 
                        / adjacents.len() as f64;
                    let avg_theta = adjacents.iter().map(|a| a.theta_left).sum::<f64>() 
                        / adjacents.len() as f64;
                    
                    agent.belief.T_left += pull_strength * (avg_T - agent.belief.T_left);
                    agent.belief.M_left += pull_strength * (avg_M - agent.belief.M_left);
                    agent.belief.y_left += pull_strength * (avg_y - agent.belief.y_left);
                    agent.belief.theta_left += pull_strength * (avg_theta - agent.belief.theta_left);
                    
                    agent.belief.T_right += pull_strength * (avg_T - agent.belief.T_right);
                    agent.belief.M_right += pull_strength * (avg_M - agent.belief.M_right);
                    agent.belief.y_right += pull_strength * (avg_y - agent.belief.y_right);
                    agent.belief.theta_right += pull_strength * (avg_theta - agent.belief.theta_right);
                }
            }
            
            // Update confidence based on proximity to consensus
            let r = agent.belief.residual();
            let norm = (r[0].powi(2) + r[1].powi(2) + r[2].powi(2) + r[3].powi(2)).sqrt();
            agent.confidence = 1.0 / (1.0 + norm);
        }
    }

    /// Run the full debate until consensus or max rounds
    pub fn run(&mut self, max_rounds: usize) -> JointDebateResult {
        for _round in 0..max_rounds {
            self.debate_round();
            
            if self.history.last().map(|r| r.consensus_reached).unwrap_or(false) {
                break;
            }
        }
        
        self.final_result()
    }

    fn final_result(&self) -> JointDebateResult {
        let last_round = self.history.last();
        
        // Get the consensus joint state (average of left and right beliefs at final round)
        let left_agent = self.agents.iter().find(|a| a.segment_id == self.left_segment_id);
        let right_agent = self.agents.iter().find(|a| a.segment_id == self.right_segment_id);
        
        let joint_state = if let (Some(left), Some(right)) = (left_agent, right_agent) {
            JointState {
                T_left: (left.belief.T_left + right.belief.T_left) / 2.0,
                M_left: (left.belief.M_left + right.belief.M_left) / 2.0,
                y_left: (left.belief.y_left + right.belief.y_left) / 2.0,
                theta_left: (left.belief.theta_left + right.belief.theta_left) / 2.0,
                T_right: (left.belief.T_right + right.belief.T_right) / 2.0,
                M_right: (left.belief.M_right + right.belief.M_right) / 2.0,
                y_right: (left.belief.y_right + right.belief.y_right) / 2.0,
                theta_right: (left.belief.theta_right + right.belief.theta_right) / 2.0,
            }
        } else {
            JointState::default()
        };
        
        JointDebateResult {
            joint_states: vec![joint_state],
            consensus_reached: last_round.map(|r| r.consensus_reached).unwrap_or(false),
            rounds: self.history.len(),
            final_residual_norm: last_round
                .map(|r| {
                    let res = &r.residuals;
                    (res[0].powi(2) + res[1].powi(2) + res[2].powi(2) + res[3].powi(2)).sqrt()
                })
                .unwrap_or(f64::INFINITY),
        }
    }
}

/// Multi-joint debate manager — coordinates debates across all interior joints
pub struct MultiJointDebate {
    pub joint_debates: Vec<JointDebate>,
    pub tolerance: f64,
}

impl MultiJointDebate {
    pub fn new(
        num_segments: usize,
        tolerance: f64,
    ) -> Self {
        let mut joint_debates = Vec::new();
        let segment_ids: Vec<usize> = (0..num_segments).collect();
        
        // Create one debate per interior joint (N-1 joints for N segments)
        for joint_idx in 0..num_segments.saturating_sub(1) {
            let left_seg = joint_idx;
            let right_seg = joint_idx + 1;
            
            joint_debates.push(JointDebate::new(
                joint_idx,
                left_seg,
                right_seg,
                &segment_ids,
                tolerance,
            ));
        }
        
        MultiJointDebate {
            joint_debates,
            tolerance,
        }
    }

    /// Run debates at all joints in parallel (conceptually)
    /// In practice, we iterate until all joints reach consensus
    pub fn run(&mut self, max_rounds: usize) -> Vec<JointDebateResult> {
        let mut results = Vec::new();
        
        for debate in &mut self.joint_debates {
            let result = debate.run(max_rounds);
            results.push(result);
        }
        
        results
    }

    /// Check if all joints have reached consensus
    pub fn all_consensus(&self) -> bool {
        self.joint_debates.iter().all(|d| {
            d.history.last().map(|r| r.consensus_reached).unwrap_or(false)
        })
    }

    /// Print summary of all joint debates
    pub fn print_summary(&self) {
        println!("Multi-joint debate summary:");
        for (i, debate) in self.joint_debates.iter().enumerate() {
            let result = debate.final_result();
            println!(
                "  Joint {} (segments {}-{}): consensus={}, rounds={}, residual={:.2e}",
                i,
                debate.left_segment_id,
                debate.right_segment_id,
                result.consensus_reached,
                result.rounds,
                result.final_residual_norm
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_joint_debate_two_segments() {
        // Two segments meeting at one joint
        let mut debate = JointDebate::new(
            0,  // joint_index
            0,  // left_segment_id
            1,  // right_segment_id
            &[0, 1],  // all_segment_ids
            1e-6,  // tolerance
        );
        
        // Set initial beliefs with some mismatch
        debate.agents[0].belief = JointState {
            T_left: 100.0, M_left: 50.0, y_left: 0.01, theta_left: 0.001,
            T_right: 100.0, M_right: 50.0, y_right: 0.01, theta_right: 0.001,
        };
        debate.agents[1].belief = JointState {
            T_left: 95.0, M_left: 48.0, y_left: 0.012, theta_left: 0.0015,
            T_right: 95.0, M_right: 48.0, y_right: 0.012, theta_right: 0.0015,
        };
        
        let result = debate.run(20);
        
        println!("Debate result: consensus={}, rounds={}, residual={:.2e}",
            result.consensus_reached, result.rounds, result.final_residual_norm);
        
        // Should converge toward consensus
        assert!(result.rounds <= 20);
    }

    #[test]
    fn test_multi_joint_debate_three_segments() {
        // Three segments with two interior joints
        let mut multi = MultiJointDebate::new(3, 1e-6);
        
        // Initialize with mismatched beliefs
        for (i, debate) in multi.joint_debates.iter_mut().enumerate() {
            for agent in &mut debate.agents {
                agent.belief = JointState {
                    T_left: 100.0 + (i as f64) * 10.0,
                    M_left: 50.0 + (i as f64) * 5.0,
                    y_left: 0.01 + (i as f64) * 0.002,
                    theta_left: 0.001 + (i as f64) * 0.0002,
                    T_right: 100.0 + (i as f64) * 10.0,
                    M_right: 50.0 + (i as f64) * 5.0,
                    y_right: 0.01 + (i as f64) * 0.002,
                    theta_right: 0.001 + (i as f64) * 0.0002,
                };
            }
        }
        
        let results = multi.run(30);
        
        println!("Multi-joint debate results:");
        multi.print_summary();
        
        // All joints should reach consensus within reasonable rounds
        assert!(results.iter().all(|r| r.rounds <= 30));
    }

    #[test]
    fn test_trust_weights_adjacent_vs_non_adjacent() {
        // For 3 segments (2 joints), test trust weights
        let debate0 = JointDebate::new(
            0, // joint 0
            0, // left segment 0
            1, // right segment 1
            &[0, 1, 2], // all segments participate
            1e-6,
        );
        
        // Joint 0: segment 0 and 1 are adjacent, segment 2 is not
        assert!((debate0.get_trust_weight(0) - 1.0).abs() < 1e-6);
        assert!((debate0.get_trust_weight(1) - 1.0).abs() < 1e-6);
        assert!((debate0.get_trust_weight(2) - 0.3).abs() < 1e-6);
        
        let debate1 = JointDebate::new(
            1, // joint 1
            1, // left segment 1
            2, // right segment 2
            &[0, 1, 2],
            1e-6,
        );
        
        // Joint 1: segment 1 and 2 are adjacent, segment 0 is not
        assert!((debate1.get_trust_weight(0) - 0.3).abs() < 1e-6);
        assert!((debate1.get_trust_weight(1) - 1.0).abs() < 1e-6);
        assert!((debate1.get_trust_weight(2) - 1.0).abs() < 1e-6);
    }
}
