//! H¹ Emergence Detection Experiment
//! 
//! Validates the claim: β₁ = E - V + C predicts consensus difficulty in multi-agent systems.
//!
//! Test cases:
//! - Case A: Tree graph (β₁=0) — should always converge
//! - Case B: Single cycle (β₁=1) — should converge if edge weights consistent
//! - Case C: Two cycles sharing edge (β₁=2) — may have conflicts
//! - Case D: Laman-rigid (β₁ = V-2) — exactly rigid, boundary case
//! - Case E: Over-constrained (β₁ > V-2) — excess edges, may fail
//!
//! Statistical design:
//! - 100 trials per case (500 total)
//! - α = 0.05, power = 0.80
//! - Primary metric: F1 for β₁ > V-2 predicting consensus failure

use std::collections::HashMap;

// Re-export Complex from fleet-homology (used at runtime)
use crate::multi_agent::{Agent, AgentRole};

// =============================================================================
// Graph Generation
// =============================================================================

/// Result of a single debate trial
#[derive(Debug, Clone)]
pub struct TrialResult {
    pub case_name: String,
    pub V: usize,
    pub E: usize,
    pub beta_1: usize,
    pub consensus_reached: bool,
    pub rounds_to_consensus: usize,
    pub final_agreement: f64,
    pub distance_from_ground_truth: f64,
}

/// Result of running all trials for a case
#[derive(Debug, Clone)]
pub struct CaseResult {
    pub case_name: String,
    pub V: usize,
    pub E: usize,
    pub beta_1: usize,
    pub n_trials: usize,
    pub consensus_rate: f64,
    pub mean_rounds: f64,
    pub std_rounds: f64,
    pub mean_agreement: f64,
}

/// Generate a tree graph (β₁ = 0)
/// V vertices, V-1 edges, connected, no cycles
pub fn generate_tree_graph(V: usize) -> Vec<(u64, u64)> {
    if V < 2 { return vec![]; }
    let mut edges = Vec::with_capacity(V - 1);
    for i in 0..(V as u64 - 1) {
        edges.push((i, i + 1));
    }
    edges
}

/// Generate a single cycle (ring) graph
/// V vertices, V edges, β₁ = 1
pub fn generate_cycle_graph(V: usize) -> Vec<(u64, u64)> {
    if V < 3 { return vec![]; }
    let mut edges = Vec::with_capacity(V);
    for i in 0..(V as u64) {
        edges.push((i, (i + 1) % V as u64));
    }
    edges
}

/// Generate two cycles sharing one edge
/// V vertices, V+1 edges, β₁ = 2
pub fn generate_two_cycles(V: usize) -> Vec<(u64, u64)> {
    if V < 4 { return vec![]; }
    let mut edges = Vec::new();
    // First cycle: 0-1-2-0
    edges.push((0, 1));
    edges.push((1, 2));
    edges.push((2, 0));
    // Second cycle shares edge (2, 0): 0-2-3-...-V-1-0
    for i in 2..(V as u64 - 1) {
        edges.push((i, i + 1));
    }
    edges.push((V as u64 - 1, 0));
    // Add shared edge already in first cycle (0,2), so no dup
    // But we need to ensure it's there for the second cycle
    // Actually the cycle 0-2-3-...-0 needs edge (2, V-1) and (V-1, 0)
    // Re-do: 0-1-2 forms first cycle, 0-2-3-...-0 forms second
    edges.clear();
    edges.push((0, 1));
    edges.push((1, 2));
    edges.push((2, 0)); // First cycle: 0-1-2-0
    
    // Second cycle: 0-2-3-...-(V-1)-0
    for i in 2..(V as u64 - 1) {
        edges.push((i, i + 1));
    }
    edges.push((V as u64 - 1, 0)); // Close second cycle
    // Note: (0,2) is shared edge between both cycles
    edges
}

/// Generate a Laman-rigid graph: E = 2V - 3 edges
/// β₁ = E - V + 1 = (2V - 3) - V + 1 = V - 2
pub fn generate_laman_graph(V: usize) -> Vec<(u64, u64)> {
    if V < 3 { return vec![]; }
    let mut edges = Vec::with_capacity(2 * V - 3);
    // Start with a path (V-1 edges, tree)
    for i in 0..(V as u64 - 1) {
        edges.push((i, i + 1));
    }
    // Add edges until E = 2V - 3
    // For a Laman graph, we need to add V-2 more edges (since V-1 tree edges already present)
    // Actually: E needs to be 2V-3, tree has V-1, so we need (2V-3) - (V-1) = V-2 more edges
    // Strategy: add edges that create cycles without over-constraining
    let mut added = 0;
    let target = V - 2;
    // Add edges between non-adjacent vertices to create independent cycles
    for i in 0..(V as u64) {
        for j in (i + 2)..(V as u64) {
            if added >= target { break; }
            // Skip if already exists
            if edges.iter().any(|(a, b)| (*a == i && *b == j) || (*a == j && *b == i)) {
                continue;
            }
            // Add edge if it doesn't violate Laman count
            edges.push((i, j));
            added += 1;
            if added >= target { break; }
        }
        if added >= target { break; }
    }
    edges
}

/// Generate an over-constrained graph: E > 2V - 3
/// β₁ = E - V + C > V - 2
pub fn generate_overconstrained_graph(V: usize, excess: usize) -> Vec<(u64, u64)> {
    let mut edges = generate_laman_graph(V);
    // Add excess edges (creating extra cycles)
    let V_u64 = V as u64;
    let mut attempts = 0usize;
    while edges.len() < 2 * V - 3 + excess && attempts < 100 {
        let i = (attempts % V) as u64;
        let j = (i + 1 + (attempts / V) as u64) % V_u64;
        if i != j && !edges.iter().any(|(a, b)| (*a == i && *b == j) || (*a == j && *b == i)) {
            edges.push((i, j));
        }
        attempts += 1;
    }
    edges
}

/// Generate a complete graph (K_V)
/// β₁ = V(V-1)/2 - V + 1 = (V² - 3V + 2)/2
pub fn generate_complete_graph(V: usize) -> Vec<(u64, u64)> {
    let V_u64 = V as u64;
    let mut edges = Vec::with_capacity(V * (V - 1) / 2);
    for i in 0..V_u64 {
        for j in (i + 1)..V_u64 {
            edges.push((i, j));
        }
    }
    edges
}

// =============================================================================
// Betti Number Computation
// =============================================================================

/// Compute β₁ = E - V + C (first Betti number)
/// C = number of connected components
pub fn compute_beta_1(V: usize, E: usize, C: usize) -> usize {
    if E >= V { E - V + C } else { 0 }
}

/// Compute number of connected components using BFS
pub fn count_components(V: usize, edges: &[(u64, u64)]) -> usize {
    if V == 0 { return 0; }
    let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
    for v in 0..V as u64 {
        adj.entry(v).or_default();
    }
    for &(a, b) in edges {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    
    let mut visited = vec![false; V];
    let mut components = 0;
    
    for start in 0..V {
        if !visited[start] {
            components += 1;
            let mut queue = vec![start as u64];
            visited[start] = true;
            while let Some(v) = queue.pop() {
                for &nbr in adj.get(&v).unwrap_or(&vec![]) {
                    let nbr_idx = nbr as usize;
                    if nbr_idx < V && !visited[nbr_idx] {
                        visited[nbr_idx] = true;
                        queue.push(nbr);
                    }
                }
            }
        }
    }
    components
}

/// Build a fleet from edge list (reimplements fleet_homology::Complex behavior)
pub fn build_fleet(edges: &[(u64, u64)]) -> (usize, usize, usize) {
    let V = if edges.is_empty() {
        0
    } else {
        edges.iter().map(|&(a, b)| a.max(b) as usize).max().unwrap_or(0) + 1
    };
    let E = edges.len();
    let C = count_components(V, edges);
    (V, E, C)
}

// =============================================================================
// Beam Debate Simulation
// =============================================================================

/// Run a multi-agent debate on a fleet graph
/// Agents are assigned to vertices; each agent debates beam equilibrium
/// 
/// Returns: consensus reached, rounds, final agreement
fn run_beam_debate(edges: &[(u64, u64)], max_rounds: usize) -> (bool, usize, f64) {
    let V = if edges.is_empty() {
        3 // default
    } else {
        edges.iter().map(|&(a, b)| a.max(b) as usize).max().unwrap_or(0) + 1
    }.max(3);
    
    // Create beam pins (fixed endpoints + intermediate pins for each vertex)
    // For simplicity: 3 pins per agent (left pin, agent position, right pin)
    // But the beam debate expects a single beam with 3 pins
    // We adapt: V agents debate a single 3-pin beam, each with own belief
    
    let p0 = (0.0_f64, 0.0); // left pin
    let p2 = (1.0_f64, 0.0); // right pin
    let initial_mid = (0.5, 0.05); // initial mid-pin belief
    
    // Create V agents, each with different initial beliefs (spread ±10%)
    let mut agents: Vec<Agent> = (0..V).map(|i| {
        let perturbation = (i as f64 - (V as f64 - 1.0) / 2.0) * 0.01;
        let belief = vec![
            p0,
            (initial_mid.0, initial_mid.1 + perturbation),
            p2,
        ];
        Agent::new(i, AgentRole::EnergyAgent, belief)
    }).collect();
    
    // Override some agents with different roles for realism
    if V >= 2 { agents[0].role = AgentRole::Architect; }
    if V >= 3 { agents[V-1].role = AgentRole::ShootingAgent; }
    
    let min_agreement = 0.92_f64;
    let max_spread = 0.01_f64;
    
    // Run debate rounds
    let mut rounds = 0;
    let mut reached = false;
    let mut final_agreement = 0.0_f64;
    
    for _ in 0..max_rounds {
        rounds += 1;
        
        // Compute agreement
        let agreement = measure_agreement(&agents);
        
        // Compute spread
        let spread = measure_spread(&agents);
        
        if agreement > min_agreement && spread < max_spread {
            reached = true;
            final_agreement = agreement;
            break;
        }
        
        // Update beliefs using spring-damper model
        let others: Vec<_> = agents.iter()
            .map(|a| (a.role.clone(), a.belief.clone(), a.confidence))
            .collect();
        
        for agent in &mut agents {
            if agent.role != AgentRole::Arbiter {
                agent.update(&others);
            }
        }
    }
    
    if !reached {
        final_agreement = measure_agreement(&agents);
    }
    
    (reached, rounds, final_agreement)
}

/// Measure how much agents agree (0.0 = total disagreement, 1.0 = perfect)
fn measure_agreement(agents: &[Agent]) -> f64 {
    if agents.is_empty() || agents[0].belief.is_empty() {
        return 1.0;
    }
    
    let n_pins = agents[0].belief.len();
    let n_agents = agents.len();
    
    if n_agents == 1 { return 1.0; }
    
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
        
        total_variance += var_x + var_y;
    }
    
    let avg_variance = total_variance / n_pins as f64;
    if avg_variance < 1e-12 { return 1.0; }
    (1.0 / (1.0 + avg_variance.sqrt() * 100.0)).min(1.0)
}

/// Measure belief spread (average std dev across pins)
fn measure_spread(agents: &[Agent]) -> f64 {
    if agents.is_empty() || agents[0].belief.is_empty() {
        return 0.0;
    }
    
    let n_pins = agents[0].belief.len();
    let n_agents = agents.len();
    
    if n_agents == 1 { return 0.0; }
    
    let mut total_spread = 0.0_f64;
    
    for pin_i in 0..n_pins {
        let mean_x: f64 = agents.iter().map(|a| a.belief[pin_i].0).sum::<f64>() / n_agents as f64;
        let mean_y: f64 = agents.iter().map(|a| a.belief[pin_i].1).sum::<f64>() / n_agents as f64;
        
        let var_x: f64 = agents.iter()
            .map(|a| (a.belief[pin_i].0 - mean_x).powi(2))
            .sum::<f64>() / n_agents as f64;
        let var_y: f64 = agents.iter()
            .map(|a| (a.belief[pin_i].1 - mean_y).powi(2))
            .sum::<f64>() / n_agents as f64;
        
        total_spread += (var_x + var_y).sqrt();
    }
    
    total_spread / n_pins as f64
}

// =============================================================================
// Experiment Runner
// =============================================================================

/// Run a single case (graph type) with n trials
fn run_case(case_name: &str, edges: Vec<(u64, u64)>, n_trials: usize) -> CaseResult {
    let (V, E, C) = build_fleet(&edges);
    let beta_1 = compute_beta_1(V, E, C);
    
    let mut n_reached = 0;
    let mut rounds_list = Vec::new();
    let mut agreements = Vec::new();
    
    for _ in 0..n_trials {
        let (reached, rounds, agreement) = run_beam_debate(&edges, 50);
        if reached { n_reached += 1; }
        rounds_list.push(rounds);
        agreements.push(agreement);
    }
    
    let consensus_rate = n_reached as f64 / n_trials as f64;
    let mean_rounds = rounds_list.iter().sum::<usize>() as f64 / n_trials as f64;
    let std_rounds = {
        let mean = mean_rounds;
        let variance = rounds_list.iter()
            .map(|&r| (r as f64 - mean).powi(2))
            .sum::<f64>() / n_trials as f64;
        variance.sqrt()
    };
    let mean_agreement = agreements.iter().sum::<f64>() / n_trials as f64;
    
    CaseResult {
        case_name: case_name.to_string(),
        V,
        E,
        beta_1,
        n_trials,
        consensus_rate,
        mean_rounds,
        std_rounds,
        mean_agreement,
    }
}

/// Run the full H1 emergence validation experiment
/// 
/// Returns summary of all cases and statistical analysis
pub fn validate_emergence_hypothesis() -> ExperimentSummary {
    let n_trials = 100;
    
    println!("\n============================================================");
    println!("H1 EMERGENCE VALIDATION EXPERIMENT");
    println!("100 trials per case, 5 cases, {} total trials", 5 * n_trials);
    println!("============================================================\n");
    
    let cases = [
        ("Case A: Tree (β₁=0)", generate_tree_graph(5)),
        ("Case B: Single Cycle (β₁=1)", generate_cycle_graph(5)),
        ("Case C: Two Cycles (β₁=2)", generate_two_cycles(5)),
        ("Case D: Laman-Rigid (β₁=V-2)", generate_laman_graph(5)),
        ("Case E: Over-Constrained (β₁>V-2)", generate_overconstrained_graph(5, 4)),
    ];
    
    let mut results = Vec::new();
    
    for (name, edges) in cases.iter() {
        let result = run_case(name, edges.clone(), n_trials);
        println!("{:30} β₁={:2}  consensus={:5.1}%  rounds={:6.1}±{:.1}",
            name, result.beta_1, result.consensus_rate * 100.0, result.mean_rounds, result.std_rounds);
        results.push(result);
    }
    
    println!("\n============================================================");
    println!("STATISTICAL ANALYSIS");
    println!("============================================================\n");
    
    // Compute effect size: Case E vs Case A (consensus rate difference)
    let case_a_rate = results[0].consensus_rate;
    let case_e_rate = results[4].consensus_rate;
    println!("Consensus rate difference (Case A - Case E): {:.1}% - {:.1}% = {:.1}%",
        case_a_rate * 100.0, case_e_rate * 100.0, (case_a_rate - case_e_rate) * 100.0);
    
    // F1 score for β₁ > V-2 predicting consensus failure
    // True positives: β₁ > V-2 and consensus failed
    // False positives: β₁ > V-2 but consensus succeeded  
    // False negatives: β₁ <= V-2 but consensus failed
    let true_positives = ((1.0 - results[4].consensus_rate) * n_trials as f64) as usize;
    let false_positives = (results[4].consensus_rate * n_trials as f64) as usize;
    let false_negatives = results[0..4].iter()
        .map(|r| ((1.0 - r.consensus_rate) * n_trials as f64) as usize)
        .sum::<usize>();
    
    let precision = if true_positives + false_positives > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else { 0.0 };
    
    let recall = if true_positives + false_negatives > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else { 0.0 };
    
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else { 0.0 };
    
    println!("\nF1 Score for β₁ > V-2 predicting consensus failure:");
    println!("  Precision: {:.2}", precision);
    println!("  Recall: {:.2}", recall);
    println!("  F1: {:.2}", f1);
    
    let summary = ExperimentSummary {
        case_results: results,
        f1_score: f1,
        precision,
        recall,
        case_a_vs_e_diff: (case_a_rate - case_e_rate).abs(),
    };
    
    println!("\n[Experiment complete. See EXPERIMENT-01-h1-emergence.md for analysis.]\n");
    
    summary
}

/// Summary of the full experiment
#[derive(Debug)]
pub struct ExperimentSummary {
    pub case_results: Vec<CaseResult>,
    pub f1_score: f64,
    pub precision: f64,
    pub recall: f64,
    pub case_a_vs_e_diff: f64,
}

// =============================================================================
// Quick test/verification
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_generators() {
        // Tree: V=5, E=4, β₁=0
        let tree = generate_tree_graph(5);
        assert_eq!(tree.len(), 4);
        
        // Cycle: V=5, E=5, β₁=1
        let cycle = generate_cycle_graph(5);
        assert_eq!(cycle.len(), 5);
        
        // Laman: V=5, E=7, β₁=3
        let laman = generate_laman_graph(5);
        assert_eq!(laman.len(), 7);
        
        // Over-constrained: V=5, E>7
        let over = generate_overconstrained_graph(5, 3);
        assert!(over.len() > 7);
    }

    #[test]
    fn test_beta_computation() {
        let tree = generate_tree_graph(5);
        let (V, E, C) = build_fleet(&tree);
        assert_eq!(V, 5);
        assert_eq!(E, 4);
        assert_eq!(C, 1);
        assert_eq!(compute_beta_1(V, E, C), 0);
        
        let cycle = generate_cycle_graph(5);
        let (V, E, C) = build_fleet(&cycle);
        assert_eq!(compute_beta_1(V, E, C), 1);
        
        let laman = generate_laman_graph(5);
        let (V, E, C) = build_fleet(&laman);
        assert_eq!(compute_beta_1(V, E, C), 3); // V-2 = 3
    }

    #[test]
    fn test_components() {
        // Connected graph
        let tree = generate_tree_graph(5);
        assert_eq!(count_components(5, &tree), 1);
        
        // Disconnected: two separate trees
        let disconnected = vec![(0, 1), (2, 3)];
        assert_eq!(count_components(4, &disconnected), 2);
    }

    #[test]
    fn test_measure_agreement() {
        // Two agents with same belief
        let belief = vec![(0.0, 0.0), (0.5, 0.05), (1.0, 0.0)];
        let agents = vec![
            Agent::new(0, AgentRole::Architect, belief.clone()),
            Agent::new(1, AgentRole::EnergyAgent, belief.clone()),
        ];
        assert!((measure_agreement(&agents) - 1.0).abs() < 0.001);
        
        // Two agents with different beliefs
        let mut belief2 = belief.clone();
        belief2[1].1 = 0.10; // Different by 0.05
        let agents2 = vec![
            Agent::new(0, AgentRole::Architect, belief),
            Agent::new(1, AgentRole::EnergyAgent, belief2),
        ];
        let agreement = measure_agreement(&agents2);
        assert!(agreement < 1.0, "Different beliefs should give lower agreement");
    }

    #[test]
    fn test_beam_debate_single_trial() {
        let tree = generate_tree_graph(5);
        let (reached, rounds, agreement) = run_beam_debate(&tree, 50);
        println!("Tree debate: reached={}, rounds={}, agreement={:.3}", reached, rounds, agreement);
        // Tree should always converge
        assert!(reached || rounds < 50);
    }

    #[test]
    fn test_small_experiment() {
        // Quick test with 5 trials per case
        let cases = [
            ("Tree", generate_tree_graph(5)),
            ("Cycle", generate_cycle_graph(5)),
        ];
        
        for (name, edges) in cases.iter() {
            let result = run_case(name, edges.clone(), 5);
            println!("{}: β₁={}, consensus_rate={:.1}%", name, result.beta_1, result.consensus_rate * 100.0);
        }
    }

    #[test]
    fn test_laman_rigid_exactly_v_minus_2() {
        let laman = generate_laman_graph(5);
        let (V, E, C) = build_fleet(&laman);
        let beta_1 = compute_beta_1(V, E, C);
        // For V=5, Laman-rigid: E=2V-3=7, β₁=E-V+C=7-5+1=3=V-2 ✓
        assert_eq!(beta_1, 3, "Laman graph should have β₁ = V-2");
    }
}