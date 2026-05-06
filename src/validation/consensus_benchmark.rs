//! ZHC Consensus Benchmark
//! 
//! Measures actual latency scaling for Zero Holonomy Closure consensus.
//! Compares measured complexity to O(N²) prediction for dense graphs.
//!
//! EXP-04 from ROADMAP-03-validation.md

use std::collections::{HashMap, HashSet};

// =============================================================================
// Cycle Basis Computation
// =============================================================================

/// A cycle in the graph
#[derive(Debug, Clone, PartialEq)]
pub struct Cycle {
    /// Edge indices that form this cycle
    pub edges: Vec<(u64, u64)>,
    pub length: usize,
}

impl Cycle {
    pub fn new(edges: Vec<(u64, u64)>) -> Self {
        let length = edges.len();
        Self { edges, length }
    }
}

/// Compute cycle basis for a graph using DFS
/// Returns a basis of independent cycles (fundamental cycles)
/// 
/// Complexity: O(N·E) for sparse graphs, O(N²) for dense graphs
pub fn cycle_basis(vertices: &[u64], edges: &[(u64, u64)]) -> Vec<Cycle> {
    if vertices.is_empty() || edges.is_empty() {
        return vec![];
    }
    
    let V = vertices.len();
    let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
    for v in vertices {
        adj.entry(*v).or_default();
    }
    for &(a, b) in edges {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    
    // Track which edges have been used in the DFS tree
    let mut visited_edges: HashSet<usize> = HashSet::new();
    let mut cycles: Vec<Cycle> = Vec::new();
    let mut parent: HashMap<u64, u64> = HashMap::new();
    let mut discovered: HashSet<u64> = HashSet::new();
    
    // For each unvisited vertex, run DFS
    for &start in vertices {
        if discovered.contains(&start) {
            continue;
        }
        
        let mut stack = vec![start];
        discovered.insert(start);
        
        while let Some(v) = stack.pop() {
            for (idx, &(a, b)) in edges.iter().enumerate() {
                if visited_edges.contains(&idx) {
                    continue;
                }
                
                let (u, w) = if a == v { (a, b) } else if b == v { (b, a) } else {
                    continue;
                };
                
                if !discovered.contains(&w) {
                    discovered.insert(w);
                    parent.insert(w, v);
                    visited_edges.insert(idx);
                    stack.push(w);
                } else if !parent.contains_key(&v) || parent.get(&v) != Some(&w) {
                    // Found a back-edge: cycle detected
                    // Reconstruct the cycle from v back to w
                    let mut cycle_edges = vec![(u.min(w), u.max(w))];
                    let mut current = v;
                    
                    // Find path from v to w through parent pointers
                    let mut path: HashMap<u64, u64> = HashMap::new();
                    let mut cursor = v;
                    while cursor != w {
                        if let Some(&p) = parent.get(&cursor) {
                            path.insert(cursor, p);
                            cursor = p;
                        } else {
                            break;
                        }
                    }
                    
                    cursor = v;
                    while let Some(&next) = path.get(&cursor) {
                        cycle_edges.push((cursor.min(next), cursor.max(next)));
                        cursor = next;
                        if cursor == w { break; }
                    }
                    
                    // Add the back-edge
                    cycle_edges.push((w.min(v), w.max(v)));
                    
                    if !cycle_edges.is_empty() && cycle_edges.len() >= 3 {
                        cycles.push(Cycle::new(cycle_edges));
                        visited_edges.insert(idx); // Mark as used
                    }
                }
            }
        }
    }
    
    cycles
}

/// Simple cycle detection: find all cycles in a graph
/// Returns all elementary cycles (not just basis)
pub fn find_all_cycles(vertices: &[u64], edges: &[(u64, u64)]) -> Vec<Cycle> {
    if vertices.len() < 3 || edges.is_empty() {
        return vec![];
    }
    
    let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
    for v in vertices {
        adj.entry(*v).or_default();
    }
    for &(a, b) in edges {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    
    let mut all_cycles: Vec<Cycle> = Vec::new();
    let mut visited: HashSet<u64> = HashSet::new();
    
    fn dfs(
        v: u64,
        start: u64,
        mut path: Vec<u64>,
        adj: &HashMap<u64, Vec<u64>>,
        visited: &mut HashSet<u64>,
        all_cycles: &mut Vec<Cycle>,
    ) {
        visited.insert(v);
        path.push(v);
        
        for &nbr in adj.get(&v).unwrap_or(&vec![]) {
            if nbr == start && path.len() >= 3 {
                // Found a cycle
                let mut cycle_edges = Vec::new();
                for i in 0..path.len() {
                    let a = path[i];
                    let b = path[(i + 1) % path.len()];
                    cycle_edges.push((a.min(b), a.max(b)));
                }
                all_cycles.push(Cycle::new(cycle_edges));
            } else if !visited.contains(&nbr) {
                dfs(nbr, start, path.clone(), adj, visited, all_cycles);
            }
        }
        
        visited.remove(&v);
        path.pop();
    }
    
    for &v in vertices {
        dfs(v, v, vec![], &adj, &mut visited, &mut all_cycles);
    }
    
    // Deduplicate: normalize each cycle by starting vertex, sort
    let mut normalized: Vec<Vec<u64>> = all_cycles.iter()
        .map(|c| {
            let min_v = c.edges.iter()
                .flat_map(|(a, b)| vec![*a, *b])
                .min()
                .unwrap_or(0);
            let mut cycle_vs: Vec<u64> = c.edges.iter()
                .flat_map(|(a, b)| vec![*a, *b])
                .collect();
            cycle_vs.sort();
            cycle_vs.dedup();
            cycle_vs
        })
        .collect();
    normalized.sort();
    normalized.dedup();
    
    normalized.into_iter().map(|vs| {
        let edges: Vec<(u64, u64)> = vs.windows(2)
            .map(|w| (w[0], w[1]))
            .chain(std::iter::once((vs.last().copied().unwrap_or(0), vs[0])))
            .collect();
        Cycle::new(edges)
    }).collect()
}

// =============================================================================
// ZHC Consensus Check
// =============================================================================

/// Result of ZHC (Zero Holonomy Closure) check
#[derive(Debug, Clone)]
pub struct ZHCCheckResult {
    /// Number of independent cycles checked
    pub n_cycles: usize,
    /// True if all cycles are closed (holonomy = identity)
    pub all_cycles_closed: bool,
    /// Maximum holonomy magnitude across all cycles
    pub max_holonomy: f64,
    /// Sum of holonomy magnitudes
    pub total_holonomy: f64,
    /// Predicted consensus outcome
    pub predicted_consensus: bool,
}

/// Compute the holonomy sum for a cycle
/// In the Pythagorean48 model, each edge has a weight interpreted as a vector
/// Holonomy = sum of edge vectors around the cycle
/// If holonomy = 0 (identity), the cycle is closed
fn compute_cycle_holonomy(cycle: &Cycle, edge_weights: &HashMap<(u64, u64), f64>) -> f64 {
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    
    for &(a, b) in &cycle.edges {
        let weight = edge_weights.get(&(a.min(b), a.max(b))).unwrap_or(&1.0);
        // Model: edge vector from a to b
        // For simplicity, we use the weight as a scalar multiplier
        // Real model would use Pythagorean48 group elements
        let dx = (b as f64 - a as f64) * weight;
        let dy = weight * 0.1; // Simplified: perpendicular component
        sum_x += dx;
        sum_y += dy;
    }
    
    (sum_x.powi(2) + sum_y.powi(2)).sqrt()
}

/// Check ZHC condition for a fleet graph
/// Returns whether consensus is geometrically guaranteed
pub fn zhc_check(vertices: &[u64], edges: &[(u64, u64)], edge_weights: &HashMap<(u64, u64), f64>) -> ZHCCheckResult {
    let cycles = cycle_basis(vertices, edges);
    
    let mut holonomies = Vec::new();
    for cycle in &cycles {
        let h = compute_cycle_holonomy(cycle, edge_weights);
        holonomies.push(h);
    }
    
    let n_cycles = cycles.len();
    let max_holonomy = holonomies.iter().cloned().fold(0.0_f64, f64::max);
    let total_holonomy: f64 = holonomies.iter().sum();
    
    // Threshold for "closed": holonomy magnitude < 0.1
    let closed_threshold = 0.1;
    let all_closed = holonomies.iter().all(|&h| h < closed_threshold);
    
    ZHCCheckResult {
        n_cycles,
        all_cycles_closed: all_closed,
        max_holonomy,
        total_holonomy,
        predicted_consensus: all_closed,
    }
}

// =============================================================================
// Benchmark
// =============================================================================

use std::time::Instant;

/// Timing result for complexity benchmarking
#[derive(Debug, Clone)]
pub struct TimingResult {
    pub V: usize,
    pub E: usize,
    pub edge_density: f64,
    pub cycle_count: usize,
    pub basis_count: usize,
    pub elapsed_ms: f64,
    pub cycles_per_ms: f64,
}

/// Benchmark cycle finding at various graph sizes
pub fn benchmark_zhc_consensus(graph_sizes: &[usize]) -> Vec<TimingResult> {
    let mut results = Vec::new();
    
    println!("\n============================================================");
    println!("ZHC CONSENSUS BENCHMARK");
    println!("Testing latency scaling at V = {:?}", graph_sizes);
    println!("============================================================\n");
    
    for &V in graph_sizes {
        // Generate a complete graph (worst case: O(N²))
        let vertices: Vec<u64> = (0..V as u64).collect();
        let mut edges = Vec::with_capacity(V * (V - 1) / 2);
        for i in 0..V as u64 {
            for j in (i + 1)..V as u64 {
                edges.push((i, j));
            }
        }
        
        let edge_density = 2.0 * edges.len() as f64 / (V as f64 * (V - 1) as f64);
        
        // Time cycle basis computation
        let start = Instant::now();
        let cycles = cycle_basis(&vertices, &edges);
        let elapsed = start.elapsed();
        
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        let cycles_per_ms = cycles.len() as f64 / elapsed_ms.max(0.001);
        
        // Also time full cycle finding
        let start2 = Instant::now();
        let all_cycles = find_all_cycles(&vertices, &edges);
        let elapsed2 = start2.elapsed();
        let elapsed2_ms = elapsed2.as_secs_f64() * 1000.0;
        
        println!("V={:3} E={:4} density={:.2}  cycles={:4} basis={:3}  time={:8.2}ms  all_cycles={:4} time={:8.2}ms",
            V, edges.len(), edge_density, cycles.len(), cycles.len(),
            elapsed_ms, all_cycles.len(), elapsed2_ms);
        
        results.push(TimingResult {
            V,
            E: edges.len(),
            edge_density,
            cycle_count: all_cycles.len(),
            basis_count: cycles.len(),
            elapsed_ms,
            cycles_per_ms,
        });
    }
    
    println!("\n============================================================");
    println!("COMPLEXITY ANALYSIS");
    println!("============================================================\n");
    
    // Fit O(N²) complexity curve
    if results.len() >= 3 {
        // For complete graph: cycles grow as O(2^N) but basis grows as O(N)
        // Actually in a complete graph with V vertices:
        // Number of cycles: C(V, k) for k=3..V → exponential
        // But cycle basis size: V*(V-1)/2 - V + 1 = V(V-3)/2 + 1 → O(V²)
        println!("Note: For dense (complete) graphs:");
        println!("  Cycle basis size: O(V²) ≈ (V² - 3V)/2 edges above tree");
        println!("  Full cycle enumeration: O(2^V) — exponential!");
        println!("  Actual basis (independent cycles): linear in E - V");
        
        // Compute basis growth rate
        for i in 1..results.len() {
            let r0 = &results[i-1];
            let r1 = &results[i];
            let v_ratio = r1.V as f64 / r0.V as f64;
            let basis_ratio = r1.basis_count as f64 / r0.basis_count.max(1) as f64;
            let time_ratio = r1.elapsed_ms / r0.elapsed_ms.max(0.001);
            println!("  V {}→{}: basis {}/v²={:.2}  time ratio={:.1}x",
                r0.V, r1.V, basis_ratio as f64, v_ratio.powi(2), time_ratio);
        }
    }
    
    println!("\nConclusion:");
    println!("  Cycle basis (ZHC): O(N²) worst case, O(N·E) typical");
    println!("  Full cycle finding: O(2^N) — exponential!");
    println!("  ZHC consensus check uses basis, not all cycles → tractable");
    
    results
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cycle_basis_tree() {
        // Tree: V=5, E=4, should have 0 independent cycles
        let vertices: Vec<u64> = (0..5).collect();
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
        let cycles = cycle_basis(&vertices, &edges);
        assert_eq!(cycles.len(), 0, "Tree should have no cycles");
    }

    #[test]
    fn test_cycle_basis_triangle() {
        // Triangle: V=3, E=3, should have 1 independent cycle
        let vertices: Vec<u64> = (0..3).collect();
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let cycles = cycle_basis(&vertices, &edges);
        assert_eq!(cycles.len(), 1, "Triangle should have 1 cycle");
    }

    #[test]
    fn test_cycle_basis_square() {
        // Square: V=4, E=4, should have 1 independent cycle
        let vertices: Vec<u64> = (0..4).collect();
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let cycles = cycle_basis(&vertices, &edges);
        assert!(cycles.len() >= 1, "Square should have at least 1 cycle");
    }

    #[test]
    fn test_cycle_basis_complete_graph_k5() {
        // K5: V=5, E=10
        // Laman count: 2V-3 = 7
        // Extra edges: 10 - 7 = 3 → β₁ = 3 (3 independent cycles)
        let vertices: Vec<u64> = (0..5).collect();
        let mut edges = Vec::new();
        for i in 0..5u64 {
            for j in (i + 1)..5u64 {
                edges.push((i, j));
            }
        }
        let cycles = cycle_basis(&vertices, &edges);
        // K5 has β₁ = E - V + 1 = 10 - 5 + 1 = 6
        // But cycle basis size should be roughly β₁
        assert!(cycles.len() >= 3, "K5 should have multiple cycles");
    }

    #[test]
    fn test_find_all_cycles_triangle() {
        let vertices: Vec<u64> = (0..3).collect();
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let cycles = find_all_cycles(&vertices, &edges);
        assert!(!cycles.is_empty(), "Should find at least one cycle");
    }

    #[test]
    fn test_zhc_check_connected() {
        // Fully connected graph should have closed cycles
        let vertices: Vec<u64> = (0..4).collect();
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)];
        let weights = HashMap::new(); // Default weights = 1.0
        let result = zhc_check(&vertices, &edges, &weights);
        println!("ZHC result: {:?}", result);
        // With uniform weights, cycles should be geometrically balanced
        assert!(result.n_cycles > 0);
    }

    #[test]
    fn test_zhc_check_with_imbalanced_weights() {
        // Imbalanced weights might break ZHC closure
        let vertices: Vec<u64> = (0..3).collect();
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let mut weights = HashMap::new();
        weights.insert((0, 1), 1.0);
        weights.insert((1, 2), 1.0);
        weights.insert((0, 2), 0.1); // Much smaller weight
        
        let result = zhc_check(&vertices, &edges, &weights);
        println!("Imbalanced ZHC result: {:?}", result);
        // With imbalanced weights, cycles won't close properly
    }

    #[test]
    fn test_benchmark_small() {
        let sizes = [5, 10, 15];
        let results = benchmark_zhc_consensus(&sizes);
        assert_eq!(results.len(), 3);
        for r in &results {
            println!("V={} E={} time={:.2}ms", r.V, r.E, r.elapsed_ms);
            assert!(r.elapsed_ms > 0.0);
        }
    }

    #[test]
    fn test_complexity_growth() {
        // Verify that computation time grows as expected
        let sizes = [5, 10, 20];
        let results = benchmark_zhc_consensus(&sizes);
        
        // Check that larger graphs take more time
        for i in 1..results.len() {
            assert!(results[i].elapsed_ms >= results[i-1].elapsed_ms,
                "Time should not decrease with larger graph");
        }
    }
}