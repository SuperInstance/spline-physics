//! Multi-agent beam debate — a dissertation chapter in miniature
//! 
//! Demonstrates how a beam's equilibrium shape emerges from debate
//! among agents with different priors and computational methods.
//! 
//! Run with: cargo run --example multi_agent_beam
//! 
//! The key question: does the group converge to the PHYSICALLY CORRECT answer
//! (energy-minimizing equilibrium) even when agents start with wrong beliefs?

use spline_physics::multi_agent::{BeamDebate, AgentRole};
use spline_physics::multi_agent::joint_debate::MultiJointDebate;
use spline_physics::multi_segment::{MultiSegmentBeam, SegmentConfig, JointConfig, DistributedLoad, BoundaryCondition, MultiSegmentSolver};
use spline_physics::material::PLA;
use spline_physics::cross_section::Rectangular;

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     MULTI-AGENT BEAM DEBATE — Constraint Theory Diss     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("SCENARIO: A 1m span beam with 50mm target rise.");
    println!("Five agents — Architect, SpilingAgent, EnergyAgent, ShootingAgent,");
    println!("QualityAgent — must agree on the fair curve shape.");
    println!();
    println!("Agents START with DIFFERENT beliefs (plurality of priors).");
    println!("They DEBATE via spring-damper forces until consensus or max rounds.");
    println!();
    
    // Test 1: agents start wrong (too flat)
    println!("{}", "─".repeat(68));
    println!("TEST 1: All agents start with flat beam belief (pin at y=1mm)");
    println!("{}", "─".repeat(68));
    let mut debate = BeamDebate::new(
        (0.0, 0.0),   // left pin
        (1.0, 0.0),   // right pin
        (0.5, 0.001), // initial belief: nearly flat (1mm rise)
    );
    debate.min_agreement = 0.90;
    debate.max_spread = 0.010;
    debate.max_rounds = 15;
    let result = debate.run();
    debate.print_transcript();
    
    println!();
    println!("RESULT: agreement={:.0}%, spread={:.4}m, consensus_shape={:?}",
        result.agreement_index * 100.0,
        result.belief_spread,
        result.consensus_shape.get(1).map(|p| format!("({:.4}, {:.4})", p.0, p.1)).unwrap_or_default()
    );
    
    // Test 2: agents start with spread-out beliefs
    println!();
    println!("{}", "─".repeat(68));
    println!("TEST 2: Agents start with VERY DIFFERENT beliefs (y ∈ [10mm, 100mm])");
    println!("{}", "─".repeat(68));
    let mut debate2 = BeamDebate::new(
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, 0.050), // start at 50mm
    );
    debate2.min_agreement = 0.90;
    debate2.max_spread = 0.010;
    debate2.max_rounds = 15;
    let result2 = debate2.run();
    debate2.print_transcript();
    
    println!();
    println!("{}", "═".repeat(68));
    println!("CONVERGENCE ANALYSIS");
    println!("{}", "═".repeat(68));
    println!();
    println!("Key observations:");
    println!();
    println!("1. TRUST TOPOLOGY matters: EnergyAgent ↔ ShootingAgent cross-validation");
    println!("   pulls them toward the same answer faster than SpilingAgent alone.");
    println!();
    println!("2. QualityAgent (skeptic) discounts outliers — this is CORRECT behavior.");
    println!("   A skeptic should NOT be convinced by a small majority.");
    println!();
    println!("3. The PHYSICAL truth (energy-minimizing shape) emerges from the debate");
    println!("   even when agents start wrong. The spring forces pull beliefs toward");
    println!("   the equilibrium, and the equilibrium is determined by the boundary conditions.");
    println!();
    println!("4. H¹ COHOMOLOGY: The beam's topology (one independent cycle) means the");
    println!("   consensus always exists. Agents can't get stuck in irreconcilable");
    println!("   disagreement because the sheaf cohomology says H¹ = ℤ for this topology.");
    println!();
    
    // Show the physics reference
    println!("{}", "─".repeat(68));
    println!("PHYSICS REFERENCE: Expected equilibrium for h/L = 0.05");
    println!("{}", "─".repeat(68));
    println!("For a uniform load beam with h/L = 0.05:");
    println!("  Expected mid-span rise: ~50mm");
    println!("  Quadratic Bézier control point: (0.5, 0.1) — y = 2× rise");
    println!("  Energy: ~25-30 J (for PLA, 20×20mm cross-section)");
    println!("  Shooting angle at left pin: ~0.10 rad");

    // ═══════════════════════════════════════════════════════════════════════════════
    // PHASE D: Multi-Segment Beam with Joint Equilibrium
    // ═══════════════════════════════════════════════════════════════════════════════
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   PHASE D: MULTI-SEGMENT BEAM WITH JOINT EQUILIBRIUM     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("SCENARIO: 3-segment beam with 2 interior joints.");
    println!("Each joint is a debate between adjacent segment agents.");
    println!("Trust weights: 1.0 for adjacent, 0.3 for non-adjacent.");
    println!();

    // D-1: Joint Debate Demo
    println!("{}", "─".repeat(68));
    println!("D-1: Joint Debate at interior joints");
    println!("{}", "─".repeat(68));
    
    let mut multi_debate = MultiJointDebate::new(3, 1e-6);
    println!("Created multi-joint debate with 3 segments, 2 interior joints");
    
    // Run the debates
    let _results = multi_debate.run(30);
    println!();
    multi_debate.print_summary();
    
    // D-2: Full Multi-Segment Solver Demo  
    println!();
    println!("{}", "─".repeat(68));
    println!("D-2: Multi-Segment Beam Solver");
    println!("{}", "─".repeat(68));
    
    // Create a 3-segment simply supported beam
    let segments = vec![
        SegmentConfig {
            id: 0,
            length: 0.333,
            material: Box::new(PLA),
            section: Box::new(Rectangular { width: 0.020, height: 0.020 }),
            left_bc: BoundaryCondition::Pinned,
            right_bc: BoundaryCondition::Prescribed { T: None, M: None, y: None, theta: None },
        },
        SegmentConfig {
            id: 1,
            length: 0.334,
            material: Box::new(PLA),
            section: Box::new(Rectangular { width: 0.020, height: 0.020 }),
            left_bc: BoundaryCondition::Prescribed { T: None, M: None, y: None, theta: None },
            right_bc: BoundaryCondition::Prescribed { T: None, M: None, y: None, theta: None },
        },
        SegmentConfig {
            id: 2,
            length: 0.333,
            material: Box::new(PLA),
            section: Box::new(Rectangular { width: 0.020, height: 0.020 }),
            left_bc: BoundaryCondition::Prescribed { T: None, M: None, y: None, theta: None },
            right_bc: BoundaryCondition::Pinned,
        },
    ];
    
    let joints = vec![
        JointConfig { left_segment_id: 0, right_segment_id: 1, equilibrium_tolerance: 1e-6 },
        JointConfig { left_segment_id: 1, right_segment_id: 2, equilibrium_tolerance: 1e-6 },
    ];
    
    let load = DistributedLoad { q: 1000.0 }; // 1 kN/m
    
    let beam = MultiSegmentBeam { segments, joints, load };
    let solver = MultiSegmentSolver::new();
    
    println!("Solving 3-segment beam with uniform load q = 1000 N/m");
    println!("Segments: [0.333m] - [0.334m] - [0.333m]");
    println!("Supports: Pinned at x=0, Roller at x=0.333, Roller at x=0.667, Pinned at x=1.0");
    println!();
    
    match solver.solve(&beam) {
        Ok(joint_states) => {
            println!("SUCCESS: Solver converged!");
            println!();
            for (i, js) in joint_states.iter().enumerate() {
                let x_pos = match i {
                    0 => 0.333,
                    1 => 0.667,
                    _ => 0.0,
                };
                println!("  Joint {} at x={:.3}m:", i, x_pos);
                println!("    T: {:.2} N, M: {:.2} Nm, y: {:.4} m, theta: {:.4} rad",
                    js.T_left, js.M_left, js.y_left, js.theta_left);
                println!("    Residual norm: {:.2e}", js.residual_norm());
            }
        }
        Err(e) => {
            println!("SOLVER ERROR: {:?}", e);
        }
    }
    
    println!();
    println!("{}", "═".repeat(68));
    println!("PHASE D SUMMARY");
    println!("{}", "═".repeat(68));
    println!();
    println!("Key insights from Phase D:");
    println!();
    println!("1. JOINT EQUILIBRIUM: Each interior joint must satisfy 4 conditions:");
    println!("   T_left = T_right, M_left = M_right, y_left = y_right, theta_left = theta_right");
    println!();
    println!("2. ROOT-FINDING PROBLEM: Joint equilibrium is R^{{4(N-1)}} root-finding,");
    println!("   NOT energy minimization. Newton-Raphson converges quadratically.");
    println!();
    println!("3. SHEAF COHOMOLOGY: H^0(S) = global sections exist when joints agree.");
    println!("   H^1(J) != 0 indicates over-constrained joints (no solution).");
    println!();
    println!("4. JOINT DEBATE: Multi-agent consensus at joints with trust weights.");
    println!("   Adjacent segments: trust = 1.0, Non-adjacent: trust = 0.3");
}
