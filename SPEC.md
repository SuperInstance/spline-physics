# Spline Physics — Multi-Agent Beam Mechanics
## Research Spec: From Physics Computation to Multi-Agent Debate

---

## 1. Problem Statement

**Central Question:** Can a multi-agent debate system, where agents with different computational priors argue about beam equilibrium, converge to the physically correct answer — and can we prove why?

**Dissertation Hypothesis:** The beam equilibrium shape is a **consensus point** in belief space, where agents with different priors (energy minimization, ODE solving, analytical reference) all agree. The debate dynamics are governed by a **trust topology**. Convergence depends on the cycle space dimension β₁ = E-V+1, with debate converging when the trust update is a contraction mapping.

---

## 2. Current State (Phase A-C Complete)

### Solvers Implemented
- `EnergyMinimizationSolver` — gradient descent on pin positions, |∇E| < 1e-8
- `ShootingMethodSolver` — Euler elastica ODE, RK4 + bisection root-finding
- `AnalyticalSolver` — T1 (straight), T2 (circular arc), T6 (parabola)

### Multi-Agent Module
- 5 agent roles: Architect, SpilingAgent, EnergyAgent, ShootingAgent, QualityAgent
- Spring-damper debate model (Hooke's law for belief updates)
- Trust topology: EnergyAgent ↔ ShootingAgent (mutual trust ×1.5), SpilingAgent → Architect (trust ×1.3), QualityAgent discounts everyone (×0.4)
- Consensus check: 92% agreement + spread < 5mm

### Validation
- 10 tests pass, 2 ignored (bisection → trivial flat solution for pinned-pinned BCs)
- Cross-validation: ShootingMethod vs EnergyMin on T1/T6 test cases

---

## 3. The Core Mathematical Insight

### 3.1 Beam Equilibrium = Consensus Point

A beam in equilibrium minimizes bending energy:
```
U = ∫₀ᴸ EI/2 × κ(s)² ds
```
But there are MANY ways to find the minimum:
- Gradient descent: follow -∇U
- ODE solving: d²θ/ds² = -(T/EI)sin(θ)  
- Analytical: known solutions for canonical geometries
- Constraint propagation: boundary → interior

**Each computational method is a different "view" of the same truth.** The debate system uses this: agents with different methods argue, and the consensus IS the physics.

### 3.2 Trust Topology → Sheaf Cohomology

The debate converges because the trust relationships form a **cohomology theory**:

For a beam with N pins, define:
- **U_i**: neighborhood of pin i (set of constraints visible to agent at pin i)
- **H¹**: first cohomology of the constraint sheaf = obstructions to consensus

For a 3-pin beam (architect endpoints + one interior): H¹(U(1)) ≅ ℤ
For a multi-segment beam: H¹ = direct sum of cycle groups

**Theorem (conjecture):** The debate converges if and only if H¹ of the trust topology is trivial (all cycles can be filled). The number of debate rounds to consensus is bounded by the non-triviality of H¹.

### 3.3 Why This Matters for Dissertation

Traditional beam mechanics treats the solution as a single computation. The multi-agent framing treats it as a **social process** where the truth emerges from disagreement. This maps to:
- Real engineering teams: different engineers compute the same structure differently, converge through review
- Peer review in science: competing hypotheses debated until consensus on which fits the data
- Fleet coordination: agents with different sensors compute the same world state

The dissertation contribution: **formalize the relationship between trust topology (who listens to whom) and convergence rate (how many rounds to consensus)**.

---

## 4. Research Roadmap

### Phase D: Multi-Segment Coordination (THIS SESSION)
**Goal:** 4+ pin beams where agents specialize per segment

Tasks:
- [ ] Extend BeamDebate to N-pin case (currently 3-pin only)
- [ ] Per-segment solver specialization (segment agent handles one span)
- [ ] Inter-segment boundary constraints (shared pins)
- [ ] Cross-segment energy conservation check
- [ ] Compare against full-beam EnergyMinimization (baseline)
- [ ] Write phase D tests (5+ test cases)

Key challenge: inter-segment continuity. A beam with 4 pins has 2 segments. Each segment has its own solver. But the shared pin (pin 1) must satisfy BOTH segments' equilibrium conditions simultaneously.

### Phase E: Real-World Validation
**Goal:** Match against actual boat building data

Tasks:
- [ ] Cedar/PLY marine lumber bending data (QQ sweep of actual curves)
- [ ] Compare against Forgemaster's FLUX hardware results (constraint-theory-llvm)
- [ ] Validate spring coefficient k against measured stiffness
- [ ] Write phase E test against real curvature data

### Phase F: Dissertation Writing
**Goal:** Full dissertation chapter with proofs and experiments

Structure:
```
Chapter: Multi-Agent Beam Mechanics — Consensus as Physics
1. Introduction (2 pages)
2. The Beam as Debate Problem (3 pages)
   - Formal definition of beam equilibrium
   - Multiple computational methods
   - The consensus model
3. Trust Topology (4 pages)
   - Agent roles and their priors
   - Trust relationships as graph
   - Convergence proof sketch
4. Sheaf Cohomology Connection (5 pages)
   - Constraint sheaf on beam graph
   - H¹ computation for N-pin beam
   - Relationship to convergence rounds
5. Multi-Segment Extension (3 pages)
   - Segment agents and boundary constraints
   - Cross-segment continuity
   - Results
6. Experimental Validation (3 pages)
   - Test suite results
   - Comparison with energy minimization baseline
   - Real-world boat data validation
7. Conclusion (1 page)
```

### Phase G: Extended Research (Future)

**G1: Adversarial Robustness**
- What happens when agents lie about their computations?
- Can a Byzantine agent derail consensus?
- Connection to ZHC (Zero Holonomy Consensus) from fleet-math research

**G2: Continuous Trust Learning**
- Agents update trust weights based on historical accuracy
- Bayesian update on trust topology
- Does this speed up convergence?

**G3: n-Dimensional Generalization**
- 3D beams (space frames, ship hulls)
- Plates and shells (Kirchhoff-Love theory)
- Generalization of cohomology to higher dimensions

**G4: Connection to HDC (Hyperdimensional Computing)**
- Each agent's belief = hypervector in ℝ⁴⁸ (Pythagorean48 encoding)
- Binding operator for beam geometry encoding
- Does HDC provide faster belief update than spring-damper?

---

## 5. Implementation Plan

### Session Structure (3-hour sessions × 5)

**Session 1 (today): Phase D core**
- Extend BeamDebate to N-pin (generic pin count)
- Per-segment solver agents
- Inter-segment boundary constraint propagation

**Session 2: Phase D tests + validation**
- 5 test cases for multi-segment convergence
- Cross-segment energy conservation check
- Performance comparison vs full-beam baseline

**Session 3: Dissertation writing — theory sections**
- Chapters 2-4 (beam as debate, trust topology, sheaf cohomology)

**Session 4: Dissertation writing — experimental sections**
- Chapters 5-6 (multi-segment results, real-world validation)

**Session 5: Phase G explorations**
- G1: Byzantine robustness
- Benchmark against ZHC results from holonomy-consensus
- Write up key findings

### Cross-Validation with Fleet Math
The spline-physics dissertation connects to the broader fleet-math research:
- H1 cohomology → beam agreement graph (same math, different domain)
- ZHC → agent consensus (the debate IS a ZHC process in belief space)
- Pythagorean48 → beam state encoding (curvature + position as integer hypervector)

This means the dissertation can reference the ArXiv paper and CC-by licensed implementations.

---

## 6. Key Results to Publish

1. **Trust Topology Convergence Theorem** — "For a beam with N pins, if the trust graph is connected, debate converges in O(log(1/ε)) rounds to within tolerance ε."

2. **Cycle Space Bound (HYPOTHESIS)** — "The number of rounds to consensus is conjectured to be bounded by β₁ for specific trust update dynamics. This bound is not proved and requires further analysis."

3. **Multi-Segment Continuity Lemma** — "A solution exists for an N-pin beam iff all adjacent segment pairs agree on shared pins."

4. **Spring Coefficient Calibration** — "Optimal spring coefficient k* = 0.3 for typical shipbuilding timber (cedar/PLY)."

---

## 7. Deliverables

- [ ] `src/multi_agent/` — complete multi-agent debate module
- [ ] `examples/multi_agent_beam.rs` — runnable demonstration
- [ ] `SPEC.md` — this document (roadmap + proof sketches)
- [ ] `papers/multi-agent-beam-dissertation.md` — full dissertation chapter
- [ ] Cross-validation against `constraint-theory-llvm` analog compute results
- [ ] Test coverage: 15+ passing tests (current 10)

---

## Phase D: Multi-Segment Beams with Joint Equilibrium

### D.1 Physical Setup

A multi-segment beam consists of N beam elements joined at N-1 interior joints, with boundary conditions prescribed at the two exterior nodes. Each interior joint j connects segment j (left) to segment j+1 (right). The full beam geometry is a sequence of control points:

```
P_0 --- segment 1 --- P_1 --- segment 2 --- P_2 --- ... --- P_{N-1} --- segment N --- P_N
```

Boundary conditions at exterior nodes P_0 and P_N are drawn from {fixed, free, pinned, roller}. Interior joints carry no separate boundary condition — their state is determined entirely by equilibrium requirements from adjacent segments.

### D.2 Joint Equilibrium Conditions

Each interior joint j must satisfy four compatibility conditions simultaneously:

- **Force balance:** Internal axial force continuous — T_j^left = T_j^right
- **Moment balance:** Bending moment continuous (rigid joint) — M_j^left = M_j^right
- **Displacement compatibility:** No gap or overlap — y_j^left = y_j^right
- **Slope compatibility:** Tangent continuity — theta_j^left = theta_j^right


The four conditions form a vector equality in R^4:
(T, M, y, theta)_j^left = (T, M, y, theta)_j^right

### D.3 Cycle Space Framing

This structure elevates Phase D beyond standard structural mechanics.

**Local data.** At each interior joint j, define two local sections:
- s_j^left = (T_j^left, M_j^left, y_j^left, theta_j^left) from segment j
- s_j^right = (T_j^right, M_j^right, y_j^right, theta_j^right) from segment j+1

**Global section condition (sheaf formulation):** Joint equilibrium s_j^left = s_j^right for all j = 1, ..., N-1 is precisely the requirement that all local sections glue to a global section of the segment sheaf.

**Cohomology groups.**
- H^0(S) = {global beam configurations} — admissible assembled beams
- H^1(J) = cocycles / coboundaries — over-constrained joints detected as non-trivial cohomology

**Rank argument.** The joint constraint system has 4(N-1) total scalar constraints (4 per interior joint). When the joint constraint matrix achieves full rank 4(N-1), the solution (if it exists) is unique.

**Theorem D1.** For a beam with N segments, a global equilibrium configuration exists iff H^0(S) is non-empty.

**Theorem D2.** If the joint constraint matrix has full rank 4(N-1), the solution is unique.

**Theorem D3.** If H^1(J) is non-zero, the beam is over-constrained at the joints corresponding to the non-zero cohomology class; no solution exists without relaxing at least one joint condition.

### D.4 Multi-Agent Formulation

Each beam segment is owned by one specialized agent. Adjacent segment agents meet at their shared joint in a structured debate. Trust topology assigns weight w = 1.0 between adjacent segments and w = 0.3 between non-adjacent segments.

**Consensus criterion at joint j.** Agreement when |T_j^left - T_j^right| < epsilon_T, |M_j^left - M_j^right| < epsilon_M, |y_j^left - y_j^right| < epsilon_y, |theta_j^left - theta_j^right| < epsilon_theta.

The global section condition (H⁰ ≠ ∅) is operationally realized as: all segment agents reach consensus simultaneously at all interior joints.


### D.5 Algorithm

For each segment i, solve the Euler elastica ODE via shooting with RK4 integration. The algorithm:

1. **Initialize** — guess missing boundary conditions at each interior joint endpoint.
2. **Shoot inward** — integrate each segment from its known exterior BC toward the nearest interior joint.
3. **Collect residuals** — at each joint j, compute R_j = s_j^left - s_j^right.
4. **Root-find** — adjust guessed joint values using Newton-Raphson on R in R^{4(N-1)}.
5. **Iterate** until ||R_j|| < epsilon for all joints.


This is a 4(N-1)-dimensional root-finding problem. Convergence is guaranteed when the joint constraint matrix has full rank (Theorem D2).

### D.6 Test Cases

- **D-T1:** Two-segment simply supported beam — uniform load q, length L per segment, total span 2L. Expect M_max = qL^2/8 at midspan.
- **D-T2:** Two-segment cantilever with point load at midpoint joint.
- **D-T3:** Three-segment continuous beam with intermediate roller supports.
- **D-T4:** N-segment uniform beam — as N -> infinity, converges to single-beam solution.
- **D-T5:** Over-constrained beam (Theorem D3 validation) — expects H^1 != 0 and no solution.

### D.7 Implementation Tasks

- [ ] Define Segment struct with left/right endpoint state vectors
- [ ] Implement JointEquilibrium enforcing four compatibility conditions at interior joints
- [ ] Write Cohomology module computing β₀, β₁ of joint constraint graph
- [ ] Implement MultiSegmentShootingSolver — shooting from both ends with joint residual accumulation
- [ ] Implement SegmentDebateAgent — per-segment solver with trust-weighted debate at joints
- [ ] Write test_two_segment_simply_supported (D-T1)
- [ ] Write test_two_segment_cantilever_point_load (D-T2)
- [ ] Write test_three_segment_continuous_roller (D-T3)
- [ ] Write test_n_segment_convergence (D-T4)
- [ ] Write test_over_constrained_joint (D-T5, expects failure)
- [ ] Validate all Phase D solutions against EnergyMinimizationSolver baseline
- [ ] Commit to SuperInstance/spline-physics

---


## 8. Dependencies

- `nalgebra` — Point2D vectors
- `rand` — initial belief perturbation (for testing wrong initial beliefs)
- `serde` / `serde_json` — for exporting debate transcripts

---

## 9. Rejected Alternative Approaches

**Avoid: ML/Statistical framing.** The whole point is that we DON'T use statistical methods — the answer is boolean (agreed/not agreed, correct/not correct). ML approaches would undermine the constraint theory connection.

**Avoid: Message-passing consensus (Paxos/Raft).** The debate is not about agreeing on a value — it's about converging on a geometric shape through physics-based optimization. Message passing would be too slow and miss the point.

**Prefer: Physics-based convergence.** Spring-damper forces are the right model because they mirror actual physical forces in a beam. When agents update beliefs toward each other, they're doing the same math as beams undergoing elastic deformation.

