# Spline Physics — Multi-Agent Beam Mechanics
## Research Spec: From Physics Computation to Multi-Agent Debate

---

## 1. Problem Statement

**Central Question:** Can a multi-agent debate system, where agents with different computational priors argue about beam equilibrium, converge to the physically correct answer — and can we prove why?

**Dissertation Hypothesis:** The beam equilibrium shape is a **consensus point** in belief space, where agents with different priors (energy minimization, ODE solving, analytical reference) all agree. The debate dynamics are governed by a **trust topology** that maps to sheaf cohomology, and convergence is guaranteed for beam topologies with H¹ ≅ ℤ.

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

2. **Sheaf Cohomology Bound** — "The number of rounds to consensus is ≤ the first Betti number β₁ of the trust topology graph."

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

## 8. Dependencies

- `nalgebra` — Point2D vectors
- `rand` — initial belief perturbation (for testing wrong initial beliefs)
- `serde` / `serde_json` — for exporting debate transcripts

---

## 9. Rejected Alternative Approaches

**Avoid: ML/Statistical framing.** The whole point is that we DON'T use statistical methods — the answer is boolean (agreed/not agreed, correct/not correct). ML approaches would undermine the constraint theory connection.

**Avoid: Message-passing consensus (Paxos/Raft).** The debate is not about agreeing on a value — it's about converging on a geometric shape through physics-based optimization. Message passing would be too slow and miss the point.

**Prefer: Physics-based convergence.** Spring-damper forces are the right model because they mirror actual physical forces in a beam. When agents update beliefs toward each other, they're doing the same math as beams undergoing elastic deformation.

