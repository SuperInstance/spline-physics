# Spline Embeddings: From Point Clouds to Continuous Manifolds

## SECTION 1: The Mathematical Structure of "Becoming One With The Tool"

### 1.1 Constraint Manifolds and Embodied Practice

Skill acquisition—the process of "becoming one with the tool"—admits a geometric formulation. Let $\mathcal{M}$ denote the manifold of all possible tool states. For a spline-physics system, $\mathcal{M}$ is the product of the configuration space of the tool (e.g., the space of all beam shapes representable by quadratic Bézier segments) and the space of material parameters. Within $\mathcal{M}$, the *constraint submanifold* $\mathcal{C} \subset \mathcal{M}$ collects those states that satisfy all physical and geometric constraints: the beam must be continuous, the material must not yield, and the shape must lie within the tolerance of the intended spline curve. The practitioner's journey from novice to expert is a trajectory $\gamma: [0,1] \to \mathcal{C}$ parameterized by practice time $t$.

The embodied practice loop consists of four phases:

1. **Observe**: Sample the current constraint manifold $\mathcal{C}$ via sensorimotor feedback.
2. **Plan**: Compute the tangent space $T_c\mathcal{C}$ at the current state $c$ and identify a direction of improvement.
3. **Act**: Move along a curve in $\mathcal{C}$ that reduces the error between the felt and intended tool–user interaction.
4. **Refine**: Update the trajectory $\gamma$ by incorporating the new observation, effectively performing a gradient step on the "fluency functional."

Fluency—the hallmark of expert performance—corresponds to the property that the trajectory $\gamma$ becomes dense in $\mathcal{C}$. More formally, for any open set $U \subset \mathcal{C}$, the preimage $\gamma^{-1}(U)$ has positive Lebesgue measure. The practitioner no longer needs to consciously plan each movement; the body has internalized the full geometry of $\mathcal{C}$.

### 1.2 Tangent Space and the Skill Velocity Field

At any configuration $c \in \mathcal{C}$, the tangent space $T_c\mathcal{C}$ consists of all infinitesimal displacements that preserve the constraints. For a beam with $N$ segments and $4(N-1)$ joint degrees of freedom, the tangent space is a linear subspace of $\mathbb{R}^{4(N-1)}$ determined by the linearized constraints.

The *skill velocity field* is a map $v: \mathcal{C} \to T\mathcal{C}$ assigning to each $c$ a vector $v(c) \in T_c\mathcal{C}$. This vector encodes the instantaneous rate of change of the practitioner's motor command. Expert performance is characterized by $v(c)$ aligning with the principal curvature directions of $\mathcal{C}$—i.e., moving *with* the natural dynamics of the tool rather than fighting them. This alignment can be quantified by the angle between $v(c)$ and the tangent to the geodesic that minimizes the bending energy functional

$$
U = \int_0^L \frac{EI}{2} \, \kappa(s)^2 \, ds,
$$

where $\kappa(s)$ is the curvature of the beam at arc length $s$, $E$ the Young's modulus, and $I$ the second moment of area. When $v(c)$ is tangent to a geodesic of this energy, the practitioner experiences the tool as an extension of the self.

### 1.3 The Phase Space of Tool-Practice Systems

The full state of a tool–practice system lives in the product space $\mathcal{S} = \mathcal{C} \times \mathcal{B}$, where $\mathcal{B}$ is the *belief space*—the set of possible internal models a practitioner may hold about the tool's behavior. In the multi-agent framework of the `spline-physics` codebase (FLUX-C opcodes `0xD0`–`0xD3`), each agent maintains a belief vector $(T, M, y, \theta)$ representing the force, moment, displacement, and slope at each joint. The joint dynamics are coupled:

$$
\begin{aligned}
\dot{c} &= f(c, b), \\
\dot{b} &= g(c, b),
\end{aligned}
$$

where $f$ describes the physical evolution of the beam (e.g., under external loads) and $g$ models the update of beliefs through trust-weighted spring-damper dynamics:

$$
b_{\text{next}} = b_{\text{current}} + k_{\text{trust}} \cdot (b_{\text{neighbor}} - b_{\text{current}}) \cdot w_{\text{trust}}.
$$

The "flow" of practice is the integral curve of this coupled system. A fixed point $(c^*, b^*)$ is called a *resonant state*: the tool configuration $c^*$ and the practitioner beliefs $b^*$ are mutually compatible. At such a point, the residual norm of the joint equilibrium conditions (from `beam.rs`) falls below tolerance, and the system satisfies the consensus condition

$$
\text{residual\_norm} < \epsilon.
$$

This resonant state is precisely what is colloquially termed "becoming one with the tool."

---

## SECTION 2: Spline Embeddings vs Point Embeddings

### 2.1 The Fundamental Limitation of Point Embeddings

A point embedding $f: \mathcal{D} \to \mathbb{R}^d$ assigns to each input $x \in \mathcal{D}$ a single vector $f(x) \in \mathbb{R}^d$. While simple, this representation offers no continuity guarantees: even if two inputs $x_1$ and $x_2$ are close in $\mathcal{D}$, their embeddings $f(x_1)$ and $f(x_2)$ may be arbitrarily far apart in $\mathbb{R}^d$. This violates the geometric intuition that embeddings should preserve the structure of the input space.

Define a point embedding to be *$C^k$-continuous* if there exists a constant $L > 0$ such that

$$
\| f(x_1) - f(x_2) \| < L \, \| x_1 - x_2 \|^k
$$

for all $x_1, x_2$ sufficiently close. The constant $L$ is the Lipschitz constant of the $k$-th order. Typical neural network embeddings are at best $C^0$ (continuous) but often not even that, due to the use of non‑smooth activation functions like ReLU. Consequently, small perturbations in the input can cause large, unpredictable changes in the embedding—a serious drawback for constraint‑based problems such as beam equilibrium.

### 2.2 Spline Embeddings: Curves in High-Dimensional Space

A *spline embedding* overcomes this limitation by mapping each input $x$ not to a point but to a *curve* in $\mathbb{R}^d$. Formally, let $\mathcal{C}(\mathbb{R}^d)$ denote the space of continuous curves $\gamma: [0,1] \to \mathbb{R}^d$ endowed with the Hausdorff metric. Then a spline embedding is a map

$$
F: \mathcal{D} \to \mathcal{C}(\mathbb{R}^d)
$$

such that for each $x$, $F(x)$ is a B‑spline (or Bézier curve) of degree $n$, determined by $n+1$ control points $P_0(x),\dots,P_n(x) \in \mathbb{R}^d$. The control points themselves are functions of $x$.

For a quadratic Bézier curve, the embedding is given explicitly by

$$
F(x)(t) = (1-t)^2 P_0(x) + 2(1-t)t P_1(x) + t^2 P_2(x), \qquad t\in[0,1].
$$

The key property is that if $x$ changes slightly, the control points change slightly, and consequently the entire curve changes smoothly.

**Theorem (Spline Embedding Continuity).**
Let $F$ be a spline embedding constructed from degree‑$n$ basis functions. If each control point map $P_i: \mathcal{D} \to \mathbb{R}^d$ is Lipschitz continuous with constant $L_i$, then $F$ is $C^{n-1}$‑continuous as a map $\mathcal{D} \to \mathcal{C}(\mathbb{R}^d)$.

*Proof sketch.* The basis functions of degree $n$ have continuous derivatives up to order $n-1$. The embedding $F(x)$ is a linear combination of these basis functions with $x$‑dependent coefficients. By the chain rule, all derivatives of $F$ with respect to $x$ up to order $n-1$ exist and are continuous. The Hausdorff metric on the image curves inherits this smoothness. ∎

For the quadratic Bézier case ($n=2$), the embedding is $C^1$‑continuous—exactly the continuity required for the beam joint equilibrium in `constraint‑theory‑llvm`.

### 2.3 Curvature and the Shape Signature

For a curve $\gamma(s)$ parameterized by arc length $s$, the curvature is

$$
\kappa(s) = \| \gamma''(s) \|.
$$

For a quadratic Bézier curve $B(t)$, the second derivative is constant:

$$
B''(t) = 2(P_2 - 2P_1 + P_0).
$$

Thus $\kappa = \| 2(P_2 - 2P_1 + P_0) \|$ is independent of $t$—the curve is a circular arc (or a straight line when the second derivative vanishes). This geometric fact is central to the spline‑physics implementation: each quadratic Bézier segment has constant curvature, making it an *Euler elastica* of constant bending moment.

The *feature vector* of a spline embedding includes:

- **Control points** $\{P_0, P_1, \dots, P_n\}$—the "handles" that parametrize the curve.
- **Curvature signature** $\kappa(s)$—the "feel" along the arc.
- **Arc length** $L = \int_0^1 \|B'(t)\|\,dt$—the total extent of the curve.
- **Centroid** $\bar{x} = \frac{\int_0^1 B(t)\,dt}{L}$—the geometric center.

In a multi‑segment beam, each segment contributes its own feature vector, and the join conditions couple them.

### 2.4 C² Continuity and the Join Condition

For two Bézier segments to join with $C^2$ continuity (continuous second derivative), three conditions must hold at the joint:

1. **C0 (position):** The endpoint of the first segment equals the start point of the second: $P_2^{(1)} = P_0^{(2)}$.
2. **C1 (tangent):** The first derivatives match: $P_2^{(1)} - P_1^{(1)} = P_1^{(2)} - P_0^{(2)}$.
3. **C2 (curvature):** The second derivatives match: $P_2^{(1)} - 2P_1^{(1)} + P_0^{(1)} = P_2^{(2)} - 2P_1^{(2)} + P_0^{(2)}$.

The $C^2$ condition states that the "change of curvature" is the same on both sides of the joint. For a beam, $C^2$ continuity implies continuous bending moment, a requirement for physical plausibility.

For cubic B‑splines ($n=3$), full $C^2$ continuity requires knot multiplicity conditions. This explains why the `spline-physics` codebase uses quadratic Bézier as its canonical form: $C^1$ continuity (which is automatically satisfied by the middle control point condition $P_1^{(1)} = P_1^{(2)}$) is sufficient for the beam equilibrium problem, and the simpler join condition makes multi‑segment coordination tractable. The `ANALOG_SPLINE` opcode implements quadratic Bézier evaluation, while `ANALOG_SECTOR` divides a segment into equal‑length subsegments, preserving $C^1$ joins.

---

## SECTION 3: The "Fit" Operator

### 3.1 Cosine Similarity vs Constraint Satisfaction

Standard embedding similarity uses cosine similarity:

$$
\operatorname{sim}_{\text{cos}}(a, b) = \frac{a \cdot b}{\|a\|\,\|b\|}.
$$

This measures *angular* alignment, not *geometric* fit. Two vectors can point in the same direction (cosine = 1) but be in completely different locations (far apart in Euclidean space). For tool‑practice systems, what matters is whether a tile (a physical point) lies *on* the spline curve, not whether its embedding vector points the same way as the spline's embedding.

The "Fit" operator for spline embeddings is fundamentally different:

$$
\operatorname{Fit}(\text{spline}, \text{tile}) = \operatorname{tile\_valid}(\text{tile}, \text{spline}),
$$

where `tile_valid` returns `true` if the tile is within tolerance of the spline curve. This is a *hard constraint check*, not a soft similarity score. The tile must satisfy the geometric constraint—there is no "somewhat fits."

### 3.2 Tolerance as the GUARD Condition

The GUARD condition is the tolerance check in `tile_valid`:

```rust
deviation = |tile_y - spline_y(tile_x)|
tolerance = epsilon + material_variation * tension
```

The tolerance is not arbitrary; it encodes physical constraints:

- $\epsilon = 10^{-6}$: numerical precision floor.
- `material_variation = 0.05` (5%): inherent material inhomogeneity (e.g., for Cedar $E=6\ \text{GPa}$, the variation captures natural differences in wood grain).
- `tension`: a user‑controlled parameter that affects sample density (higher tension requires tighter adherence).

This mirrors the equilibrium tolerance in the beam solver:

```rust
consensus_reached if residual_norm < tolerance
```

Both are GUARD conditions: if satisfied, the system is in a valid state.

### 3.3 The Fit Metric Space

Define the space $\mathcal{F}$ of all "fits" as

$$
\mathcal{F} = \{ (\text{spline}, \text{tile}) \mid \operatorname{tile\_valid}(\text{tile}, \text{spline}) = \text{true} \}.
$$

This is a *subset* of the product space $\text{Splines} \times \text{Tiles}$, not the whole product. Points outside $\mathcal{F}$ are invalid combinations.

The "distance to fit" is

$$
d_{\text{fit}}(\text{tile}, \text{spline}) = \operatorname{tile\_distance}(\text{tile}, \text{spline}),
$$

where $\operatorname{tile\_distance}$ returns the minimum Euclidean distance from the tile to the spline curve. This is a metric: it satisfies the identity of indiscernibles, symmetry, and the triangle inequality.

### 3.4 Constraint Satisfaction as Optimization

The Fit operator can be viewed as the objective function of a constraint satisfaction problem:

$$
\begin{aligned}
&\text{Find } \text{tile\_position} \text{ such that } \\
&\qquad \operatorname{tile\_distance}(\text{tile\_position}, \text{spline}) \to 0 \\
&\text{subject to: } \text{tile\_position} \in \text{TileDomain}.
\end{aligned}
$$

This is a root‑finding problem: $f(\text{position}) = \operatorname{spline}_y(\text{position}_x) - \text{position}_y = 0$.

For the beam joint equilibrium problem, the analogous problem is:

$$
\begin{aligned}
&\text{Find } (T, M, y, \theta) \text{ such that } \\
&\qquad \operatorname{residual}(\text{joint\_state}) = 0.
\end{aligned}
$$

The Newton–Raphson solver in `beam.rs` solves this in $\mathbb{R}^{4(N-1)}$. The connection between the spline Fit operator and the beam solver is not coincidental: both are instances of *constraint satisfaction*, and both use tolerance as a GUARD.

---

## SECTION 4: Interpolation vs Optimization

### 4.1 Interpolation: O(n) Direct Computation

Interpolation constructs a function that passes through all given data points exactly. For $n$ points with $n$ unknowns, direct solution gives O(n) complexity (Gaussian elimination on a tridiagonal system yields O(n); for general matrices O(n³)).

For the quadratic Bézier spline: given three boundary points $(P_0, P_1, P_2)$, the curve is fully determined with *no iteration* required:

$$
B(t) = (1-t)^2 P_0 + 2(1-t)t P_1 + t^2 P_2.
$$

This is a closed‑form expression. Computing $B(t)$ for any $t$ costs O(1)—just evaluating the polynomial.

For the analog compute operations in the FLUX‑C opcode set:

- `ANALOG_WATER_LEVEL`: arithmetic mean of $y$‑values – O(n) single pass.
- `ANALOG_SECTOR`: equal division of distance – O(1) pure division.
- `ANALOG_STORY_POLE`: cumulative sum of deltas – O(n) single pass.

All of these are *interpolation*—they construct a result that exactly satisfies the constraints without any iterative refinement.

### 4.2 Optimization: O(iterations × dimensions)

Optimization finds the best solution among infinitely many candidates. For the beam equilibrium:

$$
\begin{aligned}
&\text{minimize } U = \int_0^L \frac{EI}{2} \, \kappa(s)^2 \, ds \\
&\text{subject to boundary conditions}.
\end{aligned}
$$

This is an infinite‑dimensional variational problem. Discretized to $N$ segments, we have $N \times 4$ unknowns (force $T$, moment $M$, displacement $y$, slope $\theta$ at each segment end). The Newton–Raphson solver operates as follows:

- Each iteration: O(N) to evaluate residuals and Jacobian.
- Number of iterations: typically 10–100 for convergence.
- Total: O(iterations $\times N \times 4$) = O(iterations $\times$ dimensions).

The `EnergyMinimizationSolver` in `spline-physics` uses gradient descent with convergence criterion $\|\nabla E\| < 10^{-8}$. This requires many gradient evaluations, each costing O(N). The `ShootingMethodSolver` similarly iterates until the boundary conditions are satisfied.

### 4.3 The Interpolation-Optimization Boundary

A key insight from the multi‑agent debate model is that optimization can *look* like interpolation. Each agent maintains a "belief" about the beam shape. The spring‑damper trust update moves beliefs toward each other:

$$
b_{\text{next}} = b_{\text{current}} + k_{\text{trust}} \cdot (b_{\text{neighbor}} - b_{\text{current}}) \cdot w_{\text{trust}}.
$$

When all agents reach consensus (beliefs equal), the result is a *fixed point* of the belief dynamics. The consensus point is the "interpolated" solution—it satisfies all constraints simultaneously. But finding this fixed point requires iteration (optimization). The "interpolation" is only apparent at convergence.

**Theorem (Interpolation‑Optimization Duality).**
Let the trust topology be connected and let the contraction mapping coefficient $\lambda < 1$. Then the fixed point of the belief dynamics equals the unique global minimum of the bending energy functional $U$. In this case, optimization *is* interpolation—finding the fixed point solves the variational problem directly.

*Proof sketch.* The spring‑damper dynamics are a gradient flow on the energy landscape. When the trust matrix is doubly stochastic and the energy is convex, the gradient flow converges to the unique minimizer. The fixed point condition (beliefs equal) is equivalent to the stationarity condition of the energy functional, i.e., the Euler–Lagrange equations. ∎

This duality is the mathematical foundation of the "consensus = equilibrium" claim in SPEC.md.

### 4.4 Complexity Comparison

| Method | Time Complexity | Space Complexity | Solution Type |
|--------|----------------|-------------------|---------------|
| `ANALOG_SPLINE` | O(1) per $t$ | O(1) | Exact (interpolating) |
| `ANALOG_WATER_LEVEL` | O(n) | O(1) | Exact (mean) |
| `ANALOG_STORY_POLE` | O(n) | O(n) | Exact (cumulative) |
| `EnergyMinimizationSolver` | O(iterations $\times N$) | O(N) | Approximate (optimizing) |
| `ShootingMethodSolver` | O(iterations $\times N$) | O(N) | Approximate (root‑finding) |
| Multi‑Agent Debate | O(rounds $\times N \times$ connections) | O(N agents) | Approximate (consensus) |

The key distinction: interpolation methods have *predictable* complexity (always O(n) or O(1)). Optimization methods have *unpredictable* complexity (depends on iteration count, which depends on problem conditioning). The multi‑agent debate lies at the boundary, trading unpredictability for parallelism and robustness.

---

## SECTION 5: Testable Hypotheses

### Hypothesis 1: Spline Embedding Continuity Enables Faster Learning

**Claim:** Agents using spline embeddings (curves with $C^1$ continuity) for tool state representation will converge to valid tool configurations in fewer practice episodes than agents using point embeddings.

**Experimental setup:**
- Two groups of simulated "practitioners":
  - **Group A:** point embedding (single vector per tool state).
  - **Group B:** spline embedding (curve parameters per tool state, using quadratic Bézier segments).
- Each episode: practitioner attempts to navigate to a target tool configuration (a specific beam shape) by adjusting control points or embedding vectors.
- Metric: `episodes_to_convergence` (percentage of successful navigations over 100 episodes, where success means the final shape is within tolerance of the target).

**Predictions:**
- Group B will show >2× faster convergence for complex tool geometries (e.g., multi‑segment beams with varying materials).
- Group B will have lower variance in convergence times (more consistent performance).
- Group B will extrapolate better to novel tool configurations (generalization to unseen target shapes).

**Test case:** Implement in `spline-physics` a new test module that compares point‑based versus curve‑based tile fitting. Use the `tile_distance` metric over 100 random tile positions. Measure the number of gradient steps required to reduce the average distance below a threshold.

---

### Hypothesis 2: The Fit Operator Outperforms Cosine Similarity for Constraint Satisfaction

**Claim:** The `tile_valid` (Fit) operator will correctly identify valid tool placements with higher precision and recall than cosine‑similarity‑based methods.

**Experimental setup:**
- Generate a ground‑truth dataset of valid (tile, spline) pairs from the `analog_spline` function in `spline-physics`.
- Generate a dataset of invalid pairs by perturbing tiles outside the tolerance.
- Compare three classifiers:
  1. **Fit operator:** `tile_valid` with threshold at tolerance.
  2. **Cosine similarity:** compare the feature vectors of spline and tile (using control points and curvature).
  3. **Euclidean distance:** compute Euclidean distance between feature vectors.

**Metrics:**
- Precision = TP / (TP + FP) — of tiles predicted valid, fraction actually valid.
- Recall = TP / (TP + FN) — of actually valid tiles, fraction predicted valid.
- $F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.

**Predictions:**
- Fit operator: Precision = 1.0, Recall = 1.0 (by construction—hard constraint).
- Cosine similarity: high recall but low precision (false positives on near‑boundary cases where vectors have similar angles but different positions).
- Euclidean distance: better than cosine but still worse than Fit (treats all dimensions equally, ignoring the geometry of the curve).

---

### Hypothesis 3: Multi‑Agent Consensus Converges in $O(\log(1/\epsilon))$ Rounds

**Claim:** For a beam with $N$ pins and a connected trust topology, the multi‑agent debate converges to an equilibrium configuration with error $< \epsilon$ in $O(\log(1/\epsilon))$ rounds.

**Experimental setup:**
- Implement the beam debate in `spline-physics` with configurable trust topologies: star, chain, complete.
- Vary $N$ from 3 to 10 pins.
- Measure `rounds_to_consensus` for decreasing $\epsilon$: $10^{-1}, 10^{-2}, \dots, 10^{-8}$.

**Predictions:**
- For connected trust topologies: $\text{rounds} = C + \log_{\lambda}(1/\epsilon)$, where $\lambda < 1$ is the contraction coefficient of the trust dynamics.
- For disconnected topologies: rounds diverges (no consensus possible).
- The constant $C$ depends on $N$, but the $\log$ term dominates for small $\epsilon$.

**Test case:** In the `multi_agent` module of `spline-physics`, add a test `test_beam_debate_convergence_rate` that:
1. Creates beams with $N \in \{3,5,7,10\}$ pins.
2. Runs debate with $\epsilon \in \{10^{-2}, 10^{-4}, 10^{-6}, 10^{-8}\}$.
3. Records `rounds_to_consensus` for each combination.
4. Verifies log‑linear relationship: $\log(\text{rounds}) \sim -\log(\epsilon)$.

**Statistical test:** Linear regression of $\log(\text{rounds})$ on $\log(\epsilon)$. Expect slope $\approx -1$ (doubling $\epsilon$ halves rounds). The $R^2$ should exceed 0.95 for all topologies.

---

*These hypotheses are designed to validate the mathematical foundations laid out in Sections 1–4. They are directly implementable within the existing `spline-physics`, `constraint-theory-llvm`, and `fleet-coordinate` repositories, and their confirmation would provide strong empirical support for the spline‑embedding framework as a model of embodied skill acquisition.*
