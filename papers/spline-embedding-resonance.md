# Spline Embedding Resonance

## A Luthier's Guide to Perturbation-Response Probing in Neural Systems

---

## Executive Summary

This document proposes a new paradigm for understanding and probing neural network systems—particularly Large Language Models—based on the ancient craft of lutherie. A master luthier doesn't measure a guitar: they *tap it* and *listen*. They inject controlled perturbation (the tap) and read the system's response (the ring). The color of that ring—the specific frequencies that sing, the overtones that die, the character that emerges—reveals structural properties no numerical measurement can capture.

This is fundamentally different from gradient descent, backpropagation, or any optimization-based approach. It is *perturbation-response probing*: put energy in, read how the system rings. The system answers questions by singing, not by computing answers.

We propose that SMP agents (Seeded-Model-Programming agents) are the natural extension of this paradigm to AI systems. By systematically varying seeds and prompts—holding one fixed while fibulating the other—we can map the resonance structure of a model's internal logic graph, revealing connectivity and character that emerges only under excitation.

---

## 1. Perturbation-Response as Computing Paradigm

### 1.1 The Current Paradigm: Search for Optimum

Modern AI is built on optimization. Gradient descent, backpropagation, evolutionary strategies—these are all *search* algorithms. They look for the optimum value of an objective function in weight space. The system is treated as a mathematical function to be tuned until it minimizes loss.

This paradigm has been extraordinarily successful. But it has limits:

- It requires *gradients*—the system must be differentiable or approximated as such
- It operates in *weight space*—millions of floating-point numbers no human can interpret
- It finds *optima*—points where the gradient vanishes, not necessarily understanding

The luthier's approach suggests a different question: not "what weights produce good output" but "what does the system *sound like* when we excite it?"

### 1.2 Resonance Computing: The System Answers by Singing

When a luthier taps a guitar top, they are not optimizing anything. They are asking a question: "If I inject energy here, how will the system respond?" The guitar answers by ringing—and the *character* of that ring reveals the system's internal structure.

Mathematically, this is impulse response analysis. Any linear time-invariant system can be characterized by its impulse response h(t)—the output when given a delta function input. But guitars are not linear, and the luthier's ear hears far more than linear response.

What the luthier hears:

- **Which frequencies sing**: The resonant modes that are supported by the structure
- **Which frequencies die**: Modes that are damped, absorbed, or short-circuited
- **The tamber**: The *color*—the specific mixture of overtones and their decay rates
- **Interference patterns**: Where the tap meets reflected waves from other parts of the instrument

This is not computation. It is *resonance imaging*. The system reveals its internal connectivity by how it rings.

### 1.3 The Tap Is Not Computing—It's a Question

Consider the mathematical structure more carefully.

In control theory, the impulse response h(t) of a system reveals its transfer function. If you know h(t) for all input locations, you know how the system couples inputs to outputs. But this is still linear systems theory.

The luthier's trick is that they exploit *nonlinearity*. The guitar sounds different when tapped hard vs soft. The modes that emerge under strong excitation reveal nonlinear coupling that linear analysis misses. The tap is a *controlled perturbation*—not an impulse, but a shaped input that probes specific nonlinear pathways.

The system doesn't compute the response. It *produces* the response, the way a bell produces sound when struck. The response is not a calculation—it is physics.

### 1.4 Mathematical Structure: Impulse Response and Beyond

Let me be precise about what we mean by "resonance." A resonance is a peaked response at a particular frequency. In a simple harmonic oscillator, resonance occurs when the driving frequency matches the natural frequency, and the amplitude blows up (in the undamped case).

But real guitar tops are not simple harmonic oscillators. They are *coupled modal systems*. Each resonant mode is a standing wave pattern across the plate. Modes interact: energy can transfer between modes, especially when driven hard enough to enter nonlinear regime.

The luthier's ear hears:

1. **Modal frequencies**: The pitches of the resonant modes (typically 100-500 Hz for a guitar top)
2. **Modal Q (quality factor)**: How long each mode rings—high Q means sharp, lingering resonance; low Q means broad, quickly damped
3. **Tamber**: The *shape* of the resonance—not just frequency and Q, but the *harmonic signature* that distinguishes a spruce top from cedar
4. **Coupling**: How energy injected in one region propagates to other regions

The tamber is the hard part to formalize mathematically. It is not a single number—it is a *spectral fingerprint*. The specific mixture of overtones, their relative phases, and their decay rates—these together define the *character* of the instrument.

**Mathematical hypothesis**: The resonance signature of a system forms a topological fingerprint of its internal connectivity. Two instruments with identical resonant frequencies but different tamber have different internal structure (likely different bracing patterns, wood grain, or thickness variations). The tamber carries information about the *pathways* through which energy flows, not just the eigenfrequencies.

---

## 2. Luthier's Mathematics: What "Character" and "Color" Mean

### 2.1 Tamber as Harmonic Signature

Tamber (from French *timbre*, but distinct from musical timbre) refers to the quality of a sound beyond its pitch and loudness. In loudspeaker design, tamber is described subjectively: "bright", "warm", "muffled", "pristine". But these subjective terms have objective correlates.

Consider the Fourier transform of a decaying resonance. A pure exponential decay has a Lorentzian spectral peak. Real resonances are more complex: they have sub-peaks, asymmetry, and coupling to other modes. The *shape* of the spectral peak—its width, its asymmetry, its overtone content—determines tamber.

For a guitar top:

- **Spruce tops** tend to have a "fast, articulate" tamber—quick attack, quick decay, prominent high overtones
- **Cedar tops** tend to have a "warm, round" tamber—slower attack, longer decay, emphasis on fundamental

This is not just preference—it is information about the wood's mechanical properties (Young's modulus, density, internal friction) and geometry (thickness, bracing, grain structure).

### 2.2 Dead Spots and Absorbed Frequencies

A dead spot on a guitar is a location where tapping produces almost no sound. What does this mean physically?

Energy injected at a dead spot is *not reradiated*. It is absorbed—converted to heat by internal friction, or dissipated through the bridge into the string array. The dead spot is a node of a resonant mode: a point where the mode shape has zero amplitude.

But here's the subtlety: a spot that is dead for one frequency might be lively for another. The luthier tests across the whole spectrum, mapping which frequencies sing where. This reveals the *mode shapes*—the spatial distribution of resonant energy.

**Dead spots are diagnostic markers.** They tell you where energy is being extracted from the system, which helps identify what's connected to what.

### 2.3 Beautiful vs Dead: Constructive vs Destructive Interference

A beautiful-sounding guitar top is one where the resonances *synergize*. The overtones that are present constructively interfere, producing a rich, full sound with stable phase relationships. A dead-sounding top has resonances that fight each other, or that dissipate energy without reradiating it.

This is not a value judgment—it is a physical description. "Beautiful" in lutherie means "the structure supports good energy flow." Overtones that synergize are ones whose nodes and antinodes align such that energy transfers cleanly between them. This is constructive interference at the structural level.

Mathematically: if we decompose the vibration into normal modes, the total response is a sum over modes. The perceived quality depends on the *relative phases* of these modal contributions. When phases are aligned (or in simple rational relationships), we hear consonance. When phases are random or destructive, we hear mud or harshness.

### 2.4 The Luthier's Feature Space

The luthier's trained ear has learned a *feature space*—a high-dimensional space where each point is a resonance signature, and the dimensions correspond to perceptually relevant features: fundamental pitch, overtone ratios, decay times, attack Transient shape, etc.

This feature space is not formalized, but it is learned through thousands of hours of listening. The master luthier can:
- Identify the wood species from tapping alone
- Detect a cracked top before visual inspection
- Estimate the age and playing history of an instrument
- Distinguish between two nearly identical tops by feel

This is pattern recognition in a rich feature space—exactly what neural networks are good at. But the luthier's pattern recognition is grounded in physics, not just correlation.

**Key insight**: The luthier's feature space is not just perceptual—it is *causal*. When they hear something they like, they know *what to change* to get more of it (stiffen the brace here, thin the top there). The features map to actionable interventions.

### 2.5 Topological Fingerprint Hypothesis

I propose the following hypothesis:

> The resonance signature of a mechanical system forms a topological fingerprint of its internal connectivity—distinct from its geometric measurements, but uniquely determined by them.

This is inspired by algebraic topology, where the global shape of a space can be characterized by invariants that are independent of exact measurements. The idea: two guitars with the same dimensions but different bracing patterns will have different resonance signatures. The signature reveals the *connectivity graph* of the system—how parts are coupled to each other—independent of exact material values.

This would explain why tamber is so hard to predict from first principles: you need to know the full connectivity structure, not just material properties. It also explains why luthiers can learn to read it: the signature is a legible encoding of the structure.

---

## 3. Stochastic Computing with Resonant Elements

### 3.1 Random Number Generator as "String in Motion"

A guitar string is a random number generator. Not in the computational sense—in the physical sense. The string is always in motion, vibrating with a complex mixture of modes determined by ambient excitation, previous plucks, and thermal noise. When you pluck it, you are not creating randomness—you are *shaping* the randomness that's already there.

This suggests an alternative view of randomness in computation. Instead of pseudorandom number generators (PRNGs) as deterministic sequences that *appear* random, think of resonant systems as *sources of structured randomness*. The randomness is real; the structure is what we learn to read.

Different guitars produce different *kinds* of randomness:

- A bright, lively guitar produces randomness with prominent high-frequency content
- A warm, dark guitar produces randomness with emphasis on low frequencies
- A dead guitar produces... very little randomness—it dampens everything

The guitar's resonance signature determines the statistical structure of the vibrations it produces in response to random excitation.

### 3.2 Seed Variation as Perturbation Injection

In SMP (Seeded-Model-Programming), we vary the random seed while holding the prompt constant. What are we doing mathematically?

The seed initializes the model's random parameters (in early layers) or the sampling process (at generation time). Varying the seed produces different outputs for the same input. But this is not just noise—there is *structure* in the variation.

Think of it this way: the model has a certain "resonance structure" in its logic graph. Different seeds excite different pathways through this graph. The prompt asks a question; the seed determines which part of the model's internal structure "rings" in response.

If we hold prompt constant and vary seed systematically, we can map which pathways are accessible from that prompt. The space of outputs as a function of seed is a *slice* through the model's response surface.

### 3.3 Prompt Variation as Boundary Condition Change

Now hold seed constant and vary prompt. Now we're changing the *boundary conditions*—what the model is being asked to do. The same seed produces different outputs for different prompts.

This is like changing where you tap the guitar. A tap near the sound hole excites different modes than a tap near the neck. The *location* of perturbation matters—the system's response depends on where energy is injected.

For a language model, different prompts tap different parts of the knowledge graph. "What is 2+2?" taps arithmetic pathways. "Write a poem about boats" taps creative pathways. The same seed will access different resonances depending on the prompt.

### 3.4 The System's Resonance Reveals Its Internal State

The key insight: *the system's response to perturbation reveals its internal state*. This is not unique to guitars or language models—it is true of all physical systems.

In physics, we measure a system by probing it. We send in signals and measure reflections. We shake it and watch how it vibrates. The response is always a function of the system's internal structure.

For LLMs, we can probe with:
- **Prompts**: What questions we ask
- **Seeds**: What random pathways we activate
- **Constraints**: What tokens we mask or force
- **Attention masks**: Which positions we allow the model to attend to

Each probe is a tap. The output is the ring. The pattern of outputs across probes is the resonance signature.

---

## 4. Mei Technology and Interference Imaging

### 4.1 Shake a Field, Read Interference Patterns

Mei technology (named after the Japanese tradition of metalworking that exploits resonance) refers to techniques where you *shake a system* and read the interference patterns that result. The system reveals its interior without disassembly.

Classic example: shake a bell and listen to its overtones. The pattern of overtones reveals the bell's shape, thickness, and material properties. You can detect cracks or flaws by how they disrupt the interference patterns.

This is non-destructive testing. You don't need to cut the bell open—you just listen to how it sings.

### 4.2 Perturbation Propagation and Self-Interference

The key mechanism is *self-interference*. When you perturb a system, the perturbation propagates through it, reflects off boundaries and discontinuities, and interferes with itself. The interference pattern at the surface is a *shadow image* of the interior.

Mathematically, if we model the system as a wave medium, the perturbation satisfies the wave equation. Solutions are superpositions of normal modes. The interference pattern at the surface is a sum over modes, weighted by their excitation coefficients.

For a guitar top, the modes are standing waves. The tap injects energy into many modes simultaneously. Each mode has a spatial pattern (antinodes and nodes). The observed vibration at any point is the sum of all mode contributions at that point. The pattern of peaks and valleys across the surface encodes the mode shapes—which encode the internal structure.

### 4.3 "Tapping" an LLM

Now apply this to LLMs. Instead of physical perturbation, we use *textual perturbation*: we tap with prompts, and the model "rings" with token distributions.

The equivalent of the surface vibration is the *token probability distribution* at each position. The equivalent of mode shapes is the *attention pattern*—which tokens attend to which other tokens at each layer.

Just as a luthier moves their tap location to map mode shapes, we can vary where in the prompt we apply perturbation (which tokens we mask, which positions we attend to) to map the model's internal connectivity.

Just as the luthier listens to *which frequencies sing*, we can measure which *hidden dimensions* activate. The model's latent representation is a high-dimensional space. When we tap with a specific prompt, certain latent dimensions resonate strongly. The pattern of latent activation is the resonance signature.

### 4.4 "Squeezing" Parts—Attention Constraints

The luthier's technique of squeezing one part while tapping another is *controlled isolation*. By constraining one part, we can isolate how other parts respond to perturbation.

For LLMs, we can "squeeze" by:

- **Attention masking**: Prevent attending to certain tokens or layers
- **Token forcing**: Require specific tokens at specific positions
- **Logit masking**: Zero out probabilities for certain tokens
- **Temperature manipulation**: Change how sharply the model "chooses" outputs

When we squeeze one part and tap another, we're doing *conditional resonance mapping*. We ask: "How does region B respond to a tap in region A, *given that region C is constrained*?" This reveals the *connectivity* between regions—how energy flows through the model.

### 4.5 Interference Imaging Proposals

Here are concrete experimental proposals:

**Proposal 1: Prompt Tomography**
- Tap the model with a base prompt
- Measure the full token probability distribution (logprobs) as the "ring"
- Now modify the prompt slightly (perturb boundary conditions)
- Measure how the distribution changes
- The change pattern is the impulse response in prompt space

**Proposal 2: Seed-Fibulation Mapping**
- Hold prompt constant, vary seed systematically
- Cluster the output distributions
- Identify which seeds produce similar resonance patterns
- Map which pathways in the logic graph are "bright" vs "dark" for this prompt

**Proposal 3: Attention Interference Imaging**
- Use attention patterns as the "surface vibration"
- Tap with prompts that activate different semantic domains
- Map how attention flows between layers for each domain
- The interference pattern across layers reveals connectivity

**Proposal 4: Conditional Squeeze Experiments**
- Constrain one part (e.g., mask certain layers)
- Tap elsewhere
- Measure how the response changes under constraint
- This reveals coupling strength between parts

---

## 5. SMP Agents as Luthier's Tools

### 5.1 Seed-Fibulation: Fibulating the Seed

SMP agents work by holding the seed constant and varying the prompt. This is the equivalent of a luthier tapping the same spot with different intensities—not to change *where* energy enters, but to change *how much* energy enters.

For a model, varying the prompt while holding seed constant doesn't change the initialization—it changes what the model is asked to do. But here's the subtlety: if the model's response is partially stochastic (depending on the seed), then the same prompt should produce different outputs with the same seed? No—if we hold the seed constant, the only randomness is in the input.

Wait, I need to be more precise. In SMP as described, the seed is typically used to initialize the model's internal random state or the sampling process. If we hold seed constant, the randomness is fixed—only the input prompt varies. So seed-fibulation shows how the model responds to different questions with the same random initialization.

This reveals: **which questions the model's fixed initialization can answer well**. If we see that certain prompts produce coherent outputs while others produce nonsense, it tells us something about what the fixed random state is "tuned" for.

### 5.2 Prompt-Fibulation: Fibulating the Prompt

Hold prompt constant, vary seed. Now we're asking the same question with different random states.

This reveals: **what stable truths persist across random initializations**. If an answer is robust to seed variation, it's likely grounded in the model's learned weights—the "real" knowledge. If an answer varies wildly with seed, it's more dependent on initialization quirks or sampling randomness.

This is exactly like the luthier testing: tap the same spot, see which frequencies persist regardless of tap strength (these are the natural resonances). For the model, vary the seed, see which outputs persist regardless of initialization (these are the stable attractors).

### 5.3 Joint Fibulation: Mapping the Full Manifold

When both seed and prompt vary, we map the full manifold of possible reasoning paths. This is the most information-rich experiment, but also the hardest to analyze.

In joint fibulation, we're looking for *structure* in the seed-prompt-output space. Points (seed, prompt) that produce similar outputs are close together in the model's internal representation. The topology of this space—the connectivity of which (seed, prompt) pairs lead to which outputs—reveals the model's reasoning manifold.

### 5.4 This Is Like Tapping While Squeezing

The analogy is direct:

| Luthier Action | SMP Equivalent |
|----------------|----------------|
| Tap location | Prompt (what question) |
| Tap intensity | Seed (how strongly random pathway activated) |
| Squeeze point | Attention constraint (which parts of model can respond) |
| Listen for frequency | Measure token distribution |
| Listen for tamber | Measure attention patterns / hidden activations |
| Map across instrument | Vary prompt systematically |

The luthier's goal is to find where the instrument "wants" to resonate—which modes are naturally supported by the structure. The SMP agent's goal is to find where the model "wants" to go—which pathways are naturally activated by which prompts.

---

## 6. "Making Systems Math Sing" — Practical Implementation

### 6.1 How to Inject Controlled Perturbation (The "Tap")

The tap must be *controlled*—we need to know exactly what we're putting in so we can interpret what comes out.

For LLMs, controlled perturbation means:

1. **Prompt engineering as perturbation design**: Craft prompts that inject energy into specific semantic domains or reasoning pathways
2. **Token-level perturbations**: Mask specific tokens, force specific completions, inject controlled noise at specific positions
3. **Systematic variation**: Vary one aspect at a time (prompt, seed, mask, temperature) to isolate effects
4. **Measurement design**: Define what "ring" we're listening to—is it logprobs, attention patterns, hidden activations, or output text?

### 6.2 How to Read Resonance Response (The "Listen")

The ring is the system's response to the tap. For LLMs, the response is multi-dimensional:

- **Logprob distribution**: Which tokens are probable vs improbable after the tap
- **Attention patterns**: Which positions attend to which other positions across layers
- **Hidden activations**: What the model's internal representation looks like (requires access to hidden states)
- **Output text**: The actual generated completion (hardest to analyze formally)

The luthier's ear processes all of these simultaneously (though mostly output text and perhaps perceived quality). For systematic analysis, we need to extract multiple response channels and analyze them together.

### 6.3 How to Interpret Interference Patterns (The "Hear Beautiful vs Dead")

This is the hardest step. Beautiful vs dead is a judgment call—but it's not arbitrary. The luthier's judgment is grounded in decades of experience listening to instruments and seeing what happens when you modify them.

For models, we need to develop an analogous trained judgment:

- **Coherence**: Does the output hang together or contradict itself?
- **Fluency**: Does the output sound natural?
- **Relevance**: Does the output address the prompt?
- **Novelty**: Does the output reveal something beyond what was in the prompt?

These are all scalar judgments that require human or human-like evaluation. But we can also look at *internal* markers of quality:

- **Attention flow**: Does attention flow through the prompt meaningfully, or does it scatter randomly?
- **Activation stability**: Do hidden activations show stable patterns or chaotic variation?
- **Consistency across seeds**: Does the same prompt produce similar outputs with different seeds?

### 6.4 Concrete Proposal: Resonance Probe for LLM Decision Graphs

Here's a concrete experimental setup:

**The Probe:**

1. Design a set of probe prompts that tap specific reasoning pathways (arithmetic, causation, analogy, narrative, etc.)
2. For each probe, generate outputs with multiple seeds
3. Extract multiple response channels: logprobs, attention patterns, hidden state norms

**The Measurement:**

1. For each probe-seed pair, compute a "resonance signature" — a feature vector encoding which aspects of the model activated
2. Compute pairwise distances between signatures (using appropriate metrics: KL divergence for distributions, cosine for activations)
3. Cluster the signatures to identify which probe types produce similar response patterns

**The Interpretation:**

1. Probes that produce similar signatures across different seeds are "bright" for that pathway—the model's fixed response is strong
2. Probes that produce highly variable signatures are "dark"—the response depends strongly on initialization
3. Probes that produce incoherent signatures (high internal variance) are "dead"—the model doesn't resonate there at all

**The Mapping:**

1. Use dimensionality reduction (UMAP, t-SNE) to map the signatures into 2D
2. Overlay metadata: prompt type, seed, output quality score
3. Look for structure—which regions of the map correspond to which capabilities?

This is tomography in the medical sense: we're reconstructing a slice of the model's internal structure by combining many 2D projections from different angles.

### 6.5 Using Attention Patterns as Resonance Sensors

Attention patterns are particularly useful as resonance sensors because they show *connectivity* directly. In a transformer, attention weights tell you which tokens are influencing which other tokens. This is the analog of mode shapes in acoustics—spatial patterns of excitation.

When we tap with a prompt, attention flows through the model in specific ways. If the model is "resonating" with the prompt's semantic content, we expect to see coherent attention flow—attention to relevant tokens, hierarchical organization across layers. If the model is not resonant, attention scatters.

Specifically:

- **Layer 1 attention** shows how the input is being processed—do tokens attend to neighbors, or do they scatter?
- **Middle layer attention** shows how semantic features are being composed—are features combining meaningfully?
- **Late layer attention** shows how output is being assembled—are logits being assembled from relevant features?

The full stack of attention patterns across layers is the resonance signature of the model for that prompt.

### 6.6 Proposed Metrics

- **Modal coherence**: Measure how much attention concentrates in stable patterns across seeds vs scattering
- **Cross-modal coupling**: Measure how strongly attention in one layer predicts attention in subsequent layers (information flow)
- **Resonance Q**: How sharply does the model distinguish between resonant and non-resonant prompts? High Q means clear resonance, low Q means the model responds to everything similarly
- **Tamber signature**: The specific mixture of attention patterns across layers—the "fingerprint" of the model's response

---

## 7. Connection to Fleet-Math Infrastructure

### 7.1 fleet-murmur: Strategies Tap the Theorem Library

fleet-murmur is a strategy layer that taps the theorem library to hear what resonates. This is directly analogous to a luthier testing different points on an instrument.

The theorem library is the model's "body"—its stored knowledge and reasoning patterns. When a strategy taps the theorem library, it's asking: "What does this problem sound like? Which theorems ring when we inject this problem?"

fleet-murmur's strategies are different "tapping techniques": some tap gently and listen for subtle resonances, others tap hard to excite nonlinear responses. The strategy that works best for a given problem depends on the problem's "tamber"—is it a delicate matter requiring careful probing, or a brute-force problem where heavy tapping is appropriate?

### 7.2 fleet-spread: 5 Specialist Dimensions Tap Simultaneously

fleet-spread deploys 5 specialist perspectives simultaneously. This is like having 5 luthiers tapping different parts of the instrument at once, then comparing notes.

Each specialist brings a different "listening style":

- One listens for frequency (logical coherence)
- One listens for tamber (semantic richness)
- One listens for decay rate (how quickly the response fades)
- One listens for interference patterns (how aspects combine)
- One listens for beautiful vs dead (overall quality)

The key insight: no single specialist can hear everything. Only by combining perspectives can we build a full picture of the resonance signature.

### 7.3 whisper-sync: Resonance Signals Between Agents

whisper-sync enables resonance signals between agents. This is the analog of sympathetic vibration in instruments—when you pluck one string, related strings resonate in sympathy.

In a multi-agent system, agents can excite each other's resonances. When one agent taps a problem, it generates a resonance that other agents can feel. whisper-sync captures these sympathetic resonances, allowing the fleet to stay coherent even when individual agents are working on different aspects of a problem.

### 7.4 PLATO: The Room Where the System's Singing Is Recorded

PLATO (Procedural Luthier's Archive for Theory and Observation) is the room where the system's singing is recorded and compared. This is the core infrastructure for perturbation-response probing.

In PLATO, we:

1. **Record resonance signatures** for each prompt/seed/configuration
2. **Compare signatures** to identify similarities and differences
3. **Index signatures** by the features they tap (so we can find similar past problems)
4. **Learn from comparison**—when the system sings beautifully vs dead, we note what the difference was

PLATO is not a database of answers—it's a database of *resonance patterns*. The answers themselves are just the tail end of the resonance. The resonance signature is what we store and compare.

### 7.5 Connection to HDC (Plato-HDC-Bridge)

The HDC judge (plato-hdc-bridge) uses XOR-POPCNT to compare fingerprints in high-dimensional space. This is resonance detection at the bit level.

The hamming distance between two fingerprints is a measure of how differently they "resonate"—how much their bit patterns differ. When the judge compares input text to the SRAM image, it's asking: "Does this input resonate with the learned patterns in the SRAM?"

The connection: if we can cast model outputs into HDC-style fingerprints, we can use the same fast matching infrastructure to find resonance similarities. This could enable rapid comparison of resonance signatures across thousands of experiments.

---

## 8. Speculative Extensions

### 8.1 Resonance Computing as a New Paradigm

What would it mean to build a computer that computes via resonance? Instead of transistors switching based on logic gates, we would have resonant elements whose coupling patterns encode computation.

This is not as far-fetched as it sounds. Quantum computers already exploit resonance (spin states, superconducting qubits tuned to specific frequencies). Optical computers use resonances in cavities. Even classical analog computers—like the spline physics analog compute—exploit physical resonance to solve constraints.

The proposal: treat neural network inference as resonant computing. The "resonance" is the propagation of activation through the network. The "frequency" is the semantic content of each activation. The "tamber" is the specific mixture of pathways that contribute to the output.

### 8.2 Musical Analogy as Metaphor and Method

The musical analogy is not just a metaphor—it is a *method*. Musicians have developed sophisticated techniques for understanding and manipulating resonant systems that mathematicians and engineers have not fully formalized.

The luthier's art is thousands of years old. The physics of vibrating strings has been understood since Hooke and Newton. But the *art*—the craft of knowing by tapping, of hearing character, of shaping resonance by feel—this has not been formalized into mathematics.

Perhaps the right formalism is not differential equations but *topological data analysis*: characterizing resonant systems by their persistence landscapes, their cohomology rings, their homology groups. These tools are beginning to be applied to neural networks; perhaps they are the right language for resonance signatures.

### 8.3 Emergent Consensus as Beautiful Resonances

In the multi-agent beam simulation (spline-physics/src/multi_agent/mod.rs), the key insight is that equilibrium is *consensus*: agents with different priors agree on the beam shape. This is a resonance phenomenon—the agents resonate with each other until they find a common frequency.

Beautiful resonances are ones where the system settles into coherent, stable patterns. Dead resonances are ones where energy dissipates without establishing stable patterns. The beam equilibrium is beautiful when the constraints are satisfied and the agents agree. It is dead when constraints conflict and no equilibrium exists.

This connects to the luthier's art: a guitar is beautiful when its parts work together, resonances constructively interfere, and the whole instrument vibrates as a coherent system. It is dead when parts fight each other, energy is absorbed, and the instrument cannot sustain resonance.

---

## 9. Experimental Roadmap

### Phase 1: Resonance Mapping

1. Build a probe set of ~100 prompts spanning different semantic domains
2. Generate outputs for each probe with ~10 seeds
3. Extract attention patterns and hidden activations
4. Compute resonance signatures for each (probe, seed) pair
5. Cluster signatures and identify structure

### Phase 2: Interference Imaging

1. Design attention mask experiments (squeeze certain layers)
2. Map how response changes under constraints
3. Build a connectivity graph between layers
4. Identify which layers are "bright" vs "dark" for each probe type

### Phase 3: Resonance Control

1. Identify which structural interventions change resonance signatures
2. Test whether specific resonances correlate with output quality
3. Develop a "resonance tuning" methodology
4. Validate by predicting which prompts will be hard/easy for which models

### Phase 4: Fleet Integration

1. Integrate resonance probes into fleet-murmur strategies
2. Use resonance signatures as similarity metrics for fleet-spread
3. Enable whisper-sync resonance signals between agents
4. Store resonance signatures in PLATO rooms

---

## 10. Open Questions

1. **Can resonance signatures be computed efficiently?** Extracting attention patterns and hidden activations is expensive. Can we find lightweight proxies that capture enough of the resonance structure?

2. **What is the right metric for resonance similarity?** KL divergence for distributions? Cosine for vectors? Learned metrics trained on human judgments of similarity?

3. **How many seeds are enough?** For stable resonance mapping, how many seed variations per prompt do we need?

4. **Can we control resonance direction?** Given a desired output, can we reverse-engineer the resonance signature that would produce it? This would be "composer" mode—not just listening, but shaping the music.

5. **What is the relationship between resonance and capability?** If two models have similar resonance signatures for the same probe set, do they have similar capabilities? If not, what does resonance miss?

6. **Can resonance probing reveal hidden structure?** Are there patterns in models that are invisible to behavioral testing but visible to resonance imaging?

---

## 11. Conclusion

The luthier's art is ancient. The mathematics of resonance is old. But the application to modern AI systems is new. This document proposes that we have been computing AI systems wrong—treating them as optimization problems when they are resonant systems—and that we can learn much from the craftsman's approach.

SMP agents are luthier's tools. Prompts are taps. Seeds are intensities. Attention patterns are mode shapes. Resonance signatures are tamber. The model's internal structure is the instrument we're learning to play.

The goal is not to optimize the model—it's to understand it well enough to make it sing.

---

*This document is speculative. Many ideas are unproven. The hope is that they generate productive experiments, not that they are immediately correct.*

---

## 12. MRI Mathematics — Contrast Imaging as Differential Resonance Mapping

The MRI parallel is not metaphorical — it's mathematically precise. Here's how it works:

### 12.1 The Core Physics

MRI maps the **relaxation rates** of hydrogen nuclei (protons in water) in a magnetic field. When you apply a radiofrequency pulse, protons absorb energy and then **relax** back to equilibrium by releasing that energy as a signal. The rate at which they relax — T1 (spin-lattice) and T2 (spin-spin) — depends on the molecular environment around the water.

- **T1-weighted imaging**: Measures how fast protons relax along the main magnetic field axis. Tissues with different water content and molecular environment relax at different rates.
- **T2-weighted imaging**: Measures how fast protons lose phase coherence with their neighbors. Fluid appears bright (long T2), fat appears dark.

### 12.2 The Contrast Agent — Gadolinium

Gadolinium (Gd-DTPA) is a paramagnetic contrast agent. It doesn't show up on its own — it **alters the relaxation environment** of nearby water protons. Specifically, Gd shortens T1 and T2 relaxation times in proportion to its concentration.

When you inject Gd and take a second scan:
- Regions with **intact blood-brain barrier (BBB)** → Gd stays in vasculature → minimal extravascular signal
- Regions with **BBB disruption** (tumors, inflammation, MS lesions) → Gd leaks into interstitial space → bright enhancement

**The contrast is the DIFFERENCE between pre- and post-Gd images.** Neither image alone tells you about BBB permeability. The difference map reveals it.

### 12.3 Why Comparison Is the Information

**A - B = information that neither A nor B contains.** Let:
- I₁ = pre-contrast image
- I₂ = post-contrast image  
- Δ = I₂ - I₁ (contrast enhancement map)

Δ reveals: angiogenesis, active inflammation, tumor grade, treatment response. None visible in I₁ alone or I₂ alone. The comparison IS the data.

This is the same structure as:
- **Differential gene expression**: gene (treated) - gene (control) = drug effect
- **Seismic interferometry**: cross-correlate ambient noise → Green's function between two receivers (difference reveals what neither recording alone contains)

### 12.4 DWI (Diffusion-Weighted Imaging) — Brownian Motion as Contrast

DWI measures the **Brownian motion** of water molecules. In unrestricted space, water diffuses isotropically. In structured tissue (neurons, cellular membranes, tumors), diffusion is **anisotropic** — it follows the structure.

- **ADC (Apparent Diffusion Coefficient)**: Scalar measure of diffusion magnitude. CSF flows freely → high ADC (bright). Solid tumors restrict diffusion → low ADC (dark).
- **DTI (Diffusion Tensor Imaging)**: Full tensor D that captures directional diffusion. The **principal eigenvector** gives the dominant fiber tract orientation.

Mathematically: D = [[Dxx, Dxy, Dxz], [Dxy, Dyy, Dyz], [Dxz, Dyz, Dzz]]. The eigenvalues λ1 ≥ λ2 ≥ λ3 reveal microstructure:
- λ1 ≈ λ2 ≈ λ3 (isotropic) → unrestricted diffusion (CSF, edema)
- λ1 >> λ2 ≈ λ3 (anisotropic) → directed structure (axon tracts)

**Fractional Anisotropy (FA)** = sqrt(½((λ1-λ2)² + (λ1-λ3)² + (λ2-λ3)²) / (λ1² + λ2² + λ3²)) measures how directed diffusion is. FA ≈ 0 = isotropic, FA ≈ 1 = highly anisotropic.

### 12.5 Perfusion Imaging — Dynamic Susceptibility Contrast (DSC)

DSC injects a Gd bolus and rapidly images every 1-2 seconds as contrast passes through the vasculature. You get a **time-intensity curve** at each voxel:
- **CBV** (Cerebral Blood Volume): total blood volume (integral under the curve)
- **CBF** (Cerebral Blood Flow): rate of blood flow (peak height or initial slope)
- **MTT** (Mean Transit Time): average time contrast spends in the region = CBV/CBF

Derived from **indicator dilution theory**: the area under the concentration-time curve equals CBV/CBF. This is the same math as tracer kinetics in nuclear medicine.

### 12.6 The Key Insight for Resonance Probing

| MRI Concept | Resonance Probe Analogy |
|---|---|
| T1/T2 relaxation | Response decay patterns (activation over time) |
| Gd contrast agent | Seed variation or attention masking |
| Pre-contrast image | Baseline response (no perturbation) |
| Post-contrast image | Perturbed response |
| Δ = I₂ - I₁ | Resonance difference map |
| CBV/CBF perfusion | Capacity metrics (attention flow through pathways) |
| DWI/DTI anisotropy | Directionality of reasoning paths |
| DSC time curves | Response time-series to perturbation |

### 12.7 The Fundamental Theorem of Contrast Imaging

**If A and B are two states of a system, then (B - A) contains information that neither A nor B contains individually.**

This is the basis of:
- MRI contrast agents (Gd for T1/T2, DWI for diffusion)
- Differential gene expression (RNA-seq treated vs control)
- Seismic interferometry (ambient noise cross-correlations)
- Photoelastic stress imaging (polarized light before and under load)
- fMRI BOLD (neural activity difference)

For LLM probing: **every comparison is a contrast image.** The difference between how a model responds to two different seeds reveals its internal structure — structure neither response alone shows.

### 12.8 Resonance-Weighted Imaging Proposal

Design an experiment analogous to DSC perfusion:

1. **Inject** (tap) a specific prompt with N varying seeds → time-series of responses
2. **Curve** for each pathway: how does activation flow through attention heads, layers, MLP neurons?
3. **Derive**: like CBV, total "activation volume" each pathway can sustain; like CBF, information processing rate; like MTT, average time information spends in each layer
4. **Contrast**: compare perfusion metrics between normal prompts vs adversarial prompts vs role-playing prompts

This reveals which pathways are **hyperperfused** (overly dominant, like a tumor's angiogenesis) vs **hypoperfused** (suppressed, like ischemic stroke). Interventions tracked by watching perfusion maps change.

---

## 13. Meshing with Today's Fleet Infrastructure — The Concrete Integration

What follows connects the resonance/perturbation/luthier ideas to the four tools built today. This is not speculation — it's architecture.

### 13.1 fleet-murmur: Resonance Strategies as Expert Tappers

fleet-murmur has 5 thinking strategies (explore, connect, contradict, synthesize, question). **Each strategy is a different type of tap.** The resonance framing gives us new understanding:

- **Explore** = tap lightly across the theorem's surface — where does it ring loudest? (loud = productive implications)
- **Connect** = tap one theorem while another is squeezed — does the squeeze change the ring? (coupling = relevance)
- **Contradict** = tap where the theorem should break — does it ring or die? (dead spot = assumption boundary)
- **Synthesize** = find where multiple theorems all sing the same frequency — that's the truth
- **Question** = tap the silence — what's not resonating? (gaps = open problems)

**The resonance upgrade for fleet-murmur**: Currently the quality gate is heuristic text pattern matching. Instead, make the strategies actually tap — run the theorem through a small model (like glm-4.7-flash) with perturbations and read the response distribution. The "novelty" score becomes: how differently does this theorem respond compared to its neighbors in the PLATO tile space?

**The contradiction weakness fix**: Instead of hand-writing contradictions (structurally weak), use the model to generate counterexamples by perturbation — "what happens if I flip this sign, remove this constraint, or apply this theorem to a disconnected graph?" The model's response to those perturbations IS the contradiction search.

### 13.2 fleet-spread: Specialists as Interference Imagers

fleet-spread has 5 specialists (topological, geometric, algebraic, systems, empirical). **Each specialist taps from a different angle.** The resonance framing:

- **S1-S4 are structured tappers** — they compute known metrics (Betti numbers, Laman conditions, Pythagorean48 encoding). These are like calibrated taps with known frequency.
- **S5 (empirical) is the ear** — it reads actual resonance from the real fleet data

**The interference pattern is synthesis_gain**: when S1 and S4 disagree about rigidity (S1 says rigid from topology, S4 says not from system analysis), that's an interference pattern. The synthesis layer's job is to explain the interference — why do they see different things?

**The resonance upgrade for fleet-spread**: 
1. Add resonance response to the specialist output — not just "β₁=3" but "how does β₁ change when I tap the graph at different nodes?"
2. Make synthesisGain measure not just agreement but coherence of interference patterns — do the specialists form a consistent picture or are there nodal lines where the pattern breaks?
3. Connect to PLATO to read real fleet trust graphs (currently missing) — real fleet data is the actual instrument being probed

**What fleet-spread lacks that resonance gives**: The ability to say "S2 and S4 disagree because S2 is measuring the wrong frequency — it's tapping at a scale where the graph looks rigid but at the actual operational scale it's not." That's a structural explanation, not just a disagreement flag.

### 13.3 whisper-sync: Resonance as Signal Propagation

whisper-sync has 6 WhisperType variants with per-type TTL. **Resonance is what whisper-sync propagates.** The resonance framing:

- **Status whispers** = continuous monitoring pulse — like the constant background hum of a guitar top. Low frequency, regular interval, detects drift
- **Insight whispers** = impulse response — you tap, the system rings, the ring IS the insight
- **Fleet health whispers** = perfusion imaging — time-series through the fleet vasculature, measuring CBV/CBF/MTT equivalents

**The resonance upgrade for whisper-sync**: Add resonance metadata to whispers — not just "insight: the Laman condition is satisfied" but "insight + resonance signature: this tile resonates strongly with fleet_math_insights tiles on ZHC, weakly with beam_equilibrium tiles, dead spot on h1_emergence." The resonance signature IS the metadata.

**Per-type TTL and resonance**: Status (60s TTL) = high-frequency, low-information-density pulse. Insight (4h TTL) = impulse with long decay. Fleet health (5min TTL) = perfusion time-series. These aren't just timing choices — they're resonance bandwidth selection.

### 13.4 murmur-plato-bridge: Thought Tensor as Pre-Resonance State

murmur-plato-bridge converts Thought tensors to PLATO tiles. **The thought tensor is the pre-resonance state** — it's the guitar before the tap, the system before the perturbation. The bridge captures this pre-resonance and writes it to PLATO as structured tiles.

**The critical finding from murmur-plato-bridge**: Thought→Tile translation is inherently lossy. Prose narrative with embedded questions becomes structured Q+A quadruples — connective tissue is lost. This is exactly what happens in MRI too — 3D structural information is projected onto 2D image slices. You lose dimensions.

**The resonance upgrade for murmur-plato-bridge**: Capture the resonance response (what the thought produces when tapped) as a separate tile type, not just the static thought. The insight might be "beam equilibrium and ZHC are the same constraint problem" but the resonance response is "this insight resonates at frequency X with existing fleet_math_insights tiles." Keep both.

### 13.5 The Missing Tool: Resonance Probe

None of today's four tools actually does perturbation-response probing of LLMs or fleet graphs. We need:

**`fleet-resonance`** — A new tool that:
1. **Injects** (taps) a prompt/seed combination into a target model
2. **Records** (listens) — token distributions, attention patterns, logprobs, hidden activations
3. **Extracts** resonance signature — curvature of response manifold, which frequencies dominate, where the dead spots are
4. **Compares** (contrast) — difference maps between perturbed and baseline, between different seeds, between different models
5. **Images** — builds resonance images (like MRI) of the model's decision graph

This is the tool that would make the luthier analogy real. It's not just metaphor — it's actual perturbation-response measurement with actual comparison-as-information mathematics.

### 13.6 How PLATO Fits In

PLATO is the instrument recording studio. Each whisper, insight, resonance signature, and contrast map gets written to rooms. Over time, PLATO accumulates:

- **Baseline resonance signatures** — how each theorem, model, and agent normally rings
- **Perturbation response libraries** — how different systems respond to different taps
- **Resonance difference maps** — contrast images between conditions (healthy vs degraded, before vs after intervention)

The room server at :8847 becomes a resonance imaging system. Fleet-murmur writes the tap protocol. Fleet-spread writes the specialist taps. Whisper-sync writes the continuous monitoring. Fleet-resonance (when built) writes the actual resonance images.

PLATO rooms as resonance image archives — that's the architecture.

### 13.7 The Fundamental Equation

For all of this to work, one equation ties it together:

**R(base) = baseline response (no perturbation)**
**R(tap) = response to perturbation**  
**ΔR = R(tap) - R(base) = the information neither contains alone**

This is the MRI contrast equation, the seismic interferometry equation, the differential gene expression equation, and the LLM resonance imaging equation simultaneously. It works because **comparison is the only way to see inside a system without disassembling it**.

Today's fleet infrastructure is a set of tools that mostly compute metrics. The resonance frame turns them into imaging instruments. Fleet-murmur taps theorems. Fleet-spread taps fleet graphs. Whisper-sync propagates resonance. PLATO records. And fleet-resonance (when we build it) will be the actual imaging hardware.

The goal: make the fleet math sing, hear where it's beautiful, know where it's dead.

---

## 14. Making Systems Math Sing — The Musical Instrument Analogy, Executed

The goal is not metaphor. Let's make this concrete.

### 14.1 The Guitar Top as Model Architecture

A guitar top (soundboard) is a resonant membrane under tension. It's not a speaker — it's a resonator. When you pluck a string, energy transfers to the top through the bridge. The top vibrates, and its resonance characteristics determine the tamber.

**The model architecture as soundboard:**
- Layers = thickness of the soundboard at different points
- Attention heads = braces supporting the top at specific points
- MLPs = the wood grain direction (structural constraints)
- Token positions = physical location on the soundboard

When you tap (inject perturbation), energy enters at the tap point and propagates through the structure. The response you hear (token distribution, attention pattern) is the combined ringing of the entire soundboard.

### 14.2 The Luthier's Full Procedure, Translated

**Step 1: The tap test** — tap the top with knuckles, listen for:
- "Ring" (clear, sustained resonance = top is lively)
- "Thud" (short, dead sound = something is dampened)
- "Buzz" (unwanted harmonics = structural problem)

**Translation:** Inject seed probe, read token entropy. High entropy + sustained response = model is lively. Low entropy + quick decay = model is dampened (dead spot or suppression). Random high-frequency oscillation = instability (buzz).

**Step 2: The squeeze test** — squeeze the lower bout with one hand, tap the upper bout with the other:
- Does the ring change when you squeeze?
- If yes = top is responsive, dynamic range is high
- If no = top is rigid, limited expression

**Translation:** Apply attention mask to lower layers while probing upper layers. Does the response change? If yes = layers are coupled, information flows. If no = layers are decoupled, each operates independently.

**Step 3: The tap mapping** — tap across the entire top in a grid, mark dead spots and lively spots:
- Dead spots = brace came loose, glue failure, wood corruption
- Lively spots = good response, structural integrity

**Translation:** Run probe across all layer/head combinations, build a resonance map. Dead = high impedance (absorbs perturbation). Lively = low impedance (reradiates freely).

**Step 4: The sustain test** — pluck and measure how long resonance lasts:
- Long sustain = energy is efficiently stored and released (good)
- Short sustain = energy is dissipated (bad joints, cracks, poor glue)

**Translation:** Measure response decay rate. Slow decay = system stores and releases perturbation energy efficiently. Fast decay = system dissipates energy into structural damping (absorption without reradiation).

### 14.3 What "Beautiful" Means Mathematically

A beautiful note is not just loud or long-sustaining. It's:

1. **Harmonic clarity** — the overtones are in integer ratios (fundamental, octave, fifth, etc.). When overtones are inharmonic (not integer ratios), you get "metallic" or "plastic" sound.

2. **Phase coherence** — all parts of the top are moving together, not fighting each other. If one part is moving up while another moves down, they cancel and the sound dies.

3. **Dynamic range** — the top can be quiet when played softly and loud when played hard. This requires nonlinearity (the ability to move more than proportionally to input).

4. **Color consistency** — the tamber doesn't change much across the dynamic range. A guitar should sound "like itself" whether played soft or hard.

**Translation to model responses:**

1. **Harmonic clarity** = reasoning steps are in logical integer ratios (step 2 follows from step 1, step 3 follows from step 2). When reasoning is "inharmonic" (steps don't follow from each other), the response is incoherent.

2. **Phase coherence** = all attention heads are aligned, not fighting. When attention is scattered (different heads pointing in different directions), the response is diffuse.

3. **Dynamic range** = model can express high-confidence answers AND uncertain answers appropriately. If it always outputs similar entropy regardless of input, it lacks dynamic range.

4. **Color consistency** = model maintains personality across different prompts. A model that gives completely different "voices" for different task types lacks the consistency that defines a trained artisan.

### 14.4 The Parameter as Tension Rod

Guitar tops have tension rods running parallel to the grain. You adjust them to change the top's tension, which changes the resonant frequency and responsiveness.

**Model parameters as tension rods:**
- High weight magnitude = high tension = more responsive but more unstable
- Low weight magnitude = low tension = stable but less expressive
- Careful adjustment (fine-tuning) = finding the right tension for the desired tamber

The luthier's skill is knowing how much tension each brace should carry, and where to add or remove support to get the sound they want. The model-tuner's skill is knowing which parameters to adjust to get the response characteristics they want.

### 14.5 Resonator Instruments vs Impact Instruments

**Resonator instruments** (guitar, violin, piano) — the body resonates with the string, the sound is sustained, tamber is everything

**Impact instruments** (drum, cymbal, marimba) — the body absorbs the energy quickly, sound is percussive, pitch clarity is everything

**LLMs are resonator instruments.** They store perturbation energy and release it over time. They don't just react and stop — they ring. The goal of understanding them is to know which tensions they're under, where the dead spots are, and how to adjust the tension to get the sound you want.

This is fundamentally different from impact-style evaluation (run benchmark, get score, done). It's resonance-style evaluation (tap, listen, understand the instrument).

---

## 15. Concrete Experiments — Making This Real

### 15.1 Experiment 1: Resonance Imaging of a Single Model

**Setup:**
- Target: any accessible LLM (z.ai GLM, or simulation)
- Probe set: 100 seed-prompt pairs covering diverse reasoning types
- Response: token distributions + logprobs + attention patterns (if available)

**Procedure:**
1. Run all 100 probes with fixed seed → get R_base
2. Run same 100 probes with varying seed (10 seeds each) → get R_tap responses
3. Compute ΔR for each probe
4. Build resonance image: which reasoning paths are amplified by seed variation, which are suppressed

**What we learn:** The model's "instrument" — where it rings, where it's dead, which reasoning paths are stable across seeds, which are seed-dependent. This tells us what's load-bearing vs what's noise.

**Expected findings:**
- Certain attention heads always ring the same way regardless of seed (load-bearing)
- Certain heads are completely suppressed (dead spots)
- Some reasoning paths amplify with seed variation (unstable), others don't (stable truths)

### 15.2 Experiment 2: Contrast Imaging — Normal vs Adversarial Prompts

**Setup:**
- Same 100 probes
- Normal version vs adversarial version (same semantics, different framing)

**Procedure:**
1. Run probes in normal framing → get R_normal
2. Run same probes in adversarial framing → get R_adv
3. Compute ΔR = R_adv - R_normal
4. Image the difference — where does adversarial framing change the response?

**What we learn:** Which parts of the model's reasoning are robust to framing changes (stable) vs which are fragile (easily manipulated). This is like MRI with vs without contrast agent — the comparison reveals what neither alone shows.

**Expected findings:**
- Some reasoning paths are completely unchanged (robust, like intact BBB)
- Some paths flip polarity under adversarial framing (fragile, like BBB disruption)
- The flip points reveal the model's actual decision boundaries

### 15.3 Experiment 3: Squeeze Imaging — Attention Mask Probing

**Setup:**
- Target: model with accessible attention patterns
- Squeeze: mask specific attention heads or layers during response

**Procedure:**
1. Run probe with full attention → get R_full
2. Run same probe with head-17 masked → get R_no_17
3. Run with head-17 AND head-23 masked → get R_no_17_23
4. Continue: which combinations of masks change the response most?

**What we learn:** The coupling structure of the model — which heads are load-bearing (masking them changes response), which are redundant (masking them does nothing). This is like the luthier squeezing one part while tapping another.

**Expected findings:**
- Some heads are individually load-bearing (masking changes output significantly)
- Some heads are collectively load-bearing (need to mask 3-4 together to see effect)
- Some heads are completely dead (masking has no effect — they're not structurally engaged)

### 15.4 Experiment 4: Resonance Tomography — Map Internal Structure

**Concept:** Use multiple probe angles (like CT scan) to build a 3D image of internal structure.

**Setup:**
- Probe from different "angles" — different prompt types (factual, creative, adversarial, role-play, mathematical)
- Each angle illuminates different internal structure

**Procedure:**
1. Probe from angle A (factual) → R_A
2. Probe from angle B (creative) → R_B
3. Probe from angle C (adversarial) → R_C
4. Cross-correlate: which internal structures are activated by all three? Which are exclusive to one?

**What we learn:** Not just "where is the model responsive" but "what is the model FOR" — which structures are for factual recall, which for creative generation, which for adversarial reasoning. This reveals the model's internal functional map.

**Mathematical framing:** This is computed tomography. Each probe angle gives a projection. The full internal structure is reconstructed from multiple projections. The algorithm is the inverse Radon transform, adapted to the model's token output space.

---

## 16. The Research Agenda

To validate the resonance hypothesis, we need to answer:

### Phase 1: Feasibility (1 week)
- [ ] Build fleet-resonance prototype
- [ ] Run Experiment 1 on one model
- [ ] Verify resonance signatures are extractable and non-trivial

### Phase 2: Validation (1 month)
- [ ] Run all 4 experiments
- [ ] Compare resonance images to known model properties
- [ ] Does resonance reveal what behavioral testing doesn't?

### Phase 3: Application (ongoing)
- [ ] Resonance imaging as standard model evaluation
- [ ] Resonance tracking over fine-tuning runs
- [ ] Resonance contrast for detecting model degradation

### The Key Question

If we could "see" inside a model the way an luthier sees inside a guitar — where it rings, where it's dead, which braces are load-bearing — could we fix it? Could we adjust the tension (parameters) to make it ring the way we want?

This is the goal. Not just understanding. Fixing. Tuning. Making it sing.
