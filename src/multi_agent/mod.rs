//! Multi-agent beam simulation — constraint satisfaction as consensus
//! 
//! Key insight: a beam's equilibrium shape is what a set of agents
//! with different priors and local measurements ALL agree on.
//! 
//! Agents:
//!   - Architect: sets boundary conditions (pin endpoints, target rise)
//!   - SpilingAgent: propagates constraints from known frame positions
//!   - EnergyAgent: minimizes bending energy via gradient descent
//!   - ShootingAgent: cross-validates via Euler elastica ODE
//!   - QualityAgent: applies GD&T tolerance constraints (skeptic)
//!   - Arbiter: detects consensus or declares disagreement
//! 
//! The trust topology determines convergence:
//!   EnergyAgent <-> ShootingAgent: cross-validation (mutual trust)
//!   SpilingAgent -> Architect: trusts boundary conditions
//!   QualityAgent -> everyone: discounts (skeptic)

pub mod agent;
pub mod consensus;
pub mod debate;

pub use agent::{Agent, AgentRole};
pub use consensus::{check_consensus, measure_agreement, ConsensusResult, ConsensusState};
pub use debate::{BeamDebate, DebateRound};
