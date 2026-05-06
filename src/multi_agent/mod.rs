//! Multi-agent beam simulation — constraint satisfaction as consensus
//! 
//! Key insight: a beam's equilibrium shape is what a set of agents
//! with different priors and local measurements ALL agree on.

pub mod agent;
pub mod consensus;
pub mod debate;
pub mod segment;

pub use agent::{Agent, AgentRole};
pub use consensus::{check_consensus, measure_agreement, ConsensusResult, ConsensusState};
pub use debate::{BeamDebate, DebateRound};
pub use segment::{BeamDebateN, ConsensusResultN, PinAgent, SegmentAgent};
