//! Empirical validation for fleet mathematics claims
//! 
//! Run with: cargo test --test validation -- --nocapture
//! 
//! These experiments validate the core claims from the fleet mathematics
//! Field Report. Each experiment has a corresponding document in:
//! flux-research/roadmaps/EXPERIMENT-*.md

pub mod h1_emergence;
pub mod consensus_benchmark;