//! Shared traits and structures for dopri5 and dop853.

use std::fmt;
use thiserror::Error;

/// Trait needed to be implemented by the user.
pub trait System<V> {
    /// System of ordinary differential equations.
    fn system(&self, x: f64, y: &V, dy: &mut V);
    /// Stop function called at every successful integration step. The integration is stopped when this function returns true.
    fn solout(&mut self, _x: f64, _y: &V, _dy: &V) -> bool {
        false
    }
}

/// Enumeration of the types of the integration output.
#[derive(PartialEq, Eq)]
pub enum OutputType {
    Dense,
    Sparse,
}

/// Enumeration of the errors that may arise during integration.
#[derive(Debug, Error)]
pub enum IntegrationError {
    #[error("Stopped at x = {x}. Need more than {n_step} steps.")]
    MaxNumStepReached { x: f64, n_step: u32 },
    #[error("Stopped at x = {x}. Step size underflow.")]
    StepSizeUnderflow { x: f64 },
    #[error("The problem seems to become stiff at x = {x}.")]
    StiffnessDetected { x: f64 },
}

/// Contains some statistics of the integration.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub num_eval: u32,
    pub accepted_steps: u32,
    pub rejected_steps: u32,
}

impl Stats {
    pub(crate) fn new() -> Stats {
        Stats {
            num_eval: 0,
            accepted_steps: 0,
            rejected_steps: 0,
        }
    }

    /// Prints some statistics related to the integration process.
    #[deprecated(since = "0.2.0", note = "Use std::fmt::Display instead")]
    pub fn print(&self) {
        println!("{}", self);
    }
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Number of function evaluations: {}", self.num_eval)?;
        writeln!(f, "Number of accepted steps: {}", self.accepted_steps)?;
        write!(f, "Number of rejected steps: {}", self.rejected_steps)
    }
}
