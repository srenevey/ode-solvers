//! Shared traits and structures for dopri5 and dop853.

use nalgebra::Scalar;
use num_traits::{Float, FromPrimitive, NumCast, One, Zero};
use simba::scalar::{
    ClosedAdd, ClosedAddAssign, ClosedDiv, ClosedDivAssign, ClosedMul, ClosedMulAssign, ClosedNeg,
    ClosedSub, ClosedSubAssign, SubsetOf,
};
use std::fmt;
use thiserror::Error;

/// Trait needed to be implemented by the user
///
/// The type parameter T should be either `f32` or `f64`, the trait [FloatNumber] is used
/// internally to allow generic code.
///
/// The type parameter V is a state vector. To have an easy start it is recommended to use [nalgebra] vectors.
///
/// ```
/// use ode_solvers::{System, SVector, Vector3};
///
/// // A predefined type for a vector (works from 1..6)
/// type Precision = f64;
/// type State = Vector3<Precision>;
/// type MySystem = dyn System<Precision, State>;
///
/// // Definition of a higher dimensional vector using nalgebra
/// type AltState = SVector<Precision, 9>;
/// type MyAltSystem = dyn System<Precision, State>;
/// ```
pub trait System<T, V>
where
    T: FloatNumber,
{
    /// System of ordinary differential equations.
    fn system(&self, x: T, y: &V, dy: &mut V);

    /// Stop function called at every successful integration step. The integration is stopped when this function returns true.
    fn solout(&mut self, _x: T, _y: &V, _dy: &V) -> bool {
        false
    }
}

/// A struct that holds the result of a solver/stepper run
#[derive(Debug, Clone)]
pub struct SolverResult<T, V>(Vec<T>, Vec<V>);

/// This trait combines several traits that are useful
/// when writing generic code that shall work in f32 and f64
///
/// It is only implemented for f32 and f64 yet.
pub trait FloatNumber:
    Copy
    + Float
    + NumCast
    + FromPrimitive
    + SubsetOf<f64>
    + Scalar
    + ClosedAdd
    + ClosedMul
    + ClosedDiv
    + ClosedSub
    + ClosedNeg
    + ClosedAddAssign
    + ClosedMulAssign
    + ClosedDivAssign
    + ClosedSubAssign
    + Zero
    + One
{
}

/// Implementation of the SolverNumFloat trait for f32
impl FloatNumber for f32 {}

/// Implementation of the SolverNumFloat trait for f64
impl FloatNumber for f64 {}

impl<T, V> SolverResult<T, V> {
    pub fn new(x: Vec<T>, y: Vec<V>) -> Self {
        SolverResult(x, y)
    }

    pub fn with_capacity(n: usize) -> Self {
        SolverResult(Vec::with_capacity(n), Vec::with_capacity(n))
    }

    pub fn push(&mut self, x: T, y: V) {
        self.0.push(x);
        self.1.push(y);
    }

    pub fn append(&mut self, mut other: SolverResult<T, V>) {
        self.0.append(&mut other.0);
        self.1.append(&mut other.1);
    }

    /// Returns a pair that contains references to the internal vectors
    pub fn get(&self) -> (&Vec<T>, &Vec<V>) {
        (&self.0, &self.1)
    }
}

/// default implementation starts with empty vectors for x and y
impl<T, V> Default for SolverResult<T, V> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

/// Enumeration of the types of the integration output.
#[derive(PartialEq, Eq)]
pub enum OutputType {
    Dense,
    Sparse,
    Continuous,
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
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Number of function evaluations: {}", self.num_eval)?;
        writeln!(f, "Number of accepted steps: {}", self.accepted_steps)?;
        write!(f, "Number of rejected steps: {}", self.rejected_steps)
    }
}

pub(crate) fn sign<T: FloatNumber>(a: T, b: T) -> T {
    if b > T::zero() {
        a.abs()
    } else {
        -a.abs()
    }
}
