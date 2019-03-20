//! # ODEs Solvers
//! `ode-solvers` is a collection of numerical methods to solve ordinary differential equations (ODEs).

// Re-export from external crate
use nalgebra as na;
pub use crate::na::{Vector1, Vector2, Vector3, Vector4, Vector5, Vector6, VectorN};

// Declare modules
pub mod butcher_tableau;
pub mod controller;
pub mod dop853;
pub mod dopri5;
pub mod dop_shared;

pub use dopri5::Dopri5;
pub use dop853::Dop853;

pub use dop_shared::System;
