//! # ODEs Solvers
//! `ode-solvers` is a collection of numerical methods to solve ordinary differential equations (ODEs).

// Re-export from external crate
pub use crate::na::{
    DVector, OVector, SVector, Vector1, Vector2, Vector3, Vector4, Vector5, Vector6,
};
use nalgebra as na;

// Declare modules
pub mod butcher_tableau;
pub mod controller;
pub mod dop853;
pub mod dop_shared;
pub mod dopri5;
pub mod rk4;

pub use dop853::Dop853;
pub use dopri5::Dopri5;
pub use rk4::Rk4;

pub use dop_shared::System;
