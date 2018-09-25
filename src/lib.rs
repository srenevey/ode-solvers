//! # ODEs Solvers
//! `ode-solvers` is a collection of numerical methods to solve ordinary differential equations (ODEs).

extern crate alga;
extern crate nalgebra as na;

// Re-export from external crate
pub use na::{Vector1, Vector2, Vector3, Vector4, Vector5, Vector6, VectorN};

// Declare modules
pub mod butcher_tableau;
pub mod dopri5;
pub mod dop853;
pub mod controller;