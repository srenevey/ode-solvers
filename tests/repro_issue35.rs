//! Regression test for issue #35: Dop solvers not returning up to x_end.
//!
//! The dense-output grid is built by repeated `xd += dx`, so floating-point
//! round-off accumulates (~1e-7 for f32). The endpoint check used a hard-coded
//! absolute tolerance of 1e-9, which is smaller than the accumulated error, so
//! the final `x_end` point was silently dropped. See
//! https://github.com/srenevey/ode-solvers/issues/35

use nalgebra::Vector1;
use ode_solvers::{Dop853, Dopri5, System};

/// Reporter's minimum breaking example system: dy/dx = -y.
struct Decay;

impl System<f32, Vector1<f32>> for Decay {
    fn system(&self, _x: f32, y: &Vector1<f32>, dy: &mut Vector1<f32>) {
        dy[0] = -y[0];
    }
}

/// dy/dx = -y in f64, used for the large-|x_end| variant.
struct DecayF64;

impl System<f64, Vector1<f64>> for DecayF64 {
    fn system(&self, _x: f64, y: &Vector1<f64>, dy: &mut Vector1<f64>) {
        dy[0] = -y[0];
    }
}

/// Dopri5 dense output over [0, 1] with step 0.1 in f32 must include x_end.
///
/// Before the fix the grid accumulates round-off to ~1e-7, exceeding the old
/// 1e-9 absolute tolerance, so the last returned point is 0.9000001 and x_end
/// (1.0) is dropped.
#[test]
fn dopri5_f32_reaches_x_end() {
    let x_end = 1.0_f32;
    let y0 = Vector1::<f32>::from_vec(vec![1.0]);
    let mut stepper = Dopri5::new(Decay {}, 0.0, x_end, 0.1, y0, 1.0e-3, 1.0e-6);
    stepper.integrate().unwrap();

    let x_last = *stepper.x_out().last().unwrap();
    assert!(
        (x_end - x_last).abs() < 1.0e-4,
        "dense output dropped x_end: x_last={x_last}"
    );
}

/// Dop853 dense output over [0, 1] with step 0.1 in f32 must include x_end.
///
/// Exercises the second endpoint-check site (src/dop853.rs).
#[test]
fn dop853_f32_reaches_x_end() {
    let x_end = 1.0_f32;
    let y0 = Vector1::<f32>::from_vec(vec![1.0]);
    let mut stepper = Dop853::new(Decay {}, 0.0, x_end, 0.1, y0, 1.0e-3, 1.0e-6);
    stepper.integrate().unwrap();

    let x_last = *stepper.x_out().last().unwrap();
    assert!(
        (x_end - x_last).abs() < 1.0e-4,
        "dense output dropped x_end: x_last={x_last}"
    );
}

/// f64 with a large |x_end|: an absolute 1e-9 tolerance does not scale with the
/// magnitude of x_end, whereas the relative slack does. The endpoint must be
/// present for a large integration bound as well.
#[test]
fn dopri5_f64_large_x_end_reaches_x_end() {
    let x_end = 1000.0_f64;
    let y0 = Vector1::<f64>::from_vec(vec![1.0]);
    let mut stepper = Dopri5::new(DecayF64 {}, 0.0, x_end, 0.1, y0, 1.0e-8, 1.0e-8);
    stepper.integrate().unwrap();

    let x_last = *stepper.x_out().last().unwrap();
    assert!(
        (x_end - x_last).abs() < 1.0e-6,
        "dense output dropped x_end: x_last={x_last}"
    );
}
