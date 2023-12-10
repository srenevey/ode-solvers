//! Explicit Runge-Kutta method with Dormand-Prince coefficients of order 5(4) and dense output of order 4.

use crate::butcher_tableau::dopri54;
use crate::controller::Controller;
use crate::dop_shared::*;

use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, Scalar};
use num_traits::Zero;
use simba::scalar::{ClosedAdd, ClosedMul, ClosedNeg, ClosedSub, SubsetOf, SupersetOf};

trait DefaultController {
    fn default(x: f64, x_end: f64) -> Self;
}

impl DefaultController for Controller {
    fn default(x: f64, x_end: f64) -> Self {
        let alpha = 0.2 - 0.04 * 0.75;
        Controller::new(alpha, 0.04, 10.0, 0.2, x_end - x, 0.9, sign(1.0, x_end - x))
    }
}

/// Structure containing the parameters for the numerical integration.
pub struct Dopri5<V, F>
where
    F: System<V>,
{
    f: F,
    x: f64,
    x_old: f64,
    x_end: f64,
    xd: f64,
    dx: f64,
    y: V,
    rtol: f64,
    atol: f64,
    x_out: Vec<f64>,
    y_out: Vec<V>,
    uround: f64,
    h: f64,
    h_old: f64,
    n_max: u32,
    n_stiff: u32,
    controller: Controller,
    out_type: OutputType,
    rcont: [V; 5],
    stats: Stats,
}

impl<T, D: Dim, F> Dopri5<OVector<T, D>, F>
where
    f64: From<T>,
    T: Copy + SubsetOf<f64> + Scalar + ClosedAdd + ClosedMul + ClosedSub + ClosedNeg + Zero,
    F: System<OVector<T, D>>,
    DefaultAllocator: Allocator<T, D>,
{
    /// Default initializer for the structure
    ///
    /// # Arguments
    ///
    /// * `f`       - Structure implementing the System<V> trait
    /// * `x`       - Initial value of the independent variable (usually time)
    /// * `x_end`   - Final value of the independent variable
    /// * `dx`      - Increment in the dense output. This argument has no effect if the output type is Sparse
    /// * `y`       - Initial value of the dependent variable(s)
    /// * `rtol`    - Relative tolerance used in the computation of the adaptive step size
    /// * `atol`    - Absolute tolerance used in the computation of the adaptive step size
    ///
    pub fn new(f: F, x: f64, x_end: f64, dx: f64, y: OVector<T, D>, rtol: f64, atol: f64) -> Self {
        let (rows, cols) = y.shape_generic();
        Self {
            f,
            x,
            xd: x,
            dx,
            x_old: x,
            x_end,
            y,
            rtol,
            atol,
            x_out: Vec::new(),
            y_out: Vec::new(),
            uround: f64::EPSILON,
            h: 0.0,
            h_old: 0.0,
            n_max: 100000,
            n_stiff: 1000,
            controller: Controller::default(x, x_end),
            out_type: OutputType::Dense,
            rcont: [
                OVector::zeros_generic(rows, cols),
                OVector::zeros_generic(rows, cols),
                OVector::zeros_generic(rows, cols),
                OVector::zeros_generic(rows, cols),
                OVector::zeros_generic(rows, cols),
            ],
            stats: Stats::new(),
        }
    }

    /// Advanced initializer for the structure.
    ///
    /// # Arguments
    ///
    /// * `f`       - Structure implementing the System<V> trait
    /// * `x`       - Initial value of the independent variable (usually time)
    /// * `x_end`   - Final value of the independent variable
    /// * `dx`      - Increment in the dense output. This argument has no effect if the output type is Sparse
    /// * `y`       - Initial value of the dependent variable(s)
    /// * `rtol`    - Relative tolerance used in the computation of the adaptive step size
    /// * `atol`    - Absolute tolerance used in the computation of the adaptive step size
    /// * `safety_factor`   - Safety factor used in the computation of the adaptive step size
    /// * `beta`    - Value of the beta coefficient of the PI controller. Default is 0.04
    /// * `fac_min` - Minimum factor between two successive steps. Default is 0.2
    /// * `fac_max` - Maximum factor between two successive steps. Default is 10.0
    /// * `h_max`   - Maximum step size. Default is `x_end-x`
    /// * `h`       - Initial value of the step size. If h = 0.0, the intial value of h is computed automatically
    /// * `n_max`   - Maximum number of iterations. Default is 100000
    /// * `n_stiff` - Stifness is tested when the number of iterations is a multiple of n_stiff. Default is 1000
    /// * `out_type`    - Type of the output. Must be a variant of the OutputType enum. Default is Dense
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn from_param(
        f: F,
        x: f64,
        x_end: f64,
        dx: f64,
        y: OVector<T, D>,
        rtol: f64,
        atol: f64,
        safety_factor: f64,
        beta: f64,
        fac_min: f64,
        fac_max: f64,
        h_max: f64,
        h: f64,
        n_max: u32,
        n_stiff: u32,
        out_type: OutputType,
    ) -> Self {
        let alpha = 0.2 - beta * 0.75;
        let (rows, cols) = y.shape_generic();
        Self {
            f,
            x,
            xd: x,
            x_old: 0.0,
            x_end,
            dx,
            y,
            rtol,
            atol,
            x_out: Vec::new(),
            y_out: Vec::new(),
            uround: f64::EPSILON,
            h,
            h_old: 0.0,
            n_max,
            n_stiff,
            controller: Controller::new(
                alpha,
                beta,
                fac_max,
                fac_min,
                h_max,
                safety_factor,
                sign(1.0, x_end - x),
            ),
            out_type,
            rcont: [
                OVector::zeros_generic(rows, cols),
                OVector::zeros_generic(rows, cols),
                OVector::zeros_generic(rows, cols),
                OVector::zeros_generic(rows, cols),
                OVector::zeros_generic(rows, cols),
            ],
            stats: Stats::new(),
        }
    }

    /// Compute the initial stepsize
    fn hinit(&self) -> f64 {
        let (rows, cols) = self.y.shape_generic();
        let mut f0 = OVector::zeros_generic(rows, cols);
        self.f.system(self.x, &self.y, &mut f0);
        let posneg = sign(1.0, self.x_end - self.x);

        // Compute the norm of y0 and f0
        let dim = rows.value();
        let mut d0 = 0.0;
        let mut d1 = 0.0;
        for i in 0..dim {
            let y_i = f64::from(self.y[i]);
            let sci = self.atol + y_i.abs() * self.rtol;
            d0 += (y_i / sci) * (y_i / sci);
            let f0_i = f64::from(f0[i]);
            d1 += (f0_i / sci) * (f0_i / sci);
        }

        // Compute h0
        let mut h0 = if d0 < 1.0E-10 || d1 < 1.0E-10 {
            1.0E-6
        } else {
            0.01 * (d0 / d1).sqrt()
        };

        h0 = h0.min(self.controller.h_max());
        h0 = sign(h0, posneg);

        let y1 = &self.y + &f0 * h0.to_subset().unwrap();
        let mut f1 = OVector::zeros_generic(rows, cols);
        self.f.system(self.x + h0, &y1, &mut f1);

        // Compute the norm of f1-f0 divided by h0
        let mut d2: f64 = 0.0;
        for i in 0..dim {
            let f0_i = f64::from(f0[i]);
            let f1_i = f64::from(f1[i]);
            let y_i = f64::from(self.y[i]);
            let sci: f64 = self.atol + y_i.abs() * self.rtol;
            d2 += ((f1_i - f0_i) / sci) * ((f1_i - f0_i) / sci);
        }
        d2 = d2.sqrt() / h0;

        let h1 = if d1.sqrt().max(d2.abs()) <= 1.0E-15 {
            (1.0E-6_f64).max(h0.abs() * 1.0E-3)
        } else {
            (0.01 / (d1.sqrt().max(d2))).powf(1.0 / 5.0)
        };

        sign(
            (100.0 * h0.abs()).min(h1.min(self.controller.h_max())),
            posneg,
        )
    }

    /// Core integration method.
    pub fn integrate(&mut self) -> Result<Stats, IntegrationError> {
        // Initilization
        let (rows, cols) = self.y.shape_generic();
        self.x_old = self.x;
        let mut n_step = 0;
        let mut last = false;
        let mut h_new = 0.0;
        let dim = rows.value();
        let mut non_stiff = 0;
        let mut iasti = 0;
        let posneg = sign(1.0, self.x_end - self.x);

        if self.h == 0.0 {
            self.h = self.hinit();
            self.stats.num_eval += 2;
        }
        self.h_old = self.h;

        // Save initial values
        if self.out_type == OutputType::Sparse {
            self.x_out.push(self.x);
            self.y_out.push(self.y.clone());
        }

        let mut k = vec![OVector::zeros_generic(rows, cols); 7];
        self.f.system(self.x, &self.y, &mut k[0]);
        self.stats.num_eval += 1;

        // Main loop
        while !last {
            // Check if step number is within allowed range
            if n_step > self.n_max {
                self.h_old = self.h;
                return Err(IntegrationError::MaxNumStepReached { x: self.x, n_step });
            }

            // Check for step size underflow
            if 0.1 * self.h.abs() <= self.uround * self.x.abs() {
                self.h_old = self.h;
                return Err(IntegrationError::StepSizeUnderflow { x: self.x });
            }

            // Check if it's the last iteration
            if (self.x + 1.01 * self.h - self.x_end) * posneg > 0.0 {
                self.h = self.x_end - self.x;
                last = true;
            }
            n_step += 1;

            let h = self.h.to_subset().unwrap();
            // 6 Stages
            let mut y_next = OVector::zeros_generic(rows, cols);
            let mut y_stiff = OVector::zeros_generic(rows, cols);
            for s in 1..7 {
                y_next = self.y.clone();
                for (j, k_value) in k.iter().enumerate().take(s) {
                    y_next += k_value * h * dopri54::a(s + 1, j + 1);
                }
                self.f.system(
                    self.x + self.h * dopri54::c::<f64>(s + 1),
                    &y_next,
                    &mut k[s],
                );
                if s == 5 {
                    y_stiff = y_next.clone();
                }
            }
            k[1] = k[6].clone();
            self.stats.num_eval += 6;

            // Prepare dense output
            if self.out_type == OutputType::Dense {
                self.rcont[4] = (&k[0] * dopri54::d(1)
                    + &k[2] * dopri54::d(3)
                    + &k[3] * dopri54::d(4)
                    + &k[4] * dopri54::d(5)
                    + &k[5] * dopri54::d(6)
                    + &k[1] * dopri54::d(7))
                    * h;
            }

            // Compute error estimate
            k[3] = (&k[0] * dopri54::e(1)
                + &k[2] * dopri54::e(3)
                + &k[3] * dopri54::e(4)
                + &k[4] * dopri54::e(5)
                + &k[5] * dopri54::e(6)
                + &k[1] * dopri54::e(7))
                * h;

            // Compute error
            let mut err = 0.0;
            for i in 0..dim {
                let y_i = f64::from(self.y[i]);
                let y_next_i = f64::from(y_next[i]);
                let sc_i: f64 = self.atol + y_i.abs().max(y_next_i.abs()) * self.rtol;
                let err_est_i = f64::from(k[3][i]);
                err += (err_est_i / sc_i) * (err_est_i / sc_i);
            }
            err = (err / dim as f64).sqrt();

            // Step size control
            if self.controller.accept(err, self.h, &mut h_new) {
                self.stats.accepted_steps += 1;

                // Stifness detection
                if self.stats.accepted_steps % self.n_stiff == 0 || iasti > 0 {
                    let num = f64::from((&k[1] - &k[5]).dot(&(&k[1] - &k[5])));
                    let den = f64::from((&y_next - &y_stiff).dot(&(&y_next - &y_stiff)));
                    let h_lamb = if den > 0.0 {
                        self.h * (num / den).sqrt()
                    } else {
                        0.0
                    };

                    if h_lamb > 3.25 {
                        iasti += 1;
                        non_stiff = 0;
                        if iasti == 15 {
                            self.h_old = self.h;
                            return Err(IntegrationError::StiffnessDetected { x: self.x });
                        }
                    } else {
                        non_stiff += 1;
                        if non_stiff == 6 {
                            iasti = 0;
                        }
                    }
                }

                // Prepare dense output
                if self.out_type == OutputType::Dense {
                    let h = self.h.to_subset().unwrap();

                    let ydiff = &y_next - &self.y;
                    let bspl = &k[0] * h - &ydiff;
                    self.rcont[0] = self.y.clone();
                    self.rcont[1] = ydiff.clone();
                    self.rcont[2] = bspl.clone();
                    self.rcont[3] = -&k[1] * h + ydiff - bspl;
                }

                k[0] = k[1].clone();
                self.y = y_next.clone();
                self.x_old = self.x;
                self.x += self.h;
                self.h_old = self.h;

                self.solution_output(y_next, &k);

                if self.f.solout(self.x, self.y_out.last().unwrap(), &k[0]) {
                    last = true;
                }

                // Normal exit
                if last {
                    self.h_old = posneg * h_new;
                    return Ok(self.stats);
                }
            } else {
                last = false;
                if self.stats.accepted_steps >= 1 {
                    self.stats.rejected_steps += 1;
                }
            }
            self.h = h_new;
        }
        Ok(self.stats)
    }

    fn solution_output(&mut self, y_next: OVector<T, D>, _k: &[OVector<T, D>]) {
        if self.out_type == OutputType::Dense {
            while self.xd.abs() <= self.x.abs() {
                if self.x_old.abs() <= self.xd.abs() && self.x.abs() >= self.xd.abs() {
                    let theta = (self.xd - self.x_old) / self.h_old;
                    let theta1 = (1.0 - theta).to_subset().unwrap();
                    let theta = theta.to_subset().unwrap();
                    self.x_out.push(self.xd);
                    self.y_out.push(
                        &self.rcont[0]
                            + (&self.rcont[1]
                                + (&self.rcont[2]
                                    + (&self.rcont[3] + &self.rcont[4] * theta1) * theta)
                                    * theta1)
                                * theta,
                    );
                    self.xd += self.dx;
                }
            }
        } else {
            self.x_out.push(self.x);
            self.y_out.push(y_next);
        }
    }

    /// Getter for the independent variable's output.
    pub fn x_out(&self) -> &Vec<f64> {
        &self.x_out
    }

    /// Getter for the dependent variables' output.
    pub fn y_out(&self) -> &Vec<OVector<T, D>> {
        &self.y_out
    }
}

fn sign(a: f64, b: f64) -> f64 {
    if b > 0.0 {
        a.abs()
    } else {
        -a.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OVector, System, Vector1};
    use nalgebra::{allocator::Allocator, DefaultAllocator, Dim};

    // Same as Test3 from rk4.rs, but aborts after x is greater/equal than 0.5
    struct Test1 {}
    impl<D: Dim> System<OVector<f64, D>> for Test1
    where
        DefaultAllocator: Allocator<f64, D>,
    {
        fn system(&self, x: f64, y: &OVector<f64, D>, dy: &mut OVector<f64, D>) {
            dy[0] = (5. * x * x - y[0]) / (x + y[0]).exp();
        }

        fn solout(&mut self, x: f64, _y: &OVector<f64, D>, _dy: &OVector<f64, D>) -> bool {
            return x >= 0.5;
        }
    }

    #[test]
    fn test_integrate_test1_svector() {
        let system = Test1 {};
        let mut stepper = Dopri5::new(system, 0., 1., 0.1, Vector1::new(1.), 1e-12, 1e-6);
        let _ = stepper.integrate();

        let x = stepper.x_out();
        assert!((*x.last().unwrap() - 0.5).abs() < 1.0E-9); //

        let out = stepper.y_out();
        assert!((&out[5][0] - 0.913059243).abs() < 1.0E-9);
    }
}
