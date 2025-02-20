//! Explicit Runge-Kutta method with Dormand-Prince coefficients of order 8(5,3) and dense output of order 7.

use crate::butcher_tableau::dopri853;
use crate::controller::Controller;
use crate::dop_shared::*;

use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, MatrixSum, OVector, U1};

trait DefaultController<T: FloatNumber> {
    fn default(x: T, x_end: T) -> Self;
}

impl<T: FloatNumber> DefaultController<T> for Controller<T> {
    fn default(x: T, x_end: T) -> Self {
        let alpha = T::one() / T::from(8.0).unwrap();
        Controller::new(
            alpha,
            T::zero(),
            T::from(6.0).unwrap(),
            T::from(0.333).unwrap(),
            x_end - x,
            T::from(0.9).unwrap(),
            sign(T::one(), x_end - x),
        )
    }
}
/// Structure containing the parameters for the numerical integration.
pub struct Dop853<T, V, F>
where
    T: FloatNumber,
    F: System<T, V>,
{
    f: F,
    x: T,
    x0: T,
    x_old: T,
    x_end: T,
    xd: T,
    dx: T,
    y: V,
    rtol: T,
    atol: T,
    results: SolverResult<T, V>,
    uround: T,
    h: T,
    h_old: T,
    n_max: u32,
    n_stiff: u32,
    controller: Controller<T>,
    out_type: OutputType,
    rcont: [V; 8],
    stats: Stats,
}

impl<T, D: Dim, F> Dop853<T, OVector<T, D>, F>
where
    f64: From<T>,
    T: FloatNumber,
    F: System<T, OVector<T, D>>,
    OVector<T, D>: std::ops::Mul<T, Output = OVector<T, D>>,
    DefaultAllocator: Allocator<D>,
{
    /// Default initializer for the structure.
    ///
    /// # Arguments
    ///
    /// * `f`       - Structure implementing the [`System`] trait
    /// * `x`       - Initial value of the independent variable (usually time)
    /// * `x_end`   - Final value of the independent variable
    /// * `dx`      - Increment in the dense output. This argument has no effect if the output type is Sparse
    /// * `y`       - Initial value of the dependent variable(s)
    /// * `rtol`    - Relative tolerance used in the computation of the adaptive step size
    /// * `atol`    - Absolute tolerance used in the computation of the adaptive step size
    ///
    pub fn new(f: F, x: T, x_end: T, dx: T, y: OVector<T, D>, rtol: T, atol: T) -> Self {
        let (rows, cols) = y.shape_generic();
        Self {
            f,
            x,
            x0: x,
            xd: x,
            dx,
            x_old: x,
            x_end,
            y,
            rtol,
            atol,
            results: SolverResult::default(),
            uround: T::from(f64::EPSILON).unwrap(),
            h: T::zero(),
            h_old: T::zero(),
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
    /// * `f`       - Structure implementing the [`System`] trait
    /// * `x`       - Initial value of the independent variable (usually time)
    /// * `x_end`   - Final value of the independent variable
    /// * `dx`      - Increment in the dense output. This argument has no effect if the output type is Sparse
    /// * `y`       - Initial value of the dependent variable(s)
    /// * `rtol`    - Relative tolerance used in the computation of the adaptive step size
    /// * `atol`    - Absolute tolerance used in the computation of the adaptive step size
    /// * `safety_factor`   - Safety factor used in the computation of the adaptive step size. Default is 0.9
    /// * `beta`    - Value of the beta coefficient of the PI controller. Default is 0.0
    /// * `fac_min` - Minimum factor between two successive steps. Default is 0.333
    /// * `fac_max` - Maximum factor between two successive steps. Default is 6.0
    /// * `h_max`   - Maximum step size. Default is `x_end-x
    /// * `h`       - Initial value of the step size. If h = 0.0, the initial value of h is computed automatically
    /// * `n_max`   - Maximum number of iterations. Default is 100000
    /// * `n_stiff` - Stiffness is tested when the number of iterations is a multiple of n_stiff. Default is 1000
    /// * `out_type`    - Type of the output. Must be a variant of the OutputType enum. Default is Dense
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn from_param(
        f: F,
        x: T,
        x_end: T,
        dx: T,
        y: OVector<T, D>,
        rtol: T,
        atol: T,
        safety_factor: T,
        beta: T,
        fac_min: T,
        fac_max: T,
        h_max: T,
        h: T,
        n_max: u32,
        n_stiff: u32,
        out_type: OutputType,
    ) -> Self {
        let alpha = T::one() / T::from(8.0).unwrap() - beta * T::from(0.2).unwrap();
        let (rows, cols) = y.shape_generic();
        Self {
            f,
            x,
            x0: x,
            xd: x,
            dx,
            x_old: x,
            x_end,
            y,
            rtol,
            atol,
            results: SolverResult::default(),
            uround: T::from(f64::EPSILON).unwrap(),
            h,
            h_old: h,
            n_max,
            n_stiff,
            controller: Controller::new(
                alpha,
                beta,
                fac_max,
                fac_min,
                h_max,
                safety_factor,
                sign(T::one(), x_end - x),
            ),
            out_type,
            rcont: [
                OVector::zeros_generic(rows, cols),
                OVector::zeros_generic(rows, cols),
                OVector::zeros_generic(rows, cols),
                OVector::zeros_generic(rows, cols),
                OVector::zeros_generic(rows, cols),
                OVector::zeros_generic(rows, cols),
                OVector::zeros_generic(rows, cols),
                OVector::zeros_generic(rows, cols),
            ],
            stats: Stats::new(),
        }
    }

    /// Computes the initial step size.
    fn hinit(&self) -> T {
        let (rows, cols) = self.y.shape_generic();
        let mut f0 = OVector::zeros_generic(rows, cols);
        self.f.system(self.x, &self.y, &mut f0);
        let posneg = sign(T::one(), self.x_end - self.x);

        // Compute the norm of y0 and f0
        let dim = rows.value();
        let mut d0 = T::zero();
        let mut d1 = T::zero();
        for i in 0..dim {
            let y_i = self.y[i];
            let sci = self.atol + y_i.abs() * self.rtol;
            d0 += (y_i / sci) * (y_i / sci);
            let f0_i = f0[i];
            d1 += (f0_i / sci) * (f0_i / sci);
        }

        // Compute h0
        let tol = T::from(1.0E-10).unwrap();
        let mut h0 = if d0 < tol || d1 < tol {
            T::from(1.0E-6).unwrap()
        } else {
            T::from(0.01).unwrap() * (d0 / d1).sqrt()
        };

        h0 = h0.min(self.controller.h_max());
        h0 = sign(h0, posneg);

        let y1 = &self.y + &f0 * h0;
        let mut f1 = OVector::zeros_generic(rows, cols);
        self.f.system(self.x + h0, &y1, &mut f1);

        // Compute the norm of f1-f0 divided by h0
        let mut d2 = T::zero();
        for i in 0..dim {
            let f0_i = f0[i];
            let f1_i = f1[i];
            let y_i = self.y[i];
            let sci = self.atol + y_i.abs() * self.rtol;
            d2 += ((f1_i - f0_i) / sci) * ((f1_i - f0_i) / sci);
        }
        d2 = d2.sqrt() / h0;

        let h1 = if d1.sqrt().max(d2.abs()) <= T::from(1.0E-15).unwrap() {
            T::from(1.0E-6_f64)
                .unwrap()
                .max(h0.abs() * T::from(1.0E-3).unwrap())
        } else {
            (T::from(0.01).unwrap() / (d1.sqrt().max(d2))).powf(T::from(0.125).unwrap())
        };

        sign(
            (T::from(100.0).unwrap() * h0.abs()).min(h1.min(self.controller.h_max())),
            posneg,
        )
    }

    /// Core integration method.
    pub fn integrate(&mut self) -> Result<Stats, IntegrationError> {
        // Initialization
        let (rows, cols) = self.y.shape_generic();
        self.x_old = self.x;
        let mut n_step = 0;
        let mut last = false;
        let mut h_new = T::zero();
        let dim = rows.value();
        let mut iasti = 0;
        let mut non_stiff = 0;
        let posneg = sign(T::one(), self.x_end - self.x);

        if self.h == T::zero() {
            self.h = self.hinit();
            self.stats.num_eval += 2;
        }
        self.h_old = self.h;

        // Save initial values
        let y_tmp = self.y.clone();
        self.solution_output(y_tmp);

        let mut k = vec![OVector::zeros_generic(rows, cols); 12];
        self.f.system(self.x, &self.y, &mut k[0]);
        self.stats.num_eval += 1;

        // Main loop
        while !last {
            // Check if step number is within allowed range
            if n_step >= self.n_max {
                self.h_old = self.h;
                return Err(IntegrationError::MaxNumStepReached {
                    x: f64::from(self.x),
                    n_step,
                });
            }

            // Check for step size underflow
            if T::from(0.1).unwrap() * self.h.abs() <= self.uround * self.x.abs() {
                self.h_old = self.h;
                return Err(IntegrationError::StepSizeUnderflow {
                    x: f64::from(self.x),
                });
            }

            // Check if it's the last iteration
            if (self.x + T::from(1.01).unwrap() * self.h - self.x_end) * posneg > T::zero() {
                self.h = self.x_end - self.x;
                last = true;
            }
            n_step += 1;

            // 12 Stages
            let mut y_next = OVector::zeros_generic(rows, cols);
            for s in 1..12 {
                y_next = self.y.clone();
                for (j, k_value) in k.iter().enumerate().take(s) {
                    y_next += k_value * (self.h * dopri853::a::<T>(s + 1, j + 1));
                }
                self.f.system(
                    self.x + (self.h * dopri853::c::<T>(s + 1)),
                    &y_next,
                    &mut k[s],
                );
            }
            k[1] = k[10].clone();
            k[2] = k[11].clone();
            self.stats.num_eval += 11;

            k[3] = &k[0] * dopri853::b::<T>(1)
                + &k[5] * dopri853::b::<T>(6)
                + &k[6] * dopri853::b::<T>(7)
                + &k[7] * dopri853::b::<T>(8)
                + &k[8] * dopri853::b::<T>(9)
                + &k[9] * dopri853::b::<T>(10)
                + &k[1] * dopri853::b::<T>(11)
                + &k[2] * dopri853::b::<T>(12);
            k[4] = &self.y + &k[3] * self.h;

            // Error estimate
            let mut err_est = OVector::zeros_generic(rows, cols);
            for (i, k_value) in k.iter().enumerate().take(12) {
                err_est += k_value * dopri853::e::<T>(i + 1);
            }

            // Compute error
            let mut err = T::zero();
            let mut err2 = T::zero();
            let err_bhh = &k[3]
                - &k[0] * dopri853::bhh::<T>(1)
                - &k[8] * dopri853::bhh::<T>(2)
                - &k[2] * dopri853::bhh::<T>(3);
            for i in 0..dim {
                let y_i = self.y[i];
                let k5_i = k[4][i];
                let sc_i = self.atol + y_i.abs().max(k5_i.abs()) * self.rtol;

                let err_est_i = err_est[i];
                err += (err_est_i / sc_i) * (err_est_i / sc_i);

                let erri = err_bhh[i];
                err2 += (erri / sc_i) * (erri / sc_i);
            }
            let mut deno = err + T::from(0.01).unwrap() * err2;
            if deno <= T::zero() {
                deno = T::one();
            }

            err = self.h.abs() * err * (T::one() / (deno * T::from(dim).unwrap())).sqrt();

            // Step size control
            if self.controller.accept(err, self.h, &mut h_new) {
                self.stats.accepted_steps += 1;
                let y_tmp = k[4].clone();
                self.f.system(self.x + self.h, &y_tmp, &mut k[3]);
                self.stats.num_eval += 1;

                // Stiffness detection
                if self.stats.accepted_steps % self.n_stiff == 0 || iasti > 0 {
                    let num = T::from((&k[3] - &k[2]).dot(&(&k[3] - &k[2]))).unwrap();
                    let den = T::from((&k[4] - &y_next).dot(&(&k[4] - &y_next))).unwrap();
                    let h_lamb = if den > T::zero() {
                        self.h * (num / den).sqrt()
                    } else {
                        T::zero()
                    };

                    if h_lamb > T::from(6.1).unwrap() {
                        non_stiff = 0;
                        iasti += 1;
                        if iasti == 15 {
                            self.h_old = self.h;
                            return Err(IntegrationError::StiffnessDetected {
                                x: f64::from(self.x),
                            });
                        }
                    } else {
                        non_stiff += 1;
                        if non_stiff == 6 {
                            iasti = 0;
                        }
                    }
                }

                if self.out_type == OutputType::Dense {
                    let h = self.h;

                    self.rcont[0] = self.y.clone();
                    let y_diff = &k[4] - &self.y;
                    self.rcont[1] = y_diff.clone();
                    let bspl = &k[0] * h - &y_diff;
                    self.rcont[2] = bspl.clone();
                    self.rcont[3] = &y_diff - &k[3] * h - &bspl;
                    for i in 4..8 {
                        self.rcont[i] = OVector::zeros_generic(rows, cols);
                        for j in 1..13 {
                            self.rcont[i] += &k[j - 1] * dopri853::d::<T>(i, j);
                        }
                    }

                    // Next three function evaluations
                    let mut y_next = &self.y
                        + (&k[0] * dopri853::a::<T>(14, 1)
                            + &k[6] * dopri853::a::<T>(14, 7)
                            + &k[7] * dopri853::a::<T>(14, 8)
                            + &k[8] * dopri853::a::<T>(14, 9)
                            + &k[9] * dopri853::a::<T>(14, 10)
                            + &k[1] * dopri853::a::<T>(14, 11)
                            + &k[2] * dopri853::a::<T>(14, 12)
                            + &k[3] * dopri853::a::<T>(14, 13))
                            * h;
                    self.f
                        .system(self.x + self.h * dopri853::c::<T>(14), &y_next, &mut k[9]);

                    y_next = &self.y
                        + (&k[0] * dopri853::a::<T>(15, 1)
                            + &k[5] * dopri853::a::<T>(15, 6)
                            + &k[6] * dopri853::a::<T>(15, 7)
                            + &k[7] * dopri853::a::<T>(15, 8)
                            + &k[1] * dopri853::a::<T>(15, 11)
                            + &k[2] * dopri853::a::<T>(15, 12)
                            + &k[3] * dopri853::a::<T>(15, 13)
                            + &k[9] * dopri853::a::<T>(15, 14))
                            * h;
                    self.f
                        .system(self.x + self.h * dopri853::c::<T>(15), &y_next, &mut k[1]);

                    y_next = &self.y
                        + (&k[0] * dopri853::a::<T>(16, 1)
                            + &k[5] * dopri853::a::<T>(16, 6)
                            + &k[6] * dopri853::a::<T>(16, 7)
                            + &k[7] * dopri853::a::<T>(16, 8)
                            + &k[8] * dopri853::a::<T>(16, 9)
                            + &k[3] * dopri853::a::<T>(16, 13)
                            + &k[9] * dopri853::a::<T>(16, 14)
                            + &k[1] * dopri853::a::<T>(16, 15))
                            * h;
                    self.f
                        .system(self.x + self.h * dopri853::c::<T>(16), &y_next, &mut k[2]);

                    self.stats.num_eval += 3;

                    self.rcont[4] = (&self.rcont[4]
                        + &k[3] * dopri853::d::<T>(4, 13)
                        + &k[9] * dopri853::d::<T>(4, 14)
                        + &k[1] * dopri853::d::<T>(4, 15)
                        + &k[2] * dopri853::d::<T>(4, 16))
                        * h;
                    self.rcont[5] = (&self.rcont[5]
                        + &k[3] * dopri853::d::<T>(5, 13)
                        + &k[9] * dopri853::d::<T>(5, 14)
                        + &k[1] * dopri853::d::<T>(5, 15)
                        + &k[2] * dopri853::d::<T>(5, 16))
                        * h;
                    self.rcont[6] = (&self.rcont[6]
                        + &k[3] * dopri853::d::<T>(6, 13)
                        + &k[9] * dopri853::d::<T>(6, 14)
                        + &k[1] * dopri853::d::<T>(6, 15)
                        + &k[2] * dopri853::d::<T>(6, 16))
                        * h;
                    self.rcont[7] = (&self.rcont[7]
                        + &k[3] * dopri853::d::<T>(7, 13)
                        + &k[9] * dopri853::d::<T>(7, 14)
                        + &k[1] * dopri853::d::<T>(7, 15)
                        + &k[2] * dopri853::d::<T>(7, 16))
                        * h;
                }

                k[0] = k[3].clone();
                self.y = k[4].clone();
                self.x_old = self.x;
                self.x += self.h;
                self.h_old = self.h;

                self.solution_output(k[4].clone());

                // Early abortion check
                if self
                    .f
                    .solout(self.x, self.results.get().1.last().unwrap(), &k[0])
                {
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

    /// If a dense output is required, computes the solution and pushes it into the output vector. Else, pushes the solution into the output vector.
    fn solution_output(&mut self, y_next: OVector<T, D>) {
        if self.out_type == OutputType::Dense {
            if (self.xd - self.x0).abs() < T::from(f64::EPSILON).unwrap() {
                self.results.push(self.x0, self.y.clone());
                self.xd += self.dx;
            } else {
                while self.xd.abs() <= self.x.abs() {
                    if self.x_old.abs() <= self.xd.abs() && self.x.abs() >= self.xd.abs() {
                        let theta = (self.xd - self.x_old) / self.h_old;
                        let theta1 = T::one() - theta;
                        let y_out = self.compute_y_out(theta, theta1);
                        self.results.push(self.xd, y_out);
                        self.xd += self.dx;
                    }
                }

                // Ensure the last point is added if it's within floating point error of x_end.
                if (self.xd - self.x_end).abs() < T::from(1e-9).unwrap() {
                    let theta = (self.x_end - self.x_old) / self.h_old;
                    let theta1 = T::one() - theta;
                    let y_out = self.compute_y_out(theta, theta1);
                    self.results.push(self.x_end, y_out);
                    self.xd += self.dx;
                }
            }
        } else {
            self.results.push(self.x0, y_next.clone());
        }
    }

    /// Computes the value of y for given theta and theta1 values.
    fn compute_y_out(&mut self, theta: T, theta1: T) -> MatrixSum<T, D, U1, D, U1> {
        &self.rcont[0]
            + (&self.rcont[1]
                + (&self.rcont[2]
                    + (&self.rcont[3]
                        + (&self.rcont[4]
                            + (&self.rcont[5]
                                + (&self.rcont[6] + &self.rcont[7] * theta) * theta1)
                                * theta)
                            * theta1)
                        * theta)
                    * theta1)
                * theta
    }

    /// Getter for the independent variable's output.
    pub fn x_out(&self) -> &Vec<T> {
        self.results.get().0
    }

    /// Getter for the dependent variables' output.
    pub fn y_out(&self) -> &Vec<OVector<T, D>> {
        self.results.get().1
    }

    /// Getter for the results type, a pair of independent and dependent variables
    pub fn results(&self) -> &SolverResult<T, OVector<T, D>> {
        &self.results
    }
}

impl<T, D: Dim, F> From<Dop853<T, OVector<T, D>, F>> for SolverResult<T, OVector<T, D>>
where
    T: FloatNumber,
    F: System<T, OVector<T, D>>,
    DefaultAllocator: Allocator<D>,
{
    fn from(val: Dop853<T, OVector<T, D>, F>) -> Self {
        val.results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OVector, System, Vector1};
    use nalgebra::{allocator::Allocator, DefaultAllocator, Dim};

    // Same as Test3 from rk4.rs, but aborts after x is greater/equal than 0.5
    struct Test1 {}
    impl<D: Dim> System<f64, OVector<f64, D>> for Test1
    where
        DefaultAllocator: Allocator<D>,
    {
        fn system(&self, x: f64, y: &OVector<f64, D>, dy: &mut OVector<f64, D>) {
            dy[0] = (5. * x * x - y[0]) / (x + y[0]).exp();
        }

        fn solout(&mut self, x: f64, _y: &OVector<f64, D>, _dy: &OVector<f64, D>) -> bool {
            x >= 0.5
        }
    }

    #[test]
    fn test_integrate_test1_svector() {
        let system = Test1 {};
        let mut stepper = Dop853::new(system, 0., 1., 0.1, Vector1::new(1.), 1e-12, 1e-6);
        let _ = stepper.integrate();

        let x = stepper.x_out();
        assert!((*x.last().unwrap() - 0.5).abs() < 1.0E-9); //

        let out = stepper.y_out();
        assert!((&out[5][0] - 0.912968195).abs() < 1.0E-9);
    }
}
