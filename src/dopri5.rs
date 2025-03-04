//! Explicit Runge-Kutta method with Dormand-Prince coefficients of order 5(4) and dense output of order 4.

use crate::butcher_tableau::dopri54;
use crate::continuous_output_model::ContinuousOutputModel;
use crate::controller::Controller;
use crate::dop_shared::*;
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, MatrixSum, OVector, U1};

trait DefaultController<T: FloatNumber> {
    fn default(x: T, x_end: T) -> Self;
}

impl<T: FloatNumber> DefaultController<T> for Controller<T> {
    fn default(x: T, x_end: T) -> Self {
        let alpha = T::from(0.2 - 0.04 * 0.75).unwrap();
        Controller::new(
            alpha,
            T::from(0.04).unwrap(),
            T::from(10.0).unwrap(),
            T::from(0.2).unwrap(),
            x_end - x,
            T::from(0.9).unwrap(),
            sign(T::one(), x_end - x),
        )
    }
}

/// Structure containing the parameters for the numerical integration.
pub struct Dopri5<T, V, F>
where
    T: FloatNumber,
    F: System<T, V>,
{
    f: F,
    x: T,
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
    rcont: [V; 5],
    stats: Stats,
}

impl<T, D: Dim, F> Dopri5<T, OVector<T, D>, F>
where
    f64: From<T>,
    T: FloatNumber,
    F: System<T, OVector<T, D>>,
    OVector<T, D>: std::ops::Mul<T, Output = OVector<T, D>>,
    DefaultAllocator: Allocator<D>,
{
    /// Default initializer for the structure
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
            xd: x,
            dx,
            x_old: x,
            x_end,
            y,
            rtol,
            atol,
            results: SolverResult::default(),
            uround: T::epsilon(),
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
    /// * `safety_factor`   - Safety factor used in the computation of the adaptive step size
    /// * `beta`    - Value of the beta coefficient of the PI controller. Default is 0.04
    /// * `fac_min` - Minimum factor between two successive steps. Default is 0.2
    /// * `fac_max` - Maximum factor between two successive steps. Default is 10.0
    /// * `h_max`   - Maximum step size. Default is `x_end-x`
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
        let alpha = T::from(0.2).unwrap() - beta * T::from(0.75).unwrap();
        let (rows, cols) = y.shape_generic();
        Self {
            f,
            x,
            xd: x,
            x_old: T::zero(),
            x_end,
            dx,
            y,
            rtol,
            atol,
            results: SolverResult::default(),
            uround: T::from(f64::EPSILON).unwrap(),
            h,
            h_old: T::zero(),
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
            ],
            stats: Stats::new(),
        }
    }

    /// Computes the initial step size
    fn hinit(&self) -> T {
        let (rows, cols) = self.y.shape_generic();
        let mut f0 = OVector::zeros_generic(rows, cols);
        self.f.system(self.x, &self.y, &mut f0);
        let posneg = sign(T::from(1.0).unwrap(), self.x_end - self.x);

        // Compute the norm of y0 and f0
        let dim = rows.value();
        let mut d0 = T::zero();
        let mut d1 = T::zero();
        for i in 0..dim {
            let y_i = T::from(self.y[i]).unwrap();
            let sci = self.atol + y_i.abs() * self.rtol;
            d0 += (y_i / sci) * (y_i / sci);
            let f0_i = T::from(f0[i]).unwrap();
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
            (T::from(0.01).unwrap() / (d1.sqrt().max(d2))).powf(T::one() / T::from(5.0).unwrap())
        };

        sign(
            (T::from(100.0).unwrap() * h0.abs()).min(h1.min(self.controller.h_max())),
            posneg,
        )
    }

    /// Integrates the system and builds a continuous output model.
    pub fn integrate_with_continuous_output_model(
        &mut self,
        continuous_output: &mut ContinuousOutputModel<T, OVector<T, D>>,
    ) -> Result<Stats, IntegrationError> {
        self.out_type = OutputType::Continuous;
        self.integrate_core(Some(continuous_output))
    }

    /// Integrates the system.
    pub fn integrate(&mut self) -> Result<Stats, IntegrationError> {
        if self.out_type == OutputType::Continuous {
            panic!("Please use `integrate_with_continuous_output_model` to compute a continuous output model.");
        }
        self.integrate_core(None)
    }

    /// Core integration method.
    fn integrate_core(
        &mut self,
        mut continuous_output_model: Option<&mut ContinuousOutputModel<T, OVector<T, D>>>,
    ) -> Result<Stats, IntegrationError> {
        // Initialization
        let (rows, cols) = self.y.shape_generic();
        self.x_old = self.x;
        let mut n_step = 0;
        let mut last = false;
        let mut h_new = T::zero();
        let dim = rows.value();
        let mut non_stiff = 0;
        let mut iasti = 0;
        let posneg = sign(T::one(), self.x_end - self.x);

        // Initialize the lower bound of the continuous output model if one is present
        if let Some(output) = continuous_output_model.as_deref_mut() {
            output.set_lower_bound(self.x);
        }

        if self.h == T::zero() {
            self.h = self.hinit();
            self.stats.num_eval += 2;
        }
        self.h_old = self.h;

        // Save initial values
        if self.out_type == OutputType::Sparse {
            self.results.push(self.x, self.y.clone());
        }

        let mut k = vec![OVector::zeros_generic(rows, cols); 7];
        self.f.system(self.x, &self.y, &mut k[0]);
        self.stats.num_eval += 1;

        // Main loop
        while !last {
            // Check if step number is within allowed range
            if n_step > self.n_max {
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

            let h = self.h;
            // 6 Stages
            let mut y_next = OVector::zeros_generic(rows, cols);
            let mut y_stiff = OVector::zeros_generic(rows, cols);
            for s in 1..7 {
                y_next = self.y.clone();
                for (j, k_value) in k.iter().enumerate().take(s) {
                    y_next += k_value * h * dopri54::a(s + 1, j + 1);
                }
                self.f
                    .system(self.x + self.h * dopri54::c::<T>(s + 1), &y_next, &mut k[s]);
                if s == 5 {
                    y_stiff = y_next.clone();
                }
            }
            k[1] = k[6].clone();
            self.stats.num_eval += 6;

            // Prepare dense/continuous output
            if self.out_type == OutputType::Dense || self.out_type == OutputType::Continuous {
                self.rcont[4] = (&k[0] * dopri54::d::<T>(1)
                    + &k[2] * dopri54::d::<T>(3)
                    + &k[3] * dopri54::d::<T>(4)
                    + &k[4] * dopri54::d::<T>(5)
                    + &k[5] * dopri54::d::<T>(6)
                    + &k[1] * dopri54::d::<T>(7))
                    * h;
            }

            // Compute error estimate
            k[3] = (&k[0] * dopri54::e::<T>(1)
                + &k[2] * dopri54::e::<T>(3)
                + &k[3] * dopri54::e::<T>(4)
                + &k[4] * dopri54::e::<T>(5)
                + &k[5] * dopri54::e::<T>(6)
                + &k[1] * dopri54::e::<T>(7))
                * h;

            // Compute error
            let mut err = T::zero();
            for i in 0..dim {
                let y_i = T::from(self.y[i]).unwrap();
                let y_next_i = T::from(y_next[i]).unwrap();
                let sc_i: T = self.atol + y_i.abs().max(y_next_i.abs()) * self.rtol;
                let err_est_i = T::from(k[3][i]).unwrap();
                err += (err_est_i / sc_i) * (err_est_i / sc_i);
            }
            err = (err / T::from(dim).unwrap()).sqrt();

            // Step size control
            if self.controller.accept(err, self.h, &mut h_new) {
                self.stats.accepted_steps += 1;

                // Stiffness detection
                if self.stats.accepted_steps % self.n_stiff == 0 || iasti > 0 {
                    let num = T::from((&k[1] - &k[5]).dot(&(&k[1] - &k[5]))).unwrap();
                    let den = T::from((&y_next - &y_stiff).dot(&(&y_next - &y_stiff))).unwrap();
                    let h_lamb = if den > T::zero() {
                        self.h * (num / den).sqrt()
                    } else {
                        T::zero()
                    };

                    if h_lamb > T::from(3.25).unwrap() {
                        iasti += 1;
                        non_stiff = 0;
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

                // Prepare dense/continuous output
                if self.out_type == OutputType::Dense || self.out_type == OutputType::Continuous {
                    let h = self.h;

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

                self.solution_output(y_next, &mut continuous_output_model, &k[0]);

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

    fn solution_output(
        &mut self,
        y_next: OVector<T, D>,
        continuous_output_model: &mut Option<&mut ContinuousOutputModel<T, OVector<T, D>>>,
        dy: &OVector<T, D>,
    ) {
        if self.out_type == OutputType::Dense {
            while self.xd.abs() <= self.x.abs() {
                if self.x_old.abs() <= self.xd.abs() && self.x.abs() >= self.xd.abs() {
                    let theta = (self.xd - self.x_old) / self.h_old;
                    let theta1 = T::one() - theta;
                    let y_out = self.compute_y_out(theta, theta1);
                    self.results.push(self.xd, y_out);
                    self.xd += self.dx;
                }

                if self
                    .f
                    .solout(self.xd, self.results.get().1.last().unwrap(), dy)
                {
                    break;
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
        } else {
            self.results.push(self.x, y_next)
        }

        // Update the continuous output model if one is present
        if let Some(output) = continuous_output_model.as_deref_mut() {
            output.add_interval(self.x.abs(), self.rcont.to_vec(), self.h_old);
        }
    }

    /// Computes the value of y for given theta and theta1 values.
    fn compute_y_out(&mut self, theta: T, theta1: T) -> MatrixSum<T, D, U1, D, U1> {
        &self.rcont[0]
            + (&self.rcont[1]
                + (&self.rcont[2] + (&self.rcont[3] + &self.rcont[4] * theta1) * theta) * theta1)
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

impl<T, D: Dim, F> From<Dopri5<T, OVector<T, D>, F>> for SolverResult<T, OVector<T, D>>
where
    T: FloatNumber,
    F: System<T, OVector<T, D>>,
    DefaultAllocator: Allocator<D>,
{
    fn from(val: Dopri5<T, OVector<T, D>, F>) -> Self {
        val.results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OVector, System, Vector1};
    use nalgebra::{allocator::Allocator, DefaultAllocator, Dim};

    // Same as Test3 from rk4.rs, but aborts after x is equal to or greater than 0.5
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
        let mut stepper = Dopri5::new(system, 0., 1., 0.1, Vector1::new(1.), 1e-12, 1e-6);
        let _ = stepper.integrate();

        let x = stepper.x_out();
        assert!((*x.last().unwrap() - 0.5).abs() < 1.0E-9);

        let out = stepper.y_out();
        assert!((&out[5][0] - 0.9130611474392001).abs() < 1.0E-9);
    }

    #[test]
    #[should_panic]
    fn test_integrate_when_continuous_output_type_panic() {
        let system = Test1 {};
        let mut stepper = Dopri5::from_param(
            system,
            0.,
            1.,
            0.1,
            Vector1::new(1.),
            1e-12,
            1e-6,
            0.1,
            0.2,
            0.3,
            2.0,
            5.0,
            0.1,
            10000,
            1000,
            OutputType::Continuous,
        );
        let _ = stepper.integrate();
    }
}
