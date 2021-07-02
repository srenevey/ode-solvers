//! Explicit Runge-Kutta method of order 4 with fixed step size.

use crate::dop_shared::{IntegrationError, Stats, System};

use nalgebra::{Scalar, SVector };
use num_traits::Zero;
use simba::scalar::{ClosedAdd, ClosedMul, ClosedNeg, ClosedSub, SubsetOf};

/// Structure containing the parameters for the numerical integration.
pub struct Rk4<V, F>
    where F: System<V>
{
    f: F,
    x: f64,
    y: V,
    x_end: f64,
    step_size: f64,
    half_step: f64,
    x_out: Vec<f64>,
    y_out: Vec<V>,
    stats: Stats
}

impl<T, F, const N: usize> Rk4<SVector<T, N>, F>
    where f64: From<T>,
          T: Copy + SubsetOf<f64> + Scalar + ClosedAdd + ClosedMul + ClosedSub + ClosedNeg + Zero,
          F: System<SVector<T, N>>,
          SVector<T, N>: std::ops::Mul<f64, Output = SVector<T, N>>
{
    /// Default initializer for the structure
    ///
    /// # Arguments
    ///
    /// * `f`           - Structure implementing the System<V> trait
    /// * `x`           - Initial value of the independent variable (usually time)
    /// * `y`           - Initial value of the dependent variable(s)
    /// * `x_end`       - Final value of the independent variable
    /// * `step_size`   - Step size used in the method
    ///
    pub fn new(f: F, x: f64, y: SVector<T, N>, x_end: f64, step_size: f64) -> Self {
        Rk4 {
            f,
            x,
            y,
            x_end,
            step_size,
            half_step: step_size / 2.,
            x_out: Vec::new(),
            y_out: Vec::new(),
            stats: Stats::new()
        }
    }

    /// Core integration method.
    pub fn integrate(&mut self) -> Result<Stats, IntegrationError> {
        // Save initial values
        self.x_out.push(self.x);
        self.y_out.push(self.y);


        let num_steps = ((self.x_end - self.x) / self.step_size).ceil() as usize;
        for _ in 0..num_steps {
            let (x_new, y_new) = self.step();

            self.x_out.push(x_new);
            self.y_out.push(y_new);

            self.x = x_new;
            self.y = y_new;

            self.stats.num_eval += 4;
            self.stats.accepted_steps += 1;
        }
        Ok(self.stats)
    }

    /// Performs one step of the Runge-Kutta 4 method.
    fn step(&self) -> (f64, SVector<T, N>) {
        let mut k: Vec<SVector<T, N>> = vec![SVector::zero(); 4];

        self.f.system(self.x, &self.y, &mut k[0]);
        self.f.system(self.x + self.half_step, &(self.y + k[0] * self.half_step), &mut k[1]);
        self.f.system(self.x + self.half_step, &(self.y + k[1] * self.half_step), &mut k[2]);
        self.f.system(self.x + self.step_size, &(self.y + k[2] * self.step_size), &mut k[3]);

        let x_new = self.x + self.step_size;
        let y_new = self.y + (k[0] + k[1] * 2.0 + k[2] * 2.0 + k[3]) * (self.step_size / 6.0);
        (x_new, y_new)
    }

    /// Getter for the independent variable's output.
    pub fn x_out(&self) -> &Vec<f64> {
        &self.x_out
    }

    /// Getter for the dependent variables' output.
    pub fn y_out(&self) -> &Vec<SVector<T, N>> {
        &self.y_out
    }
}


#[cfg(test)]
mod tests {
    use crate::{System, Vector1};
    use crate::rk4::Rk4;

    struct Test1 {}
    impl System<Vector1<f64>> for Test1 {
        fn system(&self, x: f64, y: &Vector1<f64>, dy: &mut Vector1<f64>) {
            dy[0] = (x - y[0]) / 2.;
        }
    }

    struct Test2 {}
    impl System<Vector1<f64>> for Test2 {
        fn system(&self, x: f64, y: &Vector1<f64>, dy: &mut Vector1<f64>) {
            dy[0] = -2. * x - y[0];
        }
    }

    struct Test3 {}
    impl System<Vector1<f64>> for Test3 {
        fn system(&self, x: f64, y: &Vector1<f64>, dy: &mut Vector1<f64>) {
            dy[0] = (5. * x * x - y[0]) / (x + y[0]).exp();
        }
    }

    #[test]
    fn test_integrate_test1() {
        let system = Test1 {};
        let mut stepper = Rk4::new(system, 0., Vector1::new(1.), 0.2, 0.1);
        let _ = stepper.integrate();
        let out = stepper.y_out();
        assert!((&out[1][0] - 0.95369).abs() < 1.0E-5);
        assert!((&out[2][0] - 0.91451).abs() < 1.0E-5);
    }


    #[test]
    fn test_integrate_test2() {
        let system = Test2 {};
        let mut stepper = Rk4::new(system, 0., Vector1::new(-1.), 0.5, 0.1);
        let _ = stepper.integrate();
        let out = stepper.y_out();
        assert!((&out[3][0] + 0.82246).abs() < 1.0E-5);
        assert!((&out[5][0] + 0.81959).abs() < 1.0E-5);
    }

    #[test]
    fn test_integrate_test3() {
        let system = Test3 {};
        let mut stepper = Rk4::new(system, 0., Vector1::new(1.), 1., 0.1);
        let _ = stepper.integrate();
        let out = stepper.y_out();
        assert!((&out[5][0] - 0.913059839).abs() < 1.0E-9);
        assert!((&out[8][0] - 0.9838057659).abs() < 1.0E-9);
        assert!((&out[10][0] - 1.0715783953).abs() < 1.0E-9);
    }
}