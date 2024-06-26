//! Explicit Runge-Kutta method of order 4 with fixed step size.

use crate::dop_shared::{FloatNumber, IntegrationError, SolverResult, Stats, System};

use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};

/// Structure containing the parameters for the numerical integration.
pub struct Rk4<T, V, F>
where
    F: System<T, V>,
    T: FloatNumber,
{
    f: F,
    x: T,
    y: V,
    x_end: T,
    k: [V; 4],
    buffer: V, // Used for temporary copies in step
    step_size: T,
    half_step: T,
    results: SolverResult<T, V>,
    stats: Stats,
}

impl<T, D: Dim, F> Rk4<T, OVector<T, D>, F>
where
    T: FloatNumber,
    F: System<T, OVector<T, D>>,
    OVector<T, D>: std::ops::Mul<T, Output = OVector<T, D>>,
    DefaultAllocator: Allocator<D>,
{
    /// Default initializer for the structure
    ///
    /// # Arguments
    ///
    /// * `f`           - Structure implementing the [`System`] trait
    /// * `x`           - Initial value of the independent variable (usually time)
    /// * `y`           - Initial value of the dependent variable(s)
    /// * `x_end`       - Final value of the independent variable
    /// * `step_size`   - Step size used in the method
    ///
    pub fn new(f: F, x: T, y: OVector<T, D>, x_end: T, step_size: T) -> Self {
        // Cheaper push while solving due to usage of Vec::with_capacity
        let num_steps = (((x_end - x) / step_size).ceil()).to_usize().unwrap();

        // Preallocate k
        let (rows, cols) = y.shape_generic();
        let k = [
            OVector::zeros_generic(rows, cols),
            OVector::zeros_generic(rows, cols),
            OVector::zeros_generic(rows, cols),
            OVector::zeros_generic(rows, cols),
        ];

        let buffer = y.clone();
        Rk4 {
            f,
            x,
            y,
            x_end,
            k,
            buffer,
            step_size,
            half_step: step_size / T::from(2.).unwrap(),
            results: SolverResult::with_capacity(num_steps),
            stats: Stats::new(),
        }
    }

    /// Core integration method.
    pub fn integrate(&mut self) -> Result<Stats, IntegrationError> {
        // Save initial values
        self.results.push(self.x, self.y.clone());

        let num_steps = (((self.x_end - self.x) / self.step_size).ceil())
            .to_usize()
            .unwrap();

        for _ in 0..num_steps {
            let (x_new, y_new, abort) = self.step();

            self.x = x_new;
            self.y
                .iter_mut()
                .zip(y_new.iter())
                .for_each(|(y_self, y_new_elem)| {
                    *y_self = *y_new_elem;
                });

            self.results.push(x_new, y_new);

            self.stats.num_eval += 4;
            self.stats.accepted_steps += 1;

            if abort {
                break;
            }
        }
        Ok(self.stats)
    }

    /// Performs one step of the Runge-Kutta 4 method.
    fn step(&mut self) -> (T, OVector<T, D>, bool) {
        self.f.system(self.x, &self.y, &mut self.k[0]);

        self.populate_buffer(0, self.half_step);
        self.f
            .system(self.x + self.half_step, &self.buffer, &mut self.k[1]);

        self.populate_buffer(1, self.half_step);
        self.f
            .system(self.x + self.half_step, &self.buffer, &mut self.k[2]);

        self.populate_buffer(2, self.step_size);
        self.f
            .system(self.x + self.step_size, &self.buffer, &mut self.k[3]);

        let x_new = self.x + self.step_size;

        let mut y_new = self.y.clone();

        for (idx, y_elem) in y_new.iter_mut().enumerate() {
            let two = T::from(2.).unwrap();
            *y_elem +=
                (self.k[0][idx] + self.k[1][idx] * two + self.k[2][idx] * two + self.k[3][idx])
                    * (self.step_size / T::from(6.).unwrap());
        }

        // Early abortion check
        let abort = self.f.solout(x_new, &y_new, &self.k[0]);
        (x_new, y_new, abort)
    }

    /// Populate the buffer with the sum of y and the previous k-value
    fn populate_buffer(&mut self, idx: usize, step: T) {
        self.buffer
            .iter_mut()
            .zip(self.y.iter())
            .zip(self.k[idx].iter())
            .for_each(|((buffer_elem, y_elem), k_prev_elem)| {
                *buffer_elem = *y_elem + *k_prev_elem * step
            })
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

impl<T, D: Dim, F> From<Rk4<T, OVector<T, D>, F>> for SolverResult<T, OVector<T, D>>
where
    T: FloatNumber,
    F: System<T, OVector<T, D>>,
    DefaultAllocator: Allocator<D>,
{
    fn from(val: Rk4<T, OVector<T, D>, F>) -> Self {
        val.results
    }
}

#[cfg(test)]
mod tests {
    use crate::rk4::Rk4;
    use crate::{DVector, OVector, System, Vector1};
    use nalgebra::{allocator::Allocator, DefaultAllocator, Dim};

    struct Test1 {}
    impl<D: Dim> System<f64, OVector<f64, D>> for Test1
    where
        DefaultAllocator: Allocator<D>,
    {
        fn system(&self, x: f64, y: &OVector<f64, D>, dy: &mut OVector<f64, D>) {
            dy[0] = (x - y[0]) / 2.;
        }
    }

    struct Test2 {}
    impl<D: Dim> System<f64, OVector<f64, D>> for Test2
    where
        DefaultAllocator: Allocator<D>,
    {
        fn system(&self, x: f64, y: &OVector<f64, D>, dy: &mut OVector<f64, D>) {
            dy[0] = -2. * x - y[0];
        }
    }

    struct Test3 {}
    impl<D: Dim> System<f64, OVector<f64, D>> for Test3
    where
        DefaultAllocator: Allocator<D>,
    {
        fn system(&self, x: f64, y: &OVector<f64, D>, dy: &mut OVector<f64, D>) {
            dy[0] = (5. * x * x - y[0]) / (x + y[0]).exp();
        }
    }

    // Same as Test3, but aborts after x is greater/equal than 0.5
    struct Test4 {}
    impl<D: Dim> System<f64, OVector<f64, D>> for Test4
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
        let mut stepper = Rk4::new(system, 0., Vector1::new(1.), 0.2, 0.1);
        let _ = stepper.integrate();
        let x_out = stepper.x_out();
        let y_out = stepper.y_out();
        assert!((*x_out.last().unwrap() - 0.2).abs() < 1.0E-8);
        assert!((&y_out[1][0] - 0.95369).abs() < 1.0E-5);
        assert!((&y_out[2][0] - 0.91451).abs() < 1.0E-5);
    }

    #[test]
    fn test_integrate_test2_svector() {
        let system = Test2 {};
        let mut stepper = Rk4::new(system, 0., Vector1::new(-1.), 0.5, 0.1);
        let _ = stepper.integrate();
        let x_out = stepper.x_out();
        let y_out = stepper.y_out();
        assert!((*x_out.last().unwrap() - 0.5).abs() < 1.0E-8);
        assert!((&y_out[3][0] + 0.82246).abs() < 1.0E-5);
        assert!((&y_out[5][0] + 0.81959).abs() < 1.0E-5);
    }

    #[test]
    fn test_integrate_test3_svector() {
        let system = Test3 {};
        let mut stepper = Rk4::new(system, 0., Vector1::new(1.), 1., 0.1);
        let _ = stepper.integrate();
        let out = stepper.y_out();
        assert!((&out[5][0] - 0.913059839).abs() < 1.0E-9);
        assert!((&out[8][0] - 0.9838057659).abs() < 1.0E-9);
        assert!((&out[10][0] - 1.0715783953).abs() < 1.0E-9);
        assert_eq!(out.len(), 11);
    }

    #[test]
    fn test_integrate_test4_svector() {
        let system = Test4 {};
        let mut stepper = Rk4::new(system, 0., Vector1::new(1.), 1., 0.1);
        let _ = stepper.integrate();

        let x = stepper.x_out();
        assert!((*x.last().unwrap() - 0.5).abs() < 1.0E-9);

        let out = stepper.y_out();
        assert!((&out[5][0] - 0.913059839).abs() < 1.0E-9);
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn test_integrate_test1_dvector() {
        let system = Test1 {};
        let mut stepper = Rk4::new(system, 0., DVector::from(vec![1.]), 0.2, 0.1);
        let _ = stepper.integrate();
        let x_out = stepper.x_out();
        let y_out = stepper.y_out();
        assert!((*x_out.last().unwrap() - 0.2).abs() < 1.0E-8);
        assert!((&y_out[1][0] - 0.95369).abs() < 1.0E-5);
        assert!((&y_out[2][0] - 0.91451).abs() < 1.0E-5);
    }

    #[test]
    fn test_integrate_test2_dvector() {
        let system = Test2 {};
        let mut stepper = Rk4::new(system, 0., DVector::from(vec![-1.]), 0.5, 0.1);
        let _ = stepper.integrate();
        let x_out = stepper.x_out();
        let y_out = stepper.y_out();
        assert!((*x_out.last().unwrap() - 0.5).abs() < 1.0E-8);
        assert!((&y_out[3][0] + 0.82246).abs() < 1.0E-5);
        assert!((&y_out[5][0] + 0.81959).abs() < 1.0E-5);
    }

    #[test]
    fn test_integrate_test3_dvector() {
        let system = Test3 {};
        let mut stepper = Rk4::new(system, 0., DVector::from(vec![1.]), 1., 0.1);
        let _ = stepper.integrate();
        let out = stepper.y_out();
        assert!((&out[5][0] - 0.913059839).abs() < 1.0E-9);
        assert!((&out[8][0] - 0.9838057659).abs() < 1.0E-9);
        assert!((&out[10][0] - 1.0715783953).abs() < 1.0E-9);
    }
}
