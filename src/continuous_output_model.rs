use crate::dop_shared::FloatNumber;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, OVector};
use serde::{Deserialize, Serialize};

/// Stores the coefficients used to compute the dense output.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContinuousOutputModel<T, V> {
    lower_bound: T,
    breakpoints: Vec<T>,
    intervals: Vec<Interval<T, V>>,
}

impl<T, D: Dim> ContinuousOutputModel<T, OVector<T, D>>
where
    f64: From<T>,
    T: FloatNumber,
    OVector<T, D>: std::ops::Mul<T, Output = OVector<T, D>>,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new empty [`ContinuousOutputModel`].
    pub fn new() -> Self {
        Self {
            lower_bound: T::zero(),
            breakpoints: Vec::new(),
            intervals: Vec::new(),
        }
    }

    /// Evaluates the continuous output at the given value.
    pub fn evaluate(&self, x: T) -> Option<OVector<T, D>> {
        self.get_interval_index(x).map(|index| {
            let lower_bound = if index == 0 {
                self.lower_bound
            } else {
                self.breakpoints[index - 1]
            };
            let interval = &self.intervals[index];
            let theta = (x - lower_bound) / interval.step_size;
            let theta1 = T::one() - theta;

            let coefficients = &interval.coefficients;
            let mut result = coefficients[coefficients.len() - 1].clone();
            for i in (0..coefficients.len() - 1).rev() {
                let multiplier = if i == coefficients.len() - 1 {
                    T::one()
                } else if i % 2 == 0 {
                    theta
                } else {
                    theta1
                };
                result = &coefficients[i] + result * multiplier;
            }
            Some(result)
        })?
    }

    /// Returns the lower and upper bounds of the continuous output model validity range.
    pub fn bounds(&self) -> (T, T) {
        (
            self.lower_bound,
            self.breakpoints[self.breakpoints.len() - 1],
        )
    }

    /// Sets the lower bound of the continuous output.
    pub(crate) fn set_lower_bound(&mut self, lower_bound: T) {
        self.lower_bound = lower_bound;
    }

    /// Adds the coefficients valid up to a given breakpoint to the continuous output model.
    pub(crate) fn add_interval(
        &mut self,
        breakpoint: T,
        coefficients: Vec<OVector<T, D>>,
        step_size: T,
    ) {
        self.breakpoints.push(breakpoint);
        self.intervals.push(Interval {
            coefficients,
            step_size,
        });
    }

    /// Returns the index of the interval containing x.
    fn get_interval_index(&self, x: T) -> Option<usize> {
        if x < self.lower_bound {
            return None;
        }

        match self
            .breakpoints
            .binary_search_by(|probe| probe.partial_cmp(&x).unwrap())
        {
            Ok(index) => Some(index),
            Err(index) => {
                if index < self.intervals.len() {
                    Some(index)
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Interval<T, V> {
    coefficients: Vec<V>,
    step_size: T,
}

#[cfg(test)]
mod tests {
    use crate::continuous_output_model::ContinuousOutputModel;
    use approx::assert_relative_eq;
    use nalgebra::Vector1;

    type State = Vector1<f64>;

    #[test]
    fn test_evaluate_with_odd_number_of_coefficients() {
        let first_breakpoint = 2.0;
        let coefficients_first_interval = vec![State::new(0.2), State::new(1.0), State::new(-2.5)];
        let first_step_size = 0.1;
        let second_breakpoint = 5.0;
        let coefficients_second_interval = vec![State::new(-1.5), State::new(0.4), State::new(1.2)];
        let second_step_size = 0.2;

        let mut continuous_output_model = ContinuousOutputModel::default();
        continuous_output_model.add_interval(
            first_breakpoint,
            coefficients_first_interval,
            first_step_size,
        );
        continuous_output_model.add_interval(
            second_breakpoint,
            coefficients_second_interval,
            second_step_size,
        );

        assert_eq!(continuous_output_model.evaluate(-0.1), None);
        assert_eq!(continuous_output_model.evaluate(5.1), None);

        assert_relative_eq!(
            continuous_output_model.evaluate(0.0).unwrap(),
            State::new(0.2),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            continuous_output_model.evaluate(1.2).unwrap(),
            State::new(342.2),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            continuous_output_model.evaluate(2.0).unwrap(),
            State::new(970.2),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            continuous_output_model.evaluate(2.5).unwrap(),
            State::new(-5.0),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            continuous_output_model.evaluate(5.0).unwrap(),
            State::new(-247.5),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_evaluate_with_even_number_of_coefficients() {
        let first_breakpoint = 2.0;
        let coefficients_first_interval = vec![
            State::new(0.2),
            State::new(1.0),
            State::new(-2.5),
            State::new(0.3),
        ];
        let first_step_size = 0.1;
        let second_breakpoint = 5.0;
        let coefficients_second_interval = vec![
            State::new(-1.5),
            State::new(0.4),
            State::new(1.2),
            State::new(2.7),
        ];
        let second_step_size = 0.2;

        let mut continuous_output_model = ContinuousOutputModel::default();
        continuous_output_model.add_interval(
            first_breakpoint,
            coefficients_first_interval,
            first_step_size,
        );
        continuous_output_model.add_interval(
            second_breakpoint,
            coefficients_second_interval,
            second_step_size,
        );

        assert_relative_eq!(
            continuous_output_model.evaluate(0.0).unwrap(),
            State::new(0.2),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            continuous_output_model.evaluate(1.2).unwrap(),
            State::new(-133.0),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            continuous_output_model.evaluate(2.0).unwrap(),
            State::new(-1309.8),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            continuous_output_model.evaluate(2.5).unwrap(),
            State::new(-30.3125),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            continuous_output_model.evaluate(5.0).unwrap(),
            State::new(-8752.5),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_evaluate_with_no_coefficients() {
        let continuous_output_model: ContinuousOutputModel<f64, State> =
            ContinuousOutputModel::default();
        assert_eq!(continuous_output_model.evaluate(0.0), None);
        assert_eq!(continuous_output_model.evaluate(3.0), None);
    }

    #[test]
    fn test_get_interval_index() {
        let mut continuous_output_model: ContinuousOutputModel<f64, State> =
            ContinuousOutputModel::default();
        continuous_output_model.add_interval(2.0, vec![], 0.1);
        continuous_output_model.add_interval(5.0, vec![], 0.1);

        assert_eq!(continuous_output_model.get_interval_index(-0.001), None);
        assert_eq!(continuous_output_model.get_interval_index(0.0), Some(0));
        assert_eq!(continuous_output_model.get_interval_index(1.3), Some(0));
        assert_eq!(continuous_output_model.get_interval_index(2.0), Some(0));
        assert_eq!(continuous_output_model.get_interval_index(3.2), Some(1));
        assert_eq!(continuous_output_model.get_interval_index(5.0), Some(1));
        assert_eq!(continuous_output_model.get_interval_index(5.0001), None);
    }
}
