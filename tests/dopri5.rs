use approx::assert_relative_eq;
use nalgebra::Vector1;
use ode_solvers::continuous_output_model::ContinuousOutputModel;
use ode_solvers::{Dopri5, System};

const INTEGRATION_TOLERANCE: f64 = 1e-10;
const STEP_SIZE: f64 = 0.1;

type State = Vector1<f64>;
type Time = f64;

struct Ode;

// Implement the ODE dy/dt = -3 * t^2 * exp(y)
impl System<f64, State> for Ode {
    fn system(&self, t: Time, y: &State, dy: &mut State) {
        dy[0] = -3.0 * t * t * y[0].exp();
    }
}

struct OdeWithStoppingCondition;

impl System<f64, State> for OdeWithStoppingCondition {
    fn system(&self, t: Time, y: &State, dy: &mut State) {
        dy[0] = -3.0 * t * t * y[0].exp();
    }

    fn solout(&mut self, x: f64, _y: &State, _dy: &State) -> bool {
        x >= 4.1
    }
}

#[test]
fn test_dopri5() {
    let f = Ode {};
    let initial_value = State::new(0.0);
    let x_start = 0.0;
    let x_end = 6.0;
    let mut stepper = Dopri5::new(
        f,
        x_start,
        x_end,
        STEP_SIZE,
        initial_value,
        INTEGRATION_TOLERANCE,
        INTEGRATION_TOLERANCE,
    );
    let _ = stepper.integrate();

    let (times_s, states) = stepper.results().get();
    assert_eq!(times_s.len(), 61);
    assert_eq!(states.len(), 61);

    times_s
        .iter()
        .zip(states.iter())
        .for_each(|(time_s, state)| {
            let analytic = analytic_solution(*time_s);
            assert_relative_eq!(*state, analytic, epsilon = 1e-8);
        });
}

#[test]
fn test_dopri5_with_continuous_output_model() {
    let f = Ode {};
    let initial_value = State::new(0.0);
    let x_start = 0.0;
    let x_end = 6.0;
    let mut stepper = Dopri5::new(
        f,
        x_start,
        x_end,
        STEP_SIZE,
        initial_value,
        INTEGRATION_TOLERANCE,
        INTEGRATION_TOLERANCE,
    );
    let mut continuous_output_model = ContinuousOutputModel::default();
    let _ = stepper.integrate_with_continuous_output_model(&mut continuous_output_model);

    assert!(continuous_output_model.evaluate(-0.0001).is_none());
    assert!(continuous_output_model.evaluate(0.0).is_some());
    assert!(continuous_output_model.evaluate(6.0).is_some());
    assert!(continuous_output_model.evaluate(6.0001).is_none());

    let num_samples = 100;
    let step_size = (x_end - x_start) / num_samples as f64;
    for i in 0..num_samples {
        let t = i as f64 * step_size + x_start;
        let continuous_output = continuous_output_model.evaluate(t);
        let analytic = analytic_solution(t);
        assert!(continuous_output.is_some());
        assert_relative_eq!(continuous_output.unwrap(), analytic, epsilon = 1e-9);
    }
}

#[test]
fn test_dopri5_with_continuous_output_model_and_stopping_condition() {
    let f = OdeWithStoppingCondition {};
    let initial_value = State::new(0.0);
    let x_start = 0.0;
    let x_end = 6.0;
    let mut stepper = Dopri5::new(
        f,
        x_start,
        x_end,
        STEP_SIZE,
        initial_value,
        INTEGRATION_TOLERANCE,
        INTEGRATION_TOLERANCE,
    );
    let mut continuous_output_model = ContinuousOutputModel::default();
    let _ = stepper.integrate_with_continuous_output_model(&mut continuous_output_model);

    assert!(continuous_output_model.evaluate(-0.0001).is_none());
    assert!(continuous_output_model.evaluate(0.0).is_some());
    assert!(continuous_output_model.evaluate(4.1).is_some());
    assert!(continuous_output_model.evaluate(4.1 + STEP_SIZE).is_none());
    assert!(continuous_output_model.bounds().1 >= 4.1);
    assert!(continuous_output_model.bounds().1 < 4.1 + STEP_SIZE);

    let num_samples = 100;
    let step_size = (x_end - 4.1) / num_samples as f64;
    for i in 0..num_samples {
        let t = i as f64 * step_size + x_start;
        let continuous_output = continuous_output_model.evaluate(t);
        let analytic = analytic_solution(t);
        assert!(continuous_output.is_some());
        assert_relative_eq!(continuous_output.unwrap(), analytic, epsilon = 1e-9);
    }
}

/// Evaluates the analytic solution of the ODE at the given time.
fn analytic_solution(t: Time) -> State {
    State::new(-(t.powi(3) + 1.0).ln())
}
