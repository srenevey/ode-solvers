// Chemical reaction of Robertson.
// This ode is stiff and is used to test the automatic stiffness detection in dopri5 and/or dop853.

use ode_solvers::dop853::*;
use ode_solvers::*;

type State = Vector3<f64>;
type Time = f64;

fn main() {
    // Initial state.
    let y0 = State::new(1.0, 0.0, 0.0);

    // Create the structure containing the ODEs.
    let system = ChemicalReaction;

    // Create a stepper and run the integration.
    let mut stepper = Dop853::new(system, 0., 0.3, 0.3, y0, 1.0e-2, 1.0e-6);
    let res = stepper.integrate();

    // Handle result.
    match res {
        Ok(stats) => println!("{}", stats),
        Err(e) => println!("An error occured: {}", e),
    }
}

struct ChemicalReaction;

impl ode_solvers::System<State> for ChemicalReaction {
    fn system(&self, _: Time, y: &State, dy: &mut State) {
        dy[0] = -0.04 * y[0] + 10000. * y[1] * y[2];
        dy[1] = 0.04 * y[0] - 10000. * y[1] * y[2] - 3. * 10_f64.powi(7) * y[1] * y[1];
        dy[2] = 3. * 10_f64.powi(7) * y[1] * y[1];
    }
}
