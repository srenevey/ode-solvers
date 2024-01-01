// A bouncing ball example
// This shows how to couple multiple calls of the ODE solver and move the ownership of the results to a custom data structure
// This also useful for sequential simulation, e.g. for reactor cascades.

use std::{fs::File, io::BufWriter, io::Write, path::Path};

use ode_solvers::dop_shared::SolverResult;
use ode_solvers::*;

type State = Vector3<f64>; // stores location, velocity and time
type Time = f64;
type Result = SolverResult<State>;

const G: f64 = 9.81; // gravity constant on earth
const BOUNCE: f64 = 0.75;
const MAX_BOUNCES: u32 = 5;

fn main() {
    // Initial state: At 10m with zero velocity
    let mut y0 = State::new(10.0, 0., 0.);
    let mut num_bounces = 0;
    let mut combined_solver_results = Result::default();

    while num_bounces < MAX_BOUNCES && y0[0] >= 0.001 {
        // Create the structure containing the ODEs.
        let system = BouncingBall;

        // Create a stepper and run the integration.
        // Use comments to see differences with Dopri
        //let mut stepper = Dopri5::new(system, 0., 10.0, 0.01, y0, 1.0e-2, 1.0e-6);
        let mut stepper = Rk4::new(system, 0.0, y0, 10.0, 0.01);
        let res = stepper.integrate();

        // Handle result.
        match res {
            Ok(stats) => println!("{}", stats),
            Err(e) => println!("An error occured: {}", e),
        }

        num_bounces = num_bounces + 1;

        // solout may not be called and therefore end not "smooth" when observing dense values with dopri5 or dop853
        // therefore we seach for the point where the results turn zero
        let (_, y_out) = stepper.results().get();
        let f = y_out.iter().find(|y| y[0] <= 0.);
        if f.is_none() {
            // that should not happen...
            break;
        }

        let last_state = f.unwrap();
        println!("Last state: {:?}\n\n", last_state);

        y0[0] = last_state[0].abs();
        y0[1] = -1. * last_state[1] * BOUNCE;

        // beware in the case of dopri5 or dop853 the results contain a lot of invalid data with y[0] < 0
        combined_solver_results.append(stepper.into());
    }

    let path = Path::new("./outputs/bouncing_ball.dat");

    save(
        combined_solver_results.get().0,
        combined_solver_results.get().1,
        path,
    );
}

struct BouncingBall;

impl ode_solvers::System<State> for BouncingBall {
    fn system(&self, _t: Time, y: &State, dy: &mut State) {
        dy[0] = y[1]; // location is changed by v
        dy[1] = -G; // v is changed by acc of gravity
    }

    fn solout(&mut self, _x: f64, y: &State, _dy: &State) -> bool {
        y[0] < 0.
    }
}

pub fn save(times: &Vec<Time>, states: &Vec<State>, filename: &Path) {
    // Create or open file
    let file = match File::create(filename) {
        Err(e) => {
            println!("Could not open file. Error: {:?}", e);
            return;
        }
        Ok(buf) => buf,
    };
    let mut buf = BufWriter::new(file);

    // Write time and state vector in a csv format
    for (i, state) in states.iter().enumerate() {
        buf.write_fmt(format_args!("{}", times[i])).unwrap();
        for val in state.iter() {
            buf.write_fmt(format_args!(", {}", val)).unwrap();
        }
        buf.write_fmt(format_args!("\n")).unwrap();
    }
    if let Err(e) = buf.flush() {
        println!("Could not write to file. Error: {:?}", e);
    }
}