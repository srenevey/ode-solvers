use ode_solvers::dop853::*;
use ode_solvers::*;

// Define type aliases for the state and time types
type State = DVector<f64>;
type Time = f64;

use std::{fs::File, io::BufWriter, io::Write, path::Path};

fn main() {
    // Create the structure containing the problem specific constant and equations.
    let system = ThreeBodyProblem {
        mu: 0.012300118882173,
    };

    // Initial state.
    let y0 = State::from(vec![-0.271, -0.42, 0.0, 0.3, -1.0, 0.0]);

    // Create a stepper and run the integration.
    let mut stepper = Dop853::new(system, 0.0, 150.0, 0.002, y0, 1.0e-14, 1.0e-14);
    let res = stepper.integrate();

    // Handle result
    match res {
        Ok(stats) => {
            println!("{}", stats);
            let path = Path::new("./outputs/three_body_dop853_dvector.dat");
            save(stepper.x_out(), stepper.y_out(), path);
            println!("Results saved in: {:?}", path);
        }
        Err(e) => println!("An error occured: {}", e),
    }
}

struct ThreeBodyProblem {
    mu: f64,
}

impl ode_solvers::System<State> for ThreeBodyProblem {
    fn system(&self, _t: Time, y: &State, dy: &mut State) {
        let d = ((y[0] + self.mu).powi(2) + y[1].powi(2) + y[2].powi(2)).sqrt();
        let r = ((y[0] - 1.0 + self.mu).powi(2) + y[1].powi(2) + y[2].powi(2)).sqrt();

        dy[0] = y[3];
        dy[1] = y[4];
        dy[2] = y[5];
        dy[3] = y[0] + 2.0 * y[4]
            - (1.0 - self.mu) * (y[0] + self.mu) / d.powi(3)
            - self.mu * (y[0] - 1.0 + self.mu) / r.powi(3);
        dy[4] =
            -2.0 * y[3] + y[1] - (1.0 - self.mu) * y[1] / d.powi(3) - self.mu * y[1] / r.powi(3);
        dy[5] = -(1.0 - self.mu) * y[2] / d.powi(3) - self.mu * y[2] / r.powi(3);
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
