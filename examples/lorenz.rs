// Lorenz attractor

use ode_solvers::dop853::*;
use ode_solvers::*;

use std::{fs::File, io::BufWriter, io::Write, path::Path};

type State = Vector3<f64>;
type Time = f64;

fn main() {
    // Initial state
    let y0 = State::new(1.0, 1.0, 1.0);

    // Define problem specific constants
    let system = LorenzAttractor {
        sigma: 10.,
        beta: 8. / 3.,
        rho: 28.,
    };
    // Create stepper and integrate
    let mut stepper = Dop853::new(system, 0.0, 100.0, 1e-3, y0, 1e-4, 1e-4);
    let res = stepper.integrate();

    // Handle result
    match res {
        Ok(stats) => {
            println!("{}", stats);
            let path = Path::new("./outputs/lorenz_dop853.dat");
            save(stepper.x_out(), stepper.y_out(), path);
            println!("Results saved in: {:?}", path);
        }
        Err(e) => println!("An error occured: {}", e),
    }
}

struct LorenzAttractor {
    sigma: f64,
    beta: f64,
    rho: f64,
}

impl ode_solvers::System<State> for LorenzAttractor {
    fn system(&self, _t: Time, y: &State, dy: &mut State) {
        dy[0] = self.sigma * (y[1] - y[0]);
        dy[1] = y[0] * (self.rho - y[2]) - y[1];
        dy[2] = y[0] * y[1] - self.beta * y[2];
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
