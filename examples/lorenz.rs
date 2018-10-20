// Lorenz attractor

extern crate ode_solvers;
use ode_solvers::*;
use ode_solvers::dop853::*;

use std::io::prelude::*;
use std::fs::File;
use std::path::Path;

type State = Vector3<f64>;
type Time = f64;

// Define problem specific constants
const SIG: f64 = 10.0;
const BETA: f64 = 8.0/3.0;
const RHO: f64 = 28.0;

fn main() {
    // Initial state
	let y0 = State::new(1.0, 1.0, 1.0);

    // Create stepper and integrate
    let mut stepper = Dop853::new(system, 0.0, 100.0, 1e-3, y0, 1e-4, 1e-4);
    let res = stepper.integrate();

    // Handle result
    match res {
        Ok(stats) => {
            stats.print();
            let path = Path::new("./outputs/lorenz_dop853.dat");
            save(stepper.x_out(), stepper.y_out(), path);  
            println!("Results saved in: {:?}", path);
        },
        Err(_) => println!("An error occured."),
    }
}

fn system(_t: Time, y: &State, dy: &mut State) {
    dy[0] = SIG*(y[1]-y[0]);
    dy[1] = y[0]*(RHO-y[2]) - y[1];
    dy[2] = y[0]*y[1] - BETA*y[2];
}

pub fn save(times: &Vec<Time>, states: &Vec<State>, filename: &Path) {
    // Create or open file
    let mut buf = match File::create(filename) {
        Err(e) => {
            println!("Could not open file. Error: {:?}", e);
            return;
        },
        Ok(buf) => buf,
    };

    // Write time and state vector in a csv format
    for (i,state) in states.iter().enumerate() {
        buf.write_fmt(format_args!("{}", times[i])).unwrap();
        for val in state.iter() {
            buf.write_fmt(format_args!(", {}", val)).unwrap();
        }
        buf.write_fmt(format_args!("\n")).unwrap();
    }
}