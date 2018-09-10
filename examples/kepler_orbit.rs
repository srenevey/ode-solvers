extern crate ode_solvers;
use ode_solvers::*;
use ode_solvers::dopri5::*;

type State = Vector6<f64>;
type Time = f64;

use std::io::prelude::*;
use std::fs::File;
use std::path::Path;
use std::f64::consts::PI;

const MU: f64 = 398600.435436;

fn main() {
    let a: f64 = 20000.0;
    let period = 2.0*PI*(a.powi(3)/MU).sqrt();

    // Orbit with: a = 20000km, e = 0.7, i = 35 deg, raan = 100 deg, arg_per = 65 deg, true_an = 30 deg
    let y0 = State::new(-5007.248417988539, -1444.918140151374, 3628.534606178356, 0.717716656891, -10.224093784269, 0.748229399696);

    let mut stepper = Dopri5::new(system, 0.0, 5.0 * period, 60.0, y0, 1.0e-10, 1.0e-10);
    let res = stepper.integrate();
    
    // Handle result
    match res {
        Ok(stats) => {
            stats.print();
            let path = Path::new("./outputs/kepler_orbit_dopri5.dat");
            save(stepper.x_out(), stepper.y_out(), path);  
            println!("Results saved in: {:?}", path);
        },
        Err(_) => println!("An error occured."),
    }
    
}

/// Equations of motion of the system
fn system(_t: Time, y: &State) -> State {
    let r = (y[0]*y[0] + y[1]*y[1] + y[2]*y[2]).sqrt() ;
    let mut dy = State::zeros();
    dy[0] = y[3];
    dy[1] = y[4];
    dy[2] = y[5];
    dy[3] = -MU*y[0]/r.powi(3);
    dy[4] = -MU*y[1]/r.powi(3);
    dy[5] = -MU*y[2]/r.powi(3);
    dy
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