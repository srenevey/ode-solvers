use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use ode_solvers::{Rk4, System, Vector2};
fn main() {
    let k_cells = 5.7;
    let k_glucose = 1.2;

    let reactor = Bioreactor::new(k_cells, k_glucose);
    let initial = Vector2::new(0., 15.);

    let mut stepper = Rk4::new(reactor, 0., initial, 14., 0.01);

    let res = stepper.mut_integrate();

    if let Ok(val) = res {
        println!("{}", val);
        let path = Path::new("./out.csv");
        save(stepper.x_out(), stepper.y_out(), path);
        println!("Saved in: '{:?}'", path);
    } else {
        println!("an error occured:");
    }
}

struct Bioreactor {
    k_cells: f64,
    k_glucose: f64,
}
impl Bioreactor {
    fn new(k_cells: f64, k_glucose: f64) -> Self {
        Bioreactor { k_cells, k_glucose }
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
type Time = f64;
type State = Vector2<f64>;

impl System<Time, State> for Bioreactor {
    fn system(&self, x: Time, y: &State, dy: &mut State) {}

    fn mut_system(&self, x: Time, y: &mut State, dy: &mut State) {
        let k_c = self.k_cells;
        let k_g = self.k_glucose;

        // y[0] => n cells
        // y[1] => c glucose
        if (x - 2.).abs() < 1.0e-5 {
            // innoculation at day 2.
            y[0] += 0.5;
        }
        if y[1] > 0. {
            dy[1] = -y[0] * k_g;
        } else {
            dy[1] = 0.;
            y[1] = 0.;
        }

        if y[0] > 0. {
            dy[0] = y[0].powf(0.7) * y[1] - k_c;
        } else {
            dy[0] = 0.;
            y[0] = 0.;
        }

        if (x - 5.).abs() < 1.0e-5 {
            // add glucose on day 5
            y[1] = 15.;
        }
    }
}
