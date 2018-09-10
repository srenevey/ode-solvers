# ODE-solvers

Ode-solvers is a toolbox offering several methods to solve ordinary differential equations (ODEs) in Rust. The following instructions should get you up and running in no time.



## Importing the crate

To start using the crate in your project, add the following dependency in your project's Cargo.toml file:

```rust
[dependencies]
ode-solvers = "0.1.0"
```

Then, in your main file, add

```rust
extern crate ode-solvers;
use ode-solvers::*;
```



## Type alias definition

The numerical integration methods implemented in the crate support multi-dimensional systems. In order to define the dimension of the system, declare a type alias for the state vector. For instance

```rust
type State = Vector3<f64>;
```

The state representation of the system is based on the VectorN&lt;T,D&gt; structure defined in the [nalgebra](http://nalgebra.org/) crate. For convenience, ode-solvers re-exports six types to work with systems of dimension 1 to 6: Vector1&lt;T&gt;,..., Vector6&lt;T&gt;. For higher dimensions, the user should import the nalgebra crate and define a VectorN&lt;T,D&gt;  where the second type parameter of VectorN is a dimension name defined in nalgebra. Note that the type T must be f64. For instance, for a 9-dimensional system, one would have:

```rust
extern crate nalgebra as na;
type State = VectorN<f64, na::U9>;
```



## System definition

The first order ODE(s) must be defined in a function with the following signature

```rust
fn system(x: f64, y: &State) -> State
```

where the first argument is the independent variable (usually time) and the second one is a vector containing the dependent variable(s).



## Method selection

The following explicit Runge-Kutta methods are implemented in the current version (0.1.0) of the crate:

| Method         | Name   | Order | Error estimate order | Dense output order |
| -------------- | ------ | ----- | -------------------- | ------------------ |
| Dormand-Prince | Dopri5 | 5     | 4                    | 4                  |
| Dormand-Prince | Dop853 | 8     | (5,3)                | 7                  |

These methods are defined in the modules dopri5 and dop853 and feature: 

- Adaptive step size control
- Automatic initial step size selection
- Sparse or dense output

The first step is to bring the desired module into scope:

```rust
use ode_solvers::dopri5::*;
```

Then, a structure is created using the *new* or the *from_param* method of the corresponding struct. Refer to the API documentation for a description of the input arguments.

```rust
let mut stepper = Dopri5::new(system, x0, x_end, dx, y0, rtol, atol);
```

The system is integrated using

```rust
let res = stepper.integrate();
```

which returns Result&lt;Stats, IntegrationError&gt;. Upon successful completion, res = Ok(Stats) where Stats is a structure containing some information on the integration process. If an error occurs, res = Err(IntegrationError). Finally, the results can be retrieved with

```rust
let x_out = stepper.x_out();
let y_out = stepper.y_out();
```



## Acknowledgments

The algorithms implemented in this crate were originally implemented in FORTRAN by E. Hairer and G. Wanner, Université de Genève, Switzerland. This Rust implementation has been adapted from the C version written by J. Colinge, Université de Genève, Switzerland and the C++ version written by Blake Ashby, Stanford University, USA.