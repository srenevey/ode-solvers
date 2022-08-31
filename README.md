# ODE-solvers

[![Crates.io](https://img.shields.io/crates/v/ode_solvers.svg)](https://crates.io/crates/ode_solvers/) [![Docs](https://docs.rs/ode_solvers/badge.svg)](https://docs.rs/ode_solvers) [![Crates.io](https://img.shields.io/crates/l/ode_solvers.svg)](https://opensource.org/licenses/BSD-3-Clause)

[Homepage](https://srenevey.github.io/ode-solvers/)    [Documentation](https://docs.rs/ode_solvers)

Numerical methods to solve ordinary differential equations (ODEs) in Rust.



## Installation

To start using the crate in a project, the following dependency must be added in the project's Cargo.toml file:

```rust
[dependencies]
ode_solvers = "0.3.7"
```

Then, in the main file, add

```rust
use ode_solvers::*;
```



## Type alias definition

The numerical integration methods implemented in the crate support multi-dimensional systems. In order to define the dimension of the system, declare a type alias for the state vector. For instance

```rust
type State = Vector3<f64>;
```

The state representation of the system is based on the SVector&lt;T,D&gt; structure defined in the [nalgebra](https://nalgebra.org/) crate. For convenience, ode-solvers re-exports six types to work with systems of dimension 1 to 6: Vector1&lt;T&gt;,..., Vector6&lt;T&gt;. For higher dimensions, the user should import the nalgebra crate and define a SVector&lt;T,D&gt;  where the second type parameter of SVector is a dimension. Note that the type T must be f64. For instance, for a 9-dimensional system, one would have:

```rust
type State = SVector<f64, 9>;
```

Alternativly, one can also use the DVector&lt;T&gt; structure from the [nalgebra](https://nalgebra.org/) crate as the state representation. When using a DVector&lt;T&gt;, the number of rows in the DVector&lt;T&gt; defines the dimension of the system.

```rust
type State = DVector<f64>;
```


## System definition

The system of first order ODEs must be defined in the `system` method of the `System<V>` trait. Typically, this trait is defined for a structure containing some parameters of the model. The signature of the `System<V>` trait is:

```rust
pub trait System<V> {
    fn system(&self, x: f64, y: &V, dy: &mut V);
    fn solout(&self, _x: f64, _y: &V, _dy: &V) -> bool {
        false
    }
}
```

where `system` must contain the ODEs: the second argument is the independent variable (usually time), the third one is a vector containing the dependent variable(s), and the fourth one contains the derivative(s) of y with respect to x. The method `solout` is called after each successful integration step and stops the integration whenever it is evaluated as true. The implementation of that method is optional. See the examples for implementation details.



## Method selection

The following explicit Runge-Kutta methods are implemented in the current version of the crate:

| Method         | Name   | Order | Error estimate order | Dense output order |
| -------------- | ------ | ----- | -------------------- | ------------------ |
| Runge-Kutta 4  | Rk4    | 4     | N/A                  | N/A                |
| Dormand-Prince | Dopri5 | 5     | 4                    | 4                  |
| Dormand-Prince | Dop853 | 8     | (5, 3)               | 7                  |

These methods are defined in the modules rk4, dopri5, and dop853. The first step is to bring the desired module into scope:

```rust
use ode_solvers::dopri5::*;
```

Then, a structure is created using the `new` or the `from_param` method of the corresponding struct. Refer to the [API documentation](https://docs.rs/ode_solvers) for a description of the input arguments.

```rust
let mut stepper = Dopri5::new(system, x0, x_end, dx, y0, rtol, atol);
```

The system is integrated using

```rust
let res = stepper.integrate();
```

and the results are retrieved with

```rust
let x_out = stepper.x_out();
let y_out = stepper.y_out();
```

See the [homepage](https://srenevey.github.io/ode-solvers/) for more details.

## Acknowledgments

The Dopri5 and Dop853 algorithms implemented in this crate were originally implemented in FORTRAN by E. Hairer and G. Wanner, Université de Genève, Switzerland. This Rust implementation has been adapted from the C version written by J. Colinge, Université de Genève, Switzerland and the C++ version written by Blake Ashby, Stanford University, USA.
