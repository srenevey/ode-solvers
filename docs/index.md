# ODEs Solving in Rust

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



##Adaptive Step Size

The adaptive step size is computed using a PI controller

<center>$h_{n+1} = h_ns|err_n|^{-\alpha}|err_{n-1}|^\beta$</center>

where $s$ is a safety factor (usually 0.9), $|err_n|$ is the current error , and $|err_{n-1}|$ is the previous error. The coefficients $\alpha$ and $\beta$ determine the behavior of the controller. In order to make sure that $h$ doesn't change too abruptly, bounds on the factor of $h_n$ are enforced:

<center>$f_{min} \leq s|err_n|^{-\alpha}|err_{n-1}|^\beta \leq f_{max}$</center>

If $s|err_n|^{-\alpha}|err_{n-1}|^\beta \leq f_{min}$, then $h_{n+1} = h_nf_{min}$ and if $s|err_n|^{-\alpha}|err_{n-1}|^\beta \geq f_{max}$, $h_{n+1} = h_nf_{max}$.

The default values of these parameters are listed in the table below.

|           | Dopri5 | Dopri853 |
| --------- | ------ | -------- |
| $\alpha$  | 0.17   | 1/8      |
| $\beta$   | 0.04   | 0        |
| $f_{min}$ | 0.2    | 0.333    |
| $f_{max}$ | 10.0   | 6.0      |
| $s$       | 0.9    | 0.9      |

These default values can be overridden by using the method *from_param*.



For Dopri5, a 5th order approximation $y_1$ and a 4th order approximation $\hat{y}_1$ are constructed using the coefficients of the method. The error estimator is then

<center>$err=\sqrt{\frac{1}{n}\sum_{i=1}^n\frac{y_{1i}-\hat{y}_{1i}}{sc_i}}$</center>

where $n$ is the dimension of the system and $sc_i$ is a scale factor given by

<center>$sc_i=a_{tol}+r_{tol}\max{(|y_{0i}|,|y_{1i}|)}$</center>

For Dop853, an 8th order approximation $y_1$, a 5th order approximation $\hat{y}_1$, and a 3rd order approximation $\tilde{y}_1$ are constructed using the coefficients of the method. Two error estimators are computed:

<center>$err_{5}=\sqrt{\frac{1}{n}\sum_{i=1}^n\frac{y_{1i}-\hat{y}_{1i}}{sc_i}}$ &nbsp; and &nbsp;  $err_{3}=\sqrt{\frac{1}{n}\sum_{i=1}^n\frac{y_{1i}-\tilde{y}_{1i}}{sc_i}}$</center>

which are combined into a single error estimator

<center>$err=err_5\frac{err_5}{\sqrt{err_5^2+0.01err_3^2}}$</center>

For a thorough discussion, see Hairer et al. [1].


## References and Acknowledgments

The algorithms implemented in this crate are described in references [1] and [2]. They were originally implemented in FORTRAN by E. Hairer and G. Wanner, Université de Genève, Switzerland. This Rust implementation has been adapted from the C version written by J. Colinge, Université de Genève, Switzerland and the C++ version written by Blake Ashby, Stanford University, USA.



[1] Hairer, E., Nørsett, S.P., Wanner, G., *Solving Ordinary Differential Equations I, Nonstiff Problems*, Second Revised Edition, Springer, 2008

[2] Hairer, E., Wanner, G., *Solving Ordinary Differential Equations II, Stiff and Differential-Algebraic Problems*, Second Revised Edition, Springer, 2002

[3] Butcher, J.C., *Numerical Methods for Ordinary Differential Equations*, Third Edition, John Wiley and Sons, 2016

[4] Gustafsson, K., Control-Theoretic Techniques for Stepsize Selection in Explicit Runge-Kutta Methods. *ACM Transactions on Mathematical Software*, 17, 4 (December 1991), 533-554.

[5] Gustafsson, K., Control-Theoretic Techniques for Stepsize Selection in Implicit Runge-Kutta Methods. *ACM Transactions on Mathematical Software*, 20, 4 (December 1994), 496-517. 

[6] Söderlind, G., Automatic Control and Adaptive Time-Stepping, *Numerical Algorithms* (2002) 31: 281-310.