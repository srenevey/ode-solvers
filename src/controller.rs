//! Adaptive step size control.
use crate::dop_shared::FloatNumber;

/// Used for adaptive step size control
pub struct Controller<T> {
    alpha: T,
    beta: T,
    facc1: T,
    facc2: T,
    fac_old: T,
    h_max: T,
    reject: bool,
    safety_factor: T,
    posneg: T,
}

impl<T: FloatNumber> Controller<T> {
    /// Creates a controller responsible for adaptive step size control.
    ///
    /// # Arguments
    ///
    /// * `alpha`   - &#945; coefficient of the PI controller
    /// * `beta`    - &#946; coefficient of the PI controller
    /// * `fac_max` - Maximum factor between two successive steps
    /// * `fac_min` - Minimum factor between two successive steps
    /// * `h_max`   - Maximum step size
    /// * `safety_factor`   - Safety factor of the PI controller
    ///
    pub fn new(
        alpha: T,
        beta: T,
        fac_max: T,
        fac_min: T,
        h_max: T,
        safety_factor: T,
        posneg: T,
    ) -> Controller<T> {
        Controller {
            alpha,
            beta,
            facc1: T::one() / fac_min,
            facc2: T::from(1.0).unwrap() / fac_max,
            fac_old: T::from(1.0E-4).unwrap(),
            h_max: h_max.abs(),
            reject: false,
            safety_factor,
            posneg,
        }
    }

    /// Determines if the step must be accepted or rejected and adapts the step size accordingly.
    pub fn accept(&mut self, err: T, h: T, h_new: &mut T) -> bool {
        let fac11 = err.powf(self.alpha);
        let mut fac = fac11 * self.fac_old.powf(-self.beta);
        fac = (self.facc2).max((self.facc1).min(fac / self.safety_factor));
        *h_new = h / fac;

        if err <= T::from(1.0).unwrap() {
            // Accept step
            self.fac_old = err.max(T::from(1.0E-4).unwrap());

            if h_new.abs() > self.h_max {
                *h_new = self.posneg * self.h_max;
            }
            if self.reject {
                *h_new = self.posneg * h_new.abs().min(h.abs());
            }

            self.reject = false;
            true
        } else {
            // Reject step
            *h_new = h / ((self.facc1).min(fac11 / self.safety_factor));
            self.reject = true;
            false
        }
    }

    /// Returns the maximum step size allowed.
    pub fn h_max(&self) -> T {
        self.h_max
    }
}
