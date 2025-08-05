use crate::dop_shared::FloatNumber;

/// Stiffness detection thresholds for different methods
pub mod stiffness {
    use super::FloatNumber;

    /// Stiffness threshold for DOPRI5 method
    /// Based on Hairer & Wanner, "Solving Ordinary Differential Equations II"
    pub fn dopri5_threshold<T: FloatNumber>() -> T {
        T::from(3.25).unwrap()
    }

    /// Stiffness threshold for DOP853 method
    /// Based on Hairer & Wanner, "Solving Ordinary Differential Equations II"
    pub fn dop853_threshold<T: FloatNumber>() -> T {
        T::from(6.1).unwrap()
    }

    /// Maximum consecutive stiffness detections before error
    pub const MAX_STIFF_ITERATIONS: u32 = 15;

    /// Number of non-stiff steps needed to reset stiffness counter
    pub const NON_STIFF_RESET_COUNT: u32 = 6;
}

/// Initial step size computation constants
pub mod initial_step {
    use super::FloatNumber;

    /// Minimum tolerance for initial step computation
    pub fn min_tolerance<T: FloatNumber>() -> T {
        T::from(1.0e-10).unwrap()
    }

    /// Default initial step when tolerance conditions not met
    pub fn default_initial_step<T: FloatNumber>() -> T {
        T::from(1.0e-6).unwrap()
    }

    /// Safety factor for initial step estimation
    pub fn safety_factor<T: FloatNumber>() -> T {
        T::from(0.01).unwrap()
    }
}

/// Dense output interpolation constants
pub mod dense_output {
    use super::FloatNumber;

    /// Floating point tolerance for endpoint detection
    pub fn endpoint_tolerance<T: FloatNumber>() -> T {
        T::from(1.0e-9).unwrap()
    }
}
