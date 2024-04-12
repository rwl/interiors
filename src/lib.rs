//! This crate solves non-linear programming problems (NLPs) using a
//! primal-dual interior point method. It is based on the
//! [MATPOWER Interior Point Solver (MIPS)][1].
//!
//! We request that publications derived from the use of this crate
//! explicitly acknowledge the MATPOWER Interior Point Solver (MIPS)
//! by citing the following 2007 paper.
//!
//! >   H. Wang, C. E. Murillo-Sánchez, R. D. Zimmerman, R. J. Thomas, "On
//!     Computational Issues of Market-Based Optimal Power Flow," *Power Systems,
//!     IEEE Transactions on*, vol. 22, no. 3, pp. 1185-1193, Aug. 2007.
//!     doi: [10.1109/TPWRS.2007.901301](https://doi.org/10.1109/TPWRS.2007.901301)
//!
//! [1]: https://github.com/MATPOWER/mips

mod common;
mod ipm;
mod lp;
mod math;
mod qp;
#[cfg(test)]
mod tests;
mod traits;

pub use common::*;
pub use ipm::nlp;
pub use lp::lp;
pub use qp::qp;
pub use traits::*;
