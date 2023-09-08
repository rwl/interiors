use crate::common::Lambda;
use sparsetools::csr::CSR;

pub trait ObjectiveFunction {
    fn f(&self, x: &[f64], hessian: bool) -> (f64, Vec<f64>, Option<CSR<usize, f64>>);
}

pub trait NonlinearConstraint {
    fn gh(
        &self,
        x: &[f64],
        gradients: bool,
    ) -> (
        Vec<f64>,
        Vec<f64>,
        Option<CSR<usize, f64>>,
        Option<CSR<usize, f64>>,
    );

    fn hess(&self, x: &[f64], lam: &Lambda, cost_mult: f64) -> CSR<usize, f64>;
}

/// Called on each iteration of the solver with the current
/// iteration number, feasibility condition, gradient condition,
/// complementarity condition, cost condition, barrier coefficient,
/// step size, objective function value and the two update parameters.
pub trait ProgressMonitor {
    fn update(
        &self,
        i: usize,
        feas_cond: f64,
        grad_cond: f64,
        comp_cond: f64,
        cost_cond: f64,
        gamma: f64,
        step_size: f64,
        obj: f64,
        alpha_p: f64,
        alpha_d: f64,
    );
}

// pub(crate) trait Norm {
//     /// Returns the 2-norm (Euclidean) of a.
//     fn norm(&self) -> f64;
// }
//
// impl Norm for Array1<f64> {
//     fn norm(&self) -> f64 {
//         self.iter().map(|&v| v * v).sum::<f64>().sqrt()
//     }
// }
//
// pub(crate) trait Max {
//     fn maximum(&self) -> f64;
// }
//
// impl Max for Arr<f64> {
//     fn maximum(&self) -> f64 {
//         *self
//             .iter()
//             .max_by(|&a, &b| a.partial_cmp(b).unwrap())
//             .unwrap()
//     }
// }
//
// pub(crate) trait Min {
//     fn minimum(&self) -> f64;
// }
//
// impl Min for Vec<f64> {
//     fn minimum(&self) -> f64 {
//         *self
//             .iter()
//             .min_by(|&a, &b| a.partial_cmp(b).unwrap())
//             .unwrap()
//     }
// }
//
// pub(crate) trait NormInf {
//     fn norm_inf(&self) -> f64;
// }
//
// impl NormInf for Array1<f64> {
//     fn norm_inf(&self) -> f64 {
//         self.maximum().abs()
//     }
// }
//
// pub(crate) trait LnSum {
//     fn ln_sum(&self) -> f64;
// }
//
// impl LnSum for Array1<f64> {
//     fn ln_sum(&self) -> f64 {
//         self.iter().map(|v| v.ln()).sum::<f64>()
//     }
// }
