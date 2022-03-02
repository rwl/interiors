use crate::common::Lambda;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};
use sprs::{CsMat, CsMatView};

pub trait ObjectiveFunction {
    fn f(&self, x: ArrayView1<f64>, hessian: bool) -> (f64, Array1<f64>, Option<CsMat<f64>>);
}

pub trait NonlinearConstraint {
    fn gh(
        &self,
        x: ArrayView1<f64>,
        gradients: bool,
    ) -> (Array1<f64>, Array1<f64>, CsMat<f64>, CsMat<f64>);

    fn hess(&self, x: ArrayView1<f64>, lam: &Lambda, cost_mult: f64) -> CsMat<f64>;
}

pub trait LinearSolver {
    fn solve(&self, a_mat: CsMatView<f64>, b: ArrayViewMut1<f64>) -> Result<Array1<f64>, String>;
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
