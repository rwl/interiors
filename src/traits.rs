use crate::common::Lambda;
use sparsetools::csr::CSR;

pub trait ObjectiveFunction {
    /// Evaluates the objective function, its gradients and Hessian for a given value of `x`.
    fn f(&self, x: &[f64], hessian: bool) -> (f64, Vec<f64>, Option<CSR<usize, f64>>);
}

pub trait NonlinearConstraint {
    /// Evaluates the optional nonlinear constraints and their gradients
    /// for a given value of `x`.
    ///
    /// The columns of `dh` and `dg` are the gradients of the corresponding
    /// elements of `h` and `g`, i.e. `dh` and `dg` are transposes of the
    /// Jacobians of `h` and `g`, respectively.
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

    /// Computes the Hessian of the Lagrangian for given values of
    /// `x`, `lambda` and `mu`, where `lambda` and `mu` are the multipliers
    /// on the equality and inequality constraints, `g` and `h`, respectively.
    fn hess(&self, x: &[f64], lam: &Lambda, cost_mult: f64) -> CSR<usize, f64>;
}

/// Called on each iteration of the solver with the current
/// iteration number, feasibility condition, gradient condition,
/// complementarity condition, cost condition, barrier coefficient,
/// step size, objective function value (scaled by the cost multiplier)
/// and the two update parameters.
pub trait ProgressMonitor {
    fn update(
        &self,
        i: usize,
        feas_cond: f64,
        grad_cond: f64,
        comp_cond: f64,
        cost_cond: f64,
        _gamma: f64,
        step_size: f64,
        obj: f64,
        _alpha_p: f64,
        _alpha_d: f64,
    ) {
        if i == 0 {
            println!(
                " it    objective   step size   feascond     gradcond     compcond     costcond  "
            );
            println!(
                "----  ------------ --------- ------------ ------------ ------------ ------------"
            );
        }
        println!(
            "{:3}    {:10.5} {:10.5} {:12.8} {:12.8} {:12.8} {:12.8}",
            i, obj, step_size, feas_cond, grad_cond, comp_cond, cost_cond
        );
    }
}
