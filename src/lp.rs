use crate::common::{Lambda, Options};
use crate::ipm::{dot, nlp};
use crate::traits::{LinearSolver, ObjectiveFunction, ProgressMonitor};
use anyhow::Result;
use sparsetools::csr::CSR;

struct LinearObjectiveFunction {
    nx: usize,
    c: Vec<f64>,
}

impl ObjectiveFunction for LinearObjectiveFunction {
    fn f(&self, x: &[f64], hessian: bool) -> (f64, Vec<f64>, Option<CSR<usize, f64>>) {
        let f = dot(&self.c, &x);
        let df = self.c.to_owned();

        if !hessian {
            (f, df, None)
        } else {
            (f, df, Some(CSR::zeros(self.nx, self.nx)))
        }
    }
}

/// Linear Program solver based on an interior point method.
///
/// Solve the following LP (linear programming) problem:
///
///       min c'*x
///        x
///
/// subject to
///
///       l <= A*x <= u       (linear constraints)
///       xmin <= x <= xmax   (variable bounds)
pub fn lp(
    c: Option<&Vec<f64>>,
    a_mat: &CSR<usize, f64>,
    l: &[f64],
    u: &[f64],
    xmin: &[f64],
    xmax: &[f64],
    x0: &[f64],
    linsolve: &dyn LinearSolver,
    progress: Option<&dyn ProgressMonitor>,
    opt: &Options,
) -> Result<(Vec<f64>, f64, bool, usize, Lambda)> {
    // Define nx, set default values for c and x0.
    let nx = a_mat.cols();
    // let nx = if a_mat.is_some() && a_mat.unwrap().nnz() != 0 {
    //     a_mat.unwrap().cols()
    // } else if let Some(xmin) = xmin.as_ref() {
    //     xmin.len()
    // } else if let Some(xmax) = xmax.as_ref() {
    //     xmax.len()
    // } else {
    //     return Err(format_err!(
    //         "LP problem must include constraints or variable bounds"
    //     ));
    // };

    let lp = LinearObjectiveFunction {
        nx,
        c: match c {
            Some(c) => c.to_owned(),
            None => vec![0.0; nx],
        },
    };

    nlp(
        &lp, &x0, a_mat, l, u, xmin, xmax, None, linsolve, opt, progress,
    )
}
