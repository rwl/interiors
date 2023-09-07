use crate::common::{Lambda, Options};
use crate::ipm::{dot, nlp};
use crate::traits::{LinearSolver, ObjectiveFunction, ProgressMonitor};
use anyhow::Result;
use itertools::{izip, Itertools};
use sparsetools::csr::CSR;

struct QuadraticObjectiveFunction {
    c: Vec<f64>,
    h_mat: CSR<usize, f64>,
}

impl ObjectiveFunction for QuadraticObjectiveFunction {
    fn f(&self, x: &[f64], hessian: bool) -> (f64, Vec<f64>, Option<CSR<usize, f64>>) {
        let f = 0.5 * dot(&x, &(&self.h_mat * &x)) + dot(&self.c, &x);
        let df = izip!(&self.h_mat * &x, &self.c)
            .map(|(hx, c)| hx + c)
            .collect_vec();

        if !hessian {
            (f, df, None)
        } else {
            let d2f = self.h_mat.to_owned();
            (f, df, Some(d2f))
        }
    }
}

/// Quadratic Program Solver based on an interior point method.
///
/// Solve the following QP (quadratic programming) problem:
///
///       min 1/2 x'*H*x + c'*x
///        x
///
/// subject to
///
///       l <= A*x <= u       (linear constraints)
///       xmin <= x <= xmax   (variable bounds)
pub fn qp(
    h_mat: &CSR<usize, f64>,
    c: &[f64],
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
    let nx: usize = h_mat.rows();

    let qp = QuadraticObjectiveFunction {
        c: c.to_owned(),
        // c: match c {
        //     Some(c) => c.to_owned(),
        //     None => vec![0.0; nx],
        // },
        h_mat: h_mat.to_owned(),
    };
    // let x0 = match x0 {
    //     Some(x0) => x0.to_owned(),
    //     None => vec![0.0; nx],
    // };

    nlp(
        &qp, x0, a_mat, l, u, xmin, xmax, None, linsolve, opt, progress,
    )
}
