use crate::common::{Lambda, Options};
use crate::ipm::nlp;
use crate::traits::{LinearSolver, ObjectiveFunction, ProgressMonitor};
use ndarray::{Array1, ArrayView1};
use sprs::{CsMat, CsMatView};

struct QuadraticObjectiveFunction {
    c: Array1<f64>,
    h_mat: CsMat<f64>,
}

impl ObjectiveFunction for QuadraticObjectiveFunction {
    fn f(&self, x: ArrayView1<f64>, hessian: bool) -> (f64, Array1<f64>, Option<CsMat<f64>>) {
        let f = 0.5 * x.dot(&(&self.h_mat.view() * &x.view())) + self.c.dot(&x);
        let df = (&self.h_mat * &x) + &self.c;

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
    h_mat: CsMatView<f64>,
    c: Option<ArrayView1<f64>>,
    a_mat: Option<CsMatView<f64>>,
    l: Option<ArrayView1<f64>>,
    u: Option<ArrayView1<f64>>,
    xmin: Option<ArrayView1<f64>>,
    xmax: Option<ArrayView1<f64>>,
    x0: Option<ArrayView1<f64>>,
    linsolve: &dyn LinearSolver,
    progress: Option<&dyn ProgressMonitor>,
    opt: Option<Options>,
) -> Result<(Array1<f64>, f64, bool, usize, Lambda), String> {
    // Define nx, set default values for c and x0.
    let nx: usize = h_mat.rows();

    let qp = QuadraticObjectiveFunction {
        c: match c {
            Some(c) => c.to_owned(),
            None => Array1::zeros(nx),
        },
        h_mat: h_mat.to_owned(),
    };
    let x0 = match x0 {
        Some(x0) => x0.to_owned(),
        None => Array1::zeros(nx),
    };

    nlp(
        &qp,
        x0.view(),
        a_mat,
        l,
        u,
        xmin,
        xmax,
        None,
        linsolve,
        opt,
        progress,
    )
}
