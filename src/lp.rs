use crate::common::{Lambda, Options};
use crate::ipm::nlp;
use crate::traits::{LinearSolver, ObjectiveFunction, ProgressMonitor};
use ndarray::{Array1, ArrayView1};
use sprs::{CsMat, CsMatView};

struct LinearObjectiveFunction {
    nx: usize,
    c: Array1<f64>,
}

impl ObjectiveFunction for LinearObjectiveFunction {
    fn f(&self, x: ArrayView1<f64>, hessian: bool) -> (f64, Array1<f64>, Option<CsMat<f64>>) {
        let f = self.c.dot(&x);
        let df = self.c.to_owned();

        if !hessian {
            (f, df, None)
        } else {
            (f, df, Some(CsMat::zero((self.nx, self.nx))))
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
    let nx = if a_mat.is_some() && a_mat.unwrap().nnz() != 0 {
        a_mat.unwrap().cols()
    } else if let Some(xmin) = xmin.as_ref() {
        xmin.len()
    } else if let Some(xmax) = xmax.as_ref() {
        xmax.len()
    } else {
        return Err("LP problem must include constraints or variable bounds".to_string());
    };

    let lp = LinearObjectiveFunction {
        nx,
        c: match c {
            Some(c) => c.to_owned(),
            None => Array1::zeros(nx),
        },
    };
    let x0 = match x0 {
        Some(x0) => x0.to_owned(),
        None => Array1::zeros(nx),
    };

    nlp(
        &lp,
        x0.view(),
        a_mat,
        l,
        u,
        xmin,
        xmax,
        None,
        Some(linsolve),
        opt,
        progress,
    )
}
