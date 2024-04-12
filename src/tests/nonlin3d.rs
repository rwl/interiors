use std::iter::zip;

use full::{Arr, Mat};
use sparsetools::csr::CSR;
use spsolve::rlu::RLU;

use crate::math::dot;
use crate::{nlp, Lambda, NonlinearConstraint, ObjectiveFunction, Options};

struct Constrained3DNonlinear {}

impl ObjectiveFunction for Constrained3DNonlinear {
    fn f(&self, x: &[f64], hessian: bool) -> (f64, Vec<f64>, Option<CSR<usize, f64>>) {
        let f = -x[0] * x[1] - x[1] * x[2];
        let df = vec![-x[1], -(x[0] + x[2]), -x[1]];
        if !hessian {
            (f, df, None)
        } else {
            let d2f = CSR::from_dense(
                // arr2(&[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).view() * -1.0,
                &[
                    vec![0.0, 1.0, 0.0],
                    vec![1.0, 0.0, 1.0],
                    vec![0.0, 1.0, 0.0],
                ],
            );
            (f, df, Some(d2f))
        }
    }
}

impl NonlinearConstraint for Constrained3DNonlinear {
    fn gh(
        &self,
        x: &[f64],
        gradients: bool,
    ) -> (
        Vec<f64>,
        Vec<f64>,
        Option<CSR<usize, f64>>,
        Option<CSR<usize, f64>>,
    ) {
        let h0 = Mat::new(2, 3, vec![1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
            .mat_vec(&x.iter().map(|&x| x.powi(2)).collect::<Vec<_>>());
        let h = Arr::with_vec(h0) + Arr::with_vec(vec![-2.0, -10.0]);
        let g = vec![];

        if !gradients {
            (h.vec(), g, None, None)
        } else {
            let dh =
                CSR::from_dense(&[vec![x[0], x[0]], vec![-x[1], x[1]], vec![x[2], x[2]]]) * 2.0;
            let dg = CSR::with_size(3, 0);
            (h.vec(), g, Some(dh), Some(dg))
        }
    }

    fn hess(&self, _x: &[f64], lam: &Lambda, cost_mult: f64) -> CSR<usize, f64> {
        let mu = &lam.ineq_non_lin;
        let l1 = CSR::from_dense(&[
            vec![0.0, -1.0, 0.0],
            vec![-1.0, 0.0, -1.0],
            vec![0.0, -1.0, 0.0],
        ]);
        let l2 = CSR::from_dense(&[
            vec![2.0 * dot(&[1.0, 1.0], &mu), 0.0, 0.0],
            vec![0.0, 2.0 * dot(&[-1.0, 1.0], &mu), 0.0],
            vec![0.0, 0.0, 2.0 * dot(&[1.0, 1.0], &mu)],
        ]);
        let l_xx = (l1 + l2) * cost_mult;
        l_xx
    }
}

#[test]
fn constrained_3d_nonlinear() {
    let f6 = Constrained3DNonlinear {};

    let x0 = vec![1.0, 1.0, 0.0];
    let size = x0.len();

    let solver = RLU::default();
    let opt = Options::default();
    let (x, f, converged, _iterations, lambda) = nlp(
        &f6,
        &x0,
        &CSR::with_size(0, size),
        &vec![],
        &vec![],
        &vec![f64::NEG_INFINITY; size],
        &vec![f64::INFINITY; size],
        Some(&f6),
        &solver,
        &opt,
        None,
    )
    .unwrap();

    assert!(converged);
    assert_eq!(f, -5.0 * f64::sqrt(2.0));
    assert!(zip(x, vec![1.58113883, 2.23606798, 1.58113883]).all(|x| (x.0 - x.1).abs() < 1e-9));
    assert!(
        zip(lambda.ineq_non_lin, vec![0.0, f64::sqrt(2.0) / 2.0]).all(|x| (x.0 - x.1).abs() < 1e-7)
    );
}
