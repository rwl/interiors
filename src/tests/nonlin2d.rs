use float_cmp::assert_approx_eq;
use std::iter::zip;

use full::Mat;
use sparsetools::coo::Coo;
use sparsetools::csr::CSR;
use spsolve::rlu::RLU;

use crate::math::dot;
use crate::{nlp, Lambda, NonlinearConstraint, ObjectiveFunction, Options};

/// Constrained 2-d nonlinear from http://en.wikipedia.org/wiki/Nonlinear_programming#2-dimensional_example.
struct Constrained2DNonlinear {}

impl ObjectiveFunction for Constrained2DNonlinear {
    fn f(&self, x: &[f64], hessian: bool) -> (f64, Vec<f64>, Option<CSR<usize, f64>>) {
        let c = vec![-1.0, -1.0];

        let f = dot(&c, &x);
        let df = c;

        if !hessian {
            (f, df, None)
        } else {
            let d2f = CSR::with_size(2, 2);
            (f, df, Some(d2f))
        }
    }
}

impl NonlinearConstraint for Constrained2DNonlinear {
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
        let x2: Vec<f64> = x.iter().map(|v| v.powi(2)).collect();

        let h0 = Mat::new(2, 2, vec![-1.0, -1.0, 1.0, 1.0]).mat_vec(&x2);
        let h = zip(h0, vec![1.0, -2.0]).map(|(a, b)| a + b).collect();
        let g = vec![];

        if !gradients {
            return (h, g, None, None);
        } else {
            let dh = CSR::from_dense(&[vec![-x[0], x[0]], vec![-x[1], x[1]]]) * 2.0;
            let dg = CSR::with_size(2, 0);

            (h, g, Some(dh), Some(dg))
        }
    }

    fn hess(&self, _x: &[f64], lam: &Lambda, _cost_mult: f64) -> CSR<usize, f64> {
        let mu = &lam.ineq_non_lin;
        let l_xx = Coo::identity(2).to_csr() * 2.0 * dot(&[-1.0, 1.0], mu);
        l_xx
    }
}

#[test]
fn constrained_2d_nonlinear() {
    let f5 = Constrained2DNonlinear {};

    let x0 = vec![1.1, 0.0];
    let size = x0.len();
    let xmin = vec![0.0; size];
    let xmax = vec![f64::INFINITY; size];

    let solver = RLU::default();
    let opt = Options::default();
    let (x, f, converged, _iterations, lambda) = nlp(
        &f5,
        &x0,
        &CSR::with_size(0, size),
        &vec![],
        &vec![],
        &xmin,
        &xmax,
        Some(&f5),
        &solver,
        &opt,
        None,
    )
    .unwrap();

    assert!(converged);
    assert_approx_eq!(f64, f, -2.0, epsilon = 1e-6);
    zip(x, vec![1.0, 1.0]).for_each(|x| assert_approx_eq!(f64, x.0, x.1, epsilon = 1e-6));
    zip(lambda.ineq_non_lin, vec![0.0, 0.5])
        .for_each(|x| assert_approx_eq!(f64, x.0, x.1, epsilon = 1e-6));
    assert!(lambda.mu_l.is_empty());
    assert!(lambda.mu_u.is_empty());
    assert!(lambda.lower.iter().all(|&x| x == 0.0));
    assert!(lambda.upper.iter().all(|&x| x == 0.0));
}
