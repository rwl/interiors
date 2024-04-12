use std::iter::zip;

use float_cmp::assert_approx_eq;
use full::Arr;
use sparsetools::csr::CSR;
use spsolve::rlu::RLU;

use crate::math::dot;
use crate::{nlp, ObjectiveFunction, Options, ProgressMonitor};

/// Constrained 4-d QP from http://www.jmu.edu/docs/sasdoc/sashtml/iml/chap8/sect12.htm.
struct Constrained4DQP {}

impl ObjectiveFunction for Constrained4DQP {
    fn f(&self, x: &[f64], hessian: bool) -> (f64, Vec<f64>, Option<CSR<usize, f64>>) {
        let h = CSR::from_dense(&[
            vec![1003.1, 4.3, 6.3, 5.9],
            vec![4.3, 2.2, 2.1, 3.9],
            vec![6.3, 2.1, 3.5, 4.8],
            vec![5.9, 3.9, 4.8, 10.0],
        ]);
        let c = Arr::with_vec(vec![0.0; 4]);

        let f = 0.5 * dot(&(&h * &x), &x) + dot(&c, &x);
        let df = Arr::with_vec(&h * &x) + c;

        (f, df.vec(), if hessian { Some(h) } else { None })
    }
}

impl ProgressMonitor for Constrained4DQP {}

#[test]
fn constrained_4d_qp() {
    let f4 = Constrained4DQP {};

    let x0 = vec![1.0, 0.0, 0.0, 1.0];
    let a_mat = CSR::from_dense(&vec![
        vec![1.0, 1.0, 1.0, 1.0],
        vec![0.17, 0.11, 0.10, 0.18],
    ]);
    let l = vec![1.0, 0.10];
    let u = vec![1.0, f64::INFINITY];
    let xmin = vec![0.0; 4];
    let xmax = vec![f64::INFINITY; 4];

    let solver = RLU::default();
    let opt = Options::default();
    let (x, f, converged, _iterations, lambda) = nlp(
        &f4, &x0, &a_mat, &l, &u, &xmin, &xmax, None, &solver, &opt, None,
    )
    .unwrap();

    assert!(converged);
    assert_approx_eq!(f64, f, 3.29 / 3.0, epsilon = 1e-5);
    zip(x, vec![0.0, 2.8 / 3.0, 0.2 / 3.0, 0.0])
        .for_each(|x| assert_approx_eq!(f64, x.0, x.1, epsilon = 1e-6));
    zip(lambda.mu_l, vec![6.58 / 3.0, 0.0])
        .for_each(|x| assert_approx_eq!(f64, x.0, x.1, epsilon = 1e-6));
    assert!(lambda.mu_u.iter().all(|&x| x == 0.0));
    zip(lambda.lower, vec![2.24, 0.0, 0.0, 1.7667])
        .for_each(|x| assert_approx_eq!(f64, x.0, x.1, epsilon = 1e-4));
    assert!(lambda.upper.iter().all(|&x| x == 0.0));
}
