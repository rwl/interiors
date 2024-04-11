use crate::math::dot;
use crate::{nlp, ObjectiveFunction, Options};
use full::Arr;
use sparsetools::csr::CSR;
use spsolve::rlu::RLU;
use std::iter::zip;

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

/*
fn f4(x: Array1<f64>, hessian: bool) -> (f64, Array1<f64>, Option<CsMat<f64>>) {
    let h = CsMatBase::csc_from_dense(
        arr2(&[
            [1003.1, 4.3, 6.3, 5.9],
            [4.3, 2.2, 2.1, 3.9],
            [6.3, 2.1, 3.5, 4.8],
            [5.9, 3.9, 4.8, 10.0],
        ])
        .view(),
        0.0,
    );
    let c = Array1::zeros(4);

    let f = 0.5 * (&h.view() * &x.view()).dot(&x) + c.dot(&x);
    let df = (&h.view() * &x.view()) + c;

    (f, df, if hessian { Some(h) } else { None })
}
*/

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
    assert_eq!(f, 3.29 / 3.0);
    assert!(zip(x, vec![0.0, 2.8 / 3.0, 0.2 / 3.0, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-7));
    assert!(zip(lambda.mu_l, vec![6.58 / 3.0, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-7));
    assert!(lambda.mu_u.is_empty());
    assert!(zip(lambda.lower, vec![2.24, 0.0, 0.0, 1.7667]).all(|x| (x.0 - x.1).abs() < 1e-5));
    assert!(lambda.upper.iter().all(|&x| x == 0.0));
}
