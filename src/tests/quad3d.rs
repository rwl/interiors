use std::iter::zip;

use full::Arr;
use sparsetools::csr::CSR;
use spsolve::rlu::RLU;

use crate::math::dot;
use crate::{nlp, ObjectiveFunction, Options};

/// Unconstrained 3-d quadratic from http://www.akiti.ca/QuadProgEx0Constr.html.
struct Unconstrained3DQuadratic {}

impl ObjectiveFunction for Unconstrained3DQuadratic {
    fn f(&self, x: &[f64], hessian: bool) -> (f64, Vec<f64>, Option<CSR<usize, f64>>) {
        let h: CSR<usize, f64> = CSR::from_dense(&[
            vec![5.0, -2.0, -1.0],
            vec![-2.0, 4.0, 3.0],
            vec![-1.0, 3.0, 5.0],
        ]);
        // let c = vec![2.0, -35.0, -47.0];
        let c = Arr::with_vec(vec![2.0, -35.0, -47.0]);

        let f = 0.5 * dot(&(&h * &x), &x) + dot(&c, &x) + 5.0;
        // let df = zip((&h * &x), c).map(|(a, b)| a + b).collect();
        let df = Arr::with_vec(&h * &x) + c;

        (f, df.vec(), if hessian { Some(h) } else { None })
    }
}
/*
/// Unconstrained 3-d quadratic from http://www.akiti.ca/QuadProgEx0Constr.html.
fn f3(x: Array1<f64>, hessian: bool) -> (f64, Array1<f64>, Option<CsMat<f64>>) {
    let h: sprs::CsMat<f64> = CsMatBase::csc_from_dense(
        arr2(&[[5.0, -2.0, -1.0], [-2.0, 4.0, 3.0], [-1.0, 3.0, 5.0]]).view(),
        0.0,
    );
    let c = arr1(&[2.0, -35.0, -47.0]);

    let f = 0.5 * (&h.view() * &x.view()).dot(&x) + c.dot(&x) + 5.0;
    let df = (&h.view() * &x.view()) + c;

    (f, df, if hessian { Some(h) } else { None })
}
*/

#[test]
fn unconstrained_3d_quadratic() {
    let x0 = vec![0.0, 0.0, 0.0];

    let f3 = Unconstrained3DQuadratic {};

    let size = x0.len();
    let solver = RLU::default();
    let opt = Options::default();
    let (x, f, converged, _iterations, lambda) = nlp(
        &f3,
        &x0,
        &CSR::with_size(0, size),
        &vec![],
        &vec![],
        &vec![f64::NEG_INFINITY; size],
        &vec![f64::INFINITY; size],
        None,
        &solver,
        &opt,
        None,
    )
    .unwrap();

    assert!(converged);
    assert_eq!(f, -244.0);
    assert!(zip(x, vec![3.0, 5.0, 7.0]).all(|x| (x.0 - x.1).abs() < 1e-12));
    assert!(lambda.mu_l.is_empty());
    assert!(lambda.mu_u.is_empty());
    assert!(lambda.lower.iter().all(|&x| x == 0.0));
    assert!(lambda.upper.iter().all(|&x| x == 0.0));
}
