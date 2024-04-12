use sparsetools::csr::CSR;
use spsolve::rlu::RLU;

use crate::{nlp, ObjectiveFunction, Options};

/// 2-dimensional unconstrained optimization of Rosenbrock's "banana" function
/// from MATLAB Optimization Toolbox's `bandem.m`:
///
/// ```txt
///     f(x) = 100(x_2 − x_1^2) 2 + (1 − x_1)^2
/// ```
///
/// https://en.wikipedia.org/wiki/Rosenbrock_function
struct UnconstrainedBananaFunction {}

impl ObjectiveFunction for UnconstrainedBananaFunction {
    fn f(&self, x: &[f64], hessian: bool) -> (f64, Vec<f64>, Option<CSR<usize, f64>>) {
        let a = 100.0;
        let f = a * (x[1] - x[0].powi(2)).powi(2) + (1.0 - x[0]).powi(2);
        let df = vec![
            4.0 * a * (x[0].powi(3) - x[0] * x[1]) + 2.0 * x[0] - 2.0,
            2.0 * a * (x[1] - x[0].powi(2)),
        ];

        if !hessian {
            (f, df, None)
        } else {
            let d2f = CSR::from_dense(&[
                vec![3.0 * x[0].powi(2) - x[1] + 1.0 / (2.0 * a), -x[0]],
                vec![-x[0], 0.5],
            ]) * 4.0
                * a;

            (f, df, Some(d2f))
        }
    }
}

#[test]
fn unconstrained_banana() {
    let x0 = vec![-1.9, 2.0];
    // let (f, _df, _d2f) = f2(x0, false);
    // println!("{}", f);
    let f2 = UnconstrainedBananaFunction {};

    let size = 2;
    let solver = RLU::default();
    let opt = Options::default();
    let (x, f, converged, _iterations, lambda) = nlp(
        &f2,
        &x0,
        &CSR::with_size(0, size),
        &vec![],
        &vec![],
        &vec![f64::NEG_INFINITY; size],
        // &vec![-1e12; size],
        &vec![f64::INFINITY; size],
        // &vec![1e12; size],
        None,
        &solver,
        &opt,
        None,
    )
    .unwrap();

    assert!(converged);
    assert_eq!(f, 0.0);
    assert!(x.iter().all(|&x| x == 1.0));
    assert!(lambda.mu_l.is_empty());
    assert!(lambda.mu_u.is_empty());
    assert!(lambda.lower.iter().all(|&x| x == 0.0));
    assert!(lambda.upper.iter().all(|&x| x == 0.0));
}
