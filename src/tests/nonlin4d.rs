use crate::{nlp, Lambda, NonlinearConstraint, ObjectiveFunction, Options};
use full::Arr;
use sparsetools::coo::Coo;
use sparsetools::csc::CSC;
use sparsetools::csr::CSR;
use spsolve::rlu::RLU;
use std::iter::zip;

/// Hock & Schittkowski test problem #71
struct Constrained4DNonlinear {}

impl ObjectiveFunction for Constrained4DNonlinear {
    fn f(&self, x: &[f64], hessian: bool) -> (f64, Vec<f64>, Option<CSR<usize, f64>>) {
        let f = x[0] * x[3] * x[..3].iter().sum::<f64>() + x[2];
        let df = vec![
            x[0] * x[3] + x[3] * x[..3].iter().sum::<f64>(),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * x[..3].iter().sum::<f64>(),
        ];
        if !hessian {
            (f, df, None)
        } else {
            let d2f = CSR::from_dense(&[
                vec![2.0 * x[3], x[3], x[3], 2.0 * x[0] + x[1] + x[2]],
                vec![x[3], 0.0, 0.0, x[0]],
                vec![x[3], 0.0, 0.0, x[0]],
                vec![2.0 * x[0] + x[1] + x[2], x[0], x[0], 0.0],
            ]);
            (f, df, Some(d2f))
        }
    }
}

impl NonlinearConstraint for Constrained4DNonlinear {
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
        let x = Arr::with_vec(x.to_vec());

        // let g = vec![x.iter().map(|&x| x.powi(2)).sum() - 40.0];
        let g = vec![x.pow(2).sum() - 40.0];
        let h = vec![-x.prod() + 25.0];
        if !gradients {
            (g, h, None, None)
        } else {
            let dg = CSC::from_dense(&[(&x * 2.0).vec()]).t();
            let dh = CSC::from_dense(&[(1.0 / &x * -x.prod()).vec()]).t();
            (h, g, Some(dh), Some(dg))
        }
    }

    fn hess(&self, x: &[f64], lam: &Lambda, sigma: f64) -> CSR<usize, f64> {
        let lambda = lam.eq_non_lin[0];
        let mu = lam.ineq_non_lin[0];

        let (_, _, d2f) = self.f(x, true);

        let l_xx: CSR<usize, f64> = d2f.unwrap() * sigma + Coo::identity(4).to_csr() * lambda * 2.0
            - CSR::from_dense(&vec![
                vec![0.0, x[2] * x[3], x[1] * x[3], x[1] * x[2]],
                vec![x[2] * x[3], 0.0, x[0] * x[3], x[0] * x[2]],
                vec![x[1] * x[3], x[0] * x[3], 0.0, x[0] * x[1]],
                vec![x[1] * x[2], x[0] * x[2], x[0] * x[1], 0.0],
            ]) * mu;

        l_xx
    }
}

#[test]
fn constrained_4d_nonlinear() {
    let f7 = Constrained4DNonlinear {};

    let x0 = vec![1.0, 5.0, 5.0, 1.0];
    let size = x0.len();
    let xmin = vec![1.0; size];
    let xmax = vec![5.0; size];

    let solver = RLU::default();
    let opt = Options::default();
    let (x, f, converged, _iterations, lambda) = nlp(
        &f7,
        &x0,
        &CSR::with_size(0, size),
        &vec![],
        &vec![],
        &xmin,
        &xmax,
        Some(&f7),
        &solver,
        &opt,
        None,
    )
    .unwrap();

    assert!(converged);
    assert_eq!(f, 17.0140173);
    assert!(zip(x, vec![1.0, 4.7429994, 3.8211503, 1.3794082]).all(|x| (x.0 - x.1).abs() < 1e-8));
    assert!(zip(lambda.eq_non_lin, vec![0.1614686]).all(|x| (x.0 - x.1).abs() < 1e-5));
    assert!(zip(lambda.ineq_non_lin, vec![0.55229366]).all(|x| (x.0 - x.1).abs() < 1e-5));
    assert!(zip(lambda.lower, vec![1.08787121024, 0.0, 0.0, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-5));
    assert!(zip(lambda.upper, vec![0.0; size]).all(|x| (x.0 - x.1).abs() < 1e-5));
}
