use crate::{qp, Options};
use sparsetools::csr::CSR;
use spsolve::rlu::RLU;
use std::iter::zip;

/// based on example from 'doc linprog'
#[test]
pub fn lp3d() {
    let h_mat = CSR::with_size(0, 3);
    let c = vec![-5.0, -4.0, -6.0];
    let a_mat = CSR::from_dense(&vec![
        vec![1.0, -1.0, 1.0],
        vec![-3.0, -2.0, -4.0],
        vec![3.0, 2.0, 0.0],
    ]);
    let l = vec![f64::NEG_INFINITY, -42.0, f64::NEG_INFINITY];
    let u = vec![20.0, f64::INFINITY, 30.0];
    let xmin = vec![0.0; 3];
    let xmax = vec![f64::INFINITY; 3];
    let x0 = vec![0.0; 3];

    let solver = RLU::default();
    let options = Options::default();

    let (x, f, success, _out, lam) = qp(
        &h_mat, &c, &a_mat, &l, &u, &xmin, &xmax, &x0, &solver, None, &options,
    )
    .unwrap();

    assert!(success);
    assert_eq!(f, -78.0);
    assert!(zip(x, vec![0.0, 15.0, 3.0]).all(|x| (x.0 - x.1).abs() < 1e-7));
    assert!(zip(lam.mu_l, vec![0.0, 1.5, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-10));
    assert!(zip(lam.mu_u, vec![0.0, 0.0, 0.5]).all(|x| (x.0 - x.1).abs() < 1e-10));
    assert!(zip(lam.lower, vec![1.0, 0.0, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-10));
    assert!(zip(lam.upper, vec![0.0, 0.0, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-10));
}

/// from http://www.akiti.ca/QuadProgEx0Constr.html
#[test]
pub fn unconstrained_3d_qp() {
    let h_mat = CSR::from_dense(&vec![
        vec![5.0, -2.0, -1.0],
        vec![-2.0, 4.0, 3.0],
        vec![-1.0, 3.0, 5.0],
    ]);
    let c = vec![2.0, -35.0, -47.0];
    let x0 = vec![0.0, 0.0, 0.0];
    let xmin = vec![f64::NEG_INFINITY; 3];
    let xmax = vec![f64::INFINITY; 3];

    let solver = RLU::default();
    let options = Options::default();

    let (x, f, success, _out, lam) = qp(
        &h_mat,
        &c,
        &CSR::with_size(0, 3),
        &vec![],
        &vec![],
        &xmin,
        &xmax,
        &x0,
        &solver,
        None,
        &options,
    )
    .unwrap();

    assert!(success);
    assert_eq!(f, -249.0);
    assert!(zip(x, vec![3.0, 5.0, 7.0]).all(|x| (x.0 - x.1).abs() < 1e-13));
    assert!(lam.mu_l.is_empty());
    assert!(lam.mu_u.is_empty());
    assert!(zip(lam.lower, vec![0.0, 0.0, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-13));
    assert!(zip(lam.upper, vec![0.0, 0.0, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-13));
}

/// example from 'doc quadprog'
#[test]
pub fn constrained_2d_qp() {
    let h_mat = CSR::from_dense(&vec![vec![1.0, -1.0], vec![-1.0, 2.0]]);
    let c = vec![-2.0, -6.0];
    let a_mat = CSR::from_dense(&vec![vec![1.0, 1.0], vec![-1.0, 2.0], vec![2.0, 1.0]]);
    let l = vec![];
    let u = vec![2.0, 2.0, 3.0];
    let xmin = vec![0.0, 0.0];
    let xmax = vec![f64::INFINITY; 2];
    let x0 = vec![];

    let solver = RLU::default();
    let options = Options::default();

    let (x, f, success, _out, lam) = qp(
        &h_mat, &c, &a_mat, &l, &u, &xmin, &xmax, &x0, &solver, None, &options,
    )
    .unwrap();

    assert!(success);
    assert_eq!(f, -74.0 / 9.0);
    assert!(zip(x, vec![2.0 / 3.0, 4.0 / 3.0]).all(|x| (x.0 - x.1).abs() < 1e-7));
    assert!(zip(lam.mu_l, vec![0.0, 0.0, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-13));
    assert!(zip(lam.mu_u, vec![28.0 / 9.0, 4.0 / 9.0, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-5));
    assert!(zip(lam.lower, vec![0.0, 0.0, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-7));
    assert!(zip(lam.upper, vec![0.0, 0.0, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-13));
}

/// from https://v8doc.sas.com/sashtml/iml/chap8/sect12.htm
#[test]
pub fn constrained_4d_qp() {
    let h_mat = CSR::from_dense(&vec![
        vec![1003.1, 4.3, 6.3, 5.9],
        vec![4.3, 2.2, 2.1, 3.9],
        vec![6.3, 2.1, 3.5, 4.8],
        vec![5.9, 3.9, 4.8, 10.0],
    ]);
    let c = vec![0.0; 4];
    let a_mat = CSR::from_dense(&vec![
        vec![1.0, 1.0, 1.0, 1.0],
        vec![0.17, 0.11, 0.10, 0.18],
    ]);
    let l = vec![1.0, 0.10];
    let u = vec![1.0, f64::INFINITY];
    let xmin = vec![0.0; 4];
    let xmax = vec![f64::INFINITY; 4];
    let x0 = vec![1.0, 0.0, 0.0, 1.0];

    let solver = RLU::default();
    let options = Options::default();

    let (x, f, success, _out, lam) = qp(
        &h_mat, &c, &a_mat, &l, &u, &xmin, &xmax, &x0, &solver, None, &options,
    )
    .unwrap();

    assert!(success);
    assert_eq!(f, 3.29 / 3.0);
    assert!(zip(x, vec![0.0, 2.8 / 3.0, 0.2 / 3.0, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-7));
    assert!(zip(lam.mu_l, vec![6.58 / 3.0, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-6));
    assert!(zip(lam.mu_u, vec![0.0, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-13));
    assert!(zip(lam.lower, vec![2.24, 0.0, 0.0, 1.7667]).all(|x| (x.0 - x.1).abs() < 1e-5));
    assert!(zip(lam.upper, vec![0.0, 0.0, 0.0, 0.0]).all(|x| (x.0 - x.1).abs() < 1e-13));
}
