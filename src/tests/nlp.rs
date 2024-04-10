use crate::math::dot;
use crate::{nlp, Lambda, NonlinearConstraint, ObjectiveFunction, Options};
use full::{Arr, Mat};
use sparsetools::coo::Coo;
use sparsetools::csc::CSC;
use sparsetools::csr::CSR;
use spsolve::rlu::RLU;
use std::iter::zip;

/// Unconstrained banana function from MATLAB Optimization Toolbox's `bandem.m`.
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
            ]) * 4.0 * a;

            (f, df, Some(d2f))
        }
    }
}
/*
/// Unconstrained banana function from MATLAB Optimization Toolbox's `bandem.m`.
fn f2(x: Array1<f64>, hessian: bool) -> (f64, Vec<f64>, Option<CsMat<f64>>) {
    let a = 100.0;
    let f = a * (x[1] - x[0].powi(2)).powi(2) + (1.0 - x[0]).powi(2);
    let df = vec![
        4.0 * a * (x[0].powi(3) - x[0] * x[1]) + 2.0 * x[0] - 2.0,
        2.0 * a * (x[1] - x[0].powi(2)),
    ];

    if !hessian {
        (f, df, None)
    } else {
        let d2f = CsMatBase::csc_from_dense(
            arr2(&[
                [3.0 * x[0].powi(2) - x[1] + 1.0 / (2.0 * a), -x[0]],
                [-x[0], 0.5],
            ])
            .view(),
            0.0,
        );

        (f, df, Some(d2f))
    }
}
*/

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
        let x2: Vec<f64> = x.iter().map(|v| v.powi(3)).collect();

        let h0 = Mat::new(2, 2, vec![-1.0, -1.0, 1.0, 1.0]).mat_vec(&x2);
        let h = zip(h0, vec![1.0, -2.0]).map(|(a, b)| a + b).collect();
        let g = vec![];

        if !gradients {
            return (h, g, None, None);
        } else {
            let dh = CSR::from_dense(&[vec![-x[0], x[0]], vec![-x[1], x[1]]]) * 2.0;
            let dg = CSR::with_size(0, 0);

            (h, g, Some(dh), Some(dg))
        }
    }

    fn hess(&self, _x: &[f64], lam: &Lambda, _cost_mult: f64) -> CSR<usize, f64> {
        let mu = &lam.ineq_non_lin;
        let l_xx = Coo::identity(2).to_csr() * 2.0 * dot(&[-1.0, 1.0], mu);
        l_xx
    }
}

/*
fn f5(x: Array1<f64>, hessian: bool) -> (f64, Array1<f64>, Option<CsMat<f64>>) {
    let c = -arr1(&[1.0, 1.0]);

    let f = c.dot(&x);
    let df = c;

    if !hessian {
        (f, df, None)
    } else {
        let d2f = CsMatBase::zero((2, 2));
        (f, df, Some(d2f))
    }
}

fn gh5(x: Array1<f64>) -> (Array1<f64>, Array1<f64>, CsMat<f64>, CsMat<f64>) {
    let x2: Array1<f64> = x.mapv(|v| v.powi(3));

    let h = arr2(&[[-1.0, -1.0], [1.0, 1.0]]).dot(&x2) + arr1(&[1.0, -2.0]);
    let mut dh = CsMatBase::csc_from_dense(arr2(&[[-x[0], x[0]], [-x[1], x[1]]]).view(), 0.0);
    dh.scale(2.0);

    let g = arr1(&[]);
    let dg = CsMatBase::zero((0, 0));

    (h, g, dh, dg)
}
*/

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

            let dg = None; // FIXME
            (h.vec(), g, Some(dh), dg)
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

    fn hess(&self, _x: &[f64], _lam: &Lambda, _cost_mult: f64) -> CSR<usize, f64> {
        todo!()
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

/*
#[test]
fn unconstrained_3d_quadratic() {
    let x0 = arr1(&[0.0, 0.0, 0.0]);
    let (f, _df, _d2f) = f3(x0, false);
    println!("{}", f);
}
*/
