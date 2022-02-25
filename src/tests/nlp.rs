use ndarray::{arr1, arr2, Array1};
use sprs::{CsMat, CsMatBase};

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

/// Constrained 4-d QP from http://www.jmu.edu/docs/sasdoc/sashtml/iml/chap8/sect12.htm.
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

/// Constrained 2-d nonlinear from http://en.wikipedia.org/wiki/Nonlinear_programming#2-dimensional_example.
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

#[test]
fn unconstrained_banana() {
    let x0 = arr1(&[-1.9, 2.0]);
    let (f, _df, _d2f) = f2(x0, false);
    println!("{}", f);
}

#[test]
fn unconstrained_3d_quadratic() {
    let x0 = arr1(&[0.0, 0.0, 0.0]);
    let (f, _df, _d2f) = f3(x0, false);
    println!("{}", f);
}
