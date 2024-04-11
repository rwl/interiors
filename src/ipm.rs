use crate::common::{Lambda, Options};
use crate::math::*;
use crate::traits::*;
use anyhow::{format_err, Result};
use itertools::{izip, Itertools};
use sparsetools::coo::Coo;
use sparsetools::csc::CSC;
use sparsetools::csr::CSR;
use spsolve::Solver;

/// Primal-dual interior point method for NLP (nonlinear programming).
/// Minimize a function F(x) beginning from a starting point x0, subject
/// to optional linear and nonlinear constraints and variable bounds.
///
///       min F(x)
///        x
///
/// subject to
///
///       g(x) = 0            (nonlinear equalities)
///       h(x) <= 0           (nonlinear inequalities)
///       l <= A*x <= u       (linear constraints)
///       xmin <= x <= xmax   (variable bounds)
pub fn nlp<F, S>(
    f_fn: &F,
    x0: &[f64],
    a_mat: &CSR<usize, f64>,
    l: &[f64],
    u: &[f64],
    xmin: &[f64],
    xmax: &[f64],
    nonlinear: Option<&dyn NonlinearConstraint>,
    solver: &S,
    opt: &Options,
    progress: Option<&dyn ProgressMonitor>,
) -> Result<(Vec<f64>, f64, bool, usize, Lambda)>
where
    F: ObjectiveFunction,
    S: Solver<usize, f64>,
{
    let nx = x0.len();

    assert_eq!(a_mat.cols(), nx);

    // let empty_a_mat = CSR::<usize, f64>::zeros(0, nx);
    // let a_mat = if l.is_some()
    //     && u.is_some()
    //     && l.as_ref().unwrap().iter().all(|v| v.is_infinite())
    //     && u.as_ref().unwrap().iter().all(|v| v.is_infinite())
    // {
    //     // no limits => no linear constraints
    //     &empty_a_mat
    // } else {
    //     a_mat
    // };
    let na = a_mat.rows(); // number of original linear constraints

    assert_eq!(l.len(), na);
    assert_eq!(u.len(), na);
    assert_eq!(xmin.len(), nx);
    assert_eq!(xmax.len(), nx);

    // By default, linear inequalities are ...
    // let u = match u {
    //     Some(u) => u.to_owned(),
    //     None => vec![f64::INFINITY; na], // ... unbounded above and ...
    // };
    // let l = match l {
    //     Some(l) => l.to_owned(),
    //     None => vec![f64::NEG_INFINITY; na], // ... unbounded below.
    // };

    // By default, optimization variables are ...
    // let xmin = match xmin {
    //     Some(xmin) => xmin.to_owned(),
    //     None => vec![f64::NEG_INFINITY; nx], // ... unbounded below and ...
    // };
    // let xmax = match xmax {
    //     Some(xmax) => xmax.to_owned(),
    //     None => vec![f64::INFINITY; nx], // ... unbounded above.
    // };

    let gn = Vec::<f64>::default();
    let hn = Vec::<f64>::default();

    // Set up problem
    let (xi, sigma, z0, alpha_min, rho_min, rho_max, mu_threshold, max_step_size) = (
        opt.xi,
        opt.sigma,
        opt.z0,
        opt.alpha_min,
        opt.rho_min,
        opt.rho_max,
        opt.mu_threshold,
        opt.max_step_size,
    );
    if xi >= 1.0 || xi < 0.5 {
        return Err(format_err!("xi ({}) must be slightly less than 1", xi));
    }
    if sigma > 1.0 || sigma <= 0.0 {
        return Err(format_err!("sigma ({}) must be between 0 and 1", sigma));
    }

    // Add var limits to linear constraints.
    let aa_mat: CSR<usize, f64> =
        Coo::v_stack(&Coo::<usize, f64>::identity(nx), &a_mat.to_coo())?.to_csr();
    let ll: Vec<f64> = [xmin, l].concat();
    let uu: Vec<f64> = [xmax, u].concat();

    // Split up linear constraints.
    let mut ieq = Vec::<usize>::new(); // equality
    let mut igt = Vec::<usize>::new(); // greater than, unbounded above
    let mut ilt = Vec::<usize>::new(); // less than, unbounded below
    let mut ibx = Vec::<usize>::new();
    for i in 0..(nx + na) {
        let (li, ui) = (ll[i], uu[i]);
        if (ui - li).abs() <= f64::EPSILON {
            ieq.push(i);
        } else if ui >= 1e-10 && li > -1e-10 {
            igt.push(i);
        } else if li <= -1e-10 && ui < 1e10 {
            ilt.push(i);
        } else if ((ui - li).abs() > f64::EPSILON) && (ui < 1e10) && (li > -1e10) {
            ibx.push(i);
        }
    }
    let ae_mat = aa_mat.select(Some(&ieq), None)?;
    let be = ieq.iter().map(|&i| uu[i]).collect_vec();
    let ai_mat = {
        // &aa_mat;

        let a_lt = aa_mat.select(Some(&ilt), None)?;
        let a_gt = -aa_mat.select(Some(&igt), None)?;
        // floats.Neg(Agt.Data()) // inplace

        let a_bx1 = aa_mat.select(Some(&ibx), None)?;
        let a_bx2 = &a_bx1 * -1.0;
        // floats.Neg(Abx2.Data())

        let a_ieq = Coo::v_stack(&a_lt.to_coo(), &a_gt.to_coo())?;
        let a_bx = Coo::v_stack(&a_bx1.to_coo(), &a_bx2.to_coo())?;

        Coo::v_stack(&a_ieq, &a_bx)?.to_csr()
        // let a_i: CSR<usize, f64> = CSR::vstack(Aieq, Abx);
        // a_i.to_csc()
    };
    let bi: Vec<f64> = [
        ilt.iter().map(|&i| uu[i]).collect_vec(),
        igt.iter().map(|&i| -ll[i]).collect_vec(),
        ibx.iter().map(|&i| uu[i]).collect_vec(),
        ibx.iter().map(|&i| -ll[i]).collect_vec(),
    ]
    .concat();

    // Evaluate cost f(x0) and constraints g(x0), h(x0)
    let mut x = x0.to_vec();

    let (mut f, mut df, _) = f_fn.f(&x, false);

    f *= opt.cost_mult;
    df.iter_mut().for_each(|df| *df *= opt.cost_mult);

    let (h, g, dh, dg): (Vec<f64>, Vec<f64>, CSR<usize, f64>, CSR<usize, f64>) =
        if let Some(gh_fn) = nonlinear {
            let (hn, gn, dhn, dgn) = gh_fn.gh(&x, true); // nonlinear constraints
            let (dhn, dgn) = (dhn.unwrap(), dgn.unwrap());

            let h: Vec<f64> = [
                hn,
                izip!((&ai_mat * &x), &bi)
                    .map(|(xi, bi)| xi - bi)
                    .collect_vec(),
            ]
            .concat(); // inequality constraints
            let g: Vec<f64> = [
                gn,
                izip!(&ae_mat * &x, &be)
                    .map(|(xe, be)| xe - be)
                    .collect_vec(),
            ]
            .concat(); // equality constraints

            let dh: CSR<usize, f64> = Coo::h_stack(&dhn.to_coo(), &ai_mat.t().to_coo())?.to_csr(); // 1st derivative of inequalities
            let dg: CSR<usize, f64> = Coo::h_stack(&dgn.to_coo(), &ae_mat.t().to_coo())?.to_csr(); // 1st derivative of equalities

            (h, g, dh, dg)
        } else {
            let h = izip!(&ai_mat * &x, &bi)
                .map(|(xi, bi)| xi - bi)
                .collect_vec(); // inequality constraints
            let g = izip!(&ae_mat * &x, &be)
                .map(|(xe, be)| xe - be)
                .collect_vec(); // equality constraints

            let dh = ai_mat.t().to_csr(); // 1st derivative of inequalities
            let dg = ae_mat.t().to_csr(); // 1st derivative of equalities

            (h, g, dh, dg)
        };

    // Grab some dimensions.
    let neq = g.len(); // number of equality constraints
    let niq = h.len(); // number of inequality constraints
    let neqnln = gn.len(); // number of nonlinear equality constraints
    let niqnln = hn.len(); // number of nonlinear inequality constraints
    let nlt = ilt.len(); // number of upper bounded linear inequalities
    let ngt = igt.len(); // number of lower bounded linear inequalities
    let nbx = ibx.len(); // number of doubly bounded linear inequalities

    // // Initialize gamma, lam, mu, z, e.
    // let mut gamma = 1.0; // Barrier coefficient, r in Harry's code.
    // let mut lam = Vec::<f64>::zeros(neq);
    // // let mut z = Array1::from_elem(niq, z0);
    // let mut z = Array1::from(
    //     h.iter()
    //         .map(|&hk| if hk < -z0 { -hk } else { z0 })
    //         .collect::<Vec<f64>>(),
    // );
    // // let mu = z.clone();
    // let mut mu = Array1::from(
    //     z.iter()
    //         .map(|&zk| if gamma / zk > z0 { gamma / zk } else { z0 })
    //         .collect::<Vec<f64>>(),
    // );
    // let e = Array1::<f64>::ones(niq);

    // Initialize gamma, lam, mu, z, e.
    let mut gamma = 1.0; // Barrier coefficient, r in Harry's code.
    let mut lam = vec![0.0; neq];
    let mut z = vec![z0; niq];
    let mut mu = z.clone();
    izip!(&h, &mut z).for_each(|(&h, z)| {
        if h < -z0 {
            *z = -h;
        }
    });
    // (seems k is always empty if gamma = z0 = 1)
    izip!(&z, &mut mu).for_each(|(&z, mu)| {
        if gamma / z > z0 {
            *mu = gamma / z;
        }
    });
    let e = vec![1.0; niq];

    // check tolerance
    let mut f0 = f;
    let mut l_step: f64 = if opt.step_control {
        let hz = izip!(&h, &z).map(|(&h, &z)| h + z).collect_vec();
        let z_ln_sum: f64 = z.iter().map(|z| z.ln()).sum();

        f + dot(&lam, &g) + dot(&mu, &hz) - gamma * z_ln_sum
    } else {
        0.0
    };
    // let mut l_x = df + (&dg * &lam) + (&dh * &mu);
    let mut l_x = izip!(df, &dg * &lam, &dh * &mu)
        .map(|(df, dg_lam, dh_mu)| df + dg_lam + dh_mu)
        .collect_vec();

    let feascond = match max(&h) {
        None => norm_inf(&g) / (1.0 + f64::max(norm_inf(&x), norm_inf(&z))),
        Some(maxh) => f64::max(norm_inf(&g), maxh) / (1.0 + f64::max(norm_inf(&x), norm_inf(&z))),
    };
    let gradcond = norm_inf(&l_x) / (1.0 + f64::max(norm_inf(&lam), norm_inf(&mu)));
    let compcond = dot(&z, &mu) / (1.0 + norm_inf(&x));
    let costcond = (f - f0).abs() / (1.0 + f0.abs());

    let mut iterations = 0;
    if let Some(progress) = progress.as_ref() {
        progress.update(
            iterations,
            feascond,
            gradcond,
            compcond,
            costcond,
            gamma,
            0.0,
            f / opt.cost_mult,
            0.0,
            0.0,
        );
    }
    let mut failed = false;
    let mut converged = feascond < opt.feas_tol
        && gradcond < opt.grad_tol
        && compcond < opt.comp_tol
        && costcond < opt.cost_tol;

    // Newton iterations.
    while !converged && iterations < opt.max_it {
        // Update iteration counter.
        iterations += 1;

        // Compute update step.
        let lambda = Lambda {
            eq_non_lin: (0..neqnln).map(|i| lam[i]).collect_vec(),
            ineq_non_lin: (0..niqnln).map(|i| mu[i]).collect_vec(),
            ..Default::default()
        };
        let l_xx = if let Some(hess_fn) = nonlinear {
            hess_fn.hess(&x, &lambda, opt.cost_mult)
        } else {
            let (_, _, d2f) = f_fn.f(&x, true); // cost
                                                // d2f.as_mut().unwrap().scale(opt.cost_mult);
            d2f.unwrap() * opt.cost_mult
        };

        let zinvdiag = {
            let zinv = z.iter().map(|z| 1.0 / z).collect_vec();
            Coo::<usize, f64>::with_diagonal(&zinv).to_csr()
        };
        let mudiag = Coo::<usize, f64>::with_diagonal(&mu).to_csr();
        let dh_zinv = &dh * &zinvdiag;

        // M = Lxx + dh_zinv * mudiag * dh';
        let m_mat: CSR<usize, f64> = &l_xx + &((&dh_zinv * &mudiag) * &dh.t().to_csr());

        // N = Lx + dh_zinv * (mudiag * h + gamma * e);
        let n: Vec<f64> = {
            let temp = izip!(&mudiag * &h, &e)
                .map(|(mudiag_h, e)| mudiag_h + gamma * e)
                .collect_vec();
            izip!(&l_x, &dh_zinv * &temp)
                .map(|(l_x, dh_zinv_temp)| l_x + dh_zinv_temp)
                .collect_vec()
        };

        let dxdlam = {
            // let a_mat: CSR<usize, f64> = vstack(&[
            //     hstack(&[m_mat.view(), dg.view()]).view(),
            //     hstack(&[dg.transpose_view(), CsMatBase::zero((neq, neq)).view()]).view(),
            // ]);
            let a_mat: CSC<usize, f64> = Coo::compose([
                [&m_mat.to_coo(), &dg.to_coo()],
                [&dg.t().to_coo(), &Coo::with_size(neq, neq)],
            ])?
            .to_csc();
            let mut b: Vec<f64> = [
                n.iter().map(|n| -n).collect_vec(),
                g.iter().map(|g| -g).collect_vec(),
            ]
            .concat();
            // solver.solve(&a_mat, &mut b)?;
            solver.solve(
                a_mat.cols(),
                &a_mat.rowidx(),
                &a_mat.colptr(),
                &a_mat.values(),
                &mut b,
                false,
            )?;
            b
        };
        if dxdlam.iter().any(|dxdlam| dxdlam.is_nan()) || norm(&dxdlam) > max_step_size {
            failed = true;
            break;
        }
        let mut dx = (0..nx).map(|i| dxdlam[i]).collect_vec();

        let mut dlam = (nx..(nx + neq)).map(|i| dxdlam[i]).collect_vec();

        // dz = -h - z - dh' * dx;
        let mut dz = izip!(&h, &z, &dh.t().to_csr() * &dx) // fixme: to_csr()
            .map(|(h, z, dh_dx)| -h - z - dh_dx)
            .collect_vec();

        // dmu = -mu + zinvdiag *(gamma*e - mudiag * dz);
        let mut dmu = {
            let temp = izip!(&e, &mudiag * &dz)
                .map(|(e, mudiag_dz)| gamma * e - mudiag_dz)
                .collect_vec();

            izip!(&mu, &zinvdiag * &temp)
                .map(|(mu, zinvdiag_temp)| -mu + zinvdiag_temp)
                .collect_vec()
        };

        // Optional step-size control.
        let sc = if opt.step_control {
            let x1 = izip!(&x, &dx).map(|(x, dx)| x + dx).collect_vec();

            // Evaluate cost, constraints, derivatives at x1.
            let (f1, df1, _) = f_fn.f(&x1, false); // cost
            let _f1 = f1 * opt.cost_mult;
            let df1 = df1.iter().map(|df1| df1 * opt.cost_mult).collect_vec();

            let (h1, g1, dh1, dg1): (Vec<f64>, Vec<f64>, CSR<usize, f64>, CSR<usize, f64>) =
                if let Some(gh_fn) = nonlinear {
                    let (hn1, gn1, dhn1, dgn1) = gh_fn.gh(&x1, true); // nonlinear constraints
                    let (dhn1, dgn1) = (dhn1.unwrap(), dgn1.unwrap());

                    // inequality constraints
                    let h1: Vec<f64> = [
                        hn1,
                        izip!(&ai_mat * &x1, &bi)
                            .map(|(xi, bi)| xi - bi)
                            .collect_vec(),
                    ]
                    .concat();

                    // equality constraints
                    let g1: Vec<f64> = [
                        gn1,
                        izip!(&ae_mat * &x1, &be)
                            .map(|(xe, be)| xe - be)
                            .collect_vec(),
                    ]
                    .concat();

                    let dh1: CSR<usize, f64> =
                        Coo::h_stack(&dhn1.to_coo(), &ai_mat.t().to_coo())?.to_csr(); // 1st derivative of inequalities
                    let dg1: CSR<usize, f64> =
                        Coo::h_stack(&dgn1.to_coo(), &ae_mat.t().to_coo())?.to_csr(); // 1st derivative of equalities

                    (h1, g1, dh1, dg1)
                } else {
                    // inequality constraints
                    let h1 = izip!(&ai_mat * &x1, &bi)
                        .map(|(xi, bi)| xi - bi)
                        .collect_vec();

                    // equality constraints
                    let g1 = izip!(&ae_mat * &x1, &be)
                        .map(|(xe, be)| xe - be)
                        .collect_vec();

                    let dh1 = ai_mat.t().to_csr(); // dh // 1st derivative of inequalities
                    let dg1 = ae_mat.t().to_csr(); // dg // 1st derivative of equalities

                    (h1, g1, dh1, dg1)
                };

            // check tolerance
            let l_x1 = izip!(&df1, &dg1 * &lam, &dh1 * &mu)
                .map(|(df1, dg1_lam, dh1_mu)| df1 + dg1_lam + dh1_mu)
                .collect_vec();

            let feascond1 = match max(&h1) {
                None => norm_inf(&g1) / (1.0 + f64::max(norm_inf(&x1), norm_inf(&z))),
                Some(maxh1) => {
                    f64::max(norm_inf(&g1), maxh1) / (1.0 + f64::max(norm_inf(&x1), norm_inf(&z)))
                }
            };
            let gradcond1 = norm_inf(&l_x1) / (1.0 + f64::max(norm_inf(&lam), norm_inf(&mu)));

            feascond1 > feascond && gradcond1 > gradcond
        } else {
            false
        };
        if sc {
            let mut alpha = 1.0;
            for j in 0..opt.max_red {
                let dx1 = dx.iter().map(|dx| alpha * dx).collect_vec();
                let x1 = izip!(&x, &dx1).map(|(&x, &dx1)| x + dx1).collect_vec();
                let (f1, _, _) = f_fn.f(&x1, false); // cost
                let f1 = f1 * opt.cost_mult;
                let (h1, g1) = if let Some(gh_fn) = nonlinear {
                    let (hn1, gn1, _, _) = gh_fn.gh(&x1, false); // nonlinear constraints

                    // inequality constraints
                    let h1 = [
                        hn1,
                        izip!(&ai_mat * &x1, &bi)
                            .map(|(xi, bi)| xi - bi)
                            .collect_vec(),
                    ]
                    .concat();

                    // equality constraints
                    let g1 = [
                        gn1,
                        izip!(&ae_mat * &x1, &be)
                            .map(|(xe, be)| xe - be)
                            .collect_vec(),
                    ]
                    .concat();

                    (h1, g1)
                } else {
                    // inequality constraints
                    let h1 = izip!(&ai_mat * &x1, &bi)
                        .map(|(xi, bi)| xi - bi)
                        .collect_vec();

                    // equality constraints
                    let g1 = izip!(&ae_mat * &x1, &be)
                        .map(|(xe, be)| xe - be)
                        .collect_vec();

                    (h1, g1)
                };

                // L1 = f1 + lam' * g1 + mu' * (h1+z) - gamma * sum(log(z));
                let l1: f64 = {
                    let hz = izip!(&h1, &z).map(|(&h1, &z1)| h1 + z1).collect_vec();
                    let z_ln_sum: f64 = z.iter().map(|z| z.ln()).sum();

                    f1 + dot(&lam, &g1) + dot(&mu, &hz) - gamma * z_ln_sum
                };
                if opt.verbose {
                    print!("{} {}", -(j as isize), norm(&dx1));
                }
                let rho = (l1 - l_step) / (dot(&l_x, &dx1) + 0.5 * dot(&dx1, &(&l_xx * &dx1)));
                if rho > rho_min && rho < rho_max {
                    break;
                } else {
                    alpha = alpha / 2.0;
                }
            }
            dx.iter_mut().for_each(|dx| *dx *= alpha);
            dz.iter_mut().for_each(|dz| *dz *= alpha);
            dlam.iter_mut().for_each(|dlam| *dlam *= alpha);
            dmu.iter_mut().for_each(|dmu| *dmu *= alpha);
        }
        // do the update
        // let k = find(&lt(dz, 0.0));
        let k = dz
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v < 0.0 { Some(i) } else { None })
            .collect_vec();
        let alphap = if k.is_empty() {
            1.0
        } else {
            // f64::min(xi * min(z.select(&k) / -dz.select(&k)), 1.0)
            f64::min(
                xi * min(&k.iter().map(|&i| z[i] / -dz[i]).collect::<Vec<f64>>()).unwrap(),
                1.0,
            )
        };
        // let k = find(&lt(dmu, 0.0));
        let k = dmu
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v < 0.0 { Some(i) } else { None })
            .collect_vec();
        let alphad = if k.is_empty() {
            1.0
        } else {
            // f64::min(xi * min(mu.select(&k) / -dmu.select(&k)), 1.0)
            f64::min(
                xi * min(&k.iter().map(|&i| mu[i] / -dmu[i]).collect::<Vec<f64>>()).unwrap(),
                1.0,
            )
        };
        izip!(&mut x, &dx).for_each(|(x, dx)| *x += alphap * dx);
        izip!(&mut z, &dz).for_each(|(z, dz)| *z += alphad * dz);
        izip!(&mut lam, &dlam).for_each(|(lam, dlam)| *lam += alphad * dlam);
        izip!(&mut mu, &dmu).for_each(|(mu, dmu)| *mu += alphad * dmu);
        if niq > 0 {
            gamma = sigma * dot(&z, &mu) / (niq as f64);
        }

        // evaluate cost, constraints, derivatives
        (f, df, _) = f_fn.f(&x, false); // cost
        f *= opt.cost_mult;
        df.iter_mut().for_each(|df| *df *= opt.cost_mult);

        let (h, g, dh, dg): (Vec<f64>, Vec<f64>, CSR<usize, f64>, CSR<usize, f64>) =
            if let Some(gh_fn) = nonlinear {
                let (hn, gn, dhn, dgn) = gh_fn.gh(&x, true); // nonlinear constraints
                let (dhn, dgn) = (dhn.unwrap(), dgn.unwrap());

                // inequality constraints
                let h: Vec<f64> = [
                    hn,
                    izip!((&ai_mat * &x), &bi)
                        .map(|(xi, bi)| xi - bi)
                        .collect_vec(),
                ]
                .concat();
                // equality constraints
                let g: Vec<f64> = [
                    gn,
                    izip!(&ae_mat * &x, &be)
                        .map(|(xe, be)| xe - be)
                        .collect_vec(),
                ]
                .concat();

                let dh: CSR<usize, f64> =
                    Coo::h_stack(&dhn.to_coo(), &ai_mat.t().to_coo())?.to_csr(); // 1st derivative of inequalities
                let dg: CSR<usize, f64> =
                    Coo::h_stack(&dgn.to_coo(), &ae_mat.t().to_coo())?.to_csr(); // 1st derivative of equalities

                (h, g, dh, dg)
            } else {
                // inequality constraints
                let h = izip!(&ai_mat * &x, &bi)
                    .map(|(xi, bi)| xi - bi)
                    .collect_vec();
                // equality constraints
                let g = izip!(&ae_mat * &x, &be)
                    .map(|(xe, be)| xe - be)
                    .collect_vec();

                // 1st derivatives are constant, still dh = Ai', dg = Ae' TODO
                let dh = ai_mat.t().to_csr(); // 1st derivative of inequalities
                let dg = ae_mat.t().to_csr(); // 1st derivative of equalities

                (h, g, dh, dg)
            };

        l_x = izip!(&df, &dg * &lam, &dh * &mu)
            .map(|(df, dg_lam, dh_mu)| df + dg_lam + dh_mu)
            .collect_vec();

        let feascond = match max(&h) {
            None => norm_inf(&g) / (1.0 + f64::max(norm_inf(&x), norm_inf(&z))),
            Some(maxh) => {
                f64::max(norm_inf(&g), maxh) / (1.0 + f64::max(norm_inf(&x), norm_inf(&z)))
            }
        };
        let gradcond = norm_inf(&l_x) / (1.0 + f64::max(norm_inf(&lam), norm_inf(&mu)));
        let compcond = dot(&z, &mu) / (1.0 + norm_inf(&x));
        let costcond = (f - f0).abs() / (1.0 + f0.abs());

        if let Some(progress) = progress.as_ref() {
            progress.update(
                iterations,
                feascond,
                gradcond,
                compcond,
                costcond,
                gamma,
                norm(&dx),
                f / opt.cost_mult,
                alphap,
                alphad,
            );
        }
        if feascond < opt.feas_tol
            && gradcond < opt.grad_tol
            && compcond < opt.comp_tol
            && costcond < opt.cost_tol
        {
            converged = true;
            if opt.verbose {
                println!("Converged!");
            }
        } else {
            if x.iter().any(|v| v.is_nan())
                || alphap < alpha_min
                || alphad < alpha_min
                || gamma < f64::EPSILON
                || gamma > 1.0 / f64::EPSILON
            {
                failed = true;
                break;
            }

            f0 = f;
            if opt.step_control {
                l_step = {
                    let hz = izip!(&h, &z).map(|(&h, &z)| h + z).collect_vec();
                    let z_ln_sum: f64 = z.iter().map(|z| z.ln()).sum();

                    f + dot(&lam, &g) + dot(&mu, &hz) - gamma * z_ln_sum
                };
            }
        }
    }
    if opt.verbose {
        if !converged {
            println!("Did not converge in {} iterations.", iterations);
        }
    }

    // Package up results.
    if !converged {
        if failed {
            return Err(format_err!("did not converge: numerically failed"));
        } else {
            return Err(format_err!("did not converge"));
        }
    }

    // zero out multipliers on non-binding constraints
    // mu(h < -opt.feastol & mu < mu_threshold) = 0;
    izip!(&h, &mut mu).for_each(|(&h, mu)| {
        if h < -opt.feas_tol && *mu < mu_threshold {
            *mu = 0.0;
        }
    });

    // un-scale cost and prices
    let f = f / opt.cost_mult;
    lam.iter_mut().for_each(|lam| *lam /= opt.cost_mult);
    mu.iter_mut().for_each(|mu| *mu /= opt.cost_mult);

    // re-package multipliers into struct
    let lam_lin = &lam[neqnln..neq]; // lambda for linear constraints
    let mu_lin = &mu[niqnln..niq]; // mu for linear constraints

    let kl = lam_lin
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v < 0.0 { Some(i) } else { None }) // lower bound binding
        .collect_vec();
    let ku = lam_lin
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v > 0.0 { Some(i) } else { None }) // upper bound binding
        .collect_vec();

    let mut mu_l = vec![0.0; nx + na];
    kl.iter().for_each(|&kl| mu_l[ieq[kl]] = -lam_lin[kl]);
    izip!(&igt, (nlt..(nlt + ngt)).collect_vec()).for_each(|(&igt, i)| mu_l[igt] = mu_lin[i]);
    izip!(&ibx, (nlt + ngt + nbx..nlt + ngt + nbx + nbx).collect_vec())
        .for_each(|(&ibx, i)| mu_l[ibx] = mu_lin[i]);

    let mu_u = vec![0.0; nx + na];
    ku.iter().for_each(|&ku| mu_l[ieq[ku]] = lam_lin[ku]);
    izip!(&ilt, (0..nlt).collect_vec()).for_each(|(&ilt, i)| mu_l[ilt] = mu_lin[i]);
    izip!(&ibx, (nlt + ngt..nlt + ngt + nbx).collect_vec())
        .for_each(|(&ibx, i)| mu_l[ibx] = mu_lin[i]);

    let lambda = Lambda {
        mu_l: mu_l[nx..].to_vec(),
        mu_u: mu_u[nx..].to_vec(),

        lower: mu_l[..nx].to_vec(),
        upper: mu_u[..nx].to_vec(),

        ineq_non_lin: if niqnln > 0 {
            mu[..niqnln].to_vec()
        } else {
            Vec::default()
        },
        eq_non_lin: if neqnln > 0 {
            lam[..neqnln].to_vec()
        } else {
            Vec::default()
        },
    };

    Ok((x, f, converged, iterations, lambda))
}
