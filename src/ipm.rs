use std::iter::zip;

use anyhow::{format_err, Result};
use log::{debug, info, trace};
use sparsetools::coo::Coo;
use sparsetools::csc::CSC;
use sparsetools::csr::CSR;
use spsolve::Solver;

use crate::common::{Lambda, Options};
use crate::math::*;
use crate::traits::*;

/// Primal-dual interior point method for NLP (nonlinear programming).
/// Minimize a function `F(x)` beginning from a starting point `x0`, subject
/// to optional linear and nonlinear constraints and variable bounds.
///
/// ```txt
///       min F(x)
///        x
/// ```
///
/// subject to
///
/// ```txt
///       g(x) = 0            (nonlinear equalities)
///       h(x) <= 0           (nonlinear inequalities)
///       l <= A*x <= u       (linear constraints)
///       xmin <= x <= xmax   (variable bounds)
/// ```
///
/// Returns the solution vector `x`, the final objective function value `f`,
/// an exit flag indicating if the solver converged, the number of iterations
/// performed and a structure containing the Lagrange and Kuhn-Tucker
/// multipliers on the constraints.
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

    debug!("nx = {}, nA = {}", nx, na);

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

    // Set up problem
    let (xi, sigma, z0, alpha_min, mu_threshold, max_step_size) = (
        opt.xi,
        opt.sigma,
        opt.z0,
        opt.alpha_min,
        opt.mu_threshold,
        opt.max_step_size,
    );
    if xi >= 1.0 || xi < 0.5 {
        return Err(format_err!("xi ({}) must be slightly less than 1", xi));
    }
    if sigma > 1.0 || sigma <= 0.0 {
        return Err(format_err!("sigma ({}) must be between 0 and 1", sigma));
    }
    debug!(
        "xi = {}, sigma = {}, z0 = {}, max_step_size = {:e}",
        xi, sigma, z0, max_step_size
    );
    #[cfg(feature = "step-control")]
    let (rho_min, rho_max) = (opt.rho_min, opt.rho_max);
    #[cfg(feature = "step-control")]
    if opt.step_control {
        debug!(
            "step control enabled: rho_min = {}, rho_max = {}",
            rho_min, rho_max
        );
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
        }
        if ui >= 1e10 && li > -1e10 {
            igt.push(i);
        }
        if li <= -1e10 && ui < 1e10 {
            ilt.push(i);
        }
        if ((ui - li).abs() > f64::EPSILON) && (ui < 1e10) && (li > -1e10) {
            ibx.push(i);
        }
    }
    let ae_mat = aa_mat.select(Some(&ieq), None)?;
    let be: Vec<f64> = ieq.iter().map(|&i| uu[i]).collect();
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
        ilt.iter().map(|&i| uu[i]).collect::<Vec<f64>>(),
        igt.iter().map(|&i| -ll[i]).collect::<Vec<f64>>(),
        ibx.iter().map(|&i| uu[i]).collect::<Vec<f64>>(),
        ibx.iter().map(|&i| -ll[i]).collect::<Vec<f64>>(),
    ]
    .concat();

    // Evaluate cost f(x0) and constraints g(x0), h(x0)
    let mut x = x0.to_vec();

    let (mut f, mut df, _) = f_fn.f(&x, false);

    f *= opt.cost_mult;
    df.iter_mut().for_each(|df| *df *= opt.cost_mult);

    debug!("f = {}, df = {:?}", f, df);

    let (hn, gn, mut h, mut g, mut dh, mut dg): (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        CSR<usize, f64>,
        CSR<usize, f64>,
    ) = if let Some(gh_fn) = nonlinear {
        let (hn, gn, dhn, dgn) = gh_fn.gh(&x, true); // nonlinear constraints
        let (dhn, dgn) = (dhn.unwrap(), dgn.unwrap());

        let h: Vec<f64> = [
            hn.clone(),
            zip(&ai_mat * &x, &bi).map(|(xi, bi)| xi - bi).collect(),
        ]
        .concat(); // inequality constraints
        let g: Vec<f64> = [
            gn.clone(),
            zip(&ae_mat * &x, &be).map(|(xe, be)| xe - be).collect(),
        ]
        .concat(); // equality constraints

        let dh: CSR<usize, f64> = Coo::h_stack(&dhn.to_coo(), &ai_mat.t().to_coo())?.to_csr(); // 1st derivative of inequalities
        let dg: CSR<usize, f64> = Coo::h_stack(&dgn.to_coo(), &ae_mat.t().to_coo())?.to_csr(); // 1st derivative of equalities

        (hn, gn, h, g, dh, dg)
    } else {
        let gn = Vec::<f64>::default();
        let hn = Vec::<f64>::default();

        let h = zip(&ai_mat * &x, &bi).map(|(xi, bi)| xi - bi).collect(); // inequality constraints
        let g = zip(&ae_mat * &x, &be).map(|(xe, be)| xe - be).collect(); // equality constraints

        let dh = ai_mat.t().to_csr(); // 1st derivative of inequalities
        let dg = ae_mat.t().to_csr(); // 1st derivative of equalities

        (hn, gn, h, g, dh, dg)
    };

    // Grab some dimensions.
    let neq = g.len(); // number of equality constraints
    let niq = h.len(); // number of inequality constraints
    let neqnln = gn.len(); // number of nonlinear equality constraints
    let niqnln = hn.len(); // number of nonlinear inequality constraints
    let nlt = ilt.len(); // number of upper bounded linear inequalities
    let ngt = igt.len(); // number of lower bounded linear inequalities
    let nbx = ibx.len(); // number of doubly bounded linear inequalities

    debug!(
        "neq = {}, niq = {}, neqnln = {}, niqnln = {}, nlt = {}, ngt = {}, nbx = {}",
        neq, niq, neqnln, niqnln, nlt, ngt, nbx
    );

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
    zip(&h, &mut z).for_each(|(&h, z)| {
        if h < -z0 {
            *z = -h;
        }
    });
    // (seems k is always empty if gamma = z0 = 1)
    zip(&z, &mut mu).for_each(|(&z, mu)| {
        if gamma / z > z0 {
            *mu = gamma / z;
        }
    });
    let e = vec![1.0; niq];

    // check tolerance
    let mut f0 = f;
    #[cfg(feature = "step-control")]
    let mut l_step: f64 = if opt.step_control {
        let hz: Vec<f64> = zip(&h, &z).map(|(&h, &z)| h + z).collect();
        let z_ln_sum: f64 = z.iter().map(|z| z.ln()).sum();

        f + dot(&lam, &g) + dot(&mu, &hz) - gamma * z_ln_sum
    } else {
        0.0
    };
    // let mut l_x = df + (&dg * &lam) + (&dh * &mu);
    let mut l_x: Vec<f64> = zip(df, &dg * &lam)
        .zip(&dh * &mu)
        .map(|((df, dg_lam), dh_mu)| df + dg_lam + dh_mu)
        .collect();
    trace!("Lx = {:?}", l_x);

    let feascond = match max(&h) {
        None => norm_inf(&g) / (1.0 + f64::max(norm_inf(&x), norm_inf(&z))),
        Some(maxh) => f64::max(norm_inf(&g), maxh) / (1.0 + f64::max(norm_inf(&x), norm_inf(&z))),
    };
    let gradcond = norm_inf(&l_x) / (1.0 + f64::max(norm_inf(&lam), norm_inf(&mu)));
    let compcond = dot(&z, &mu) / (1.0 + norm_inf(&x));
    let costcond = (f - f0).abs() / (1.0 + f0.abs());

    debug!(
        "feascond = {}, gradcond = {}, compcond = {}, costcond = {}",
        feascond, gradcond, compcond, costcond
    );

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
        debug!("Newton iteration {}...", iterations);

        // Compute update step.
        let lambda = Lambda {
            eq_non_lin: (0..neqnln).map(|i| lam[i]).collect(),
            ineq_non_lin: (0..niqnln).map(|i| mu[i]).collect(),
            ..Default::default()
        };
        let l_xx = if let Some(hess_fn) = nonlinear {
            hess_fn.hess(&x, &lambda, opt.cost_mult)
        } else {
            let (_, _, d2f) = f_fn.f(&x, true); // cost
            d2f.unwrap() * opt.cost_mult
        };
        trace!("Lxx:\n{}", l_xx.to_table());

        let zinvdiag = {
            let zinv: Vec<f64> = z.iter().map(|z| 1.0 / z).collect();
            Coo::<usize, f64>::with_diagonal(&zinv).to_csr()
        };
        let mudiag = Coo::<usize, f64>::with_diagonal(&mu).to_csr();
        let dh_zinv = &dh * &zinvdiag;

        // M = Lxx + dh_zinv * mudiag * dh';
        // let m_mat: CSR<usize, f64> = &l_xx + &((&dh_zinv * &mudiag) * &dh.t().to_csr());
        let m_mat: CSR<usize, f64> = &l_xx + &(&dh_zinv * (&mudiag * &dh.t().to_csr()));

        // N = Lx + dh_zinv * (mudiag * h + gamma * e);
        let n: Vec<f64> = {
            let temp: Vec<f64> = zip(&mudiag * &h, &e)
                .map(|(mudiag_h, e)| mudiag_h + gamma * e)
                .collect();
            zip(&l_x, &dh_zinv * &temp)
                .map(|(l_x, dh_zinv_temp)| l_x + dh_zinv_temp)
                .collect()
        };

        let dxdlam = {
            let a_mat: CSC<usize, f64> = Coo::compose([
                [&m_mat.to_coo(), &dg.to_coo()],
                [&dg.t().to_coo(), &Coo::with_size(neq, neq)],
            ])?
            .to_csc();
            let mut b: Vec<f64> = [
                n.iter().map(|n| -n).collect::<Vec<f64>>(),
                g.iter().map(|g| -g).collect::<Vec<f64>>(),
            ]
            .concat();
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
        // let mut dx: Vec<f64> = (0..nx).map(|i| dxdlam[i]).collect();
        let dx = dxdlam[0..nx].to_vec();
        #[cfg(feature = "step-control")]
        let mut dx = dx;
        trace!("dx = {:?}", dx);

        // let mut dlam: Vec<f64> = (nx..(nx + neq)).map(|i| dxdlam[i]).collect();
        let dlam = dxdlam[nx..(nx + neq)].to_vec();
        #[cfg(feature = "step-control")]
        let mut dlam = dlam;
        trace!("dlam = {:?}", dlam);

        // dz = -h - z - dh' * dx;
        let dz: Vec<f64> = zip(&h, &z)
            .zip(&dh.t().to_csr() * &dx) // fixme: to_csr()
            .map(|((h, z), dh_dx)| -h - z - dh_dx)
            .collect();
        #[cfg(feature = "step-control")]
        let mut dz = dz;
        trace!("dz = {:?}", dz);

        // dmu = -mu + zinvdiag *(gamma*e - mudiag * dz);
        let dmu: Vec<f64> = {
            let temp: Vec<f64> = zip(&e, &mudiag * &dz)
                .map(|(e, mudiag_dz)| gamma * e - mudiag_dz)
                .collect();

            zip(&mu, &zinvdiag * &temp)
                .map(|(mu, zinvdiag_temp)| -mu + zinvdiag_temp)
                .collect()
        };
        #[cfg(feature = "step-control")]
        let mut dmu = dmu;
        trace!("dmu = {:?}", dmu);

        // Optional step-size control.
        #[cfg(feature = "step-control")]
        let sc = if opt.step_control {
            let x1: Vec<f64> = zip(&x, &dx).map(|(x, dx)| x + dx).collect();

            // Evaluate cost, constraints, derivatives at x1.
            let (mut _f1, mut df1, _) = f_fn.f(&x1, false); // cost
            _f1 *= opt.cost_mult;
            df1.iter_mut().for_each(|df1| *df1 *= opt.cost_mult);

            let (h1, g1, dh1, dg1): (Vec<f64>, Vec<f64>, CSR<usize, f64>, CSR<usize, f64>) =
                if let Some(gh_fn) = nonlinear {
                    let (hn1, gn1, dhn1, dgn1) = gh_fn.gh(&x1, true); // nonlinear constraints
                    let (dhn1, dgn1) = (dhn1.unwrap(), dgn1.unwrap());

                    // inequality constraints
                    let h1: Vec<f64> = [
                        hn1,
                        zip(&ai_mat * &x1, &bi).map(|(xi, bi)| xi - bi).collect(),
                    ]
                    .concat();

                    // equality constraints
                    let g1: Vec<f64> = [
                        gn1,
                        zip(&ae_mat * &x1, &be).map(|(xe, be)| xe - be).collect(),
                    ]
                    .concat();

                    let dh1: CSR<usize, f64> =
                        Coo::h_stack(&dhn1.to_coo(), &ai_mat.t().to_coo())?.to_csr(); // 1st derivative of inequalities
                    let dg1: CSR<usize, f64> =
                        Coo::h_stack(&dgn1.to_coo(), &ae_mat.t().to_coo())?.to_csr(); // 1st derivative of equalities

                    (h1, g1, dh1, dg1)
                } else {
                    // inequality constraints
                    let h1 = zip(&ai_mat * &x1, &bi).map(|(xi, bi)| xi - bi).collect();

                    // equality constraints
                    let g1 = zip(&ae_mat * &x1, &be).map(|(xe, be)| xe - be).collect();

                    let dh1 = ai_mat.t().to_csr(); // dh // 1st derivative of inequalities
                    let dg1 = ae_mat.t().to_csr(); // dg // 1st derivative of equalities

                    (h1, g1, dh1, dg1)
                };

            // check tolerance
            let l_x1: Vec<f64> = zip(&df1, &dg1 * &lam)
                .zip(&dh1 * &mu)
                .map(|((df1, dg1_lam), dh1_mu)| df1 + dg1_lam + dh1_mu)
                .collect();

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
        #[cfg(feature = "step-control")]
        if sc {
            let mut alpha = 1.0;
            for j in 0..opt.max_red {
                let dx1: Vec<f64> = dx.iter().map(|dx| alpha * dx).collect();
                let x1: Vec<f64> = zip(&x, &dx1).map(|(&x, &dx1)| x + dx1).collect();
                let (mut f1, _, _) = f_fn.f(&x1, false); // cost
                f1 *= opt.cost_mult;
                let (h1, g1) = if let Some(gh_fn) = nonlinear {
                    let (hn1, gn1, _, _) = gh_fn.gh(&x1, false); // nonlinear constraints

                    // inequality constraints
                    let h1 = [
                        hn1,
                        zip(&ai_mat * &x1, &bi).map(|(xi, bi)| xi - bi).collect(),
                    ]
                    .concat();

                    // equality constraints
                    let g1 = [
                        gn1,
                        zip(&ae_mat * &x1, &be).map(|(xe, be)| xe - be).collect(),
                    ]
                    .concat();

                    (h1, g1)
                } else {
                    // inequality constraints
                    let h1 = zip(&ai_mat * &x1, &bi).map(|(xi, bi)| xi - bi).collect();

                    // equality constraints
                    let g1 = zip(&ae_mat * &x1, &be).map(|(xe, be)| xe - be).collect();

                    (h1, g1)
                };

                // L1 = f1 + lam' * g1 + mu' * (h1+z) - gamma * sum(log(z));
                let l1: f64 = {
                    let hz: Vec<f64> = zip(&h1, &z).map(|(&h1, &z1)| h1 + z1).collect();
                    let z_ln_sum: f64 = z.iter().map(|z| z.ln()).sum();

                    f1 + dot(&lam, &g1) + dot(&mu, &hz) - gamma * z_ln_sum
                };
                debug!("reduction {}: {}", j + 1, norm(&dx1));

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
        let k: Vec<usize> = dz
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v < 0.0 { Some(i) } else { None })
            .collect();
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
        let k: Vec<usize> = dmu
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v < 0.0 { Some(i) } else { None })
            .collect();
        let alphad = if k.is_empty() {
            1.0
        } else {
            // f64::min(xi * min(mu.select(&k) / -dmu.select(&k)), 1.0)
            f64::min(
                xi * min(&k.iter().map(|&i| mu[i] / -dmu[i]).collect::<Vec<f64>>()).unwrap(),
                1.0,
            )
        };
        zip(&mut x, &dx).for_each(|(x, dx)| *x += alphap * dx);
        zip(&mut z, &dz).for_each(|(z, dz)| *z += alphap * dz);
        zip(&mut lam, &dlam).for_each(|(lam, dlam)| *lam += alphad * dlam);
        zip(&mut mu, &dmu).for_each(|(mu, dmu)| *mu += alphad * dmu);
        if niq > 0 {
            gamma = sigma * dot(&z, &mu) / (niq as f64);
        }
        trace!("x = {:?}", x);
        trace!("z = {:?}", z);
        trace!("lam = {:?}", lam);
        trace!("mu = {:?}", mu);

        // evaluate cost, constraints, derivatives
        (f, df, _) = f_fn.f(&x, false); // cost
        f *= opt.cost_mult;
        df.iter_mut().for_each(|df| *df *= opt.cost_mult);

        debug!("f = {}, df = {:?}", f, df);

        (h, g, dh, dg) = if let Some(gh_fn) = nonlinear {
            let (hn, gn, dhn, dgn) = gh_fn.gh(&x, true); // nonlinear constraints
            let (dhn, dgn) = (dhn.unwrap(), dgn.unwrap());

            // inequality constraints
            let h: Vec<f64> =
                [hn, zip(&ai_mat * &x, &bi).map(|(xi, bi)| xi - bi).collect()].concat();
            // equality constraints
            let g: Vec<f64> =
                [gn, zip(&ae_mat * &x, &be).map(|(xe, be)| xe - be).collect()].concat();

            let dh: CSR<usize, f64> = Coo::h_stack(&dhn.to_coo(), &ai_mat.t().to_coo())?.to_csr(); // 1st derivative of inequalities
            let dg: CSR<usize, f64> = Coo::h_stack(&dgn.to_coo(), &ae_mat.t().to_coo())?.to_csr(); // 1st derivative of equalities

            (h, g, dh, dg)
        } else {
            // inequality constraints
            let h = zip(&ai_mat * &x, &bi).map(|(xi, bi)| xi - bi).collect();
            // equality constraints
            let g = zip(&ae_mat * &x, &be).map(|(xe, be)| xe - be).collect();

            // 1st derivatives are constant, still dh = Ai', dg = Ae' TODO
            let dh = ai_mat.t().to_csr(); // 1st derivative of inequalities
            let dg = ae_mat.t().to_csr(); // 1st derivative of equalities

            (h, g, dh, dg)
        };

        l_x = zip(&df, &dg * &lam)
            .zip(&dh * &mu)
            .map(|((df, dg_lam), dh_mu)| df + dg_lam + dh_mu)
            .collect();
        trace!("Lx: {:?}", l_x);

        let feascond = match max(&h) {
            None => norm_inf(&g) / (1.0 + f64::max(norm_inf(&x), norm_inf(&z))),
            Some(maxh) => {
                f64::max(norm_inf(&g), maxh) / (1.0 + f64::max(norm_inf(&x), norm_inf(&z)))
            }
        };
        let gradcond = norm_inf(&l_x) / (1.0 + f64::max(norm_inf(&lam), norm_inf(&mu)));
        let compcond = dot(&z, &mu) / (1.0 + norm_inf(&x));
        let costcond = (f - f0).abs() / (1.0 + f0.abs());

        debug!(
            "feascond = {}, gradcond = {}, compcond = {}, costcond = {}",
            feascond, gradcond, compcond, costcond
        );

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
            info!("Converged in {} iterations", iterations);
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
            #[cfg(feature = "step-control")]
            if opt.step_control {
                l_step = {
                    let hz: Vec<f64> = zip(&h, &z).map(|(&h, &z)| h + z).collect();
                    let z_ln_sum: f64 = z.iter().map(|z| z.ln()).sum();

                    f + dot(&lam, &g) + dot(&mu, &hz) - gamma * z_ln_sum
                };
            }
        }
    }
    if !converged {
        info!("Did not converge in {} iterations.", iterations);
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
    zip(&h, &mut mu).for_each(|(&h, mu)| {
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

    let kl: Vec<usize> = lam_lin
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v < 0.0 { Some(i) } else { None }) // lower bound binding
        .collect();
    let ku: Vec<usize> = lam_lin
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v > 0.0 { Some(i) } else { None }) // upper bound binding
        .collect();

    let mut mu_l = vec![0.0; nx + na];
    kl.iter().for_each(|&kl| mu_l[ieq[kl]] = -lam_lin[kl]);
    zip(&igt, &mu_lin[nlt..(nlt + ngt)]).for_each(|(&igt, &mu_lin)| mu_l[igt] = mu_lin);
    zip(&ibx, &mu_lin[nlt + ngt + nbx..nlt + ngt + nbx + nbx])
        .for_each(|(&ibx, &mu_lin)| mu_l[ibx] = mu_lin);

    let mu_u = vec![0.0; nx + na];
    ku.iter().for_each(|&ku| mu_l[ieq[ku]] = lam_lin[ku]);
    zip(&ilt, &mu_lin[0..nlt]).for_each(|(&ilt, &mu_lin)| mu_l[ilt] = mu_lin);
    zip(&ibx, &mu_lin[nlt + ngt..nlt + ngt + nbx]).for_each(|(&ibx, &mu_lin)| mu_l[ibx] = mu_lin);

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

    debug!("mu_l: {:?}", lambda.mu_l);
    debug!("mu_u: {:?}", lambda.mu_u);
    debug!("lower: {:?}", lambda.lower);
    debug!("upper: {:?}", lambda.upper);
    debug!("eq_non_lin: {:?}", lambda.eq_non_lin);
    debug!("ineq_non_lin: {:?}", lambda.ineq_non_lin);

    Ok((x, f, converged, iterations, lambda))
}
