use crate::common::*;
use crate::linsol::RLUSolver;
use crate::traits::*;
use ndarray::{concatenate, s, Array1, ArrayView1, Axis};
use sprs::{hstack, vstack, CsMat, CsMatBase, CsMatView, TriMat};

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
pub fn nlp(
    f_fn: &dyn ObjectiveFunction,
    x0: ArrayView1<f64>,
    a_mat: Option<CsMatView<f64>>,
    l: Option<ArrayView1<f64>>,
    u: Option<ArrayView1<f64>>,
    xmin: Option<ArrayView1<f64>>,
    xmax: Option<ArrayView1<f64>>,
    nonlinear: Option<&dyn NonlinearConstraint>,
    solver: Option<&dyn LinearSolver>,
    opt: Option<Options>,
    progress: Option<&dyn ProgressMonitor>,
) -> Result<(Array1<f64>, f64, bool, usize, Lambda), String> {
    let nx = x0.len();

    let a_mat = match a_mat {
        Some(a_mat) => a_mat.to_owned(),
        None => CsMat::<f64>::zero((0, nx)),
    };
    let a_mat = if l.is_some()
        && u.is_some()
        && l.as_ref().unwrap().iter().all(|v| v.is_infinite())
        && u.as_ref().unwrap().iter().all(|v| v.is_infinite())
    {
        // no limits => no linear constraints
        CsMat::zero((0, nx))
    } else {
        a_mat
    };
    let na = a_mat.rows(); // number of original linear constraints

    // By default, linear inequalities are ...
    let u = match u {
        Some(u) => u.to_owned(),
        None => Array1::from_elem(na, f64::INFINITY), // ... unbounded above and ...
    };
    let l = match l {
        Some(l) => l.to_owned(),
        None => Array1::from_elem(na, f64::NEG_INFINITY), // ... unbounded below.
    };

    // By default, optimization variables are ...
    let xmin = match xmin {
        Some(xmin) => xmin.to_owned(),
        None => Array1::from_elem(nx, f64::NEG_INFINITY), // ... unbounded below and ...
    };
    let xmax = match xmax {
        Some(xmax) => xmax.to_owned(),
        None => Array1::from_elem(nx, f64::INFINITY), // ... unbounded above.
    };

    let (gn, hn) = (vec![0.0; 0], vec![0.0; 0]);

    // Set up problem
    let opt = opt.unwrap_or(Options::default());
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
        return Err(format!("xi ({}) must be slightly less than 1", xi));
    }
    if sigma > 1.0 || sigma <= 0.0 {
        return Err(format!("sigma ({}) must be between 0 and 1", sigma));
    }
    // Add var limits to linear constraints.
    let aa_mat = vstack(&[CsMat::<f64>::eye(nx).view(), a_mat.view()]);
    let ll: Array1<f64> = concatenate![Axis(1), xmin, l];
    let uu: Array1<f64> = concatenate![Axis(1), xmax, u];

    // Split up linear constraints.
    let mut ieq = Vec::<usize>::new(); // equality
    let mut igt = Vec::<usize>::new(); // greater than, unbounded above
    let mut ilt = Vec::<usize>::new(); // less than, unbounded below
    let mut ibx = Vec::<usize>::new();
    for i in 0..nx {
        let (ui, li) = (uu[i], ll[i]);
        if (ui - li).abs() <= f64::EPSILON {
            ieq.push(i);
        } else if ui >= 1e-10 && li > -1e-10 {
            igt.push(i);
        } else if li <= -1e-10 && ui < 1e10 {
            ilt.push(i);
        } else {
            ibx.push(i);
        }
    }
    let ae_mat = &aa_mat; // FIXME: select(ieq)
    let be: Array1<f64> = ieq.iter().map(|&i| uu[i]).collect();
    let ai_mat = &aa_mat; // FIXME: select
    let bi: Array1<f64> = Array1::from(
        [
            ilt.iter().map(|i| uu[*i]).collect::<Vec<f64>>(),
            igt.iter().map(|i| -ll[*i]).collect::<Vec<f64>>(),
            ibx.iter().map(|i| uu[*i]).collect::<Vec<f64>>(),
            ibx.iter().map(|i| -ll[*i]).collect::<Vec<f64>>(),
        ]
        .concat(),
    );

    // Evaluate cost f(x0) and constraints g(x0), h(x0)
    let mut x = x0.to_owned();

    let (f, df, _) = f_fn.f(x.view(), false);

    let f = f * opt.cost_mult;
    let df: Array1<f64> = df * opt.cost_mult;

    let (h, g, dh, dg): (Array1<f64>, Array1<f64>, CsMat<f64>, CsMat<f64>) =
        if let Some(gh_fn) = nonlinear {
            let (hn, gn, dhn, dgn) = gh_fn.gh(x.view(), true); // nonlinear constraints

            let h: Array1<f64> = concatenate![Axis(1), hn, (&ai_mat.view() * &x.view()) - &bi]; // inequality constraints
            let g: Array1<f64> = concatenate![Axis(1), gn, (&ae_mat.view() * &x.view()) - &be]; // equality constraints

            let dh: CsMat<f64> = hstack(&[dhn.unwrap().view(), ai_mat.transpose_view()]); // 1st derivative of inequalities
            let dg: CsMat<f64> = hstack(&[dgn.unwrap().view(), ae_mat.transpose_view()]); // 1st derivative of equalities

            (h, g, dh, dg)
        } else {
            let h = (&ai_mat.view() * &x.view()) - &bi; // inequality constraints
            let g = (&ae_mat.view() * &x.view()) - &be; // equality constraints

            let dh = ai_mat.view().transpose_into().to_csr(); // 1st derivative of inequalities
            let dg = ae_mat.view().transpose_into().to_csr(); // 1st derivative of equalities

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

    // Initialize gamma, lam, mu, z, e.
    let mut gamma = 1.0; // Barrier coefficient, r in Harry's code.
    let mut lam = Array1::<f64>::zeros(neq);
    // let mut z = Array1::from_elem(niq, z0);
    let mut z = Array1::from(
        h.iter()
            .map(|&hk| if hk < -z0 { -hk } else { z0 })
            .collect::<Vec<f64>>(),
    );
    // let mu = z.clone();
    let mut mu = Array1::from(
        z.iter()
            .map(|&zk| if gamma / zk > z0 { gamma / zk } else { z0 })
            .collect::<Vec<f64>>(),
    );
    let e = Array1::<f64>::ones(niq);

    // check tolerance
    let mut f0 = f.clone();
    let mut l_step: f64 = if opt.step_control {
        f + lam.dot(&g) + mu.dot(&(&h + &z)) - gamma * z.ln_sum()
    } else {
        0.0
    };
    let mut l_x = df + (&dg.view() * &lam.view()) + (&dh.view() * &mu.view());
    let maxh: f64 = h.maximum();

    let feascond = g.norm_inf().max(maxh) / (1.0 + x.norm_inf().max(z.norm_inf()));
    let gradcond = l_x.norm_inf() / (1.0 + lam.norm_inf().max(mu.norm_inf()));
    let compcond = z.dot(&mu.view()) / (1.0 + x.norm_inf());
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
            eq_non_lin: (0..neqnln).map(|i| lam[i]).collect(),
            ineq_non_lin: (0..niqnln).map(|i| mu[i]).collect(),
            ..Default::default()
        };
        let l_xx = if let Some(hess_fn) = nonlinear {
            hess_fn.hess(x.view(), &lambda, opt.cost_mult)
        } else {
            let (_, _, mut d2f) = f_fn.f(x.view(), true); // cost
            d2f.as_mut().unwrap().scale(opt.cost_mult);
            d2f.unwrap()
        };

        let zinvdiag = TriMat::<f64>::from_triplets(
            (niq, niq),
            Vec::from_iter(0..niq),
            Vec::from_iter(0..niq),
            z.iter().map(|v| v.recip()).collect(),
        )
        .to_csr::<usize>();
        let mudiag = TriMat::<f64>::from_triplets(
            (niq, niq),
            Vec::from_iter(0..niq),
            Vec::from_iter(0..niq),
            mu.to_vec(),
        )
        .to_csr::<usize>();
        let dh_zinv = &dh.view() * &zinvdiag.view();

        // M = Lxx + dh_zinv * mudiag * dh';
        let m_mat: CsMat<f64> =
            &l_xx.view() + &(&(&dh_zinv.view() * &mudiag.view()).view() * &dh.transpose_view());

        // N = Lx + dh_zinv * (mudiag * h + gamma * e);
        let n: Array1<f64> = &l_x.view()
            + (&dh_zinv.view() * &((&mudiag.view() * &h.view()) + (gamma * &e.view())).view());

        let dxdlam = {
            let a_mat: CsMat<f64> = vstack(&[
                hstack(&[m_mat.view(), dg.view()]).view(),
                hstack(&[dg.transpose_view(), CsMatBase::zero((neq, neq)).view()]).view(),
            ]);
            let mut b: Array1<f64> = Array1::from(
                [
                    n.iter().map(|v| -v).collect::<Vec<f64>>(),
                    g.iter().map(|v| -v).collect::<Vec<f64>>(),
                ]
                .concat(),
            );
            solver
                .unwrap_or(&RLUSolver::default())
                .solve(a_mat.view(), b.view_mut())?;
            b
        };
        if dxdlam.iter().any(|v| v.is_nan()) || dxdlam.norm() > max_step_size {
            failed = true;
            break;
        }
        let mut dx = (0..nx).map(|i| dxdlam[i]).collect::<Array1<f64>>();
        let mut dlam = (nx..nx + neq).map(|i| dxdlam[i]).collect::<Array1<f64>>();
        let mut dz = -&h - &z - (&dh.transpose_view() * &dx.view());
        let mut dmu = -&mu + (&zinvdiag.view() * &((gamma * &e) - (&mudiag.view() * &dz.view())));

        // Optional step-size control.
        let sc = if opt.step_control {
            let x1 = &x + &dx;

            // Evaluate cost, constraints, derivatives at x1.
            let (f1, df1, _) = f_fn.f(x1.view(), false); // cost
            let f1 = f1 * opt.cost_mult;
            let df1 = df1 * opt.cost_mult;

            let (h1, g1, dh1, dg1): (Array1<f64>, Array1<f64>, CsMat<f64>, CsMat<f64>) =
                if let Some(gh_fn) = nonlinear {
                    let (hn1, gn1, dhn1, dgn1) = gh_fn.gh(x1.view(), true); // nonlinear constraints

                    let h1: Array1<f64> =
                        concatenate![Axis(1), hn1, (&ai_mat.view() * &x1.view()) - &bi]; // inequality constraints
                    let g1: Array1<f64> =
                        concatenate![Axis(1), gn1, (&ae_mat.view() * &x1.view()) - &be]; // equality constraints

                    let dh1: CsMat<f64> = hstack(&[dhn1.unwrap().view(), ai_mat.transpose_view()]); // 1st derivative of inequalities
                    let dg1: CsMat<f64> = hstack(&[dgn1.unwrap().view(), ae_mat.transpose_view()]); // 1st derivative of equalities

                    (h1, g1, dh1, dg1)
                } else {
                    let h1 = (&ai_mat.view() * &x1.view()) - &bi; // inequality constraints
                    let g1 = (&ae_mat.view() * &x1.view()) - &be; // equality constraints

                    let dh1 = ai_mat.view().transpose_into().to_csr(); // dh // 1st derivative of inequalities
                    let dg1 = ae_mat.view().transpose_into().to_csr(); // dg // 1st derivative of equalities

                    (h1, g1, dh1, dg1)
                };

            // check tolerance
            let l_x1 = df1 + (&dg1.view() * &lam.view()) + (&dh1.view() * &mu.view());
            let maxh1: f64 = h1.maximum();

            let feascond1 = g1.norm_inf().max(maxh1) / (1.0 + x1.norm_inf().max(z.norm_inf()));
            let gradcond1 = l_x1.norm_inf() / (1.0 + lam.norm_inf().max(mu.norm_inf()));

            feascond1 > feascond && gradcond1 > gradcond
        } else {
            false
        };
        if sc {
            let mut alpha = 1.0;
            for j in 0..opt.max_red {
                let dx1 = alpha * &dx;
                let x1 = &x + &dx1;
                let (f1, _, _) = f_fn.f(x1.view(), false); // cost
                let f1 = f1 * opt.cost_mult;
                let (h1, g1) = if let Some(gh_fn) = nonlinear {
                    let (hn1, gn1, _, _) = gh_fn.gh(x1.view(), false); // nonlinear constraints
                    let h1 = concatenate![Axis(1), hn1, (&ai_mat.view() * &x1.view()) - &bi]; // inequality constraints
                    let g1 = concatenate![Axis(1), gn1, (&ae_mat.view() * &x1.view()) - &be]; // equality constraints
                    (h1, g1)
                } else {
                    let h1 = (&ai_mat.view() * &x1.view()) - &bi; // inequality constraints
                    let g1 = (&ae_mat.view() * &x1.view()) - &be; // equality constraints
                    (h1, g1)
                };
                let l1 = f1 + lam.dot(&g1) + mu.dot(&(&h1 + &z)) - gamma * z.ln_sum();
                if opt.verbose {
                    print!("{} {}", -(j as isize), dx1.norm());
                }
                let rho =
                    (l1 - l_step) / (l_x.dot(&dx1) + 0.5 * dx1.dot(&(&l_xx.view() * &dx1.view())));
                if rho > rho_min && rho < rho_max {
                    break;
                } else {
                    alpha = alpha / 2.0;
                }
            }
            dx = alpha * dx;
            dz = alpha * dz;
            dlam = alpha * dlam;
            dmu = alpha * dmu;
        }
        // do the update
        // k = find(dz < 0);
        let k = dz
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v < 0.0 { Some(i) } else { None })
            .collect::<Vec<usize>>();
        let alphap = if k.is_empty() {
            1.0
        } else {
            // alphap = min( [xi * min(z(k) ./ -dz(k)) 1] );
            (xi * (&k.iter().map(|&i| z[i] / -dz[i]).collect::<Vec<f64>>()).minimum()).min(1.0)
        };
        // k = find(dmu < 0);
        let k = dmu
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v < 0.0 { Some(i) } else { None })
            .collect::<Vec<usize>>();
        let alphad = if k.is_empty() {
            1.0
        } else {
            // alphad = min( [xi * min(mu(k) ./ -dmu(k)) 1] );
            (xi * (&k.iter().map(|&i| mu[i] / -dmu[i]).collect::<Vec<f64>>()).minimum()).min(1.0)
        };
        x = x + alphap * &dx;
        z = z + alphap * &dz;
        lam = lam + alphad * &dlam;
        mu = mu + alphad * &dmu;
        if niq > 0 {
            gamma = sigma * z.dot(&mu) / (niq as f64);
        }

        // evaluate cost, constraints, derivatives
        let (f, df, _) = f_fn.f(x.view(), false); // cost
        let f = f * opt.cost_mult;
        let df = df * opt.cost_mult;

        let (h, g, dh, dg): (Array1<f64>, Array1<f64>, CsMat<f64>, CsMat<f64>) =
            if let Some(gh_fn) = nonlinear {
                let (hn, gn, dhn, dgn) = gh_fn.gh(x.view(), true); // nonlinear constraints

                let h: Array1<f64> = concatenate![Axis(1), hn, (&ai_mat.view() * &x.view()) - &bi]; // inequality constraints
                let g: Array1<f64> = concatenate![Axis(1), gn, (&ae_mat.view() * &x.view()) - &be]; // equality constraints

                let dh: CsMat<f64> = hstack(&[dhn.unwrap().view(), ai_mat.transpose_view()]); // 1st derivative of inequalities
                let dg: CsMat<f64> = hstack(&[dgn.unwrap().view(), ae_mat.transpose_view()]); // 1st derivative of equalities

                (h, g, dh, dg)
            } else {
                let h = (&ai_mat.view() * &x.view()) - &bi; // inequality constraints
                let g = (&ae_mat.view() * &x.view()) - &be; // equality constraints

                // 1st derivatives are constant, still dh = Ai', dg = Ae' TODO
                let dh = ai_mat.view().transpose_into().to_csr(); // 1st derivative of inequalities
                let dg = ae_mat.view().transpose_into().to_csr(); // 1st derivative of equalities

                (h, g, dh, dg)
            };

        l_x = df + (&dg.view() * &lam.view()) + (&dh.view() * &mu.view());
        let maxh: f64 = h.maximum();

        let feascond = g.norm_inf().max(maxh) / (1.0 + x.norm_inf().max(z.norm_inf()));
        let gradcond = l_x.norm_inf() / (1.0 + lam.norm_inf().max(mu.norm_inf()));
        let compcond = z.dot(&mu.view()) / (1.0 + x.norm_inf());
        let costcond = (f - f0).abs() / (1.0 + f0.abs());

        if let Some(progress) = progress.as_ref() {
            progress.update(
                iterations,
                feascond,
                gradcond,
                compcond,
                costcond,
                gamma,
                dx.norm(),
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
                l_step = f + lam.dot(&g) + mu.dot(&(&h + &z)) - gamma * z.ln_sum()
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
            return Err("did not converge: numerically failed".to_string());
        } else {
            return Err("did not converge".to_string());
        }
    }

    // zero out multipliers on non-binding constraints
    // mu(h < -opt.feastol & mu < mu_threshold) = 0;
    mu = mu
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if h[i] < -opt.feas_tol && v < mu_threshold {
                0.0
            } else {
                v
            }
        })
        .collect();

    // un-scale cost and prices
    let f = f / opt.cost_mult;
    lam = lam / opt.cost_mult;
    mu = mu / opt.cost_mult;

    // re-package multipliers into struct
    let lam_lin: ArrayView1<f64> = lam.slice(s![neqnln..neq]); // lambda for linear constraints
    let mu_lin: ArrayView1<f64> = mu.slice(s![niqnln..niq]); // mu for linear constraints

    let kl = lam_lin
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v < 0.0 { Some(i) } else { None }) // lower bound binding
        .collect::<Vec<usize>>();
    let ku = lam_lin
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v > 0.0 { Some(i) } else { None }) // upper bound binding
        .collect::<Vec<usize>>();

    let mu_l = Array1::zeros(nx + na);
    // FIXME

    let mu_u = Array1::zeros(nx + na);

    let lambda = Lambda {
        mu_l: mu_l.slice(s![nx..]).to_owned(),
        mu_u: mu_u.slice(s![nx..]).to_owned(),

        lower: mu_l.slice(s![..nx]).to_owned(),
        upper: mu_u.slice(s![..nx]).to_owned(),

        ineq_non_lin: if niqnln > 0 {
            mu.slice(s![..niqnln]).to_owned()
        } else {
            Array1::zeros(0)
        },
        eq_non_lin: if neqnln > 0 {
            lam.slice(s![..neqnln]).to_owned()
        } else {
            Array1::zeros(0)
        },
    };

    Ok((x, f, converged, iterations, lambda))
}
