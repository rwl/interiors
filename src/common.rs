/// Lagrange and Kuhn-Tucker multipliers on the constraints.
pub struct Lambda {
    /// Multipliers on the equality constraints.
    pub eq_non_lin: Vec<f64>,
    /// Multipliers on the inequality constraints.
    pub ineq_non_lin: Vec<f64>,

    /// Lower (left-hand) limit on linear constraints.
    pub mu_l: Vec<f64>,
    /// Upper (right-hand) limit on linear constraints.
    pub mu_u: Vec<f64>,

    /// Lower bound on optimization variables.
    pub lower: Vec<f64>,
    /// Upper bound on optimization variables.
    pub upper: Vec<f64>,
}

impl Default for Lambda {
    fn default() -> Self {
        Self {
            eq_non_lin: Vec::new(),
            ineq_non_lin: Vec::new(),

            mu_l: Vec::new(),
            mu_u: Vec::new(),

            lower: Vec::new(),
            upper: Vec::new(),
        }
    }
}

pub struct Options {
    /// Termination tolerance for feasibility condition.
    pub feas_tol: f64,
    /// Termination tolerance for gradient condition.
    pub grad_tol: f64,
    /// Termination tolerance for complementarity condition.
    pub comp_tol: f64,
    /// Termination tolerance for cost condition.
    pub cost_tol: f64,

    /// Maximum number of iterations.
    pub max_it: usize,

    /// Set to enable step-size control.
    pub step_control: bool,
    /// Maximum number of step-size reductions if step-control is on.
    pub max_red: usize,

    /// Cost multiplier used to scale the objective function for improved
    /// conditioning.
    ///
    /// Note: This value is also passed as the 3rd argument to the Hessian
    /// evaluation function so that it can appropriately scale the objective
    /// function term in the Hessian of the Lagrangian.
    pub cost_mult: f64,

    /// Constant used in alpha updates.
    pub xi: f64,
    /// Centering parameter.
    pub sigma: f64,
    /// Used to initialize slack variables.
    pub z0: f64,
    /// Exits if either alpha parameter becomes smaller than this value.
    pub alpha_min: f64,
    /// Lower bound on rho_t.
    pub rho_min: f64,
    /// Upper bound on rho_t.
    pub rho_max: f64,
    /// KT multipliers smaller than this value for non-binding constraints are forced to zero.
    pub mu_threshold: f64,
    /// Exits if the 2-norm of the reduced Newton step exceeds this value.
    pub max_step_size: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            feas_tol: 1e-6,
            grad_tol: 1e-6,
            comp_tol: 1e-6,
            cost_tol: 1e-6,

            max_it: 150,

            step_control: false,
            max_red: 20,

            cost_mult: 1.0,

            xi: 0.99995,
            sigma: 0.1,
            z0: 1.0,
            alpha_min: 1e-8,
            rho_min: 0.95,
            rho_max: 1.05,
            mu_threshold: 1e-5,
            max_step_size: 1e10,
        }
    }
}
