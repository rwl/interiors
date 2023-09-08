/// Computes the dot-product of `a` and `b`.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    return a
        .iter()
        .zip(b)
        .map(|(&ai, &bi)| ai * bi)
        .reduce(|x, y| x + y)
        .unwrap_or(0.0);
}

/// Returns the maximum value of `a`.
pub fn max(a: &[f64]) -> f64 {
    *a.iter().max_by(|&a, b| a.partial_cmp(b).unwrap()).unwrap()
}

/// Returns the minimum value of `a`.
pub fn min(a: &[f64]) -> f64 {
    *a.iter().min_by(|&a, b| a.partial_cmp(b).unwrap()).unwrap()
}

/// Computes the infinity norm: `max(abs(a))`
pub fn norm_inf(a: &[f64]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for i in 0..a.len() {
        let absvi = a[i].abs();
        if absvi > max {
            max = absvi
        }
    }
    max
}

/// Returns the 2-norm (Euclidean) of `a`.
pub fn norm(a: &[f64]) -> f64 {
    let mut sqsum = 0.0;
    for i in 0..a.len() {
        sqsum += a[i] * a[i];
    }
    f64::sqrt(sqsum)
}
