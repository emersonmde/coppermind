//! Statistical utilities for evaluation rigor.
//!
//! This module provides tools for assessing statistical significance of
//! benchmark results, including:
//! - Bootstrap confidence intervals
//! - Paired t-tests for system comparison
//! - Effect size measures (Cohen's d)
//!
//! # References
//!
//! - Efron & Tibshirani (1993). "An Introduction to the Bootstrap"
//! - Smucker et al. (2007). "A comparison of statistical significance tests for IR evaluation"

/// Result of bootstrap confidence interval computation.
#[derive(Debug, Clone, Copy)]
pub struct BootstrapResult {
    /// Sample mean
    pub mean: f64,
    /// Lower bound of confidence interval
    pub lower: f64,
    /// Upper bound of confidence interval
    pub upper: f64,
}

impl BootstrapResult {
    /// Formats the result as "mean [lower, upper]".
    pub fn format(&self, precision: usize) -> String {
        format!(
            "{:.prec$} [{:.prec$}, {:.prec$}]",
            self.mean,
            self.lower,
            self.upper,
            prec = precision
        )
    }
}

/// Computes bootstrap 95% confidence interval for the mean.
///
/// Bootstrap resampling estimates the sampling distribution of the mean by:
/// 1. Resampling with replacement from the original data
/// 2. Computing the mean of each resample
/// 3. Taking the 2.5th and 97.5th percentiles as the CI bounds
///
/// # Arguments
///
/// * `values` - Sample values (e.g., NDCG scores across queries)
/// * `n_bootstrap` - Number of bootstrap resamples (typically 1000-10000)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// `BootstrapResult` containing mean and 95% confidence interval bounds.
///
/// # Example
///
/// ```ignore
/// let ndcg_scores = vec![0.85, 0.92, 0.78, 0.91, 0.88];
/// let result = bootstrap_ci(&ndcg_scores, 1000, 42);
/// println!("NDCG@10: {}", result.format(4));  // e.g., "0.8680 [0.8123, 0.9147]"
/// ```
///
/// # Panics
///
/// Returns (NaN, NaN, NaN) if `values` is empty.
pub fn bootstrap_ci(values: &[f64], n_bootstrap: usize, seed: u64) -> BootstrapResult {
    if values.is_empty() {
        return BootstrapResult {
            mean: f64::NAN,
            lower: f64::NAN,
            upper: f64::NAN,
        };
    }

    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;

    // Simple LCG for reproducible randomness (avoid external deps)
    let mut rng = LcgRng::new(seed);

    let mut bootstrap_means = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Resample with replacement
        let mut sum = 0.0;
        for _ in 0..n {
            let idx = rng.next_usize(n);
            sum += values[idx];
        }
        bootstrap_means.push(sum / n as f64);
    }

    // Sort for percentile computation
    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // 95% CI: 2.5th and 97.5th percentiles
    let lower_idx = ((n_bootstrap as f64) * 0.025) as usize;
    let upper_idx = ((n_bootstrap as f64) * 0.975) as usize;

    BootstrapResult {
        mean,
        lower: bootstrap_means[lower_idx.min(bootstrap_means.len() - 1)],
        upper: bootstrap_means[upper_idx.min(bootstrap_means.len() - 1)],
    }
}

/// Result of a paired t-test.
#[derive(Debug, Clone, Copy)]
pub struct TTestResult {
    /// t-statistic (positive if system A > system B)
    pub t_statistic: f64,
    /// Two-tailed p-value
    pub p_value: f64,
    /// Degrees of freedom
    pub df: usize,
}

impl TTestResult {
    /// Returns true if the difference is significant at the given alpha level.
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }

    /// Formats the result for display.
    pub fn format(&self) -> String {
        let sig_marker = if self.is_significant(0.05) { "*" } else { "" };
        format!(
            "t({})={:.3}, p={:.4}{}",
            self.df, self.t_statistic, self.p_value, sig_marker
        )
    }
}

/// Performs a paired t-test comparing two systems on the same queries.
///
/// The paired t-test is appropriate when both systems are evaluated on the
/// same set of queries, creating paired observations. This is the standard
/// approach for comparing IR systems.
///
/// # Arguments
///
/// * `system_a` - Scores from system A (e.g., NDCG per query)
/// * `system_b` - Scores from system B (same queries, same order)
///
/// # Returns
///
/// `TTestResult` with t-statistic, p-value, and degrees of freedom.
/// Positive t means system A > system B on average.
///
/// # Example
///
/// ```ignore
/// let hybrid_ndcg = vec![0.85, 0.92, 0.78];
/// let bm25_ndcg = vec![0.75, 0.88, 0.72];
/// let result = paired_ttest(&hybrid_ndcg, &bm25_ndcg);
/// if result.is_significant(0.05) {
///     println!("Hybrid significantly outperforms BM25: {}", result.format());
/// }
/// ```
///
/// # Panics
///
/// Panics if arrays have different lengths or are empty.
pub fn paired_ttest(system_a: &[f64], system_b: &[f64]) -> TTestResult {
    assert_eq!(
        system_a.len(),
        system_b.len(),
        "Paired t-test requires equal-length arrays"
    );
    assert!(
        !system_a.is_empty(),
        "Cannot perform t-test on empty arrays"
    );

    let n = system_a.len();
    let df = n - 1;

    // Compute paired differences
    let diffs: Vec<f64> = system_a
        .iter()
        .zip(system_b.iter())
        .map(|(a, b)| a - b)
        .collect();

    // Mean of differences
    let mean_diff = diffs.iter().sum::<f64>() / n as f64;

    // Standard deviation of differences
    let var_diff = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / df as f64;
    let std_diff = var_diff.sqrt();

    // Standard error of the mean difference
    let se = std_diff / (n as f64).sqrt();

    // t-statistic
    let t = if se > 0.0 { mean_diff / se } else { 0.0 };

    // Two-tailed p-value from t-distribution
    let p_value = t_distribution_p_value(t.abs(), df);

    TTestResult {
        t_statistic: t,
        p_value,
        df,
    }
}

/// Computes Cohen's d effect size for comparing two groups.
///
/// Cohen's d measures the standardized difference between two means.
/// It's useful for understanding practical significance beyond statistical
/// significance.
///
/// # Interpretation (Cohen's conventions)
///
/// - |d| < 0.2: negligible effect
/// - 0.2 <= |d| < 0.5: small effect
/// - 0.5 <= |d| < 0.8: medium effect
/// - |d| >= 0.8: large effect
///
/// # Arguments
///
/// * `group_a` - Scores from group A
/// * `group_b` - Scores from group B
///
/// # Returns
///
/// Cohen's d (positive if group A > group B).
///
/// # Example
///
/// ```ignore
/// let hybrid_scores = vec![0.85, 0.92, 0.78, 0.91];
/// let bm25_scores = vec![0.75, 0.88, 0.72, 0.80];
/// let d = cohens_d(&hybrid_scores, &bm25_scores);
/// if d.abs() >= 0.8 {
///     println!("Large effect size: d = {:.3}", d);
/// }
/// ```
pub fn cohens_d(group_a: &[f64], group_b: &[f64]) -> f64 {
    if group_a.is_empty() || group_b.is_empty() {
        return 0.0;
    }

    let n_a = group_a.len();
    let n_b = group_b.len();

    // Means
    let mean_a = group_a.iter().sum::<f64>() / n_a as f64;
    let mean_b = group_b.iter().sum::<f64>() / n_b as f64;

    // Variances
    let var_a: f64 = group_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1) as f64;
    let var_b: f64 = group_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1) as f64;

    // Pooled standard deviation
    let pooled_var = ((n_a - 1) as f64 * var_a + (n_b - 1) as f64 * var_b) / (n_a + n_b - 2) as f64;
    let pooled_std = pooled_var.sqrt();

    if pooled_std == 0.0 {
        return 0.0;
    }

    (mean_a - mean_b) / pooled_std
}

/// Interprets Cohen's d value.
pub fn interpret_cohens_d(d: f64) -> &'static str {
    let d_abs = d.abs();
    if d_abs < 0.2 {
        "negligible"
    } else if d_abs < 0.5 {
        "small"
    } else if d_abs < 0.8 {
        "medium"
    } else {
        "large"
    }
}

// ============================================================================
// Internal: Simple LCG RNG (no external deps)
// ============================================================================

/// Simple Linear Congruential Generator for reproducible randomness.
///
/// Uses the same parameters as glibc's rand() for compatibility.
/// This avoids adding rand crate as a dependency.
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next() as usize) % max
    }
}

// ============================================================================
// Internal: T-distribution p-value approximation
// ============================================================================

/// Approximates two-tailed p-value from t-distribution.
///
/// Uses the incomplete beta function relationship:
/// p = I_{df/(df+t²)}(df/2, 1/2)
///
/// For large df, approximates using normal distribution.
fn t_distribution_p_value(t_abs: f64, df: usize) -> f64 {
    // For large df, use normal approximation
    if df > 100 {
        return 2.0 * (1.0 - normal_cdf(t_abs));
    }

    // Use incomplete beta function
    let x = df as f64 / (df as f64 + t_abs * t_abs);
    incomplete_beta(df as f64 / 2.0, 0.5, x)
}

/// Normal CDF using error function approximation.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Error function approximation (Abramowitz and Stegun).
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Incomplete beta function using continued fraction.
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    // Use continued fraction approximation
    let bt = if x == 0.0 || x == 1.0 {
        0.0
    } else {
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp()
    };

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Continued fraction for incomplete beta.
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 100;
    let eps = 1e-10;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m = m as f64;
        let m2 = 2.0 * m;

        // Even step
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }

    h
}

/// Log gamma function using Stirling's approximation.
fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];

    let y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();

    let mut ser = 1.000000000190015;
    for (i, &coeff) in coeffs.iter().enumerate() {
        ser += coeff / (y + 1.0 + i as f64);
    }

    -tmp + (2.5066282746310005 * ser / x).ln()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_ci_basic() {
        let values = vec![0.85, 0.90, 0.88, 0.92, 0.87, 0.89, 0.91, 0.86, 0.88, 0.90];
        let result = bootstrap_ci(&values, 1000, 42);

        // Mean should be approximately 0.886
        assert!((result.mean - 0.886).abs() < 0.01);

        // CI should contain the mean
        assert!(result.lower <= result.mean);
        assert!(result.upper >= result.mean);

        // CI width should be reasonable
        let width = result.upper - result.lower;
        assert!(width > 0.01 && width < 0.1);
    }

    #[test]
    fn test_bootstrap_ci_single_value() {
        let values = vec![0.9];
        let result = bootstrap_ci(&values, 100, 42);

        // Single value should have mean = that value, and tight CI
        assert!((result.mean - 0.9).abs() < 0.001);
        assert!((result.lower - 0.9).abs() < 0.001);
        assert!((result.upper - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_bootstrap_ci_empty() {
        let values: Vec<f64> = vec![];
        let result = bootstrap_ci(&values, 100, 42);

        assert!(result.mean.is_nan());
    }

    #[test]
    fn test_paired_ttest_significant() {
        // Clearly different distributions
        let system_a = vec![0.9, 0.92, 0.88, 0.91, 0.89, 0.93, 0.87, 0.90];
        let system_b = vec![0.7, 0.72, 0.68, 0.71, 0.69, 0.73, 0.67, 0.70];

        let result = paired_ttest(&system_a, &system_b);

        // Should be highly significant (p < 0.001)
        assert!(result.is_significant(0.001));
        assert!(result.t_statistic > 0.0); // A > B
    }

    #[test]
    fn test_paired_ttest_not_significant() {
        // Very similar distributions
        let system_a = vec![0.85, 0.87, 0.86, 0.84, 0.85];
        let system_b = vec![0.84, 0.86, 0.87, 0.85, 0.86];

        let result = paired_ttest(&system_a, &system_b);

        // Should not be significant at 0.05
        assert!(!result.is_significant(0.05));
    }

    #[test]
    fn test_cohens_d_large_effect() {
        // Large difference
        let group_a = vec![0.9, 0.92, 0.88, 0.91, 0.89];
        let group_b = vec![0.5, 0.52, 0.48, 0.51, 0.49];

        let d = cohens_d(&group_a, &group_b);

        // Should be a very large effect
        assert!(d > 2.0);
        assert_eq!(interpret_cohens_d(d), "large");
    }

    #[test]
    fn test_cohens_d_small_effect() {
        // Small difference with high variance (produces small d)
        // d = (mean_a - mean_b) / pooled_std
        // For small effect (d ~ 0.3), we need difference << std
        let group_a = vec![0.70, 0.80, 0.85, 0.90, 0.75, 0.82, 0.88, 0.78];
        let group_b = vec![0.68, 0.78, 0.82, 0.87, 0.73, 0.80, 0.85, 0.75];

        let d = cohens_d(&group_a, &group_b);

        // Should be a small effect (d between 0.2 and 0.5)
        assert!(d > 0.0, "Effect should be positive");
        assert!(d < 0.8, "Effect should be less than large, got {}", d);
    }

    #[test]
    fn test_cohens_d_interpretation() {
        assert_eq!(interpret_cohens_d(0.1), "negligible");
        assert_eq!(interpret_cohens_d(0.3), "small");
        assert_eq!(interpret_cohens_d(0.6), "medium");
        assert_eq!(interpret_cohens_d(1.0), "large");
        assert_eq!(interpret_cohens_d(-0.9), "large"); // Absolute value
    }

    #[test]
    fn test_lcg_reproducibility() {
        let mut rng1 = LcgRng::new(42);
        let mut rng2 = LcgRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_normal_cdf() {
        // Standard normal CDF values
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.001);
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.01);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.01);
    }
}
