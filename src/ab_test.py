"""A/B test analysis toolkit — the statistics half of the project.

The order of operations matters and mirrors how a real experiment is run:
  1. Power analysis BEFORE the test (how many users do we need?)
  2. Sample ratio mismatch check (is the randomisation broken?)
  3. Hypothesis test with confidence intervals and effect size
  4. Multiple-comparison correction if testing more than one metric
  5. Optional CUPED variance reduction using pre-experiment data
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import (proportion_confint,
                                          proportion_effectsize,
                                          proportions_ztest)
from statsmodels.stats.multitest import multipletests

from src.config import ALPHA, POWER, RANDOM_SEED


# ------------------------------------------------- 1. power / sample size
def required_sample_size(baseline_rate: float, mde_relative: float,
                         alpha: float = ALPHA, power: float = POWER) -> int:
    """Users needed PER ARM to detect a relative lift of `mde_relative`.

    Run this before the experiment. If you can't reach this sample size,
    the experiment is not worth running.
    """
    treated_rate = baseline_rate * (1 + mde_relative)
    effect = proportion_effectsize(treated_rate, baseline_rate)
    n = NormalIndPower().solve_power(
        effect_size=effect, alpha=alpha, power=power, ratio=1.0,
        alternative="two-sided",
    )
    return int(np.ceil(n))


def sample_size_table(baseline_rate: float,
                      mdes=(0.01, 0.02, 0.05, 0.10, 0.20)) -> pd.DataFrame:
    return pd.DataFrame([
        {"MDE_relative_%": m * 100,
         "n_per_arm": required_sample_size(baseline_rate, m)}
        for m in mdes
    ])


# ------------------------------------------------- 2. sanity checks
def srm_check(n_control: int, n_treatment: int,
              expected_ratio: float = 0.5) -> dict:
    """Sample Ratio Mismatch. If p < 0.001 the assignment is broken and
    NOTHING else in the analysis can be trusted. Always run this first."""
    total = n_control + n_treatment
    expected = [total * (1 - expected_ratio), total * expected_ratio]
    chi2, p = stats.chisquare([n_control, n_treatment], f_exp=expected)
    return {
        "n_control": n_control, "n_treatment": n_treatment,
        "observed_split": n_treatment / total,
        "chi2": chi2, "p_value": p,
        "srm_detected": bool(p < 0.001),
    }


# ------------------------------------------------- 3. the tests
def proportion_test(conv_c: int, n_c: int, conv_t: int, n_t: int,
                    alpha: float = ALPHA) -> dict:
    """Two-proportion z-test for a binary metric (conversion, click, signup)."""
    stat, p = proportions_ztest([conv_t, conv_c], [n_t, n_c])
    rate_c, rate_t = conv_c / n_c, conv_t / n_t

    lo_c, hi_c = proportion_confint(conv_c, n_c, alpha=alpha, method="wilson")
    lo_t, hi_t = proportion_confint(conv_t, n_t, alpha=alpha, method="wilson")

    # CI on the difference (normal approximation)
    se = np.sqrt(rate_c * (1 - rate_c) / n_c + rate_t * (1 - rate_t) / n_t)
    z = stats.norm.ppf(1 - alpha / 2)
    diff = rate_t - rate_c

    return {
        "metric": "conversion_rate",
        "control_rate": rate_c, "treatment_rate": rate_t,
        "control_ci": (lo_c, hi_c), "treatment_ci": (lo_t, hi_t),
        "absolute_lift": diff,
        "relative_lift_%": (diff / rate_c * 100) if rate_c else np.nan,
        "diff_ci": (diff - z * se, diff + z * se),
        "z_stat": stat, "p_value": p,
        "significant": bool(p < alpha),
        "cohens_h": proportion_effectsize(rate_t, rate_c),
    }


def continuous_test(control: np.ndarray, treatment: np.ndarray,
                    alpha: float = ALPHA, metric: str = "revenue_per_user") -> dict:
    """Welch's t-test for a continuous metric. NEVER assume equal variance —
    treatment usually changes spread as well as mean."""
    control = np.asarray(control, float)
    treatment = np.asarray(treatment, float)
    t, p = stats.ttest_ind(treatment, control, equal_var=False)

    mc, mt = control.mean(), treatment.mean()
    se = np.sqrt(control.var(ddof=1) / len(control) + treatment.var(ddof=1) / len(treatment))
    df = (se ** 4) / (
        (control.var(ddof=1) / len(control)) ** 2 / (len(control) - 1)
        + (treatment.var(ddof=1) / len(treatment)) ** 2 / (len(treatment) - 1)
    )
    crit = stats.t.ppf(1 - alpha / 2, df)
    diff = mt - mc
    pooled_sd = np.sqrt((control.var(ddof=1) + treatment.var(ddof=1)) / 2)

    return {
        "metric": metric,
        "control_mean": mc, "treatment_mean": mt,
        "absolute_lift": diff,
        "relative_lift_%": diff / mc * 100 if mc else np.nan,
        "diff_ci": (diff - crit * se, diff + crit * se),
        "t_stat": t, "p_value": p,
        "significant": bool(p < alpha),
        "cohens_d": diff / pooled_sd if pooled_sd else np.nan,
    }


def bootstrap_ci(control, treatment, stat_fn=np.mean,
                 n_boot: int = 5000, alpha: float = ALPHA, seed: int = RANDOM_SEED):
    """Distribution-free CI on the difference. Useful for revenue, which is
    heavily skewed and zero-inflated."""
    rng = np.random.default_rng(seed)
    control, treatment = np.asarray(control, float), np.asarray(treatment, float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        c = rng.choice(control, len(control), replace=True)
        t = rng.choice(treatment, len(treatment), replace=True)
        diffs[i] = stat_fn(t) - stat_fn(c)
    lo, hi = np.percentile(diffs, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return {"bootstrap_diff": diffs.mean(), "ci": (lo, hi),
            "p_two_sided": 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())}


# ------------------------------------------------- 4. multiple comparisons
def correct_pvalues(results: list[dict], method: str = "fdr_bh",
                    alpha: float = ALPHA) -> pd.DataFrame:
    """Testing 5 metrics at alpha=0.05 gives ~23% chance of a false positive
    somewhere. Correct for it."""
    df = pd.DataFrame(results)
    reject, adj, _, _ = multipletests(df["p_value"], alpha=alpha, method=method)
    df["p_adjusted"] = adj
    df["significant_after_correction"] = reject
    return df


# ------------------------------------------------- 5. CUPED
def cuped_adjust(y_post: np.ndarray, y_pre: np.ndarray) -> np.ndarray:
    """Variance reduction using a pre-experiment covariate.
    Typically cuts variance 30-50%, meaning you need far fewer users.
    theta = cov(post, pre) / var(pre);  y_adj = y_post - theta*(y_pre - mean(y_pre))
    """
    y_post, y_pre = np.asarray(y_post, float), np.asarray(y_pre, float)
    theta = np.cov(y_post, y_pre)[0, 1] / np.var(y_pre, ddof=1)
    return y_post - theta * (y_pre - y_pre.mean())


# ------------------------------------------------- reporting
def summarise(result: dict) -> str:
    sig = "SIGNIFICANT" if result["significant"] else "not significant"
    lo, hi = result["diff_ci"]
    return (
        f"{result['metric']}: {result['relative_lift_%']:+.2f}% relative lift "
        f"(95% CI on absolute diff: [{lo:+.4f}, {hi:+.4f}]), "
        f"p={result['p_value']:.4f} — {sig}"
    )
