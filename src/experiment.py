"""Run the full experiment analysis.

Two data modes — pick one and be explicit about it in your README:

  --source kaggle   Real experiment data. Download the Kaggle
                    "Marketing A/B Testing" dataset (marketing_AB.csv) into
                    data/raw/. Columns: user id, test group (ad/psa),
                    converted, total ads, most ads day/hour.

  --source simulate Simulated with a KNOWN ground-truth effect. Honest and
                    useful for validating the pipeline — but you MUST label it
                    as simulated in the README. Never present it as real.

    python -m src.experiment --source simulate
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.ab_test import (bootstrap_ci, continuous_test, correct_pvalues,
                         cuped_adjust, proportion_test, required_sample_size,
                         sample_size_table, srm_check, summarise)


# ------------------------------------------------------------- data sources
def simulate(n: int = 40_000, baseline_rate: float = 0.04,
             true_lift: float = 0.12, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Generate an experiment with a known 12% relative lift in conversion,
    plus a correlated pre-period covariate so CUPED has something to work with."""
    rng = np.random.default_rng(seed)
    group = rng.choice(["control", "treatment"], size=n, p=[0.5, 0.5])

    # pre-experiment spend: the covariate CUPED will exploit
    pre_spend = rng.gamma(shape=2.0, scale=25.0, size=n)
    propensity = (pre_spend - pre_spend.mean()) / pre_spend.std()

    rate = np.where(group == "treatment",
                    baseline_rate * (1 + true_lift), baseline_rate)
    # higher past spenders convert more - realistic, and creates the correlation
    rate = np.clip(rate * (1 + 0.25 * propensity), 0.001, 0.9)
    converted = rng.binomial(1, rate)

    order_value = rng.lognormal(mean=3.6, sigma=0.6, size=n)
    revenue = converted * order_value
    sessions = rng.poisson(3.2 + 0.2 * (group == "treatment"), size=n)

    return pd.DataFrame({
        "user_id": np.arange(n),
        "group": group,
        "converted": converted,
        "revenue": revenue,
        "sessions": sessions,
        "pre_spend": pre_spend,
        "_true_lift": true_lift,
    })


def load_kaggle() -> pd.DataFrame:
    path = C.DATA_RAW / "marketing_AB.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the Kaggle 'Marketing A/B Testing' "
            "dataset into data/raw/, or run with --source simulate."
        )
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"test_group": "group", "total_ads": "sessions"})
    df["group"] = df["group"].map({"psa": "control", "ad": "treatment"}).fillna(df["group"])
    df["converted"] = df["converted"].astype(int)
    return df


# ------------------------------------------------------------- analysis
def run(df: pd.DataFrame, simulated: bool) -> pd.DataFrame:
    control = df[df.group == "control"]
    treatment = df[df.group == "treatment"]

    print("=" * 72)
    print("STEP 1 — POWER ANALYSIS (what we could have detected)")
    print("=" * 72)
    baseline = control["converted"].mean()
    print(f"Baseline conversion rate: {baseline:.4f}")
    print(sample_size_table(baseline).to_string(index=False))
    have = min(len(control), len(treatment))
    need_5pct = required_sample_size(baseline, 0.05)
    print(f"\nWe have {have:,} per arm. Detecting a 5% relative lift needs "
          f"{need_5pct:,} per arm — {'adequately powered' if have >= need_5pct else 'UNDERPOWERED'}.")

    print("\n" + "=" * 72)
    print("STEP 2 — SAMPLE RATIO MISMATCH")
    print("=" * 72)
    srm = srm_check(len(control), len(treatment))
    print(pd.Series(srm).to_string())
    if srm["srm_detected"]:
        print("\n!! SRM DETECTED — randomisation is broken. Stop here.\n")

    print("\n" + "=" * 72)
    print("STEP 3 — HYPOTHESIS TESTS")
    print("=" * 72)
    results = []

    conv = proportion_test(
        int(control.converted.sum()), len(control),
        int(treatment.converted.sum()), len(treatment),
    )
    results.append(conv)
    print(summarise(conv))

    if "revenue" in df.columns:
        rev = continuous_test(control.revenue, treatment.revenue,
                              metric="revenue_per_user")
        results.append(rev)
        print(summarise(rev))
        boot = bootstrap_ci(control.revenue, treatment.revenue)
        print(f"  bootstrap CI on mean diff: "
              f"[{boot['ci'][0]:+.3f}, {boot['ci'][1]:+.3f}]")

    if "sessions" in df.columns:
        ses = continuous_test(control.sessions, treatment.sessions,
                              metric="sessions_per_user")
        results.append(ses)
        print(summarise(ses))

    print("\n" + "=" * 72)
    print("STEP 4 — MULTIPLE COMPARISON CORRECTION (Benjamini-Hochberg)")
    print("=" * 72)
    table = correct_pvalues(results)
    cols = ["metric", "control_mean" if "control_mean" in table else "control_rate",
            "relative_lift_%", "p_value", "p_adjusted",
            "significant_after_correction"]
    cols = [c for c in cols if c in table.columns]
    print(table[cols].to_string(index=False))

    if "pre_spend" in df.columns and "revenue" in df.columns:
        print("\n" + "=" * 72)
        print("STEP 5 — CUPED VARIANCE REDUCTION")
        print("=" * 72)
        adj = cuped_adjust(df["revenue"].to_numpy(), df["pre_spend"].to_numpy())
        df = df.assign(revenue_cuped=adj)
        raw_var = df["revenue"].var()
        cuped_var = df["revenue_cuped"].var()
        print(f"Variance: {raw_var:.2f} -> {cuped_var:.2f} "
              f"({(1 - cuped_var / raw_var) * 100:.1f}% reduction)")
        c2 = df[df.group == "control"]["revenue_cuped"]
        t2 = df[df.group == "treatment"]["revenue_cuped"]
        print(summarise(continuous_test(c2, t2, metric="revenue_cuped")))

    if simulated:
        print(f"\n[simulated data — ground truth relative lift was "
              f"{df['_true_lift'].iloc[0] * 100:.0f}%]")

    table.to_csv(C.REPORTS / "ab_results.csv", index=False)
    plot_results(conv)
    return table


def plot_results(conv: dict):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rates = [conv["control_rate"], conv["treatment_rate"]]
    cis = [conv["control_ci"], conv["treatment_ci"]]
    err = [[r - lo for r, (lo, hi) in zip(rates, cis)],
           [hi - r for r, (lo, hi) in zip(rates, cis)]]
    ax.bar(["Control", "Treatment"], rates, yerr=err, capsize=8,
           color=["#888", "#2e7d32"])
    ax.set_ylabel("Conversion rate")
    ax.set_title(f"Conversion rate with 95% CI (p={conv['p_value']:.4f})")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "ab_conversion.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["kaggle", "simulate"], default="simulate")
    args = ap.parse_args()
    simulated = args.source == "simulate"
    df = simulate() if simulated else load_kaggle()
    run(df, simulated)


if __name__ == "__main__":
    main()
