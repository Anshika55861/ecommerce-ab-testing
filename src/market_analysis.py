"""Market intelligence on the collected listings. python -m src.market_analysis"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src import config as C


def load() -> pd.DataFrame:
    if not C.LISTINGS_FILE.exists():
        raise FileNotFoundError("Run `python -m src.collect` first.")
    df = pd.read_csv(C.LISTINGS_FILE)
    df = df[df["price"].between(1, 5000)]              # drop junk listings
    df["free_shipping"] = (df["shipping_cost"].fillna(0) == 0).astype(int)
    df["is_auction"] = df["buying_options"].str.contains("AUCTION", na=False).astype(int)
    df["is_new"] = df["condition"].str.contains("New", case=False, na=False).astype(int)
    df["log_price"] = np.log1p(df["price"])
    df["title_len"] = df["title"].str.len()
    df["title_words"] = df["title"].str.split().str.len()
    return df


def price_distributions(df):
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df, x="search_term", y="price", ax=ax)
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Price distribution by category (log scale)")
    fig.tight_layout(); fig.savefig(C.FIGURES / "01_price_by_category.png", dpi=150)
    plt.close(fig)


def shipping_premium(df) -> pd.DataFrame:
    """Do free-shipping listings carry a price premium? Test it, don't eyeball it."""
    rows = []
    for term, sub in df.groupby("search_term"):
        free = sub.loc[sub.free_shipping == 1, "price"]
        paid = sub.loc[sub.free_shipping == 0, "price"]
        if len(free) < 30 or len(paid) < 30:
            continue
        t, p = stats.ttest_ind(free, paid, equal_var=False)   # Welch's
        rows.append({
            "category": term,
            "n_free": len(free), "n_paid": len(paid),
            "median_free": free.median(), "median_paid": paid.median(),
            "premium_%": (free.median() / paid.median() - 1) * 100,
            "t_stat": t, "p_value": p,
        })
    return pd.DataFrame(rows)


def condition_gap(df) -> pd.DataFrame:
    rows = []
    for term, sub in df.groupby("search_term"):
        new = sub.loc[sub.is_new == 1, "price"]
        used = sub.loc[sub.is_new == 0, "price"]
        if len(new) < 30 or len(used) < 30:
            continue
        u, p = stats.mannwhitneyu(new, used)     # prices are skewed, use a rank test
        rows.append({
            "category": term,
            "median_new": new.median(), "median_used": used.median(),
            "new_premium_%": (new.median() / used.median() - 1) * 100,
            "p_value": p,
        })
    return pd.DataFrame(rows)


def seller_rating_effect(df):
    sub = df.dropna(subset=["seller_feedback_pct", "price"])
    if sub.empty:
        return None
    r, p = stats.spearmanr(sub["seller_feedback_pct"], sub["log_price"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=sub.sample(min(3000, len(sub)), random_state=0),
                    x="seller_feedback_pct", y="price", s=8, alpha=0.4, ax=ax)
    ax.set_yscale("log")
    ax.set_title(f"Seller feedback % vs price (Spearman r={r:.3f}, p={p:.3g})")
    fig.tight_layout(); fig.savefig(C.FIGURES / "02_seller_rating.png", dpi=150)
    plt.close(fig)
    return {"spearman_r": r, "p_value": p}


def keyword_effect(df, keywords=("bundle", "lot", "oem", "genuine", "refurbished",
                                 "sealed", "warranty", "fast")) -> pd.DataFrame:
    """Does a keyword in the title associate with higher price? Controls only
    for category — this is correlational, and the README should say so."""
    rows = []
    for kw in keywords:
        has = df["title"].str.contains(kw, case=False, na=False)
        if has.sum() < 30 or (~has).sum() < 30:
            continue
        # compare within category then average, to avoid Simpson's paradox
        diffs = []
        for term, sub in df.groupby("search_term"):
            h = sub.loc[sub["title"].str.contains(kw, case=False, na=False), "price"]
            n = sub.loc[~sub["title"].str.contains(kw, case=False, na=False), "price"]
            if len(h) >= 15 and len(n) >= 15:
                diffs.append(h.median() / n.median() - 1)
        if not diffs:
            continue
        t, p = stats.ttest_ind(df.loc[has, "log_price"],
                               df.loc[~has, "log_price"], equal_var=False)
        rows.append({"keyword": kw, "n_listings": int(has.sum()),
                     "median_price_lift_%": np.mean(diffs) * 100, "p_value": p})
    return pd.DataFrame(rows).sort_values("median_price_lift_%", ascending=False)


def main():
    sns.set_theme(style="whitegrid")
    df = load()
    print(f"Loaded {len(df):,} listings across {df.search_term.nunique()} categories\n")

    price_distributions(df)

    ship = shipping_premium(df)
    print("FREE SHIPPING PREMIUM\n", ship.to_string(index=False), "\n")
    ship.to_csv(C.REPORTS / "shipping_premium.csv", index=False)

    cond = condition_gap(df)
    print("NEW vs USED\n", cond.to_string(index=False), "\n")
    cond.to_csv(C.REPORTS / "condition_gap.csv", index=False)

    kw = keyword_effect(df)
    print("TITLE KEYWORD EFFECTS\n", kw.to_string(index=False), "\n")
    kw.to_csv(C.REPORTS / "keyword_effects.csv", index=False)

    print("SELLER RATING\n", seller_rating_effect(df))


if __name__ == "__main__":
    main()
