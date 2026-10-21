# =============================================================================
# phase4_feature_analysis.py
# Location: src/phase4_feature_analysis.py
#
# Purpose: DEV Mini Project — PHASE 4: Feature Engineering & Statistical
# Observations. Finds out which features actually help predict the character:
#   1. Correlation matrix + heatmap          (relationships between features)
#   2. Multicollinearity check               (highly correlated pairs + VIF)
#   3. Statistical significance testing      (ANOVA across classes, t-tests)
#   4. Feature importance                    (Random Forest: Gini + permutation)
#   5. Feature selection                     (shape features AND pixels)
#   6. Interaction features                  (kept only if they improve a model)
#
# Output: data/selected_features.json — which columns Phase 5 should use.
#
# WHY ONLY THE TRAINING SPLIT IS USED HERE:
# Feature selection is part of building the model. If it looked at the test
# rows, the test set would have influenced which features were chosen, and
# the Phase 5 test scores would be optimistic (data leakage). So everything
# below runs on the 80% training split from split_train_test(), the SAME
# split Phase 5 uses.
#
# Run from the project root (after Phase 2) with:
#     python src/phase4_feature_analysis.py
# =============================================================================

# --- Step 1: Import the libraries we need ---
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from scipy import stats                     # ANOVA and t-tests
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.append(os.path.dirname(__file__))  # So Python can find our own modules
from phase1_data_summary import PIXEL_COLUMNS
from phase2_preprocessing import PROCESSED_CSV_PATH, FEATURE_COLUMNS
from phase3_eda import (FEATURE_LABELS, INK_CMAP, TEXT, TEXT_SECONDARY, NEUTRAL,
                        load_processed, print_header, set_chart_style)


# =============================================================================
# SETTINGS
# =============================================================================

SEED = 42
TEST_SIZE = 0.2                     # Same split as the Phase 5 specification
RESULTS_DIR = "results"
SELECTED_FEATURES_PATH = "data/selected_features.json"

CORRELATION_THRESHOLD = 0.8         # |r| above this = "highly correlated"
VIF_THRESHOLD = 10                  # Common rule of thumb for serious multicollinearity
ALPHA = 0.05                        # Significance level for all tests
INTERACTION_MIN_GAIN = 0.01         # Keep interactions only if CV accuracy rises >= 1 pp
                                    # (smaller gains are within run-to-run noise)
CV_SAMPLE_SIZE = 15_000             # Rows used for the interaction experiment (speed)

FEATURE_COLOR = "#2a78d6"           # single-series bars
DROPPED_COLOR = "#c3c2b7"           # features removed by selection

# Diverging map for correlations: blue = negative, gray = 0, red = positive.
CORR_CMAP = LinearSegmentedColormap.from_list(
    "corr", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#a12a2a"])


def save(fig, name):
    path = os.path.join(RESULTS_DIR, f"phase4_{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# SECTION 1: TRAIN / TEST SPLIT (shared with Phase 5)
# =============================================================================

def split_train_test(df):
    """
    The one train/test split used by Phases 4 and 5.
    stratify keeps every class's share equal in both parts, which matters
    because some classes have fewer than 200 images.
    """
    return train_test_split(df, test_size=TEST_SIZE, random_state=SEED,
                            stratify=df["label"])


# =============================================================================
# SECTION 2: CORRELATION & MULTICOLLINEARITY
# =============================================================================

def analyze_correlation(train):
    """1. Pearson correlation matrix and heatmap of the shape features."""
    print_header("1. CORRELATION MATRIX (shape features, Pearson r)")
    corr = train[FEATURE_COLUMNS].corr()
    print(corr.round(2).to_string())

    labels = [FEATURE_LABELS[f] for f in FEATURE_COLUMNS]
    # Hide the upper triangle: it mirrors the lower one and doubles the clutter.
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(corr, mask=mask, cmap=CORR_CMAP, vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"fontsize": 8},
                linewidths=2, linecolor="white", square=True,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Pearson correlation (r)", "shrink": 0.8}, ax=ax)
    ax.set_title("Correlation between shape features (training split)", fontsize=13, color=TEXT)
    ax.tick_params(axis="x", rotation=40)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.grid(False)
    save(fig, "01_correlation_heatmap")
    return corr


def find_correlated_pairs(corr):
    """2a. Every pair of features with |r| above CORRELATION_THRESHOLD."""
    pairs = []
    for i, a in enumerate(FEATURE_COLUMNS):
        for b in FEATURE_COLUMNS[i + 1:]:
            if abs(corr.loc[a, b]) > CORRELATION_THRESHOLD:
                pairs.append((a, b, corr.loc[a, b]))
    return pairs


def compute_vif(data, columns):
    """
    2b. Variance Inflation Factor: regress each feature on all the others.
    VIF = 1 / (1 - R^2). VIF 1 = independent; above 10 = the feature is
    almost fully predictable from the others (serious multicollinearity).
    """
    X = StandardScaler().fit_transform(data[columns])
    vif = {}
    for i, col in enumerate(columns):
        others = np.delete(X, i, axis=1)
        r2 = LinearRegression().fit(others, X[:, i]).score(others, X[:, i])
        vif[col] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(vif)


def analyze_multicollinearity(train, corr):
    print_header("2. MULTICOLLINEARITY")
    pairs = find_correlated_pairs(corr)
    print(f"  a) Feature pairs with |r| > {CORRELATION_THRESHOLD}:")
    for a, b, r in pairs:
        print(f"     {a:<16} <-> {b:<16} r = {r:+.3f}")
    if not pairs:
        print("     none")

    vif = compute_vif(train, FEATURE_COLUMNS)
    print(f"\n  b) Variance Inflation Factor (VIF > {VIF_THRESHOLD} = serious):")
    for col, v in vif.sort_values(ascending=False).items():
        flag = "  <- high" if v > VIF_THRESHOLD else ""
        print(f"     {col:<16}: {v:8.2f}{flag}")
    return pairs, vif


# =============================================================================
# SECTION 3: STATISTICAL SIGNIFICANCE
# =============================================================================

def eta_squared(f_stat, k, n):
    """Effect size for one-way ANOVA: share of variance explained by the class."""
    return (f_stat * (k - 1)) / (f_stat * (k - 1) + (n - k))


def cohens_d(a, b):
    """Effect size for a t-test: difference of means in units of std deviation."""
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def run_anova(train):
    """
    3a. One-way ANOVA per feature across all 62 classes.
    H0: the feature has the same mean for every character.

    With ~40,000 rows almost ANY difference becomes "significant", so the
    p-value alone says little. We also report eta-squared (effect size):
    how much of the feature's variation is explained by the character.
    """
    print_header("3a. ONE-WAY ANOVA ACROSS 62 CLASSES  (H0: equal means for all characters)")
    k, n = train["label"].nunique(), len(train)
    # Bonferroni: we run 10 tests, so each must pass alpha / 10 to keep the
    # overall chance of a false positive at alpha.
    alpha_corrected = ALPHA / len(FEATURE_COLUMNS)
    rows = []
    for feature in FEATURE_COLUMNS:
        samples = [g[feature].to_numpy() for _, g in train.groupby("label")]
        f_stat, p = stats.f_oneway(*samples)
        rows.append({"feature": feature, "F": f_stat, "p_value": p,
                     "eta_squared": eta_squared(f_stat, k, n),
                     "significant": p < alpha_corrected})
    anova = pd.DataFrame(rows).set_index("feature").sort_values("eta_squared", ascending=False)

    print(f"  Bonferroni-corrected alpha = {ALPHA} / {len(FEATURE_COLUMNS)} = {alpha_corrected:.4f}")
    printable = anova.copy()
    printable["p_value"] = printable["p_value"].map(lambda p: f"{p:.2e}")
    printable["F"] = printable["F"].round(1)
    printable["eta_squared"] = printable["eta_squared"].round(3)
    print(printable.to_string())
    print("  eta-squared guide: 0.01 small, 0.06 medium, 0.14+ large")

    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = anova["eta_squared"].sort_values()
    ax.barh([FEATURE_LABELS[f] for f in ordered.index], ordered.values,
            color=FEATURE_COLOR, height=0.6)
    for y, v in enumerate(ordered.values):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=8, color=TEXT_SECONDARY)
    for x, name in [(0.06, "medium"), (0.14, "large")]:
        ax.axvline(x, color=NEUTRAL, linestyle="--", linewidth=1)
        ax.text(x, len(ordered) - 0.4, f" {name}", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, ordered.max() * 1.15)
    ax.set_xlabel("Effect size η² (share of variance explained by the character)")
    ax.set_title("ANOVA effect size of each feature across 62 classes", color=TEXT)
    save(fig, "02_anova_effect_size")
    return anova


def run_t_tests(train):
    """
    3b. Welch's t-tests (no equal-variance assumption).
      - uppercase vs lowercase, for every feature
      - look-alike character pairs, for the feature that should separate them
    """
    print_header("3b. WELCH'S T-TESTS")
    alpha_corrected = ALPHA / len(FEATURE_COLUMNS)

    upper = train[train["group"] == "uppercase"]
    lower = train[train["group"] == "lowercase"]
    print(f"  i) Uppercase (n={len(upper)}) vs lowercase (n={len(lower)})  "
          f"(H0: equal means; Bonferroni alpha {alpha_corrected:.4f})")
    print(f"     {'feature':<16} {'mean upper':>10} {'mean lower':>10} {'t':>8} "
          f"{'p-value':>10} {'Cohen d':>8}  significant")
    for feature in FEATURE_COLUMNS:
        t, p = stats.ttest_ind(upper[feature], lower[feature], equal_var=False)
        d = cohens_d(upper[feature], lower[feature])
        print(f"     {feature:<16} {upper[feature].mean():>10.3f} {lower[feature].mean():>10.3f} "
              f"{t:>8.2f} {p:>10.2e} {d:>8.2f}  {'yes' if p < alpha_corrected else 'no'}")
    print("     Cohen d guide: 0.2 small, 0.5 medium, 0.8 large")

    print("\n  ii) Look-alike pairs (can a shape feature tell them apart?)")
    pairs = [("1", "l", "aspect_ratio"), ("I", "l", "ink_density"),
             ("O", "o", "bbox_height"), ("C", "c", "bbox_height"),
             ("S", "s", "ink_pixel_count"), ("0", "O", "aspect_ratio")]
    for a, b, feature in pairs:
        xa = train.loc[train["char"] == a, feature]
        xb = train.loc[train["char"] == b, feature]
        t, p = stats.ttest_ind(xa, xb, equal_var=False)
        d = cohens_d(xa, xb)
        verdict = "significant" if p < ALPHA else "NOT significant"
        print(f"     '{a}' vs '{b}' on {feature:<16}: means {xa.mean():7.3f} vs {xb.mean():7.3f}, "
              f"p = {p:.2e}, d = {d:+.2f}  -> {verdict}")


# =============================================================================
# SECTION 4: FEATURE IMPORTANCE
# =============================================================================

def analyze_importance(train, test_like):
    """
    4. Random Forest feature importance on the shape features, two ways:
       - Gini importance: how much each feature reduces impurity inside the
         trees. Fast, but biased toward features with many distinct values.
       - Permutation importance: shuffle one feature and measure how much
         accuracy drops on held-out data. Slower but more trustworthy.
    The held-out data is a validation slice of the TRAINING split, so the
    real test set stays unseen.
    """
    print_header("4. RANDOM FOREST FEATURE IMPORTANCE (shape features)")
    fit_part, val_part = test_like
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(fit_part[FEATURE_COLUMNS], fit_part["label"])
    val_acc = rf.score(val_part[FEATURE_COLUMNS], val_part["label"])
    print(f"  Random Forest on the 10 shape features only: validation accuracy {val_acc:.1%}")

    perm = permutation_importance(rf, val_part[FEATURE_COLUMNS], val_part["label"],
                                  n_repeats=5, random_state=SEED, n_jobs=-1)
    importance = pd.DataFrame({
        "gini": rf.feature_importances_,
        "permutation": perm.importances_mean,
        "permutation_std": perm.importances_std,
    }, index=FEATURE_COLUMNS).sort_values("permutation", ascending=False)
    print("  (permutation = drop in validation accuracy when the feature is shuffled)")
    print(importance.round(4).to_string())
    return importance


def plot_importance(importance, dropped):
    order = importance.sort_values("permutation").index
    labels = [FEATURE_LABELS[f] + ("  (dropped)" if f in dropped else "") for f in order]
    colors = [DROPPED_COLOR if f in dropped else FEATURE_COLOR for f in order]

    fig, (ax_gini, ax_perm) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    ax_gini.barh(labels, importance.loc[order, "gini"], color=colors, height=0.6)
    ax_gini.set_title("Gini importance (impurity reduction)", color=TEXT)
    ax_gini.set_xlabel("Importance (sums to 1)")

    ax_perm.barh(labels, importance.loc[order, "permutation"], color=colors, height=0.6,
                 xerr=importance.loc[order, "permutation_std"],
                 error_kw={"ecolor": TEXT_SECONDARY, "linewidth": 1, "capsize": 3})
    ax_perm.set_title("Permutation importance (accuracy drop when shuffled)", color=TEXT)
    ax_perm.set_xlabel("Drop in validation accuracy")

    fig.suptitle("Random Forest feature importance — shape features", fontsize=13, color=TEXT)
    fig.tight_layout()
    fig.legend(handles=[Patch(color=FEATURE_COLOR, label="kept"),
                        Patch(color=DROPPED_COLOR, label="dropped (collinear)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    save(fig, "03_feature_importance")


# =============================================================================
# SECTION 5: FEATURE SELECTION
# =============================================================================

def select_shape_features(pairs, importance):
    """
    5a. From every highly correlated pair, drop the feature with the LOWER
    permutation importance — it adds the least information the other one
    doesn't already carry.

    A feature that was KEPT in an earlier pair is protected: it was kept as
    the stand-in for the feature dropped there, so dropping it later would
    lose that information completely (e.g. dropping bbox_width for
    aspect_ratio, then aspect_ratio for bbox_height, leaves no width at all).
    """
    print_header("5a. FEATURE SELECTION — shape features")
    dropped, protected = set(), set()
    for a, b, r in pairs:
        if a in dropped or b in dropped:
            print(f"  {a} <-> {b} (r = {r:+.2f}): already resolved")
            continue
        if a in protected or b in protected:
            stronger = a if a in protected else b
            weaker = b if stronger == a else a
            reason = f"'{stronger}' already kept as a stand-in"
        else:
            weaker = a if importance.loc[a, "permutation"] < importance.loc[b, "permutation"] else b
            stronger = b if weaker == a else a
            reason = "lower permutation importance"
        dropped.add(weaker)
        protected.add(stronger)
        print(f"  {a} <-> {b} (r = {r:+.2f}): drop '{weaker}', keep '{stronger}'  ({reason})")
    kept = [f for f in FEATURE_COLUMNS if f not in dropped]
    print(f"  Kept {len(kept)} of {len(FEATURE_COLUMNS)}: {kept}")
    return kept, dropped


def select_pixels(train, fit_val):
    """
    5b. Pixel selection, in two filters:
      1. Remove zero-variance pixels (identical in every image -> no information).
      2. ANOVA F-test per pixel (sklearn's f_classif); keep pixels whose mean
         differs significantly between classes (Bonferroni over all pixels).
    Also trains a Random Forest on all pixels to show WHERE the informative
    pixels are, as a 28x28 importance map.
    """
    print_header("5b. FEATURE SELECTION — pixels")
    X = train[PIXEL_COLUMNS].to_numpy()
    y = train["label"].to_numpy()

    variance = X.var(axis=0)
    nonzero_var = variance > 0
    print(f"  Filter 1 — zero-variance pixels removed: {int((~nonzero_var).sum())}")

    f_scores = np.zeros(len(PIXEL_COLUMNS))
    p_values = np.ones(len(PIXEL_COLUMNS))
    f_scores[nonzero_var], p_values[nonzero_var] = f_classif(X[:, nonzero_var], y)
    alpha_corrected = ALPHA / len(PIXEL_COLUMNS)
    significant = nonzero_var & (p_values < alpha_corrected)
    print(f"  Filter 2 — ANOVA F-test, Bonferroni alpha {alpha_corrected:.1e}: "
          f"{int((nonzero_var & ~significant).sum())} more removed")
    print(f"  Pixels kept: {int(significant.sum())} of {len(PIXEL_COLUMNS)}")

    fit_part, _ = fit_val
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(fit_part[PIXEL_COLUMNS], fit_part["label"])
    rf_importance = rf.feature_importances_
    top_share = np.sort(rf_importance)[::-1][:200].sum()
    print(f"  Random Forest on pixels: the top 200 pixels carry {top_share:.0%} of total importance")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (np.log10(f_scores + 1).reshape(28, 28), "ANOVA F-score per pixel (log10)", INK_CMAP, "log10(F + 1)"),
        (rf_importance.reshape(28, 28), "Random Forest importance per pixel", INK_CMAP, "Gini importance"),
    ]
    for ax, (data, title, cmap, cbar_label) in zip(axes[:2], panels):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, color=TEXT)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, color=TEXT_SECONDARY)
    axes[2].imshow(significant.reshape(28, 28), cmap=ListedColormap(["#e1e0d9", FEATURE_COLOR]))
    axes[2].set_title(f"Selected pixels ({int(significant.sum())} of 784)", color=TEXT)
    axes[2].legend(handles=[Patch(color=FEATURE_COLOR, label="kept"),
                            Patch(color="#e1e0d9", label="removed")],
                   loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("Which pixels carry information? (training split)", fontsize=13, color=TEXT)
    fig.tight_layout()
    save(fig, "04_pixel_importance_maps")

    return [col for col, keep in zip(PIXEL_COLUMNS, significant) if keep]


# =============================================================================
# SECTION 6: INTERACTION FEATURES
# =============================================================================

def test_interactions(train, shape_features):
    """
    6. Interaction features = products of two features (e.g. aspect_ratio x
    ink_density). They let a LINEAR model capture combined effects it
    otherwise can't. Tree models (Random Forest) find such combinations on
    their own, so we test both and only keep interactions if they help.

    Measured with 3-fold cross-validation on a sample of the training split.
    """
    print_header("6. INTERACTION FEATURES")
    sample = train.sample(min(CV_SAMPLE_SIZE, len(train)), random_state=SEED)
    X, y = sample[shape_features], sample["label"]
    n = len(shape_features)
    print(f"  Pairwise interactions of {n} shape features: {n * (n - 1) // 2} new columns")

    def logreg():
        return LogisticRegression(max_iter=2000)

    candidates = {
        "Logistic Regression": (
            make_pipeline(StandardScaler(), logreg()),
            make_pipeline(StandardScaler(),
                          PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                          StandardScaler(), logreg()),
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1),
            make_pipeline(PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                          RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1)),
        ),
    }

    results = {}
    print(f"  3-fold CV accuracy on {len(sample)} training images:")
    for name, (without, with_inter) in candidates.items():
        scores_without = cross_val_score(without, X, y, cv=3, n_jobs=-1)
        scores_with = cross_val_score(with_inter, X, y, cv=3, n_jobs=-1)
        acc_without, acc_with = scores_without.mean(), scores_with.mean()
        gain = acc_with - acc_without
        # Both versions see the same 3 folds, so the per-fold differences show
        # how stable the gain is.
        gain_std = (scores_with - scores_without).std()
        keep = gain >= INTERACTION_MIN_GAIN
        results[name] = {"without": acc_without, "with": acc_with, "keep": keep}
        print(f"    {name:<20}: without {acc_without:.1%}  with {acc_with:.1%}  "
              f"gain {gain * 100:+.1f} pp (fold-to-fold std {gain_std * 100:.1f} pp)  "
              f"-> {'KEEP' if keep else 'do not keep'}")
    print(f"  Rule: keep only if the gain is at least {INTERACTION_MIN_GAIN * 100:.1f} pp.")
    return results


# =============================================================================
# SECTION 7: MAIN — RUN EVERYTHING IN ORDER
# =============================================================================

if __name__ == "__main__":
    if not os.path.exists(PROCESSED_CSV_PATH):
        raise FileNotFoundError(
            f"'{PROCESSED_CSV_PATH}' not found. Run 'python src/phase2_preprocessing.py' first."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_chart_style()
    df = load_processed()
    train, _ = split_train_test(df)   # the test part is NOT touched in this phase
    print(f"  Using the training split only: {len(train)} of {len(df)} rows")

    # A validation slice of the training split, for importance measurements.
    fit_val = train_test_split(train, test_size=0.2, random_state=SEED, stratify=train["label"])

    corr = analyze_correlation(train)
    pairs, vif = analyze_multicollinearity(train, corr)
    anova = run_anova(train)
    run_t_tests(train)
    importance = analyze_importance(train, fit_val)
    shape_features, dropped = select_shape_features(pairs, importance)
    plot_importance(importance, dropped)

    vif_after = compute_vif(train, shape_features)
    print(f"  Max VIF after selection: {vif_after.max():.2f}  (was {vif.max():.2f})")

    pixel_features = select_pixels(train, fit_val)
    interactions = test_interactions(train, shape_features)

    selection = {
        "shape_features": shape_features,
        "dropped_shape_features": sorted(dropped),
        "pixel_features": pixel_features,
        "use_interactions": {name: bool(r["keep"]) for name, r in interactions.items()},
    }
    with open(SELECTED_FEATURES_PATH, "w") as f:
        json.dump(selection, f, indent=2)

    print_header("PHASE 4 SUMMARY")
    print(f"  Shape features kept : {len(shape_features)} of {len(FEATURE_COLUMNS)}  "
          f"(dropped: {sorted(dropped) or 'none'})")
    print(f"  Pixels kept         : {len(pixel_features)} of {len(PIXEL_COLUMNS)}")
    print(f"  Strongest feature   : {anova.index[0]} (eta² {anova['eta_squared'].iloc[0]:.3f}), "
          f"top permutation importance: {importance.index[0]}")
    print(f"  Interactions kept   : {selection['use_interactions']}")
    print(f"  Saved               : {SELECTED_FEATURES_PATH}")
    print("\nphase4_feature_analysis.py finished. Next: Phase 5 (machine learning models).")
