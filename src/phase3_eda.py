# =============================================================================
# phase3_eda.py
# Location: src/phase3_eda.py
#
# Purpose: DEV Mini Project — PHASE 3: Exploratory Data Analysis (EDA).
# Explores the processed dataset from Phase 2 through visualizations:
#   1. Target variable distribution      -> bar charts
#   2. Feature distributions             -> histograms + KDE plots
#   3. Outlier detection                 -> box plots + IQR counts + example images
#   4. Numerical features vs target      -> box plots per group and per character
#   5. Relationships between features    -> scatter plots + pair plot
#   6. Image-level patterns              -> mean image per class, ink heatmaps
#
# Every figure is saved to results/phase3_*.png, and the key numbers behind
# each figure are printed so they can be quoted in the report.
#
# Run from the project root (after Phase 2) with:
#     python src/phase3_eda.py
# =============================================================================

# --- Step 1: Import the libraries we need ---
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns                       # Statistical plots on top of matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

sys.path.append(os.path.dirname(__file__))  # So Python can find our own modules
from label_map import get_class_names
from phase1_data_summary import PIXEL_COLUMNS
from phase2_preprocessing import PROCESSED_CSV_PATH, FEATURE_COLUMNS


# =============================================================================
# SETTINGS & CHART STYLE
# =============================================================================

SEED = 42
RESULTS_DIR = "results"

# One fixed color per character group, used in EVERY figure so a reader
# learns it once: blue = digit, orange = uppercase, aqua = lowercase.
# (Palette checked for color-blind separation; every chart also has a legend
# or text labels so color is never the only cue.)
GROUP_ORDER = ["digit", "uppercase", "lowercase"]
GROUP_COLORS = {"digit": "#2a78d6", "uppercase": "#eb6834", "lowercase": "#1baf7a"}
GROUP_MARKERS = {"digit": "o", "uppercase": "s", "lowercase": "^"}

NEUTRAL = "#898781"          # single-series marks (all data, not split by group)
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Character labels use a serif font: in the default sans-serif font capital
# 'I' and lowercase 'l' are drawn identically, which defeats the point of
# charts that compare exactly those characters.
CHAR_FONT = "DejaVu Serif"

# Sequential map for "how often" heatmaps: one hue, light -> dark.
INK_CMAP = LinearSegmentedColormap.from_list("ink", ["#fcfcfb", "#86b6ef", "#2a78d6", "#0d366b"])

# Readable names for the features on chart axes.
FEATURE_LABELS = {
    "ink_pixel_count": "Ink pixel count",
    "mean_intensity":  "Mean intensity",
    "bbox_width":      "Bounding-box width (px)",
    "bbox_height":     "Bounding-box height (px)",
    "aspect_ratio":    "Aspect ratio (w / h)",
    "ink_density":     "Ink density",
    "centroid_x":      "Centre of mass x (px)",
    "centroid_y":      "Centre of mass y (px)",
    "lr_symmetry":     "Left-right symmetry",
    "tb_symmetry":     "Top-bottom symmetry",
}


def set_chart_style():
    """Quiet axes and grid so the data stands out; same look in every figure."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": AXIS,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.fontsize": 8,
        "legend.frameon": False,
    })


def group_legend_handles(marker="s"):
    """Legend entries for the three groups (used where seaborn adds none)."""
    return [Line2D([0], [0], marker=marker, linestyle="", markersize=8,
                   markerfacecolor=GROUP_COLORS[g], markeredgecolor="white", label=g)
            for g in GROUP_ORDER]


def save(fig, name):
    """Saves a figure into results/ and frees its memory."""
    path = os.path.join(RESULTS_DIR, f"phase3_{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# =============================================================================
# SECTION 1: LOAD THE PROCESSED DATA
# =============================================================================

def load_processed():
    """Loads the Phase 2 output with sensible column types."""
    print(f"Loading processed dataset from '{PROCESSED_CSV_PATH}'...")
    dtypes = {col: "uint8" for col in PIXEL_COLUMNS}
    dtypes.update({"char": "string", "group": "string"})
    df = pd.read_csv(PROCESSED_CSV_PATH, dtype=dtypes)
    # Ordered categories make every plot list classes and groups in the same order.
    df["char"] = pd.Categorical(df["char"], categories=get_class_names(), ordered=True)
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# =============================================================================
# SECTION 2: TARGET VARIABLE
# =============================================================================

def plot_target_distribution(df):
    """1. How many images per class (62 bars) and per group (3 bars)."""
    print_header("1. TARGET VARIABLE DISTRIBUTION")

    fig, (ax_class, ax_group) = plt.subplots(
        1, 2, figsize=(18, 4.8), gridspec_kw={"width_ratios": [5, 1]})

    counts = df["char"].value_counts(sort=False)
    char_groups = df.groupby("char", observed=True)["group"].first()
    ax_class.bar(range(len(counts)), counts.values, width=0.8,
                 color=[GROUP_COLORS[g] for g in char_groups[counts.index]])
    ax_class.set_xticks(range(len(counts)))
    ax_class.set_xticklabels(counts.index, fontsize=8, fontfamily=CHAR_FONT)
    ax_class.set_xlim(-0.6, len(counts) - 0.4)
    ax_class.set_title("Images per character (62 classes)")
    ax_class.set_xlabel("Character")
    ax_class.set_ylabel("Number of images")
    ax_class.legend(handles=group_legend_handles(), loc="upper right")

    group_counts = df["group"].value_counts(sort=False)
    ax_group.bar(group_counts.index.astype(str), group_counts.values, width=0.65,
                 color=[GROUP_COLORS[g] for g in group_counts.index])
    for i, n in enumerate(group_counts.values):
        ax_group.text(i, n, f"{n:,}\n({n / len(df):.0%})", ha="center", va="bottom",
                      fontsize=8, color=TEXT_SECONDARY)
    ax_group.set_ylim(0, group_counts.max() * 1.2)
    ax_group.set_title("Images per group")
    ax_group.set_ylabel("Number of images")

    fig.tight_layout()
    save(fig, "01_target_distribution")

    print("  Images per group:")
    for g, n in group_counts.items():
        print(f"    {g:<10}: {n:>6}  ({n / len(df):.1%})")
    sorted_counts = counts.sort_values(ascending=False)
    print(f"  Top 5 classes   : {sorted_counts.head(5).to_dict()}")
    print(f"  Bottom 5 classes: {sorted_counts.tail(5).to_dict()}")
    print(f"  Imbalance ratio : {sorted_counts.iloc[0] / sorted_counts.iloc[-1]:.1f}x")


# =============================================================================
# SECTION 3: FEATURE DISTRIBUTIONS
# =============================================================================

def plot_feature_distributions(df):
    """
    2. Histogram of each feature over ALL images (gray bars), with one KDE
    curve per group on top, so you see both the overall shape and how the
    groups differ.
    """
    print_header("2. FEATURE DISTRIBUTIONS")
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.5))

    for ax, feature in zip(axes.flat, FEATURE_COLUMNS):
        sns.histplot(df[feature], bins=40, stat="density", color=NEUTRAL,
                     alpha=0.25, edgecolor="none", ax=ax)
        sns.kdeplot(data=df, x=feature, hue="group", hue_order=GROUP_ORDER,
                    palette=GROUP_COLORS, common_norm=False, linewidth=2,
                    ax=ax, legend=(ax is axes.flat[0]))
        ax.set_xlabel(FEATURE_LABELS[feature])
        ax.set_ylabel("Density" if ax in axes[:, 0] else "")
        ax.set_title(f"skew {df[feature].skew():+.2f}", fontsize=9, color=TEXT_SECONDARY)

    fig.suptitle("Feature distributions — gray: all images, lines: KDE per group",
                 fontsize=13, color=TEXT)
    fig.tight_layout()
    save(fig, "02_feature_distributions")

    print("  Skewness of each feature (0 = symmetric, >0 = long right tail):")
    for feature in FEATURE_COLUMNS:
        print(f"    {feature:<16}: {df[feature].skew():+.2f}")


# =============================================================================
# SECTION 4: OUTLIERS
# =============================================================================

def iqr_outlier_mask(series):
    """True for values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] (the box-plot rule)."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)


def plot_outliers(df):
    """
    3. Box plot of every feature, with the number of IQR outliers in the
    title, then a figure of actual images at the extremes so we can judge
    whether the "outliers" are errors or genuine handwriting.
    """
    print_header("3. OUTLIER DETECTION (IQR rule)")
    fig, axes = plt.subplots(2, 5, figsize=(20, 5.5))
    outlier_counts = {}

    for ax, feature in zip(axes.flat, FEATURE_COLUMNS):
        n_out = int(iqr_outlier_mask(df[feature]).sum())
        outlier_counts[feature] = n_out
        sns.boxplot(x=df[feature], color="#86b6ef", width=0.5, linewidth=1,
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.3,
                                "markerfacecolor": NEUTRAL, "markeredgecolor": "none"},
                    ax=ax)
        ax.set_xlabel(FEATURE_LABELS[feature])
        ax.set_title(f"{n_out:,} outliers ({n_out / len(df):.1%})", fontsize=9,
                     color=TEXT_SECONDARY)

    fig.suptitle("Outlier detection — box plots (dots = beyond 1.5 × IQR)",
                 fontsize=13, color=TEXT)
    fig.tight_layout()
    save(fig, "03_outlier_boxplots")

    any_outlier = np.zeros(len(df), dtype=bool)
    for feature in FEATURE_COLUMNS:
        any_outlier |= iqr_outlier_mask(df[feature]).to_numpy()
    print("  Outliers per feature:")
    for feature, n in outlier_counts.items():
        print(f"    {feature:<16}: {n:>5}  ({n / len(df):.1%})")
    print(f"  Images that are an outlier on at least one feature: "
          f"{int(any_outlier.sum())} ({any_outlier.mean():.1%})")

    # --- Example images at the extremes ---
    extremes = [
        ("aspect_ratio", "widest"), ("aspect_ratio", "narrowest"),
        ("ink_pixel_count", "most ink"), ("ink_pixel_count", "least ink"),
    ]
    n_show = 8
    fig, axes = plt.subplots(len(extremes), n_show, figsize=(n_show * 1.6, len(extremes) * 2.0))
    for row, (feature, description) in enumerate(extremes):
        ascending = description in ("narrowest", "least ink")
        picks = df.sort_values(feature, ascending=ascending).head(n_show)
        for col, (_, rec) in enumerate(picks.iterrows()):
            ax = axes[row, col]
            ax.imshow(rec[PIXEL_COLUMNS].to_numpy(dtype=np.uint8).reshape(28, 28), cmap="gray")
            ax.set_title(f"'{rec['char']}'  {rec[feature]:.2f}" if feature == "aspect_ratio"
                         else f"'{rec['char']}'  {int(rec[feature])}",
                         fontsize=8, fontfamily=CHAR_FONT)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        axes[row, 0].set_ylabel(f"{description}\n({FEATURE_LABELS[feature]})", fontsize=8)

    fig.suptitle("What the outliers look like — label and feature value above each image",
                 fontsize=12, color=TEXT)
    fig.tight_layout()
    save(fig, "04_outlier_examples")

    print("  Decision: outliers are KEPT. The example images show they are real,")
    print("  readable handwriting (very wide 'm'/'w', thin '1'/'l', bold strokes),")
    print("  not data errors — removing them would remove natural variation the")
    print("  model must learn. Invalid records were already removed in Phase 2.")


# =============================================================================
# SECTION 5: FEATURES VS TARGET
# =============================================================================

def plot_features_vs_target(df):
    """
    4. How each feature differs between the three groups (box plots), and
    how aspect ratio differs across all 62 characters.
    """
    print_header("4. FEATURES VS TARGET")

    # --- 4a. Every feature by group ---
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.5))
    for ax, feature in zip(axes.flat, FEATURE_COLUMNS):
        sns.boxplot(data=df, x="group", y=feature, hue="group", order=GROUP_ORDER,
                    hue_order=GROUP_ORDER, palette=GROUP_COLORS, width=0.55,
                    linewidth=1, showfliers=False, legend=False, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(FEATURE_LABELS[feature])
    fig.suptitle("Shape features by character group (outliers hidden; see figure 03)",
                 fontsize=13, color=TEXT)
    fig.tight_layout()
    save(fig, "05_features_vs_group")

    medians = df.groupby("group", observed=True)[FEATURE_COLUMNS].median().T
    print("  Median of each feature per group:")
    print("    " + medians.round(3).to_string().replace("\n", "\n    "))

    # --- 4b. Aspect ratio for every character, sorted by median ---
    order = (df.groupby("char", observed=True)["aspect_ratio"].median()
               .sort_values().index.astype(str).tolist())
    char_group = df.groupby("char", observed=True)["group"].first()
    palette = {str(ch): GROUP_COLORS[g] for ch, g in char_group.items()}

    fig, ax = plt.subplots(figsize=(18, 5))
    sns.boxplot(data=df.assign(char=df["char"].astype(str)), x="char", y="aspect_ratio",
                hue="char", order=order, palette=palette, width=0.7, linewidth=0.8,
                showfliers=False, legend=False, ax=ax)
    ax.axhline(1.0, color=AXIS, linewidth=1, linestyle="--", zorder=0)
    ax.set_title("Aspect ratio per character, sorted by median — tall/thin on the left, wide on the right")
    ax.set_xlabel("Character")
    ax.set_ylabel(FEATURE_LABELS["aspect_ratio"])
    for tick in ax.get_xticklabels():
        tick.set_fontfamily(CHAR_FONT)
    square_handle = Line2D([0], [0], color=AXIS, linewidth=1, linestyle="--", label="square (w = h)")
    ax.legend(handles=group_legend_handles() + [square_handle], loc="upper left")
    fig.tight_layout()
    save(fig, "06_aspect_ratio_by_character")

    print(f"  Narrowest characters (median aspect ratio): {order[:6]}")
    print(f"  Widest characters    (median aspect ratio): {order[-6:]}")


# =============================================================================
# SECTION 6: RELATIONSHIPS BETWEEN FEATURES
# =============================================================================

def plot_relationships(df):
    """
    5a. Scatter plots for two sets of look-alike characters. Each set has one
    digit, one uppercase and one lowercase character, so the group colors
    apply directly. If the clouds overlap, these features cannot tell the
    characters apart — a confusion we should expect to see in Phase 5.
    5b. Pair plot of four features to show pairwise relationships per group.
    """
    print_header("5. RELATIONSHIPS BETWEEN FEATURES")
    # char -> (group, spelled-out name so 'I' and 'l' can't be mixed up)
    lookalike_sets = [
        ({"1": ("digit", "one"), "I": ("uppercase", "capital i"), "l": ("lowercase", "small L")},
         "aspect_ratio", "ink_density"),
        ({"0": ("digit", "zero"), "O": ("uppercase", "capital o"), "o": ("lowercase", "small o")},
         "bbox_height", "aspect_ratio"),
    ]
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (chars, x, y) in zip(axes, lookalike_sets):
        for ch, (group, name) in chars.items():
            subset = df[df["char"] == ch]
            subset = subset.sample(min(len(subset), 300), random_state=SEED)
            # Integer features (like bbox_height) stack points on top of each
            # other; a small random jitter makes the density visible.
            x_vals = subset[x] + (rng.uniform(-0.3, 0.3, len(subset))
                                  if x.startswith("bbox") else 0)
            ax.scatter(x_vals, subset[y], s=22, alpha=0.55, color=GROUP_COLORS[group],
                       marker=GROUP_MARKERS[group], edgecolors="white", linewidths=0.4,
                       label=f"'{ch}' {name} ({group}) — median {df.loc[df['char'] == ch, x].median():.2f}")
            print(f"  '{ch}': median {x} = {df.loc[df['char'] == ch, x].median():.3f}, "
                  f"median {y} = {df.loc[df['char'] == ch, y].median():.3f}")
        ax.set_xlabel(FEATURE_LABELS[x])
        ax.set_ylabel(FEATURE_LABELS[y])
        ax.set_title("Look-alikes: " + " vs ".join(f"{name}" for _, name in chars.values()))
        legend = ax.legend(loc="best", markerscale=1.3, title=f"median {FEATURE_LABELS[x].lower()}",
                           title_fontsize=8)
        for text in legend.get_texts():
            text.set_fontfamily(CHAR_FONT)
    fig.tight_layout()
    save(fig, "07_lookalike_scatter")

    # --- 5b. Pair plot ---
    pair_features = ["aspect_ratio", "ink_pixel_count", "centroid_y", "lr_symmetry"]
    sample = df.sample(3000, random_state=SEED)
    grid = sns.pairplot(sample, vars=pair_features, hue="group", hue_order=GROUP_ORDER,
                        palette=GROUP_COLORS, corner=True, height=2.3,
                        plot_kws={"s": 10, "alpha": 0.4, "edgecolor": "none"},
                        diag_kws={"common_norm": False, "linewidth": 2, "fill": False})
    for ax in grid.axes.flat:
        if ax is not None:
            ax.set_xlabel(FEATURE_LABELS.get(ax.get_xlabel(), ax.get_xlabel()), fontsize=8)
            ax.set_ylabel(FEATURE_LABELS.get(ax.get_ylabel(), ax.get_ylabel()), fontsize=8)
    grid.figure.suptitle("Pair plot of four shape features (3,000-image random sample)",
                         fontsize=12, color=TEXT, y=1.02)
    path = os.path.join(RESULTS_DIR, "phase3_08_pairplot.png")
    grid.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(grid.figure)
    print(f"  Saved: {path}")


# =============================================================================
# SECTION 7: IMAGE-LEVEL PATTERNS
# =============================================================================

def plot_image_patterns(df):
    """
    6a. The average image of every class — shows the "typical" shape the
        model has to learn for each character.
    6b. How often each pixel contains ink, overall and per group — shows
        which pixels carry information and which are always background.
    """
    print_header("6. IMAGE-LEVEL PATTERNS")
    images = df[PIXEL_COLUMNS].to_numpy().reshape(-1, 28, 28)
    class_names = get_class_names()
    chars = df["char"].astype(str).to_numpy()

    # --- 6a. Mean image per class ---
    cols = 11
    rows = int(np.ceil(len(class_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25, rows * 1.45))
    for ax in axes.flat:
        ax.axis("off")
    for ax, ch in zip(axes.flat, class_names):
        ax.imshow(images[chars == ch].mean(axis=0), cmap="gray")
        ax.set_title(f"'{ch}'  n={int((chars == ch).sum())}", fontsize=7, fontfamily=CHAR_FONT)
    fig.suptitle("Average image of each character", fontsize=13, color=TEXT)
    fig.tight_layout()
    save(fig, "09_mean_image_per_class")

    # --- 6b. Ink frequency heatmaps ---
    ink = images > 0
    panels = [("All images", np.ones(len(df), dtype=bool))]
    panels += [(g, (df["group"] == g).to_numpy()) for g in GROUP_ORDER]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
    for ax, (title, mask) in zip(axes, panels):
        im = ax.imshow(ink[mask].mean(axis=0), cmap=INK_CMAP, vmin=0, vmax=1)
        ax.set_title(f"{title}  (n={int(mask.sum()):,})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Share of images with ink at this pixel", color=TEXT_SECONDARY)
    fig.suptitle("Where ink appears — darker = inked in more images", fontsize=13, color=TEXT)
    save(fig, "10_ink_frequency_heatmap")

    ink_share = ink.mean(axis=0)
    print(f"  Pixels never inked in any image         : {int((ink_share == 0).sum())} of 784")
    print(f"  Pixels inked in fewer than 1% of images : {int((ink_share < 0.01).sum())} of 784")
    print(f"  Pixels inked in more than 50% of images : {int((ink_share > 0.5).sum())} of 784")
    print("  -> Border pixels carry almost no information; Phase 4 uses this for")
    print("     feature selection.")


# =============================================================================
# SECTION 8: MAIN — RUN EVERYTHING IN ORDER
# =============================================================================

if __name__ == "__main__":
    if not os.path.exists(PROCESSED_CSV_PATH):
        raise FileNotFoundError(
            f"'{PROCESSED_CSV_PATH}' not found. Run 'python src/phase2_preprocessing.py' first."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_chart_style()
    df = load_processed()

    plot_target_distribution(df)
    plot_feature_distributions(df)
    plot_outliers(df)
    plot_features_vs_target(df)
    plot_relationships(df)
    plot_image_patterns(df)

    print("\nphase3_eda.py finished. Figures are in results/phase3_*.png.")
    print("Next: Phase 4 (feature engineering & statistical observations).")
