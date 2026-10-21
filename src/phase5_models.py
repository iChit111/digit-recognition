# =============================================================================
# phase5_models.py
# Location: src/phase5_models.py
#
# Purpose: DEV Mini Project — PHASE 5: Machine Learning Models.
# Builds, evaluates and compares five classifiers on the same data:
#   1. Logistic Regression   (linear baseline, + interaction features)
#   2. Random Forest         (ensemble of decision trees)
#   3. SVM, RBF kernel       (on 50 PCA components of the pixels)
#   4. KNN                   (on the same 50 PCA components)
#   5. Dense Neural Network  (the architecture from model.py)
#
# Metrics: Accuracy, Precision, Recall, F1-score (macro and weighted),
# confusion matrix, per-group accuracy, and training / prediction time.
#
# HOW THE COMPARISON IS KEPT FAIR:
#   - Same train/test split for every model: split_train_test() from Phase 4
#     (test_size=0.2, random_state=42, stratified). Phase 4 selected features
#     from the training part only, so the test part is still unseen.
#   - Same input columns for every model: the pixels and shape features
#     selected in Phase 4 (data/selected_features.json).
#   - All scaling / PCA is inside a scikit-learn Pipeline, so it is fitted on
#     the training split only and merely APPLIED to the test split.
#   - Default hyperparameters for every model, no class weights for any —
#     no model gets extra tuning the others didn't.
#
# WHY THE NEURAL NETWORK IS RETRAINED HERE (and results/model.h5 is not used):
# model.h5 was trained on the FULL EMNIST training split, which contains every
# image in our 50,000 sample — including this phase's test rows. Scoring it
# here would test it on images it has already seen. So we train the same
# architecture on this phase's 40,000 training rows instead. model.h5's own
# results (full dataset, official EMNIST test split) come from evaluate.py.
#
# Run from the project root (after Phase 4) with:
#     python src/phase5_models.py
# Takes roughly 10 minutes.
# =============================================================================

# --- Step 1: Import the libraries we need ---
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest           # For McNemar's test
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

sys.path.append(os.path.dirname(__file__))  # So Python can find our own modules
from label_map import (LABEL_TO_CHAR, NUM_CLASSES, DIGIT_LABELS, UPPERCASE_LABELS,
                       LOWERCASE_LABELS, get_class_names)
from phase3_eda import (CHAR_FONT, GROUP_COLORS, GROUP_ORDER, INK_CMAP, TEXT,
                        TEXT_SECONDARY, load_processed, print_header, set_chart_style)
from phase4_feature_analysis import SELECTED_FEATURES_PATH, split_train_test


# =============================================================================
# SETTINGS
# =============================================================================

SEED = 42
PCA_COMPONENTS = 50                 # Pixel dimensions kept for SVM and KNN
NN_MAX_EPOCHS = 100                 # EarlyStopping ends training much sooner
NN_BATCH_SIZE = 128
NN_PATIENCE = 5
RESULTS_DIR = "results"
COMPARISON_CSV_PATH = "results/phase5_model_comparison.csv"

MODEL_COLOR = "#2a78d6"             # single-series bars
BEST_COLOR = "#0d366b"              # the best model, darker step of the same hue

GROUPS = {"digit": DIGIT_LABELS, "uppercase": UPPERCASE_LABELS, "lowercase": LOWERCASE_LABELS}


def save(fig, name):
    path = os.path.join(RESULTS_DIR, f"phase5_{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# SECTION 1: DATA
# =============================================================================

def load_data_and_selection():
    """Loads the processed data, the Phase 4 feature selection, and splits it."""
    with open(SELECTED_FEATURES_PATH) as f:
        selection = json.load(f)
    df = load_processed()
    train, test = split_train_test(df)

    pixel_cols = selection["pixel_features"]
    shape_cols = selection["shape_features"]
    feature_cols = pixel_cols + shape_cols
    print(f"  Train: {len(train)} rows   Test: {len(test)} rows")
    print(f"  Features: {len(pixel_cols)} pixels + {len(shape_cols)} shape features "
          f"= {len(feature_cols)} columns")
    return (train[feature_cols], train["label"].to_numpy(),
            test[feature_cols], test["label"].to_numpy(),
            pixel_cols, shape_cols, selection)


# =============================================================================
# SECTION 2: MODEL DEFINITIONS
# =============================================================================

def scale_pixels(x):
    """Pixel normalization 0-255 -> 0.0-1.0 (a fixed formula; nothing to learn)."""
    return np.asarray(x, dtype=np.float32) / 255.0


def build_sklearn_models(pixel_cols, shape_cols, use_interactions):
    """
    Each model is a Pipeline: preprocessing steps + classifier. A
    ColumnTransformer treats the pixel columns and the shape columns
    differently, because they need different preprocessing.
    """
    def interactions_then_scale():
        return make_pipeline(StandardScaler(),
                             PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                             StandardScaler())

    # Linear model: normalized pixels + standardized shape features
    # (+ their pairwise interactions, if Phase 4 found they help).
    logreg_prep = ColumnTransformer([
        ("pixels", FunctionTransformer(scale_pixels), pixel_cols),
        ("shape", interactions_then_scale() if use_interactions["Logistic Regression"]
                  else StandardScaler(), shape_cols),
    ])

    # Trees split on thresholds, so scaling changes nothing for them.
    forest_prep = ColumnTransformer([
        ("pixels", "passthrough", pixel_cols),
        ("shape", interactions_then_scale() if use_interactions["Random Forest"]
                  else "passthrough", shape_cols),
    ])

    # Distance-based models: in 687 dimensions every point is roughly equally
    # far from every other ("curse of dimensionality") and prediction is slow.
    # PCA compresses the pixels to 50 components that keep most of the variance.
    def distance_prep():
        return ColumnTransformer([
            ("pixels", make_pipeline(FunctionTransformer(scale_pixels),
                                     PCA(n_components=PCA_COMPONENTS, random_state=SEED)),
             pixel_cols),
            ("shape", StandardScaler(), shape_cols),
        ])

    return {
        "Logistic Regression": make_pipeline(logreg_prep, LogisticRegression(max_iter=1000)),
        "Random Forest": make_pipeline(forest_prep, RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1)),
        "SVM (RBF)": make_pipeline(distance_prep(), SVC(kernel="rbf", random_state=SEED)),
        "KNN (k=5)": make_pipeline(distance_prep(), KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1)),
    }


def train_neural_network(X_train, y_train, X_test, pixel_cols, shape_cols):
    """
    Trains the model.py architecture on this phase's training split.
    Preprocessing: normalized pixels + standardized shape features (fitted on
    the training split only). 10% of the training split is held out for
    EarlyStopping; the test split is never seen during training.
    """
    import keras                                # TensorFlow loads only when needed
    from model import build_model

    keras.utils.set_random_seed(SEED)
    prep = ColumnTransformer([
        ("pixels", FunctionTransformer(scale_pixels), pixel_cols),
        ("shape", StandardScaler(), shape_cols),
    ])
    X_tr_all = prep.fit_transform(X_train).astype(np.float32)
    X_te = prep.transform(X_test).astype(np.float32)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_all, y_train, test_size=0.1, random_state=SEED, stratify=y_train)

    model = build_model(input_dim=X_tr.shape[1])
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=NN_PATIENCE, restore_best_weights=True)

    start = time.perf_counter()
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=NN_MAX_EPOCHS,
                        batch_size=NN_BATCH_SIZE, callbacks=[early_stop], verbose=0)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    predict_seconds = time.perf_counter() - start

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"    stopped after {len(history.history['loss'])} epochs (best: {best_epoch})")
    return y_pred, fit_seconds, predict_seconds


# =============================================================================
# SECTION 3: EVALUATION
# =============================================================================

def evaluate(name, y_true, y_pred, fit_seconds, predict_seconds):
    """All metrics for one model, as one dict (one row of the comparison table)."""
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    row = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_m,
        "recall_macro": recall_m,
        "f1_macro": f1_m,
        "f1_weighted": f1_w,
    }
    for group, labels in GROUPS.items():
        mask = np.isin(y_true, labels)
        row[f"acc_{group}"] = float(np.mean(y_pred[mask] == y_true[mask]))

    # Share of errors that are only a case mix-up ('o' predicted as 'O' etc.).
    wrong = y_pred != y_true
    case_only = sum(LABEL_TO_CHAR[int(t)].lower() == LABEL_TO_CHAR[int(p)].lower()
                    for t, p in zip(y_true[wrong], y_pred[wrong]))
    row["case_only_error_share"] = case_only / max(int(wrong.sum()), 1)
    row["fit_seconds"] = fit_seconds
    row["predict_seconds"] = predict_seconds
    return row


def print_comparison(results):
    print_header("MODEL COMPARISON (test split, 10,000 images)")
    table = results.set_index("model")
    shown = table[["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]]
    print((shown * 100).round(2).to_string())
    print("  (all values in %; 'macro' = every class counts equally, so rare classes matter;")
    print("   'weighted' = classes count by how often they occur)")

    print("\n  Accuracy per character group (%) and share of errors that are case-only:")
    groups = table[[f"acc_{g}" for g in GROUP_ORDER] + ["case_only_error_share"]]
    print((groups * 100).round(1).to_string())

    print("\n  Time (seconds):")
    print(table[["fit_seconds", "predict_seconds"]].round(1).to_string())


# =============================================================================
# SECTION 4: PLOTS
# =============================================================================

def plot_metric_comparison(results, best):
    """Small multiples: one panel per metric, models in the same order everywhere."""
    metrics = [("accuracy", "Accuracy"), ("precision_macro", "Precision (macro)"),
               ("recall_macro", "Recall (macro)"), ("f1_macro", "F1-score (macro)")]
    ordered = results.sort_values("accuracy")
    colors = [BEST_COLOR if m == best else MODEL_COLOR for m in ordered["model"]]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4.2), sharey=True)
    for ax, (col, title) in zip(axes, metrics):
        ax.barh(ordered["model"], ordered[col] * 100, color=colors, height=0.6)
        for y, v in enumerate(ordered[col] * 100):
            ax.text(v + 1, y, f"{v:.1f}%", va="center", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xlim(0, 105)
        ax.set_title(title, color=TEXT)
        ax.set_xlabel("%")
    fig.suptitle(f"Model comparison on the test split (darker bar = highest accuracy: {best})",
                 fontsize=13, color=TEXT)
    fig.tight_layout()
    save(fig, "01_model_comparison")


def plot_group_accuracy(results):
    """Accuracy per character group for every model (grouped bars)."""
    ordered = results.sort_values("accuracy")
    x = np.arange(len(ordered))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, group in enumerate(GROUP_ORDER):
        values = ordered[f"acc_{group}"] * 100
        # A 2px white edge separates neighbouring bars.
        ax.bar(x + (i - 1) * width, values, width=width, color=GROUP_COLORS[group],
               edgecolor="white", linewidth=2, label=group)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["model"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy per character group — letters are harder than digits", color=TEXT)
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    save(fig, "02_accuracy_per_group")


def plot_confusion(y_true, y_pred, name):
    """Row-normalized 62x62 confusion matrix of the best model."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    names = get_class_names()

    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(cm_norm, cmap=INK_CMAP, vmin=0, vmax=1, square=True,
                xticklabels=names, yticklabels=names,
                cbar_kws={"label": "Share of the true class", "shrink": 0.7}, ax=ax)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily(CHAR_FONT)
        tick.set_fontsize(7)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    # Lines between the digit / uppercase / lowercase blocks.
    for boundary in (10, 36):
        ax.axhline(boundary, color=TEXT_SECONDARY, linewidth=0.8)
        ax.axvline(boundary, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_xlabel("Predicted character")
    ax.set_ylabel("True character")
    ax.set_title(f"Confusion matrix — {name} (row-normalized; off-diagonal = mistakes)",
                 fontsize=13, color=TEXT)
    save(fig, "03_confusion_matrix_best")
    return cm


def report_best_model(y_true, y_pred, cm, name):
    """Per-class F1 extremes and the most frequent mistakes of the best model."""
    print_header(f"BEST MODEL IN DETAIL — {name}")
    names = get_class_names()
    report = classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                   target_names=names, output_dict=True, zero_division=0)
    f1 = pd.Series({ch: report[ch]["f1-score"] for ch in names}).sort_values()
    support = pd.Series({ch: report[ch]["support"] for ch in names})
    print("  Easiest classes (highest F1):")
    for ch in f1.index[::-1][:8]:
        print(f"    '{ch}': F1 {f1[ch]:.2f}  (n={int(support[ch])})")
    print("  Hardest classes (lowest F1):")
    for ch in f1.index[:8]:
        print(f"    '{ch}': F1 {f1[ch]:.2f}  (n={int(support[ch])})")

    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    top = np.argsort(off_diag.ravel())[::-1][:12]
    print("  Most frequent mistakes (true -> predicted):")
    for flat in top:
        t, p = divmod(int(flat), NUM_CLASSES)
        case_note = "  (case only)" if names[t].lower() == names[p].lower() else ""
        print(f"    '{names[t]}' -> '{names[p]}': {off_diag[t, p]} times{case_note}")


def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar's test: are two classifiers' accuracies really different on the
    same test images, or could the gap be chance?

    Only images where exactly ONE model is right matter:
        b = A right, B wrong      c = A wrong, B right
    H0: both models are equally good -> b and c are equally likely, so b
    follows Binomial(b + c, 0.5). The exact binomial test gives the p-value.
    """
    a_right = pred_a == y_true
    b_right = pred_b == y_true
    b = int((a_right & ~b_right).sum())
    c = int((~a_right & b_right).sum())
    p_value = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, p_value


# =============================================================================
# SECTION 5: MAIN — TRAIN, EVALUATE, COMPARE
# =============================================================================

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_chart_style()

    print_header("DATA")
    X_train, y_train, X_test, y_test, pixel_cols, shape_cols, selection = load_data_and_selection()

    models = build_sklearn_models(pixel_cols, shape_cols, selection["use_interactions"])
    rows, predictions = [], {}

    print_header("TRAINING")
    for name, pipeline in models.items():
        print(f"  {name}...")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            print("    note: optimizer hit max_iter before fully converging")

        start = time.perf_counter()
        y_pred = pipeline.predict(X_test)
        predict_seconds = time.perf_counter() - start

        if name == "SVM (RBF)":
            pca = pipeline[0].named_transformers_["pixels"][-1]
            print(f"    PCA: {PCA_COMPONENTS} components keep "
                  f"{pca.explained_variance_ratio_.sum():.1%} of pixel variance")

        predictions[name] = y_pred
        rows.append(evaluate(name, y_test, y_pred, fit_seconds, predict_seconds))
        print(f"    accuracy {rows[-1]['accuracy']:.2%}  (fit {fit_seconds:.0f}s)")

    print("  Dense Neural Network...")
    y_pred, fit_seconds, predict_seconds = train_neural_network(
        X_train, y_train, X_test, pixel_cols, shape_cols)
    predictions["Dense Neural Network"] = y_pred
    rows.append(evaluate("Dense Neural Network", y_test, y_pred, fit_seconds, predict_seconds))
    print(f"    accuracy {rows[-1]['accuracy']:.2%}  (fit {fit_seconds:.0f}s)")

    results = pd.DataFrame(rows)
    results.round(4).to_csv(COMPARISON_CSV_PATH, index=False)
    print(f"\n  Saved: {COMPARISON_CSV_PATH}")
    print_comparison(results)

    best = results.loc[results["accuracy"].idxmax(), "model"]
    print_header("PLOTS")
    plot_metric_comparison(results, best)
    plot_group_accuracy(results)
    cm = plot_confusion(y_test, predictions[best], best)
    report_best_model(y_test, predictions[best], cm, best)

    # --- Is the best model really better than the runner-up? ---
    ranked = results.sort_values("accuracy", ascending=False)
    best_row, runner_up = ranked.iloc[0], ranked.iloc[1]
    print_header(f"McNEMAR'S TEST — {best} vs {runner_up['model']}")
    b, c, p_value = mcnemar_test(y_test, predictions[best], predictions[runner_up["model"]])
    print("  H0: both models have the same accuracy on this test set")
    print(f"  Images only {best} got right            : {b}")
    print(f"  Images only {runner_up['model']} got right : {c}")
    print(f"  p-value = {p_value:.3f}  -> "
          f"{'significant: reject H0' if p_value < 0.05 else 'NOT significant: the gap could be chance'}")

    print_header("CONCLUSION")
    gap = (best_row["accuracy"] - runner_up["accuracy"]) * 100
    if p_value < 0.05:
        print(f"  Best model : {best} — accuracy {best_row['accuracy']:.2%}, significantly "
              f"ahead of {runner_up['model']} ({gap:.1f} pp, p = {p_value:.3f}).")
    else:
        print(f"  Top models : {best} ({best_row['accuracy']:.2%}) and {runner_up['model']} "
              f"({runner_up['accuracy']:.2%}) are statistically tied ({gap:.1f} pp, p = {p_value:.3f}).")
        f1_leader = ranked.iloc[:2].sort_values("f1_macro", ascending=False).iloc[0]
        print(f"               {f1_leader['model']} has the higher macro F1 "
              f"({f1_leader['f1_macro']:.2%}), i.e. it handles rare classes better.")
    worst = results.loc[results["accuracy"].idxmin()]
    print(f"  Weakest    : {worst['model']} ({worst['accuracy']:.2%}) — a linear model cannot")
    print("               capture curved stroke shapes even with interaction features.")
    print("  Digits are recognised far better than letters in every model. The errors come")
    print("  from characters that are drawn the same way: l / I -> 1, O <-> 0, S <-> 5, and")
    print(f"  case-only pairs (m/M, c/C, u/U...: {best_row['case_only_error_share']:.0%} of "
          f"{best}'s errors). Lowercase")
    print("  look-alikes are almost never predicted, because their uppercase twin is more")
    print("  common in the training data — the class imbalance found in Phases 1 and 3.")
    print("\nphase5_models.py finished.")
