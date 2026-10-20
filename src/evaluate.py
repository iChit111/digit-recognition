# =============================================================================
# evaluate.py
# Location: src/evaluate.py
# Purpose: Load the trained model and measure how well it performs on the
# EMNIST ByClass test set — data it has NEVER seen during training.
# We use several tools to understand performance across all 62 classes:
#   1. Overall accuracy & loss  — one number that summarises everything
#   2. Per-group accuracy       — Digits (0-9), Uppercase (A-Z), Lowercase (a-z)
#   3. Confusion matrix         — row-normalised 62x62 heatmap of confusions
#   4. Top-15 confused pairs    — most frequent (true -> predicted) mistakes
#   5. Case-only confusions     — how many errors are just uppercase vs lowercase
#   6. Misclassified examples   — actual images the model got wrong
# =============================================================================

# --- Import the libraries we need ---
import numpy as np                          # For numerical operations
import matplotlib.pyplot as plt             # For plotting charts and images
import tensorflow as tf                     # The main deep learning library
import keras                # High-level API for loading models
from sklearn.metrics import confusion_matrix, classification_report
# sklearn (scikit-learn) has ready-made tools for measuring classifier performance.
# confusion_matrix: shows a grid of predicted vs. actual labels.
# classification_report: prints precision, recall, and F1-score per character.
import sys
import os
sys.path.append(os.path.dirname(__file__))  # So Python can find data_loader.py and label_map.py
from data_loader import load_data, preprocess_data
from label_map import (
    NUM_CLASSES,
    LABEL_TO_CHAR,
    DIGIT_LABELS,
    UPPERCASE_LABELS,
    LOWERCASE_LABELS,
    get_class_names,
)


# =============================================================================
# SECTION 1: LOAD THE SAVED MODEL
# =============================================================================

def load_trained_model(filepath="results/model.h5"):
    """
    Loads a previously trained and saved Keras model from disk.
    We saved the model at the end of model.py using model.save("results/model.h5").
    keras.models.load_model() reads that file and reconstructs the exact same
    model — architecture, weights, and all — ready to make predictions.

    Args:
        filepath: Path to the saved .h5 model file (default: "results/model.h5").
    Returns:
        model: The loaded Keras model, ready for evaluation.
    """
    print(f"Loading trained model from '{filepath}'...")
    # If the file doesn't exist, give a helpful error instead of a cryptic crash.
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n  ERROR: Model file '{filepath}' not found.\n"
            f"  Please run 'python src/model.py' first to train and save the model."
        )
    model = keras.models.load_model(filepath)
    print("  Model loaded successfully!")
    model.summary()   # Print a reminder of the model architecture
    return model


# =============================================================================
# SECTION 2: EVALUATE OVERALL ACCURACY AND LOSS
# =============================================================================

def evaluate_model(model, x_test, y_test):
    """
    Runs the model on the entire test set and reports overall performance.
    model.evaluate() feeds every test image through the network, compares
    each prediction to the true label, and returns the average loss and
    accuracy across all 116,323 test images.

    Args:
        model:  The loaded Keras model.
        x_test: Preprocessed test images — shape (116323, 784)
        y_test: True test labels (integers 0-61) — shape (116323,)
    Returns:
        test_loss:     Average loss on the test set (lower is better).
        test_accuracy: Fraction of correct predictions (higher is better).
    """
    print("\nEvaluating model on the test set...")
    print(f"  (This may take a few seconds — the model is checking {len(x_test)} images)")
    # verbose=0 suppresses the progress bar so our output stays clean.
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("\n========== OVERALL PERFORMANCE ==========")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_accuracy * 100:.2f}%")
    print("  (Accuracy = correct predictions / total predictions)")
    print("=========================================")
    return test_loss, test_accuracy


# =============================================================================
# SECTION 3: GENERATE PREDICTIONS & GROUP / CASE BREAKDOWNS
# =============================================================================

def get_predictions(model, x_test):
    """
    Runs the model on all test images and returns the predicted class for each.
    model.predict() returns a 62-element probability array for each image.
    np.argmax() picks the index (0-61) with the highest probability — that's
    the model's final predicted character class.

    Args:
        model:  The loaded Keras model.
        x_test: Preprocessed test images — shape (116323, 784)
    Returns:
        y_pred:  Array of predicted labels (integers 0-61) — shape (116323,)
        y_proba: Full probability arrays for all images — shape (116323, 62)
    """
    print(f"\nGenerating predictions for all {len(x_test)} test images...")
    # model.predict() returns a 2D array: one row per image, 62 columns (one per class).
    y_proba = model.predict(x_test, verbose=0)
    # np.argmax(axis=1) takes the column index with the highest value in each row.
    y_pred = np.argmax(y_proba, axis=1)
    print(f"  Predictions generated for {len(y_pred)} images.")
    return y_pred, y_proba


def print_per_group_accuracy(y_test, y_pred):
    """
    Measures and prints accuracy per character group (Digits 0-9,
    Uppercase A-Z, Lowercase a-z) using the label_map groupings,
    matching the breakdown in model.py.
    """
    groups = {
        "Digits (0-9)":     DIGIT_LABELS,
        "Uppercase (A-Z)":  UPPERCASE_LABELS,
        "Lowercase (a-z)":  LOWERCASE_LABELS,
    }
    print("\n========== PER-GROUP TEST ACCURACY ==========")
    for name, labels in groups.items():
        mask = np.isin(y_test, list(labels))
        acc = float(np.mean(y_pred[mask] == y_test[mask]))
        count = int(mask.sum())
        print(f"  {name:<18}: {acc * 100:.2f}%  ({count} test images)")
    print("=============================================")


def print_case_only_confusions(y_test, y_pred):
    """
    Counts how many errors are 'case-only' confusions — where the model
    predicted the right letter, just the wrong case (e.g. 'c' vs 'C',
    's' vs 'S', 'o' vs 'O') — and prints it as a percentage of all errors.
    """
    wrong_mask = (y_pred != y_test)
    total_errors = int(wrong_mask.sum())
    if total_errors == 0:
        print("\nNo misclassifications found!")
        return

    wrong_true = y_test[wrong_mask]
    wrong_pred = y_pred[wrong_mask]

    # Compare lowercase versions of the true and predicted characters.
    # Since wrong_mask guarantees true != pred, any match here is a case-only error.
    case_only_errors = sum(
        1 for t, p in zip(wrong_true, wrong_pred)
        if LABEL_TO_CHAR[int(t)].lower() == LABEL_TO_CHAR[int(p)].lower()
    )
    pct = (case_only_errors / total_errors) * 100.0

    print("\n========== CASE-ONLY CONFUSION ANALYSIS ==========")
    print(f"  Total errors       : {total_errors}")
    print(f"  Case-only errors   : {case_only_errors} ({pct:.2f}% of all errors)")
    print("  (e.g. predicting 'C' for 'c' or 's' for 'S')")
    print("==================================================")


# =============================================================================
# SECTION 4: CONFUSION MATRIX & TOP-15 CONFUSED PAIRS
# =============================================================================

def plot_confusion_matrix(y_test, y_pred):
    """
    Builds and visualises the 62x62 row-normalised confusion matrix,
    prints the top 15 most-confused character pairs, and prints the
    full 62-class classification report.

    Args:
        y_test: True test labels — shape (116323,)
        y_pred: Predicted labels — shape (116323,)
    """
    print("\nGenerating confusion matrix...")
    class_names = get_class_names()
    all_labels = list(range(len(class_names)))

    # Build the raw counts 62x62 confusion matrix.
    # cm[i, j] = number of times true class i was predicted as class j.
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)

    # Row-normalise the matrix so each row sums to 1.0 (fraction of that true class).
    # Guard against division by zero in case a class has 0 samples.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=(row_sums != 0),
    )

    # --- Plot the row-normalised 62x62 confusion matrix ---
    # A 62x62 grid with per-cell numbers is unreadable, so we omit cell text
    # annotations, use a larger figure, and label ticks with characters at fontsize ~6.
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of True Class", fontsize=11)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)

    ax.set_title(
        "Row-Normalised Confusion Matrix — Predicted vs. True Character (62 Classes)",
        fontsize=14,
        pad=15,
    )
    ax.set_ylabel("True Label (actual character)", fontsize=12)
    ax.set_xlabel("Predicted Label (model's guess)", fontsize=12)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    save_path = "results/confusion_matrix.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved: {save_path}")

    # --- Print the Top 15 most-confused (true -> predicted) pairs by count ---
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)   # Zero out correct predictions on the diagonal

    # Flatten and sort in descending order to get the highest off-diagonal counts
    flat_top_indices = np.argsort(cm_off_diag.ravel())[::-1][:15]
    n_classes = cm.shape[0]

    print("\n========== TOP 15 MOST-CONFUSED PAIRS (True -> Predicted) ==========")
    for rank, flat_idx in enumerate(flat_top_indices, start=1):
        true_idx = int(flat_idx // n_classes)
        pred_idx = int(flat_idx % n_classes)
        count = int(cm_off_diag[true_idx, pred_idx])
        true_char = LABEL_TO_CHAR[true_idx]
        pred_char = LABEL_TO_CHAR[pred_idx]
        print(f"  {rank:2d}. {true_char} -> {pred_char}: {count} times")
    print("====================================================================")

    # --- Also print per-character stats as a table ---
    print("\nPer-character classification report:")
    print("  (precision = of all times the model predicted char X, how often was it right?)")
    print("  (recall    = of all actual char X images, how many did the model catch?)")
    print("  (f1-score  = harmonic mean of precision and recall — a balanced summary)\n")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=all_labels,
            target_names=class_names,
            zero_division=0,
        )
    )


# =============================================================================
# SECTION 5: SHOW MISCLASSIFIED EXAMPLES
# =============================================================================

def show_misclassified(x_test_raw, y_test, y_pred, y_proba, num_examples=15):
    """
    Displays images that the model got WRONG, along with what character it
    predicted and how confident it was.

    Args:
        x_test_raw:   ORIGINAL (unflattened) test images — shape (116323, 28, 28)
        y_test:       True test labels — shape (116323,)
        y_pred:       Predicted labels — shape (116323,)
        y_proba:      Full probability arrays — shape (116323, 62)
        num_examples: How many misclassified images to show (default: 15).
    """
    print("\nFinding misclassified examples...")
    wrong_indices = np.where(y_pred != y_test)[0]
    print(
        f"  Total misclassified: {len(wrong_indices)} out of {len(y_test)} "
        f"({len(wrong_indices) / len(y_test) * 100:.1f}% error rate)"
    )

    # Limit to the first `num_examples` mistakes for display.
    display_indices = wrong_indices[:num_examples]

    # --- Plot the misclassified images in a grid ---
    cols = 5
    rows = (num_examples + cols - 1) // cols   # Round up to fit all images
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.5))
    fig.suptitle(f"Misclassified Examples (first {num_examples})", fontsize=14, y=1.02)

    for i, idx in enumerate(display_indices):
        ax = axes[i // cols][i % cols]
        # Display the original 28x28 image (not the flattened version)
        ax.imshow(x_test_raw[idx], cmap="gray")
        # Map integer labels to characters via LABEL_TO_CHAR
        true_char = LABEL_TO_CHAR[int(y_test[idx])]
        pred_char = LABEL_TO_CHAR[int(y_pred[idx])]
        confidence = y_proba[idx][y_pred[idx]] * 100   # Confidence % for the wrong prediction
        ax.set_title(
            f"True: {true_char}  |  Pred: {pred_char}\n({confidence:.0f}% confident)",
            fontsize=8,
            color="red",   # Red text to highlight that these are errors
        )
        ax.axis("off")

    # Hide any unused subplot panels (if num_examples isn't a multiple of cols)
    for i in range(len(display_indices), rows * cols):
        axes[i // cols][i % cols].axis("off")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    save_path = "results/misclassified_examples.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


# =============================================================================
# SECTION 6: MAIN — RUN THE FULL EVALUATION PIPELINE
# =============================================================================

if __name__ == "__main__":
    # This block runs only when you execute this file directly, e.g.:
    #   python src/evaluate.py
    print("=" * 60)
    print("  EMNIST CHARACTER CLASSIFIER — EVALUATION PIPELINE")
    print("=" * 60)

    # Ensure the results/ directory exists for saving plots
    os.makedirs("results", exist_ok=True)

    # --- Step 1: Load the raw EMNIST ByClass test data ---
    # We need both the raw (28x28) images for display AND the preprocessed
    # (flattened + normalised) versions for feeding into the model.
    x_train_raw, y_train, x_test_raw, y_test = load_data()

    # --- Step 2: Preprocess — normalise + flatten ---
    # Must match EXACTLY how data was prepared during training.
    _, x_test = preprocess_data(x_train_raw, x_test_raw)

    # --- Step 3: Load the trained model ---
    model = load_trained_model(filepath="results/model.h5")

    # --- Step 4: Overall accuracy + loss ---
    test_loss, test_accuracy = evaluate_model(model, x_test, y_test)

    # --- Step 5: Generate predictions for all test images ---
    y_pred, y_proba = get_predictions(model, x_test)

    # --- Step 6: Per-group accuracy & case-only confusion breakdown ---
    print_per_group_accuracy(y_test, y_pred)
    print_case_only_confusions(y_test, y_pred)

    # --- Step 7: Confusion matrix, top-15 confusions & per-class report ---
    plot_confusion_matrix(y_test, y_pred)

    # --- Step 8: Show some images the model got wrong ---
    show_misclassified(x_test_raw, y_test, y_pred, y_proba, num_examples=15)

    # --- Final summary ---
    misclassified_count = int((y_pred != y_test).sum())
    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Test samples     : {len(y_test)}")
    print(f"  Test Loss        : {test_loss:.4f}")
    print(f"  Test Accuracy    : {test_accuracy * 100:.2f}%")
    print(f"  Misclassified    : {misclassified_count} images")
    print("  Saved plots      : results/confusion_matrix.png, results/misclassified_examples.png")
    print("=" * 60)
    print("\nevaluate.py finished. Check the saved plots in results/ for a full picture of model performance!")