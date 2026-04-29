# =============================================================================
# evaluate.py
# Location: src/evaluate.py
#
# Purpose: Load the trained model and measure how well it performs on the
# MNIST test set — data it has NEVER seen during training.
#
# We use three tools to understand performance:
#   1. Overall accuracy & loss  → one number that summarises everything
#   2. Confusion matrix         → shows WHICH digits the model confuses
#   3. Misclassified examples   → actual images the model got wrong
# =============================================================================

# --- Import the libraries we need ---
import numpy as np                          # For numerical operations
import matplotlib.pyplot as plt             # For plotting charts and images
import tensorflow as tf                     # The main deep learning library
import keras               # High-level API for loading models
from sklearn.metrics import confusion_matrix, classification_report
# sklearn (scikit-learn) has ready-made tools for measuring classifier performance.
# confusion_matrix: shows a grid of predicted vs. actual labels.
# classification_report: prints precision, recall, and F1-score per digit.

import sys
import os
sys.path.append(os.path.dirname(__file__))  # So Python can find data_loader.py
from data_loader import load_data, preprocess_data


# =============================================================================
# SECTION 1: LOAD THE SAVED MODEL
# =============================================================================

def load_trained_model(filepath="results/model.h5"):
    """
    Loads a previously trained and saved Keras model from disk.

    We saved the model at the end of model.py using model.save("model.h5").
    keras.models.load_model() reads that file and reconstructs the exact same
    model — architecture, weights, and all — ready to make predictions.

    Args:
        filepath: Path to the saved .h5 model file (default: "model.h5").

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
    accuracy across all 10,000 test images.

    Args:
        model:  The loaded Keras model.
        x_test: Preprocessed test images → shape (10000, 784)
        y_test: True test labels (integers 0–9) → shape (10000,)

    Returns:
        test_loss:     Average loss on the test set (lower is better).
        test_accuracy: Fraction of correct predictions (higher is better).
    """

    print("\nEvaluating model on the test set...")
    print("  (This may take a few seconds — the model is checking 10,000 images)")

    # verbose=0 suppresses the progress bar so our output stays clean.
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    print("\n========== OVERALL PERFORMANCE ==========")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_accuracy * 100:.2f}%")
    print("  (Accuracy = correct predictions ÷ total predictions)")
    print("=========================================")

    return test_loss, test_accuracy


# =============================================================================
# SECTION 3: GENERATE PREDICTIONS
# =============================================================================

def get_predictions(model, x_test):
    """
    Runs the model on all test images and returns the predicted digit for each.

    model.predict() returns a probability array for each image, e.g.:
        [0.01, 0.02, 0.03, 0.91, 0.01, 0.00, 0.01, 0.00, 0.01, 0.00]
    This means the model is 91% confident the digit is a 3.

    np.argmax() picks the index (0–9) with the highest probability — that's
    the model's final predicted digit.

    Args:
        model:  The loaded Keras model.
        x_test: Preprocessed test images → shape (10000, 784)

    Returns:
        y_pred:  Array of predicted digits (integers 0–9) → shape (10000,)
        y_proba: Full probability arrays for all images → shape (10000, 10)
    """

    print("\nGenerating predictions for all 10,000 test images...")

    # model.predict() returns a 2D array: one row per image, 10 columns (one per digit).
    y_proba = model.predict(x_test, verbose=0)

    # np.argmax(axis=1) takes the column index with the highest value in each row.
    # axis=1 means "look across the 10 columns" (not down the rows).
    y_pred = np.argmax(y_proba, axis=1)

    print(f"  Predictions generated for {len(y_pred)} images.")
    return y_pred, y_proba


# =============================================================================
# SECTION 4: CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(y_test, y_pred):
    """
    Builds and visualises the confusion matrix.

    A confusion matrix is a 10×10 grid where:
      - Rows = the TRUE digit (what the image actually shows)
      - Columns = the PREDICTED digit (what the model guessed)
      - Diagonal cells (top-left to bottom-right) = correct predictions ✓
      - Off-diagonal cells = mistakes — e.g., the model predicted 7 but it was 1 ✗

    Reading the matrix:
      - A bright square on the diagonal = many correct predictions for that digit.
      - A bright square OFF the diagonal = a common mistake.
        e.g., row 4, col 9 being bright means "the model often thinks 4s are 9s."

    Args:
        y_test: True test labels → shape (10000,)
        y_pred: Predicted labels → shape (10000,)
    """

    print("\nGenerating confusion matrix...")

    # sklearn builds the raw counts matrix for us.
    # cm[i][j] = number of times a digit of class i was predicted as class j.
    cm = confusion_matrix(y_test, y_pred)

    # --- Plot the confusion matrix as a colour-coded heatmap ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # imshow displays the matrix as an image; "Blues" makes higher values darker blue.
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    # Add a colour bar on the right to show the scale.
    plt.colorbar(im, ax=ax)

    # Label the axes with the digit classes.
    tick_marks = np.arange(10)   # [0, 1, 2, ..., 9]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(tick_marks)
    ax.set_yticklabels(tick_marks)

    ax.set_title("Confusion Matrix — Predicted vs. True Digit", fontsize=14, pad=15)
    ax.set_ylabel("True Label (actual digit)", fontsize=12)
    ax.set_xlabel("Predicted Label (model's guess)", fontsize=12)

    # --- Write the count number inside each cell ---
    # We automatically choose white or black text so it's readable on any background.
    thresh = cm.max() / 2.0   # Cells darker than this get white text; lighter get black.
    for i in range(10):
        for j in range(10):
            ax.text(
                j, i,                        # Column (x), Row (y) position
                str(cm[i, j]),               # The count to display
                ha="center", va="center",    # Centre the text in the cell
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("  Saved: confusion_matrix.png")

    # --- Also print per-digit stats as a table ---
    print("\nPer-digit classification report:")
    print("  (precision = of all times the model predicted digit X, how often was it right?)")
    print("  (recall    = of all actual digit X images, how many did the model catch?)")
    print("  (f1-score  = harmonic mean of precision and recall — a balanced summary)\n")
    print(classification_report(y_test, y_pred, target_names=[str(d) for d in range(10)]))


# =============================================================================
# SECTION 5: SHOW MISCLASSIFIED EXAMPLES
# =============================================================================

def show_misclassified(x_test_raw, y_test, y_pred, y_proba, num_examples=15):
    """
    Displays images that the model got WRONG, along with what it predicted
    and how confident it was.

    Seeing real mistakes is one of the best ways to understand a model's
    weaknesses. For example, you might notice:
      - It often confuses 4 and 9 (they look similar)
      - It struggles with unusual handwriting styles
      - It's very confident even when wrong ("overconfident mistakes")

    Args:
        x_test_raw:   ORIGINAL (unflattened) test images → shape (10000, 28, 28)
                      We use these for display (the 28×28 pixel grid looks nicer).
        y_test:       True test labels → shape (10000,)
        y_pred:       Predicted labels → shape (10000,)
        y_proba:      Full probability arrays → shape (10000, 10)
        num_examples: How many misclassified images to show (default: 15).
    """

    print(f"\nFinding misclassified examples...")

    # np.where returns the indices where the condition is True.
    # Here: find every position where the prediction does NOT match the true label.
    wrong_indices = np.where(y_pred != y_test)[0]

    print(f"  Total misclassified: {len(wrong_indices)} out of {len(y_test)} "
          f"({len(wrong_indices)/len(y_test)*100:.1f}% error rate)")

    # Limit to the first `num_examples` mistakes for display.
    display_indices = wrong_indices[:num_examples]

    # --- Plot the misclassified images in a grid ---
    cols = 5
    rows = (num_examples + cols - 1) // cols   # Round up to fit all images

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.5))
    fig.suptitle(f"Misclassified Examples (first {num_examples})", fontsize=14, y=1.02)

    for i, idx in enumerate(display_indices):
        ax = axes[i // cols][i % cols]

        # Display the original 28×28 image (not the flattened version)
        ax.imshow(x_test_raw[idx], cmap="gray")

        # Show: true label, predicted label, and the model's confidence in its (wrong) guess
        confidence = y_proba[idx][y_pred[idx]] * 100   # Confidence % for the wrong prediction
        ax.set_title(
            f"True: {y_test[idx]}  →  Pred: {y_pred[idx]}\n({confidence:.0f}% confident)",
            fontsize=8,
            color="red"   # Red text to highlight that these are errors
        )
        ax.axis("off")

    # Hide any unused subplot panels (if num_examples isn't a multiple of cols)
    for i in range(len(display_indices), rows * cols):
        axes[i // cols][i % cols].axis("off")

    plt.tight_layout()
    plt.savefig("misclassified_examples.png")
    plt.show()
    print("  Saved: misclassified_examples.png")


# =============================================================================
# SECTION 6: MAIN — RUN THE FULL EVALUATION PIPELINE
# =============================================================================

if __name__ == "__main__":
    # This block runs only when you execute this file directly, e.g.:
    #   python src/evaluate.py

    print("=" * 60)
    print("  MNIST DIGIT CLASSIFIER — EVALUATION PIPELINE")
    print("=" * 60)

    # --- Step 1: Load the raw MNIST test data ---
    # We need both the raw (28×28) images for display AND the preprocessed
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

    # --- Step 6: Confusion matrix + per-digit report ---
    plot_confusion_matrix(y_test, y_pred)

    # --- Step 7: Show some images the model got wrong ---
    show_misclassified(x_test_raw, y_test, y_pred, y_proba, num_examples=15)

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Test samples     : {len(y_test)}")
    print(f"  Test Loss        : {test_loss:.4f}")
    print(f"  Test Accuracy    : {test_accuracy * 100:.2f}%")
    print(f"  Misclassified    : {int((1 - test_accuracy) * len(y_test))} images")
    print(f"  Saved plots      : confusion_matrix.png, misclassified_examples.png")
    print("=" * 60)
    print("\nevaluate.py finished. Check the saved plots for a full picture of model performance!")