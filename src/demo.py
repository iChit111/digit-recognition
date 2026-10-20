# =============================================================================
# demo.py
# Location: src/demo.py
# Purpose: A fun, interactive demo that lets you pick any image from the EMNIST
# ByClass test set and watch the model predict what character it is — in real time!
# This is the "show-off" script. Run it to impress your teammates, professor,
# or anyone curious about what Neural Ninjas built. :)
# How to run:
#   python src/demo.py                          random EMNIST test image
#   python src/demo.py --index 42               EMNIST test image #42
#   python src/demo.py --random --count 5       5 random EMNIST images
# =============================================================================

# --- Standard library imports ---
import argparse    # Lets us read command-line arguments like --index and --count
import sys
import os

# --- Third-party library imports ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec   # For creating custom subplot layouts
import tensorflow as tf
import keras

# --- Our own project files ---
sys.path.append(os.path.dirname(__file__))   # So Python can find data_loader.py and label_map.py
from data_loader import load_data, preprocess_data
from label_map import LABEL_TO_CHAR


# =============================================================================
# SECTION 1: LOAD THE TRAINED MODEL
# =============================================================================

def load_trained_model(filepath="results/model.h5"):
    """
    Loads the saved Keras model from disk.
    This is the same model that was trained in model.py and evaluated in
    evaluate.py. All three scripts share the same model file so there is
    one single source of truth for the weights.

    Args:
        filepath: Path to the .h5 model file (default: "results/model.h5").
    Returns:
        model: The loaded Keras model, ready for predictions.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n  ERROR: Model file '{filepath}' not found.\n"
            f"  Please run 'python src/model.py' first to train and save the model."
        )
    print(f"Loading model from '{filepath}'... ", end="")
    model = keras.models.load_model(filepath)
    print("done!")
    return model


# =============================================================================
# SECTION 2: PREDICT A SINGLE IMAGE
# =============================================================================

def predict_single_image(model, image_flat):
    """
    Feeds ONE image through the model and returns the prediction.
    The model's final layer uses softmax activation, which means its output
    is a list of 62 probabilities — one for each character class (0-9, A-Z, a-z).
    All 62 probabilities always add up to exactly 1.0 (100%).

    Args:
        model:      The loaded Keras model.
        image_flat: A single preprocessed image — shape (784,)
                    (flattened and normalised by preprocess_data)
    Returns:
        predicted_label: The class index the model is most confident about (0-61).
        confidence:      The model's confidence for that class (0.0 - 1.0).
        all_probs:       Full probability array for all 62 classes — shape (62,)
    """
    # model.predict() expects a BATCH of images, not a single image.
    # So we use np.expand_dims to add a "batch dimension":
    #   image_flat shape: (784,)  ->  after expand_dims: (1, 784)
    image_batch = np.expand_dims(image_flat, axis=0)   # shape: (1, 784)

    # Run the image through the model.
    # verbose=0 suppresses the progress bar for a single image.
    all_probs_batch = model.predict(image_batch, verbose=0)   # shape: (1, 62)

    # Remove the batch dimension so we get a simple array of 62 probabilities.
    all_probs = all_probs_batch[0]   # shape: (62,)

    # The predicted class is the index with the highest probability.
    predicted_label = int(np.argmax(all_probs))

    # The confidence is the probability at that index.
    confidence = float(all_probs[predicted_label])

    return predicted_label, confidence, all_probs


# =============================================================================
# SECTION 3: VISUALISE THE PREDICTION
# =============================================================================

def visualise_prediction(image_raw, true_label, predicted_label, confidence, all_probs, index):
    """
    Creates a clear, two-panel visualisation:
      LEFT PANEL  — the actual 28x28 character image
      RIGHT PANEL — a horizontal bar chart of the TOP 5 predicted characters

    The bar for the top predicted character is highlighted in green (correct)
    or red (incorrect) so you can immediately see whether the model was right.

    Args:
        image_raw:       The original 28x28 pixel image (for display).
        true_label:      The actual integer label (0-61).
        predicted_label: What class index the model guessed (0-61).
        confidence:      Model's confidence in its guess (0.0 - 1.0).
        all_probs:       Probability for every class 0-61 — shape (62,)
        index:           Which test image this is (for the title).
    """
    is_correct   = (predicted_label == true_label)
    result_color = "#2ecc71" if is_correct else "#e74c3c"   # Green or red
    result_text  = "CORRECT" if is_correct else "WRONG"

    true_char = LABEL_TO_CHAR[int(true_label)]
    pred_char = LABEL_TO_CHAR[int(predicted_label)]

    # --- Set up the figure with two panels side by side ---
    fig = plt.figure(figsize=(12, 5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Neural Ninjas — Handwritten Character Recognition")
    fig.patch.set_facecolor("#1a1a2e")   # Dark navy background
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], figure=fig)

    # -------------------------
    # LEFT PANEL: the character image
    # -------------------------
    ax_img = fig.add_subplot(gs[0])
    ax_img.set_facecolor("#16213e")
    ax_img.imshow(image_raw, cmap="gray", interpolation="nearest")
    ax_img.axis("off")
    ax_img.set_title(
        f"Test Image #{index}\nTrue Label: '{true_char}'",
        color="white", fontsize=13, pad=10
    )

    # -------------------------
    # RIGHT PANEL: Top 5 confidence bar chart
    # -------------------------
    ax_bar = fig.add_subplot(gs[1])
    ax_bar.set_facecolor("#16213e")

    # Pick the top 5 predictions sorted highest -> lowest probability
    top5_indices = np.argsort(all_probs)[::-1][:5]
    top5_probs   = [float(all_probs[i]) for i in top5_indices]
    top5_labels  = [f"'{LABEL_TO_CHAR[int(i)]}'" for i in top5_indices]
    y_positions  = list(range(len(top5_indices)))

    bar_colors = [
        result_color if i == predicted_label else "#4a90d9"
        for i in top5_indices
    ]
    bars = ax_bar.barh(
        y_positions, top5_probs, color=bar_colors, edgecolor="none", height=0.6
    )

    # Add percentage labels at the end of each bar
    for bar, prob in zip(bars, top5_probs):
        ax_bar.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%",
            va="center", ha="left", color="white", fontsize=9
        )

    ax_bar.set_xlim(0, 1.15)
    ax_bar.set_yticks(y_positions)
    ax_bar.set_yticklabels(top5_labels, color="white", fontsize=10)
    ax_bar.invert_yaxis()   # Highest probability bar at the top
    ax_bar.tick_params(axis="x", colors="white")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color("#444466")
    ax_bar.spines["bottom"].set_color("#444466")
    ax_bar.set_xlabel("Model Confidence (probability) — Top 5", color="white", fontsize=11)
    ax_bar.set_title(
        f"Prediction: '{pred_char}'   |   Confidence: {confidence * 100:.1f}%   |   {result_text}",
        color=result_color, fontsize=13, pad=12, fontweight="bold"
    )

    plt.suptitle(
        "Neural Ninjas — Handwritten Character Recognition",
        color="white", fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    save_path = f"results/demo_prediction_{index}.png"
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved visualisation -> {save_path}")
    plt.show()


# =============================================================================
# SECTION 4: PRINT A SUMMARY TO THE TERMINAL
# =============================================================================

def print_prediction_summary(index, true_label, predicted_label, confidence, all_probs):
    """
    Prints a clean, readable summary of the prediction (and top 5 probabilities)
    to the terminal.

    Args:
        index:           Test image index.
        true_label:      The actual class label (0-61).
        predicted_label: What class label the model guessed (0-61).
        confidence:      Model's confidence (0.0 - 1.0).
        all_probs:       Probability for all 62 classes.
    """
    is_correct  = (predicted_label == true_label)
    result_text = "CORRECT" if is_correct else "WRONG"
    true_char   = LABEL_TO_CHAR[int(true_label)]
    pred_char   = LABEL_TO_CHAR[int(predicted_label)]

    top5_indices = np.argsort(all_probs)[::-1][:5]

    print("\n" + "=" * 50)
    print(f"  TEST IMAGE #{index}")
    print("=" * 50)
    print(f"  True label          : '{true_char}' (class {true_label})")
    print(f"  Predicted character : '{pred_char}' (class {predicted_label})")
    print(f"  Confidence          : {confidence * 100:.1f}%")
    print(f"  Result              : {result_text}")
    print("-" * 50)
    print("  Top 5 probabilities:")
    for cls_idx in top5_indices:
        prob   = float(all_probs[cls_idx])
        char   = LABEL_TO_CHAR[int(cls_idx)]
        bar    = "█" * int(prob * 30)
        marker = "  <-- predicted" if cls_idx == predicted_label else ""
        print(f"    '{char}' (class {int(cls_idx):2d}): {prob * 100:5.1f}%  {bar}{marker}")
    print("=" * 50)


# =============================================================================
# SECTION 5: COMMAND-LINE ARGUMENT PARSER
# =============================================================================

def parse_arguments():
    """
    Reads command-line arguments so the user can customise the demo run.
    Usage examples:
        python src/demo.py                      # Random image
        python src/demo.py --index 99           # Image #99
        python src/demo.py --index 99 --count 5 # 5 images starting from #99
        python src/demo.py --random --count 10  # 10 random images

    Returns:
        args: A namespace object with attributes: index, count, random, model
    """
    parser = argparse.ArgumentParser(
        description="Neural Ninjas — EMNIST Handwritten Character Recognition Demo"
    )
    parser.add_argument(
        "--index", type=int, default=None,
        help="Index of the test image to predict (0-116322). Default: random."
    )
    parser.add_argument(
        "--count", type=int, default=1,
        help="Number of predictions to run in sequence. Default: 1."
    )
    parser.add_argument(
        "--random", action="store_true",
        help="Use random images (ignores --index). Default: False."
    )
    parser.add_argument(
        "--model", type=str, default="results/model.h5",
        help="Path to the saved model file. Default: results/model.h5"
    )
    return parser.parse_args()


# =============================================================================
# SECTION 6: MAIN — RUN THE DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  NEURAL NINJAS — HANDWRITTEN CHARACTER RECOGNITION DEMO")
    print("=" * 60)
    args = parse_arguments()

    # --- Step 1: Load EMNIST test data ---
    print("\nLoading EMNIST ByClass test data...")
    x_train_raw, y_train, x_test_raw, y_test = load_data()
    # We need both:
    #   x_test_raw — shape (116323, 28, 28) — original images for display
    #   x_test     — shape (116323, 784)    — flattened + normalised for the model
    _, x_test = preprocess_data(x_train_raw, x_test_raw)
    print(f"  Test set loaded: {len(x_test)} images ready.")

    # --- Step 2: Validate --index if provided ---
    num_test_images = len(x_test)
    if not args.random and args.index is not None:
        if args.index < 0 or args.index >= num_test_images:
            raise ValueError(
                f"--index must be between 0 and {num_test_images - 1} (got {args.index})."
            )

    # --- Step 3: Load the trained model ---
    model = load_trained_model(filepath=args.model)

    # --- Step 4: Decide which images to predict ---
    if args.random or args.index is None:
        indices = np.random.randint(0, num_test_images, size=args.count).tolist()
    else:
        start   = args.index
        indices = [(start + i) % num_test_images for i in range(args.count)]

    # --- Step 5: Run predictions ---
    print(f"\nRunning {len(indices)} prediction(s)...\n")
    correct_count = 0
    for idx in indices:
        image_raw  = x_test_raw[idx]       # shape: (28, 28) — for display
        image_flat = x_test[idx]           # shape: (784,)   — for the model
        true_label = int(y_test[idx])

        predicted_label, confidence, all_probs = predict_single_image(model, image_flat)
        if predicted_label == true_label:
            correct_count += 1

        print_prediction_summary(idx, true_label, predicted_label, confidence, all_probs)
        visualise_prediction(image_raw, true_label, predicted_label, confidence, all_probs, idx)

    # --- Step 6: Session summary (only shown when --count > 1) ---
    if len(indices) > 1:
        accuracy = correct_count / len(indices) * 100
        print("\nDemo session complete!")
        print(f"  Predictions : {len(indices)}")
        print(f"  Correct     : {correct_count}")
        print(f"  Accuracy    : {accuracy:.1f}%")

    print("\nThank you for trying Neural Ninjas!")