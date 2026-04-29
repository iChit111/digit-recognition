# =============================================================================
# demo.py
# Location: src/demo.py
#
# Purpose: A fun, interactive demo that lets you pick any image from the MNIST
# test set and watch the model predict what digit it is — in real time!
#
# This is the "show-off" script. Run it to impress your teammates, professor,
# or anyone curious about what Neural Ninjas built. :)
#
# How to run:
#   python src/demo.py                  → picks a RANDOM test image
#   python src/demo.py --index 42       → uses test image #42 (0–9999)
#   python src/demo.py --index 42 --count 5  → shows 5 predictions in a row
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
sys.path.append(os.path.dirname(__file__))   # So Python can find data_loader.py
from data_loader import load_data, preprocess_data


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
        filepath: Path to the .h5 model file (default: "model.h5").

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
    is a list of 10 probabilities — one for each digit (0 through 9).
    All 10 probabilities always add up to exactly 1.0 (100%).

    Example output:
        [0.00, 0.01, 0.02, 0.95, 0.01, 0.00, 0.00, 0.00, 0.01, 0.00]
         ^0    ^1    ^2    ^3    ^4    ^5    ^6    ^7    ^8    ^9

    Here the model is 95% confident the digit is a 3.

    Args:
        model:      The loaded Keras model.
        image_flat: A single preprocessed image → shape (784,)
                    (flattened and normalised by preprocess_data)

    Returns:
        predicted_digit: The digit the model is most confident about (0–9).
        confidence:      The model's confidence for that digit (0.0 – 1.0).
        all_probs:       Full probability array for all 10 digits → shape (10,)
    """

    # model.predict() expects a BATCH of images, not a single image.
    # So we use np.expand_dims to add a "batch dimension":
    #   image_flat shape: (784,)  →  after expand_dims: (1, 784)
    # This tells the model "I'm giving you a batch of 1 image."
    image_batch = np.expand_dims(image_flat, axis=0)   # shape: (1, 784)

    # Run the image through the model.
    # verbose=0 suppresses the progress bar for a single image.
    all_probs_batch = model.predict(image_batch, verbose=0)   # shape: (1, 10)

    # Remove the batch dimension so we get a simple array of 10 probabilities.
    all_probs = all_probs_batch[0]   # shape: (10,)

    # The predicted digit is the index with the highest probability.
    predicted_digit = int(np.argmax(all_probs))

    # The confidence is the probability at that index.
    confidence = float(all_probs[predicted_digit])

    return predicted_digit, confidence, all_probs


# =============================================================================
# SECTION 3: VISUALISE THE PREDICTION
# =============================================================================

def visualise_prediction(image_raw, true_label, predicted_digit, confidence, all_probs, index):
    """
    Creates a clear, two-panel visualisation:
      LEFT PANEL  → the actual 28×28 digit image
      RIGHT PANEL → a horizontal bar chart of confidence for all 10 digits

    The bar for the predicted digit is highlighted in green (correct) or red
    (incorrect) so you can immediately see whether the model was right.

    Args:
        image_raw:       The original 28×28 pixel image (for display).
        true_label:      The actual digit label (ground truth).
        predicted_digit: What the model guessed.
        confidence:      Model's confidence in its guess (0.0 – 1.0).
        all_probs:       Probability for every digit 0–9 → shape (10,)
        index:           Which test image this is (for the title).
    """

    # Determine if the prediction was correct.
    is_correct = (predicted_digit == true_label)

    # Choose colour scheme based on correctness.
    result_color = "#2ecc71" if is_correct else "#e74c3c"   # Green or red
    result_text  = "CORRECT ✓" if is_correct else "WRONG ✗"

    # --- Set up the figure with two panels side by side ---
    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")   # Dark navy background — looks sleek!

    # GridSpec lets us control the relative widths of the two panels.
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], figure=fig)

    # -------------------------
    # LEFT PANEL: the digit image
    # -------------------------
    ax_img = fig.add_subplot(gs[0])
    ax_img.set_facecolor("#16213e")

    # Display the 28×28 greyscale image.
    ax_img.imshow(image_raw, cmap="gray", interpolation="nearest")
    ax_img.axis("off")   # Hide axes — we don't need tick marks on a digit image

    ax_img.set_title(
        f"Test Image #{index}\nTrue Label: {true_label}",
        color="white", fontsize=13, pad=10
    )

    # -------------------------
    # RIGHT PANEL: confidence bar chart
    # -------------------------
    ax_bar = fig.add_subplot(gs[1])
    ax_bar.set_facecolor("#16213e")

    digits = list(range(10))   # [0, 1, 2, ..., 9]

    # Assign a colour to each bar:
    #   Predicted digit → green (correct) or red (wrong)
    #   All other bars  → steel blue
    bar_colors = []
    for d in digits:
        if d == predicted_digit:
            bar_colors.append(result_color)
        else:
            bar_colors.append("#4a90d9")   # Muted blue for non-predicted digits

    # Draw horizontal bars (barh = bar horizontal).
    bars = ax_bar.barh(digits, all_probs, color=bar_colors, edgecolor="none", height=0.6)

    # Add a percentage label at the end of each bar.
    for bar, prob in zip(bars, all_probs):
        label = f"{prob * 100:.1f}%"
        ax_bar.text(
            bar.get_width() + 0.01,   # Slightly to the right of the bar's end
            bar.get_y() + bar.get_height() / 2,   # Vertically centred on the bar
            label,
            va="center", ha="left",
            color="white", fontsize=9
        )

    # Style the axes for dark background.
    ax_bar.set_xlim(0, 1.15)          # A bit of room on the right for labels
    ax_bar.set_yticks(digits)
    ax_bar.set_yticklabels([f"Digit {d}" for d in digits], color="white", fontsize=10)
    ax_bar.tick_params(axis="x", colors="white")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color("#444466")
    ax_bar.spines["bottom"].set_color("#444466")
    ax_bar.set_xlabel("Model Confidence (probability)", color="white", fontsize=11)
    ax_bar.set_title(
        f"Prediction: {predicted_digit}   |   Confidence: {confidence * 100:.1f}%   |   {result_text}",
        color=result_color, fontsize=13, pad=12, fontweight="bold"
    )

    plt.suptitle(
        "Neural Ninjas — Handwritten Digit Recognition",
        color="white", fontsize=15, fontweight="bold", y=1.02
    )

    plt.tight_layout()

    # Save the figure to results/ folder.
    os.makedirs("results", exist_ok=True)
    save_path = f"results/demo_prediction_{index}.png"
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved visualisation → {save_path}")

    plt.show()


# =============================================================================
# SECTION 4: PRINT A SUMMARY TO THE TERMINAL
# =============================================================================

def print_prediction_summary(index, true_label, predicted_digit, confidence, all_probs):
    """
    Prints a clean, readable summary of the prediction to the terminal.

    This is useful even without looking at the plot — for quick checks or
    when running in a headless environment (e.g., a server with no display).

    Args:
        index:           Test image index.
        true_label:      The actual digit.
        predicted_digit: What the model guessed.
        confidence:      Model's confidence (0.0 – 1.0).
        all_probs:       Probability for all 10 digits.
    """

    is_correct = (predicted_digit == true_label)
    result_text = "CORRECT ✓" if is_correct else "WRONG ✗"

    print("\n" + "=" * 50)
    print(f"  TEST IMAGE #{index}")
    print("=" * 50)
    print(f"  True label      : {true_label}")
    print(f"  Predicted digit : {predicted_digit}")
    print(f"  Confidence      : {confidence * 100:.1f}%")
    print(f"  Result          : {result_text}")
    print("-" * 50)
    print("  All probabilities:")
    for digit, prob in enumerate(all_probs):
        bar = "█" * int(prob * 30)   # A mini text bar to show relative confidence
        marker = " ← predicted" if digit == predicted_digit else ""
        print(f"    Digit {digit}: {prob * 100:5.1f}%  {bar}{marker}")
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
        description="Neural Ninjas — MNIST Handwritten Digit Demo"
    )

    parser.add_argument(
        "--index", type=int, default=None,
        help="Index of the test image to predict (0–9999). Default: random."
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
        help="Path to the saved model file. Default: model.h5"
    )

    return parser.parse_args()


# =============================================================================
# SECTION 6: MAIN — RUN THE DEMO
# =============================================================================

if __name__ == "__main__":
    # This block runs only when you execute this file directly, e.g.:
    #   python src/demo.py

    print("=" * 60)
    print("  NEURAL NINJAS — HANDWRITTEN DIGIT RECOGNITION DEMO")
    print("=" * 60)

    # --- Parse command-line arguments ---
    args = parse_arguments()

    # --- Step 1: Load MNIST data ---
    print("\nLoading MNIST test data...")
    x_train_raw, y_train, x_test_raw, y_test = load_data()

    # We need both:
    #   x_test_raw → shape (10000, 28, 28) — the original images for display
    #   x_test     → shape (10000, 784)    — flattened + normalised for the model
    _, x_test = preprocess_data(x_train_raw, x_test_raw)
    print(f"  Test set loaded: {len(x_test)} images ready.")

    # --- Step 2: Load the trained model ---
    model = load_trained_model(filepath=args.model)

    # --- Step 3: Decide which image(s) to predict ---
    num_test_images = len(x_test)   # Usually 10,000 for MNIST

    # Build the list of indices to predict.
    if args.random or args.index is None:
        # Pick `count` random indices from 0 to 9999.
        indices = np.random.randint(0, num_test_images, size=args.count).tolist()
    else:
        # Start from --index and go up to --index + count.
        start = args.index
        indices = [
            (start + i) % num_test_images   # Wrap around if we go past 9999
            for i in range(args.count)
        ]

    # --- Step 4: Run predictions and show results ---
    print(f"\nRunning {len(indices)} prediction(s)...\n")

    correct_count = 0

    for idx in indices:
        # Get the raw image (for display) and the flattened image (for the model).
        image_raw  = x_test_raw[idx]   # shape: (28, 28)
        image_flat = x_test[idx]       # shape: (784,)
        true_label = int(y_test[idx])

        # Run the model.
        predicted_digit, confidence, all_probs = predict_single_image(model, image_flat)

        # Track accuracy across multiple predictions.
        if predicted_digit == true_label:
            correct_count += 1

        # Print terminal summary.
        print_prediction_summary(idx, true_label, predicted_digit, confidence, all_probs)

        # Show the visual plot.
        visualise_prediction(image_raw, true_label, predicted_digit, confidence, all_probs, idx)

    # --- Step 5: Final summary (useful when --count > 1) ---
    if len(indices) > 1:
        accuracy = correct_count / len(indices) * 100
        print(f"\nDemo session complete!")
        print(f"  Predictions : {len(indices)}")
        print(f"  Correct     : {correct_count}")
        print(f"  Accuracy    : {accuracy:.1f}%")

    print("\nThank you for trying our software! 🎉")