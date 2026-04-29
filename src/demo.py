# =============================================================================
# demo.py
# Location: src/demo.py
#
# Purpose: A fun, interactive demo that lets you test the Neural Ninjas model
# in THREE different ways:
#
#   1. MNIST test set image  → pick any of the 10,000 test images by index
#   2. Custom image file     → supply your own PNG/JPG of a handwritten digit
#   3. Live drawing canvas   → draw a digit with your mouse and predict live!
#
# How to run:
#   python src/demo.py                        → random MNIST test image
#   python src/demo.py --index 42             → MNIST test image #42
#   python src/demo.py --random --count 5     → 5 random MNIST images
#   python src/demo.py --file my_digit.png    → your own image file
#   python src/demo.py --draw                 → open the drawing canvas
# =============================================================================

# --- Standard library imports ---
import argparse    # Reads command-line arguments like --index, --file, --draw
import sys
import os

# --- Third-party library imports ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import keras
from PIL import Image          # Pillow: for loading and resizing custom image files
                               # Install with: pip install pillow

# --- Our own project files ---
sys.path.append(os.path.dirname(__file__))
from data_loader import load_data, preprocess_data


# =============================================================================
# SECTION 1: LOAD THE TRAINED MODEL
# =============================================================================

def load_trained_model(filepath="results/model.h5"):
    """
    Loads the saved Keras model from disk.

    Args:
        filepath: Path to the .h5 model file.

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
    Feeds ONE preprocessed image through the model and returns the prediction.

    Args:
        model:      The loaded Keras model.
        image_flat: A single preprocessed image → shape (784,)

    Returns:
        predicted_digit: The digit the model is most confident about (0–9).
        confidence:      The model's confidence for that digit (0.0 – 1.0).
        all_probs:       Full probability array for all 10 digits → shape (10,)
    """

    # Add a batch dimension: (784,) → (1, 784)
    image_batch = np.expand_dims(image_flat, axis=0)

    # Run through the model — output is shape (1, 10)
    all_probs_batch = model.predict(image_batch, verbose=0)

    # Remove the batch dimension to get shape (10,)
    all_probs = all_probs_batch[0]

    predicted_digit = int(np.argmax(all_probs))
    confidence = float(all_probs[predicted_digit])

    return predicted_digit, confidence, all_probs


# =============================================================================
# SECTION 3: PREPROCESS A CUSTOM IMAGE FILE
# =============================================================================

def load_and_preprocess_custom_image(filepath):
    """
    Loads a custom image (PNG, JPG, etc.) from disk and prepares it so the
    model can make a prediction on it.

    The MNIST model was trained on:
      - Greyscale images (no colour)
      - 28×28 pixels
      - White digit on a BLACK background
      - Pixel values normalised to 0.0–1.0
      - Flattened to a 784-element vector

    So we must apply ALL of those same steps to any custom image.

    Important tip for best results:
      - Draw your digit in BLACK ink on a WHITE background (like Paint)
      - Make the digit large and centred in the image
      - Save as PNG before passing to --file

    Args:
        filepath: Path to the image file (PNG, JPG, etc.)

    Returns:
        image_28x28: The resized greyscale image → shape (28, 28), values 0.0–1.0
                     (used for display)
        image_flat:  The preprocessed, flattened image → shape (784,)
                     (used as model input)
    """

    print(f"\nLoading custom image from '{filepath}'...")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"  ERROR: Image file '{filepath}' not found.")

    # --- Step 1: Open the image and convert to greyscale ---
    # "L" mode = 8-bit greyscale. This drops any colour or transparency channels.
    img = Image.open(filepath).convert("L")

    # --- Step 2: Resize to 28×28 pixels ---
    # LANCZOS is a high-quality filter — it slightly blurs the edges, which
    # actually helps because MNIST digits have soft, blurry edges too.
    img_resized = img.resize((28, 28), Image.LANCZOS)

    # --- Step 3: Convert to numpy array ---
    image_array = np.array(img_resized)   # shape: (28, 28), values 0–255

    # --- Step 4: Invert if necessary ---
    # MNIST has WHITE digits on a BLACK background.
    # Most people draw BLACK digits on a WHITE background (e.g., in Paint).
    # If the image is mostly white (mean pixel value > 127), we invert it
    # so the digit becomes white and the background becomes black.
    if image_array.mean() > 127:
        print("  Detected light background — inverting colours to match MNIST format.")
        image_array = 255 - image_array

    # --- Step 5: Normalise pixel values to 0.0–1.0 ---
    image_28x28 = image_array / 255.0

    # --- Step 6: Flatten from (28, 28) to (784,) ---
    image_flat = image_28x28.reshape(784)

    print(f"  Image loaded and preprocessed successfully.")
    print(f"  Shape after preprocessing: {image_flat.shape}")

    return image_28x28, image_flat


# =============================================================================
# SECTION 4: VISUALISE THE PREDICTION
# =============================================================================

def visualise_prediction(image_raw, true_label, predicted_digit, confidence, all_probs, title="Test Image"):
    """
    Creates a two-panel visualisation:
      LEFT  → the digit image
      RIGHT → horizontal confidence bar chart for all 10 digits

    Args:
        image_raw:       The 28×28 pixel image (for display).
        true_label:      The actual digit (or None if unknown, e.g. custom image).
        predicted_digit: What the model guessed.
        confidence:      Model's confidence (0.0 – 1.0).
        all_probs:       Probability for every digit 0–9 → shape (10,)
        title:           Title for the left panel.
    """

    # For custom images we don't know the true label, so can't say correct/wrong.
    if true_label is not None:
        is_correct = (predicted_digit == true_label)
        result_color = "#2ecc71" if is_correct else "#e74c3c"
        result_text  = "CORRECT ✓" if is_correct else "WRONG ✗"
    else:
        result_color = "#f39c12"   # Orange = neutral (custom/unknown input)
        result_text  = "CUSTOM IMAGE"

    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], figure=fig)

    # --- LEFT: digit image ---
    ax_img = fig.add_subplot(gs[0])
    ax_img.set_facecolor("#16213e")
    ax_img.imshow(image_raw, cmap="gray", interpolation="nearest")
    ax_img.axis("off")
    label_line = f"\nTrue Label: {true_label}" if true_label is not None else "\n(custom input)"
    ax_img.set_title(f"{title}{label_line}", color="white", fontsize=13, pad=10)

    # --- RIGHT: confidence bar chart ---
    ax_bar = fig.add_subplot(gs[1])
    ax_bar.set_facecolor("#16213e")

    digits = list(range(10))
    bar_colors = [result_color if d == predicted_digit else "#4a90d9" for d in digits]
    bars = ax_bar.barh(digits, all_probs, color=bar_colors, edgecolor="none", height=0.6)

    for bar, prob in zip(bars, all_probs):
        ax_bar.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%",
            va="center", ha="left", color="white", fontsize=9
        )

    ax_bar.set_xlim(0, 1.15)
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

    os.makedirs("results", exist_ok=True)
    safe_title = title.replace(" ", "_").replace("#", "").lower()
    save_path = f"results/demo_prediction_{safe_title}.png"
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved visualisation → {save_path}")

    plt.show()


# =============================================================================
# SECTION 5: PRINT TERMINAL SUMMARY
# =============================================================================

def print_prediction_summary(title, true_label, predicted_digit, confidence, all_probs):
    """
    Prints a clean text summary of the prediction to the terminal.

    Args:
        title:           Label for this prediction (e.g. "Test Image #42").
        true_label:      The actual digit, or None for custom images.
        predicted_digit: What the model guessed.
        confidence:      Model's confidence (0.0 – 1.0).
        all_probs:       Probability for all 10 digits.
    """

    if true_label is not None:
        result_text = "CORRECT ✓" if predicted_digit == true_label else "WRONG ✗"
    else:
        result_text = "CUSTOM IMAGE (no ground truth)"

    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)
    if true_label is not None:
        print(f"  True label      : {true_label}")
    print(f"  Predicted digit : {predicted_digit}")
    print(f"  Confidence      : {confidence * 100:.1f}%")
    print(f"  Result          : {result_text}")
    print("-" * 50)
    print("  All probabilities:")
    for digit, prob in enumerate(all_probs):
        bar = "█" * int(prob * 30)
        marker = " ← predicted" if digit == predicted_digit else ""
        print(f"    Digit {digit}: {prob * 100:5.1f}%  {bar}{marker}")
    print("=" * 50)


# =============================================================================
# SECTION 6: DRAWING CANVAS (tkinter)
# =============================================================================

def run_drawing_canvas(model):
    """
    Opens an interactive drawing window where you can draw a digit with your
    mouse and the model will predict it when you click 'Predict'.

    How it works:
      - A black canvas (280×280 pixels) opens in a new window.
      - You draw by clicking and dragging with the left mouse button.
      - Click 'Predict' → the canvas is captured, resized to 28×28,
        preprocessed, and fed to the model. The prediction appears in the
        window AND as a full confidence chart.
      - Click 'Clear' → wipe the canvas and start over.
      - Close the window to exit.

    Why 280×280?
      The model expects 28×28 input, but 280×280 is 10× bigger — much easier
      to draw on! We shrink it down to 28×28 just before prediction.

    Args:
        model: The loaded Keras model.
    """

    try:
        import tkinter as tk
        from PIL import ImageDraw
    except ImportError:
        print("\n  ERROR: tkinter or Pillow is not available.")
        print("  Install Pillow with: pip install pillow")
        print("  tkinter usually comes with Python — reinstall Python if it's missing.")
        return

    print("\nOpening drawing canvas...")
    print("  → Draw a digit with your mouse (left click + drag)")
    print("  → Click 'Predict' to see the model's guess")
    print("  → Click 'Clear' to start over\n")

    # --- Canvas settings ---
    CANVAS_SIZE  = 280    # Display size (10× the MNIST 28×28 for easy drawing)
    BRUSH_RADIUS = 10     # Brush thickness in pixels
    MNIST_SIZE   = 28     # What the model actually expects

    # --- We maintain two parallel drawing surfaces ---
    # 1. tk_canvas  → the widget the user sees (tkinter, can't export easily)
    # 2. pil_image  → a Pillow image we draw on simultaneously (exportable to numpy)
    pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)  # Black bg
    pil_draw  = ImageDraw.Draw(pil_image)

    last_x, last_y = None, None   # Track previous mouse position for smooth lines

    # --- Build the window ---
    root = tk.Tk()
    root.title("Neural Ninjas — Draw a Digit")
    root.resizable(False, False)
    root.configure(bg="#1a1a2e")

    tk.Label(
        root, text="✏️  Draw a digit (0–9) below",
        font=("Helvetica", 14, "bold"), fg="white", bg="#1a1a2e"
    ).pack(pady=(15, 5))

    tk_canvas = tk.Canvas(
        root, width=CANVAS_SIZE, height=CANVAS_SIZE,
        bg="black", cursor="crosshair",
        highlightthickness=2, highlightbackground="#4a90d9"
    )
    tk_canvas.pack(padx=20, pady=10)

    result_var = tk.StringVar(value="Draw a digit, then click Predict!")
    result_label = tk.Label(
        root, textvariable=result_var,
        font=("Helvetica", 13), fg="#f39c12", bg="#1a1a2e"
    )
    result_label.pack(pady=5)

    # --- Mouse event handlers ---

    def on_mouse_press(event):
        nonlocal last_x, last_y
        last_x, last_y = event.x, event.y

    def on_mouse_drag(event):
        nonlocal last_x, last_y
        x, y = event.x, event.y

        if last_x is not None:
            # Draw on tkinter canvas (what user sees)
            tk_canvas.create_line(
                last_x, last_y, x, y,
                fill="white", width=BRUSH_RADIUS * 2,
                capstyle=tk.ROUND, smooth=True
            )
            # Draw the same stroke on the Pillow image (what we process)
            pil_draw.ellipse(
                [x - BRUSH_RADIUS, y - BRUSH_RADIUS,
                 x + BRUSH_RADIUS, y + BRUSH_RADIUS],
                fill=255
            )
        last_x, last_y = x, y

    def on_mouse_release(event):
        nonlocal last_x, last_y
        last_x, last_y = None, None

    tk_canvas.bind("<ButtonPress-1>",   on_mouse_press)
    tk_canvas.bind("<B1-Motion>",       on_mouse_drag)
    tk_canvas.bind("<ButtonRelease-1>", on_mouse_release)

    # --- Predict button ---

    def on_predict():
        # Resize 280×280 → 28×28
        img_small  = pil_image.resize((MNIST_SIZE, MNIST_SIZE), Image.LANCZOS)
        img_array  = np.array(img_small) / 255.0   # Normalise to 0.0–1.0
        img_flat   = img_array.reshape(784)         # Flatten

        predicted_digit, confidence, all_probs = predict_single_image(model, img_flat)

        # Update result label inside the window
        result_var.set(
            f"Prediction: {predicted_digit}   |   Confidence: {confidence * 100:.1f}%"
        )
        result_label.config(fg="#2ecc71")

        # Print terminal summary and show full chart
        print_prediction_summary("Drawing Canvas", None, predicted_digit, confidence, all_probs)
        visualise_prediction(
            image_raw=img_array,
            true_label=None,
            predicted_digit=predicted_digit,
            confidence=confidence,
            all_probs=all_probs,
            title="Drawing Canvas"
        )

    def on_clear():
        nonlocal pil_image, pil_draw
        tk_canvas.delete("all")
        pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        pil_draw  = ImageDraw.Draw(pil_image)
        result_var.set("Canvas cleared — draw again!")
        result_label.config(fg="#f39c12")

    # --- Buttons ---
    button_frame = tk.Frame(root, bg="#1a1a2e")
    button_frame.pack(pady=(5, 20))

    tk.Button(
        button_frame, text="🔮  Predict", command=on_predict,
        font=("Helvetica", 12, "bold"), fg="white", bg="#2ecc71",
        activebackground="#27ae60", activeforeground="white",
        padx=20, pady=8, bd=0, cursor="hand2"
    ).pack(side=tk.LEFT, padx=10)

    tk.Button(
        button_frame, text="🗑️  Clear", command=on_clear,
        font=("Helvetica", 12, "bold"), fg="white", bg="#e74c3c",
        activebackground="#c0392b", activeforeground="white",
        padx=20, pady=8, bd=0, cursor="hand2"
    ).pack(side=tk.LEFT, padx=10)

    root.mainloop()


# =============================================================================
# SECTION 7: COMMAND-LINE ARGUMENT PARSER
# =============================================================================

def parse_arguments():
    """
    Reads command-line arguments so the user can choose which demo mode to run.

    Usage examples:
        python src/demo.py                        # Random MNIST test image
        python src/demo.py --index 42             # MNIST test image #42
        python src/demo.py --random --count 5     # 5 random MNIST images
        python src/demo.py --file my_digit.png    # Your own image file
        python src/demo.py --draw                 # Open the drawing canvas
    """

    parser = argparse.ArgumentParser(
        description="Neural Ninjas — MNIST Handwritten Digit Demo"
    )
    parser.add_argument(
        "--index", type=int, default=None,
        help="Index of the MNIST test image to predict (0–9999). Default: random."
    )
    parser.add_argument(
        "--count", type=int, default=1,
        help="Number of MNIST predictions to run. Default: 1."
    )
    parser.add_argument(
        "--random", action="store_true",
        help="Use random MNIST test images."
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a custom image file to predict. e.g. --file my_digit.png"
    )
    parser.add_argument(
        "--draw", action="store_true",
        help="Open the interactive drawing canvas."
    )
    parser.add_argument(
        "--model", type=str, default="results/model.h5",
        help="Path to the saved model file. Default: results/model.h5"
    )
    return parser.parse_args()


# =============================================================================
# SECTION 8: MAIN — RUN THE DEMO
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  NEURAL NINJAS — HANDWRITTEN DIGIT RECOGNITION DEMO")
    print("=" * 60)

    args = parse_arguments()

    # Load the model first — needed for all three modes
    model = load_trained_model(filepath=args.model)

    # =========================================================================
    # MODE A: Drawing canvas  (--draw)
    # =========================================================================
    if args.draw:
        print("\nMode: Drawing Canvas")
        run_drawing_canvas(model)

    # =========================================================================
    # MODE B: Custom image file  (--file path/to/image.png)
    # =========================================================================
    elif args.file:
        print(f"\nMode: Custom Image File → '{args.file}'")
        image_28x28, image_flat = load_and_preprocess_custom_image(args.file)
        predicted_digit, confidence, all_probs = predict_single_image(model, image_flat)
        print_prediction_summary("Custom Image", None, predicted_digit, confidence, all_probs)
        visualise_prediction(
            image_raw=image_28x28,
            true_label=None,
            predicted_digit=predicted_digit,
            confidence=confidence,
            all_probs=all_probs,
            title="Custom Image"
        )

    # =========================================================================
    # MODE C: MNIST test set  (default)
    # =========================================================================
    else:
        print("\nMode: MNIST Test Set")
        print("\nLoading MNIST test data...")
        x_train_raw, y_train, x_test_raw, y_test = load_data()
        _, x_test = preprocess_data(x_train_raw, x_test_raw)
        print(f"  Test set loaded: {len(x_test)} images ready.")

        num_test_images = len(x_test)

        if args.random or args.index is None:
            indices = np.random.randint(0, num_test_images, size=args.count).tolist()
        else:
            start = args.index
            indices = [(start + i) % num_test_images for i in range(args.count)]

        print(f"\nRunning {len(indices)} prediction(s)...\n")

        correct_count = 0

        for idx in indices:
            image_raw  = x_test_raw[idx]
            image_flat = x_test[idx]
            true_label = int(y_test[idx])

            predicted_digit, confidence, all_probs = predict_single_image(model, image_flat)

            if predicted_digit == true_label:
                correct_count += 1

            print_prediction_summary(f"Test Image #{idx}", true_label, predicted_digit, confidence, all_probs)
            visualise_prediction(image_raw, true_label, predicted_digit, confidence, all_probs, title=f"Test Image #{idx}")

        if len(indices) > 1:
            accuracy = correct_count / len(indices) * 100
            print(f"\nDemo session complete!")
            print(f"  Predictions : {len(indices)}")
            print(f"  Correct     : {correct_count}")
            print(f"  Accuracy    : {accuracy:.1f}%")

    print("\nThank you for trying Neural Ninjas! 🎉")