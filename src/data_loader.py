# =============================================================================
# data_loader.py
# Location: src/data_loader.py
#
# Purpose: Load, explore, visualize, and preprocess the MNIST dataset.
# The MNIST dataset contains 70,000 grayscale images of handwritten digits
# (0–9), each image being 28x28 pixels.
# =============================================================================

# --- Step 1: Import the libraries we need ---
import numpy as np                          # For working with numbers and arrays
import matplotlib.pyplot as plt             # For drawing charts and showing images
import tensorflow as tf                     # The main deep learning library
from tensorflow.keras.datasets import mnist # The MNIST dataset, built into Keras


# =============================================================================
# SECTION 1: LOAD THE DATA
# =============================================================================

def load_data():
    """
    Loads the MNIST dataset and splits it into training and test sets.

    Returns:
        x_train: Training images  → shape: (60000, 28, 28)
        y_train: Training labels  → shape: (60000,)
        x_test:  Test images      → shape: (10000, 28, 28)
        y_test:  Test labels      → shape: (10000,)
    """

    print("Loading MNIST dataset...")

    # Keras downloads the dataset automatically the first time you run this.
    # It splits the 70,000 images into:
    #   - 60,000 images for training (teaching the model)
    #   - 10,000 images for testing (checking how well the model learned)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(f"  Training images: {x_train.shape}  → {x_train.shape[0]} images, each {x_train.shape[1]}x{x_train.shape[2]} pixels")
    print(f"  Training labels: {y_train.shape}")
    print(f"  Test images:     {x_test.shape}")
    print(f"  Test labels:     {y_test.shape}")
    print(f"  Pixel value range (before preprocessing): {x_train.min()} – {x_train.max()}")

    return x_train, y_train, x_test, y_test


# =============================================================================
# SECTION 2: EXPLORE AND VISUALIZE THE DATA
# =============================================================================

def explore_data(x_train, y_train):
    """
    Visualizes the MNIST data in two ways:
      1. A grid of 25 sample images from the training set.
      2. A bar chart showing how many images exist for each digit (0–9).

    Args:
        x_train: Training images (numpy array)
        y_train: Training labels (numpy array)
    """

    # --- 2a. Show a 5x5 grid of sample images ---
    print("\nVisualizing 25 sample images from the training set...")

    fig, axes = plt.subplots(5, 5, figsize=(8, 8))   # Create a 5x5 grid of subplots
    fig.suptitle("Sample MNIST Images", fontsize=16)  # Add a title to the whole figure

    for i, ax in enumerate(axes.flat):  # Loop through each subplot (25 total)
        ax.imshow(x_train[i], cmap="gray")  # Show the image in grayscale
        ax.set_title(f"Label: {y_train[i]}")  # Show the correct digit above each image
        ax.axis("off")                        # Hide the axis lines for a cleaner look

    plt.tight_layout()       # Automatically adjust spacing between subplots
    plt.savefig("sample_images.png")  # Save the figure as a PNG file
    plt.show()               # Display the figure on screen
    print("  Saved: sample_images.png")

    # --- 2b. Show a bar chart of label distribution ---
    print("\nVisualizing label distribution...")

    # np.bincount counts how many times each digit (0–9) appears in y_train
    label_counts = np.bincount(y_train)   # e.g., [5923, 6742, ...] for digits 0–9
    digits = np.arange(10)                # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    plt.figure(figsize=(8, 5))
    plt.bar(digits, label_counts, color="steelblue", edgecolor="black")
    plt.title("Label Distribution in Training Set")
    plt.xlabel("Digit (0–9)")
    plt.ylabel("Number of Images")
    plt.xticks(digits)  # Make sure all digit labels 0–9 appear on the x-axis

    # Add the exact count on top of each bar for easy reading
    for digit, count in zip(digits, label_counts):
        plt.text(digit, count + 50, str(count), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("label_distribution.png")  # Save the chart as a PNG file
    plt.show()
    print("  Saved: label_distribution.png")


# =============================================================================
# SECTION 3: PREPROCESS THE DATA
# =============================================================================

def preprocess_data(x_train, x_test):
    """
    Prepares the raw images so a neural network can use them.

    Two steps:
      1. NORMALIZE: Scale pixel values from the range [0, 255] to [0.0, 1.0].
                    Neural networks learn faster and more reliably with small numbers.

      2. FLATTEN:   Reshape each 28x28 image into a single row of 784 numbers.
                    A basic neural network expects a 1D input, not a 2D grid.
                    (28 × 28 = 784)

    Args:
        x_train: Raw training images  → shape: (60000, 28, 28)
        x_test:  Raw test images      → shape: (10000, 28, 28)

    Returns:
        x_train_processed: shape (60000, 784), values between 0.0 and 1.0
        x_test_processed:  shape (10000, 784),  values between 0.0 and 1.0
    """

    print("\nPreprocessing data...")

    # --- Step 3a: Normalize pixel values ---
    # Each pixel is an integer from 0 (black) to 255 (white).
    # Dividing by 255.0 converts all values to floats between 0.0 and 1.0.
    # The ".0" makes sure Python does decimal division, not integer division.
    x_train_normalized = x_train / 255.0
    x_test_normalized  = x_test  / 255.0

    print(f"  Pixel value range (after normalization): "
          f"{x_train_normalized.min():.1f} – {x_train_normalized.max():.1f}")

    # --- Step 3b: Flatten each 28x28 image into a 784-element vector ---
    # .reshape(num_images, 784) rearranges the data:
    #   - -1 means "figure out this dimension automatically" (keeps image count the same)
    #   - 784 = 28 × 28 (all pixels in one long row)
    x_train_processed = x_train_normalized.reshape(-1, 784)
    x_test_processed  = x_test_normalized.reshape(-1, 784)

    print(f"  Training data shape (after flattening): {x_train_processed.shape}")
    print(f"  Test data shape     (after flattening): {x_test_processed.shape}")
    print("  Preprocessing complete!")

    return x_train_processed, x_test_processed


# =============================================================================
# SECTION 4: MAIN — RUN EVERYTHING IN ORDER
# =============================================================================

if __name__ == "__main__":
    # This block runs only when you execute this file directly, e.g.:
    #   python src/data_loader.py
    # It won't run when another file imports functions from here.

    # Step 1: Load the data
    x_train, y_train, x_test, y_test = load_data()

    # Step 2: Explore and visualize the data
    explore_data(x_train, y_train)

    # Step 3: Preprocess the data
    x_train_processed, x_test_processed = preprocess_data(x_train, x_test)

    # Step 4: Quick final summary
    print("\n========== DATA PIPELINE SUMMARY ==========")
    print(f"  x_train_processed shape : {x_train_processed.shape}")
    print(f"  x_test_processed  shape : {x_test_processed.shape}")
    print(f"  y_train shape           : {y_train.shape}")
    print(f"  y_test  shape           : {y_test.shape}")
    print(f"  Pixel value range       : {x_train_processed.min()} – {x_train_processed.max()}")
    print("============================================")
    print("\ndata_loader.py finished. Your data is ready for model training!")