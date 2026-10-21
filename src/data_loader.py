# =============================================================================
# data_loader.py
# Location: src/data_loader.py
#
# Purpose: Load, explore, visualize, and preprocess the EMNIST ByClass
# dataset. EMNIST ByClass contains 814,255 grayscale images of handwritten
# characters across 62 classes: digits 0-9, uppercase A-Z, and lowercase a-z,
# each image being 28x28 pixels.
#
# CHANGE LOG FROM THE ORIGINAL (MNIST, 10-class) VERSION:
#   - Data source: keras.datasets.mnist -> tensorflow_datasets "emnist/byclass"
#   - Classes: 10 -> 62 (see src/label_map.py for the official mapping)
#   - Added: orientation fix (EMNIST images ship rotated/flipped vs. MNIST)
#   - Added: class_weight computation, since ByClass is heavily imbalanced
#     (this project does NOT merge in MNIST digits — see the note in
#     load_data() for why)
#   - Function names/structure kept identical on purpose so this stays a
#     clean diff for teammates reviewing it, not a rewrite from scratch.
# =============================================================================

# --- Step 1: Import the libraries we need ---
import numpy as np                          # For working with numbers and arrays
import matplotlib.pyplot as plt             # For drawing charts and showing images
import tensorflow as tf                     # The main deep learning library
import tensorflow_datasets as tfds          # Gives us access to EMNIST (not in plain Keras)

import sys
import os
sys.path.append(os.path.dirname(__file__))  # So Python can find label_map.py
from label_map import NUM_CLASSES, LABEL_TO_CHAR, get_class_names


# =============================================================================
# SECTION 1: LOAD THE DATA
# =============================================================================

def load_data():
    """
    Loads the EMNIST ByClass dataset and splits it into training and test sets.

    WHY tensorflow-datasets (TFDS) INSTEAD OF THE `emnist` PIP PACKAGE:
    We evaluated both options. The `emnist` pip package is a thin wrapper
    that downloads a NIST-hosted zip directly; it has no official maintainer
    guarantee, and it gives you back raw numpy arrays with NO structured
    class-to-character mapping — you'd have to hand-parse NIST's mapping
    file yourself to get labels right, which is exactly the kind of
    hand-rolled guess we're trying to avoid. TFDS is actively maintained by
    Google, documents this dataset's known quirks in its own catalog entry,
    and hands you back an integer label (0-61) in the standard byclass
    ordering, which we mirror exactly in src/label_map.py.
    Install with: pip install tensorflow-datasets (see requirements.txt).

    WHY WE DO **NOT** MERGE IN MNIST DIGITS:
    It's tempting to bolster the digit classes with the original MNIST
    dataset for "more data." We're deliberately NOT doing this. Both MNIST
    and EMNIST ByClass are resamples of the exact same underlying source —
    NIST Special Database 19 — as documented in the EMNIST paper (Cohen et
    al., 2017). Merging them doesn't introduce a new handwriting
    distribution; it mostly adds near-duplicate-distribution samples, with a
    real risk of overlap between what ends up in your "train" set from one
    source and your "test" set from the other (i.e. leakage, which would
    make your test accuracy look better than it actually is). ByClass's own
    digit classes already have ~344,000 samples combined, so this project
    is not short on digit data. If you want to revisit this later, the
    place to do it is here, with an explicit, deduplicated merge — not a
    silent concat.

    Returns:
        x_train: Training images  → shape: (697932, 28, 28), uint8
        y_train: Training labels  → shape: (697932,), integers 0-61
        x_test:  Test images      → shape: (116323, 28, 28), uint8
        y_test:  Test labels      → shape: (116323,), integers 0-61
    """

    print("Loading EMNIST ByClass dataset (this can take a while on first run"
          " — ~350MB download, cached afterward)...")

    # tfds.load with batch_size=-1 pulls back the WHOLE split as a single
    # batch, and as_supervised=True gives us clean (image, label) pairs
    # instead of a dict. tfds.as_numpy converts the TF tensors to numpy
    # arrays so the rest of this file (and the rest of the team's code) can
    # keep working with plain numpy, exactly like the MNIST version did.
    train_ds = tfds.load(
        "emnist/byclass", split="train", as_supervised=True, batch_size=-1
    )
    test_ds = tfds.load(
        "emnist/byclass", split="test", as_supervised=True, batch_size=-1
    )

    x_train, y_train = tfds.as_numpy(train_ds)
    x_test,  y_test  = tfds.as_numpy(test_ds)

    # TFDS gives images shape (N, 28, 28, 1) — drop the trailing channel
    # dimension so downstream code sees the same (N, 28, 28) shape the
    # original MNIST version used.
    x_train = x_train.squeeze(axis=-1)
    x_test  = x_test.squeeze(axis=-1)

    # --- Fix EMNIST's known orientation bug ---
    # TFDS's own catalog page for this dataset states it plainly: EMNIST
    # images are stored rotated 90 degrees anti-clockwise and flipped
    # horizontally relative to how a human (or MNIST) would expect them to
    # look. If we don't correct this, every image in our sample grid (and
    # every image the demo shows a human) will look wrong, even though the
    # model can technically still learn from the rotated version.
    x_train = _fix_emnist_orientation(x_train)
    x_test  = _fix_emnist_orientation(x_test)

    y_train = y_train.astype(np.int64)
    y_test  = y_test.astype(np.int64)

    print(f"  Training images: {x_train.shape}  → {x_train.shape[0]} images, each {x_train.shape[1]}x{x_train.shape[2]} pixels")
    print(f"  Training labels: {y_train.shape}  → {NUM_CLASSES} classes (0-9, A-Z, a-z)")
    print(f"  Test images:     {x_test.shape}")
    print(f"  Test labels:     {y_test.shape}")
    print(f"  Pixel value range (before preprocessing): {x_train.min()} – {x_train.max()}")

    return x_train, y_train, x_test, y_test


def _fix_emnist_orientation(images):
    """
    (Private helper) Corrects EMNIST's known 90-degree-rotation + horizontal-flip
    orientation bug so images display the way a human expects to see them.

    This bug is documented directly on TFDS's own catalog page for this
    dataset. The fix is: transpose the height/width axes (undoes the
    rotation and mirroring in one step, since the two together are exactly a
    transpose). We do this with plain numpy since we've already pulled the
    data out of TF tensors.

    Args:
        images: array of shape (N, 28, 28), as loaded from TFDS.

    Returns:
        Corrected array of shape (N, 28, 28).
    """
    # axes=(0, 2, 1) swaps the row/column axes for every image in the batch
    # (leaves the batch dimension, axis 0, alone) — this is the "un-rotate" step.
    # A transpose alone fixes EMNIST: "rotated 90 + mirrored" IS a transpose,
    # so no extra flip is needed (adding one mirrors the images).
    return np.transpose(images, axes=(0, 2, 1))


# =============================================================================
# SECTION 2: EXPLORE AND VISUALIZE THE DATA
# =============================================================================

def explore_data(x_train, y_train):
    """
    Visualizes the EMNIST ByClass data in two ways:
      1. A grid of 25 sample images from the training set, labeled with their
         actual character (not just a raw integer label).
      2. A bar chart showing how many images exist for each of the 62 classes.

    Args:
        x_train: Training images (numpy array)
        y_train: Training labels (numpy array), integers 0-61
    """

    # --- 2a. Show a 5x5 grid of sample images ---
    print("\nVisualizing 25 sample images from the training set...")

    fig, axes = plt.subplots(5, 5, figsize=(8, 8))   # Create a 5x5 grid of subplots
    fig.suptitle("Sample EMNIST ByClass Images", fontsize=16)  # Add a title to the whole figure

    for i, ax in enumerate(axes.flat):  # Loop through each subplot (25 total)
        ax.imshow(x_train[i], cmap="gray")  # Show the image in grayscale
        # Look up the human-readable character instead of showing a bare integer —
        # "Label: 36" means nothing at a glance, but "Label: a" does.
        char_label = LABEL_TO_CHAR[int(y_train[i])]
        ax.set_title(f"Label: {char_label}")
        ax.axis("off")                        # Hide the axis lines for a cleaner look

    plt.tight_layout()       # Automatically adjust spacing between subplots
    plt.savefig("sample_images.png")  # Save the figure as a PNG file
    plt.show()               # Display the figure on screen
    print("  Saved: sample_images.png")

    # --- 2b. Show a bar chart of label distribution across all 62 classes ---
    print("\nVisualizing label distribution across all 62 classes...")

    # np.bincount counts how many times each label (0-61) appears in y_train.
    # minlength=NUM_CLASSES guards against a rare class not appearing at all
    # in a given run, which would otherwise silently shrink the array.
    label_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_names  = get_class_names()   # ['0', '1', ..., '9', 'A', ..., 'Z', 'a', ..., 'z']

    # 62 bars is a lot wider than the original 10, so we widen the figure and
    # rotate the x-axis labels so they don't overlap into an unreadable mess.
    plt.figure(figsize=(20, 6))
    bar_positions = np.arange(NUM_CLASSES)

    # Color-code the three groups (digits / uppercase / lowercase) so the
    # imbalance pattern is visually obvious at a glance.
    colors = (
        ["steelblue"] * 10 +      # digits: 0-9
        ["darkorange"] * 26 +     # uppercase: A-Z
        ["seagreen"] * 26         # lowercase: a-z
    )

    plt.bar(bar_positions, label_counts, color=colors, edgecolor="black", linewidth=0.3)
    plt.title("Label Distribution in Training Set — EMNIST ByClass (62 classes)")
    plt.xlabel("Class (0-9, A-Z, a-z)")
    plt.ylabel("Number of Images")
    plt.xticks(bar_positions, class_names, fontsize=8)

    # Unlike the 10-class version, we skip printing the exact count on top of
    # every single bar — with 62 bars that just becomes visual noise. The
    # color-coded groups tell the real story: digits and uppercase are far
    # better represented than most lowercase letters.
    plt.tight_layout()
    plt.savefig("label_distribution.png")  # Save the chart as a PNG file
    plt.show()
    print("  Saved: label_distribution.png")

    # Print the min/max class counts so the imbalance is stated explicitly,
    # not just implied by a chart someone might skim past.
    most_common  = class_names[np.argmax(label_counts)]
    least_common = class_names[np.argmin(label_counts)]
    print(f"\n  Most common class : '{most_common}' with {label_counts.max()} samples")
    print(f"  Least common class: '{least_common}' with {label_counts.min()} samples")
    print(f"  Imbalance ratio   : {label_counts.max() / label_counts.min():.1f}x")


# =============================================================================
# SECTION 3: PREPROCESS THE DATA
# =============================================================================

def preprocess_data(x_train, x_test):
    """
    Prepares the raw images so a neural network can use them.

    Two steps (unchanged from the original MNIST version):
      1. NORMALIZE: Scale pixel values from the range [0, 255] to [0.0, 1.0].
                    Neural networks learn faster and more reliably with small numbers.

      2. FLATTEN:   Reshape each 28x28 image into a single row of 784 numbers.
                    A basic neural network expects a 1D input, not a 2D grid.
                    (28 × 28 = 784)

    NOTE FOR PERSON 2 (model.py): the OUTPUT SHAPE of x_train/x_test is
    unchanged from the original MNIST version — still (N, 784). What changes
    is the LABEL shape: y_train/y_test now range 0-61 instead of 0-9, so your
    model's final Dense layer needs units=62 instead of units=10 (see the
    "What to tell your teammates" notes for the full list).

    Args:
        x_train: Raw training images  → shape: (697932, 28, 28)
        x_test:  Raw test images      → shape: (116323, 28, 28)

    Returns:
        x_train_processed: shape (697932, 784), values between 0.0 and 1.0
        x_test_processed:  shape (116323, 784), values between 0.0 and 1.0
    """

    print("\nPreprocessing data...")

    # --- Step 3a: Normalize pixel values ---
    # float32 on purpose: dividing a uint8 array by 255.0 gives float64 by
    # default, which is ~4.4 GB for 698k images (and train_test_split copies it).
    # Keras trains in float32 anyway, so float64 is pure waste.
    x_train_normalized = x_train.astype("float32") / 255.0
    x_test_normalized  = x_test.astype("float32") / 255.0

    print(f"  Pixel value range (after normalization): "
          f"{x_train_normalized.min():.1f} – {x_train_normalized.max():.1f}")

    # --- Step 3b: Flatten each 28x28 image into a 784-element vector ---
    x_train_processed = x_train_normalized.reshape(-1, 784)
    x_test_processed  = x_test_normalized.reshape(-1, 784)

    print(f"  Training data shape (after flattening): {x_train_processed.shape}")
    print(f"  Test data shape     (after flattening): {x_test_processed.shape}")
    print("  Preprocessing complete!")

    return x_train_processed, x_test_processed


# =============================================================================
# SECTION 4: CLASS WEIGHTS (NEW — ByClass is heavily imbalanced)
# =============================================================================

def compute_class_weights(y_train):
    """
    Computes a Keras-compatible class_weight dict to counteract ByClass's
    known class imbalance (e.g. digits and uppercase letters have far more
    samples than several lowercase letters like 'l', 'i', or 'q' — real
    handwriting samples of some lowercase letters are simply scarcer in the
    source data).

    Uses the standard "balanced" heuristic (same formula sklearn's
    `compute_class_weight(class_weight='balanced', ...)` uses under the
    hood, reimplemented directly here in plain numpy so this file doesn't
    take on a new sklearn dependency just for one formula):

        weight[c] = n_samples / (n_classes * count[c])

    This gives rare classes a proportionally larger weight, so the model's
    loss function penalizes mistakes on rare letters more heavily than
    mistakes on common ones — without this, a model can rack up deceptively
    high overall accuracy just by nailing digits and never learning rare
    lowercase letters at all.

    Args:
        y_train: Training labels → shape (N,), integers 0-61

    Returns:
        class_weight: dict mapping {0: weight, 1: weight, ..., 61: weight},
                      ready to pass directly as model.fit(..., class_weight=class_weight)
    """

    print("\nComputing class weights to counteract ByClass's known imbalance...")

    label_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    n_samples    = len(y_train)

    # Guard against a pathological empty class (shouldn't happen with the
    # real dataset, but this keeps the function from throwing a divide-by-
    # zero if it's ever run on a filtered/sampled subset during debugging).
    safe_counts = np.where(label_counts == 0, 1, label_counts)

    class_weight = {
        label: float(n_samples / (NUM_CLASSES * safe_counts[label]))
        for label in range(NUM_CLASSES)
    }

    weight_values = np.array(list(class_weight.values()))
    print(f"  Class weight range: {weight_values.min():.2f} (most common class) "
          f"to {weight_values.max():.2f} (rarest class)")

    return class_weight


# =============================================================================
# SECTION 5: MAIN — RUN EVERYTHING IN ORDER
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

    # Step 4: Compute class weights (new — needed because ByClass is imbalanced)
    class_weight = compute_class_weights(y_train)

    # Step 5: Quick final summary
    print("\n========== DATA PIPELINE SUMMARY ==========")
    print(f"  x_train_processed shape : {x_train_processed.shape}")
    print(f"  x_test_processed  shape : {x_test_processed.shape}")
    print(f"  y_train shape           : {y_train.shape}")
    print(f"  y_test  shape           : {y_test.shape}")
    print(f"  Number of classes       : {NUM_CLASSES}")
    print(f"  Pixel value range       : {x_train_processed.min()} – {x_train_processed.max()}")
    print("============================================")
    print("\ndata_loader.py finished. Your data is ready for model training!")