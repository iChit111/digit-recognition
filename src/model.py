# =============================================================================
# model.py
# Location: src/model.py
#
# Purpose: Build, train, evaluate, and save a dense (fully connected) neural
# network that recognizes handwritten characters from EMNIST ByClass.
#
# EMNIST ByClass has 62 classes: digits 0-9, uppercase A-Z, lowercase a-z.
# This is a DENSE BASELINE. A CNN will be tried later as a separate experiment,
# so we deliberately keep the architecture simple here.
#
# Run from the project root with:
#     python src/model.py
# =============================================================================

# --- Step 1: Import the libraries we need ---
import os                                    # For creating folders / building paths
import numpy as np                           # For working with numbers and arrays
import matplotlib.pyplot as plt              # For drawing the training-history chart
from tensorflow import keras                 # The deep learning library (Keras API)
from tensorflow.keras import layers          # Building blocks (Dense, Dropout, ...)
from sklearn.model_selection import train_test_split  # Shuffled/stratified split

# Our own project files (they live in the same src/ folder).
from data_loader import load_data, preprocess_data, compute_class_weights
from label_map import (
    NUM_CLASSES,          # 62 -> single source of truth, never hardcode 62 here
    DIGIT_LABELS,         # class indices for 0-9
    UPPERCASE_LABELS,     # class indices for A-Z
    LOWERCASE_LABELS,     # class indices for a-z
)


# =============================================================================
# SETTINGS (change values here, not deep inside the functions)
# =============================================================================

SEED = 42                          # Makes the split and weight init repeatable
VALIDATION_FRACTION = 0.10         # 10% of the training data is held out for validation
MAX_EPOCHS = 100                   # Generous cap; EarlyStopping will stop sooner
BATCH_SIZE = 128
EARLY_STOP_PATIENCE = 5            # Stop if val_loss hasn't improved for 5 epochs
USE_CLASS_WEIGHTS = True           # Flip to False to compare "with vs without" weighting

# Output paths. Other scripts (evaluate.py, demo.py) load this exact file,
# so do NOT change the path or the .h5 format without telling the team.
RESULTS_DIR = "results"
MODEL_PATH = "results/model.h5"
HISTORY_PLOT_PATH = "results/training_history.png"


# =============================================================================
# SECTION 1: BUILD THE MODEL
# =============================================================================

def build_model():
    """
    Builds the dense neural network (not trained yet).

    Architecture:
        Input (784 pixels)
        -> Dense 512 + ReLU -> Dropout 0.3
        -> Dense 256 + ReLU -> Dropout 0.2
        -> Dense NUM_CLASSES (62) + Softmax

    Returns:
        A compiled Keras model, ready for training.
    """

    model = keras.Sequential([
        # An explicit Input layer tells Keras the shape of one example:
        # 784 numbers (a flattened 28x28 image). This replaces the old,
        # deprecated "input_shape=" argument on the first Dense layer.
        keras.Input(shape=(784,)),

        # Hidden layer 1: 512 "neurons". Wider than the old MNIST model (128)
        # because telling apart 62 classes needs more capacity than 10.
        layers.Dense(512, activation="relu"),

        # Dropout randomly switches off 30% of neurons during training,
        # which stops the network from simply memorizing the training images.
        layers.Dropout(0.3),

        # Hidden layer 2: 256 neurons (was 64).
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),

        # Output layer: one neuron per class. Softmax turns the 62 outputs
        # into probabilities that add up to 1.
        # NUM_CLASSES comes from label_map.py so it can never drift out of sync.
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    # Our labels are whole numbers (0..61), not one-hot vectors, so we use
    # "sparse" categorical crossentropy.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


# =============================================================================
# SECTION 2: TRAIN THE MODEL
# =============================================================================

def train_model(model, x_train, y_train):
    """
    Trains the model using a shuffled, seeded, stratified validation split
    and EarlyStopping.

    Why not validation_split=0.1 in model.fit()?
        Keras' validation_split takes the LAST 10% of the data WITHOUT
        shuffling. If the data is ordered in any way, the validation set would
        not be representative. Here we split ourselves instead.

    Args:
        model:   The compiled Keras model.
        x_train: Training images, shape (N, 784).
        y_train: Training labels, shape (N,).

    Returns:
        history: Keras History object (loss/accuracy for every epoch).
        n_train: Number of images actually used for training (after the split).
        n_val:   Number of images held out for validation.
    """

    # --- Step 2a: Shuffled + stratified train/validation split ---
    # shuffle=True       -> mix the data before splitting
    # random_state=SEED  -> same split every run (repeatable results)
    # stratify=y_train   -> every class keeps (almost) the same share in both
    #                       parts. Important here because some classes are rare.
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        shuffle=True,
        stratify=y_train,
    )
    print(f"\nTrain/validation split: {len(x_tr)} train, {len(x_val)} validation")

    # --- Step 2b: Class weights ---
    # EMNIST ByClass is HEAVILY IMBALANCED: some classes (like digits) have
    # tens of thousands of examples while others (some lowercase letters, e.g.
    # 'k' or 'z') have only a few thousand. Without correction, the model can
    # score well overall by mostly getting the common classes right and
    # ignoring the rare ones.
    # Class weights fix this: a mistake on a rare class counts MORE in the loss,
    # a mistake on a common class counts LESS. We compute them from the
    # training portion only, using the helper in data_loader.py.
    class_weight = compute_class_weights(y_tr) if USE_CLASS_WEIGHTS else None
    print(f"Using class weights: {USE_CLASS_WEIGHTS}")

    # --- Step 2c: EarlyStopping ---
    # Watches the validation loss. If it hasn't improved for
    # EARLY_STOP_PATIENCE epochs, training stops. restore_best_weights=True
    # then rolls the model back to its best epoch (not the last one).
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    print("\nTraining...")
    history = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,
    )

    return history, len(x_tr), len(x_val)


# =============================================================================
# SECTION 3: PLOT TRAINING HISTORY
# =============================================================================

def plot_history(history):
    """
    Draws accuracy and loss curves (train vs validation) and saves them to
    results/training_history.png.

    Args:
        history: The History object returned by train_model().
    """

    # results/ is gitignored, so it won't exist on a fresh clone. Create it.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left chart: accuracy
    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # Right chart: loss
    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH)
    plt.close(fig)   # Free memory; also avoids blocking on a window
    print(f"  Saved: {HISTORY_PLOT_PATH}")


# =============================================================================
# SECTION 4: EVALUATE THE MODEL
# =============================================================================

def evaluate_model(model, x_test, y_test):
    """
    Measures accuracy on the untouched test set, overall and per group
    (digits / uppercase / lowercase).

    The per-group numbers show whether class weighting actually helps the rare
    lowercase letters. Tip: run once with USE_CLASS_WEIGHTS = True and once
    with False, then compare the groups.

    Args:
        model:  The trained Keras model.
        x_test: Test images, shape (N, 784).
        y_test: Test labels, shape (N,).

    Returns:
        test_loss, test_acc, group_acc (dict of group name -> accuracy)
    """

    print("\nEvaluating on the test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # The model outputs 62 probabilities per image; argmax picks the most likely class.
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)

    groups = {
        "Digits (0-9)":     DIGIT_LABELS,
        "Uppercase (A-Z)":  UPPERCASE_LABELS,
        "Lowercase (a-z)":  LOWERCASE_LABELS,
    }

    group_acc = {}
    for name, labels in groups.items():
        # Select only the test images whose TRUE class belongs to this group...
        mask = np.isin(y_test, list(labels))
        # ...and check what fraction the model got right.
        group_acc[name] = float(np.mean(predictions[mask] == y_test[mask]))
        group_acc[name + " count"] = int(mask.sum())

    return test_loss, test_acc, group_acc


# =============================================================================
# SECTION 5: SAVE THE MODEL
# =============================================================================

def save_model(model, path=MODEL_PATH):
    """
    Saves the trained model so evaluate.py and demo.py can load it later.

    Args:
        model: The trained Keras model.
        path:  Where to save it (default: results/model.h5).
    """

    # results/ is gitignored and missing on a fresh clone, so create it first.
    # exist_ok=True means "don't complain if it's already there".
    os.makedirs(os.path.dirname(path), exist_ok=True)

    model.save(path)   # The .h5 extension makes Keras use the HDF5 format
    print(f"  Saved: {path}")


# =============================================================================
# SECTION 6: MAIN — RUN EVERYTHING IN ORDER
# =============================================================================

if __name__ == "__main__":
    # Runs only when you execute this file directly:  python src/model.py
    # Seed FIRST. Weight initialization happens inside build_model(), so
    # seeding after it (as train_model() used to) leaves the starting weights random.
    keras.utils.set_random_seed(SEED)

    # Step 1: Load and preprocess the data (normalized, flattened to 784)
    x_train, y_train, x_test, y_test = load_data()
    x_train, x_test = preprocess_data(x_train, x_test)

    # Step 2: Build the model
    model = build_model()

    # Step 3: Train it
    history, n_train, n_val = train_model(model, x_train, y_train)
    epochs_trained = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    # Step 4: Plot and save the training history
    print("\nSaving training history plot...")
    plot_history(history)

    # Step 5: Evaluate on the test set
    test_loss, test_acc, group_acc = evaluate_model(model, x_test, y_test)

    # Step 6: Save the model
    print("\nSaving model...")
    save_model(model, MODEL_PATH)

    # Step 7: Final summary (everything printed here is a real variable)
    print("\n========== MODEL SUMMARY ==========")
    print(f"  Number of classes        : {NUM_CLASSES}")
    print(f"  Training images used     : {n_train}")
    print(f"  Validation images used   : {n_val}")
    print(f"  Class weights used       : {USE_CLASS_WEIGHTS}")
    print(f"  Epochs trained           : {epochs_trained} (max {MAX_EPOCHS})")
    print(f"  Best epoch (restored)    : {best_epoch}")
    print(f"  Test loss                : {test_loss:.4f}")
    print(f"  Test accuracy (overall)  : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print("  --- Per-group test accuracy ---")
    for name in ("Digits (0-9)", "Uppercase (A-Z)", "Lowercase (a-z)"):
        print(f"  {name:<24}: {group_acc[name] * 100:.2f}%  "
              f"({group_acc[name + ' count']} test images)")
    print(f"  Model saved to           : {MODEL_PATH}")
    print(f"  History plot saved to    : {HISTORY_PLOT_PATH}")
    print("===================================")
    print("\nmodel.py finished.")