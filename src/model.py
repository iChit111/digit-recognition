# =============================================================================
# model.py
# Location: src/model.py
#
# Purpose: Define, compile, train, and save a neural network that classifies
# handwritten digits (0–9) using the preprocessed MNIST data from data_loader.py.
#
# A neural network is inspired by the human brain — it's made of layers of
# "neurons" that pass signals to each other. By showing it tens of thousands
# of labeled examples, it learns which patterns of pixels correspond to which
# digits. Once trained, it can look at a brand-new image and predict the digit.
# =============================================================================

# --- Import the libraries we need ---
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np                          # For numerical operations
import matplotlib.pyplot as plt             # For plotting training history
import tensorflow as tf                     # The main deep learning library
import keras               # High-level API for building models


# =============================================================================
# SECTION 1: BUILD THE MODEL
# =============================================================================

def build_model():
    """
    Constructs and returns a Keras Sequential neural network.

    Architecture (the layers stacked on top of each other):
    ┌─────────────────────────────────────────────────────────┐
    │  Input       → 784 units  (one per pixel)               │
    │  Hidden #1   → 128 units  (ReLU activation)             │
    │  Hidden #2   →  64 units  (ReLU activation)             │
    │  Output      →  10 units  (Softmax activation)          │
    └─────────────────────────────────────────────────────────┘

    Why these choices?
    - Sequential: layers are stacked one after the other in a straight line.
    - Dense layers: every neuron is connected to every neuron in the next layer.
    - ReLU (Rectified Linear Unit): a simple activation function — it outputs
      the input if it's positive, otherwise outputs 0. This helps the network
      learn non-linear patterns (e.g., curves and edges in digits).
    - Softmax: squashes the 10 raw output scores into probabilities that sum
      to 1.0, so the network's prediction is "I'm X% sure this is a 3."

    Returns:
        model: An uncompiled Keras Sequential model.
    """

    print("Building the neural network model...")

    # keras.Sequential lets us build a model by adding layers one by one.
    # Think of it like stacking Lego bricks — each layer transforms the data
    # and passes the result to the next layer.
    model = keras.Sequential([

        # --- Layer 1: Input + First Hidden Layer ---
        # units=128: this layer has 128 neurons.
        # activation="relu": each neuron fires only if its input is positive.
        # input_shape=(784,): tells Keras the shape of ONE training example —
        #   a flat array of 784 pixel values (28×28 image, already flattened).
        #   This first layer automatically acts as our input layer.
        keras.layers.Dense(units=128, activation="relu", input_shape=(784,)),

        # --- Layer 2: Second Hidden Layer ---
        # A smaller second hidden layer (64 neurons) helps the network
        # compress what it learned in layer 1 into more abstract features.
        # For example, layer 1 might detect edges; layer 2 might detect curves.
        keras.layers.Dense(units=64, activation="relu"),

        # --- Layer 3: Output Layer ---
        # units=10: one output neuron for each possible digit class (0 through 9).
        # activation="softmax": converts the 10 raw scores into probabilities.
        #   e.g., [0.02, 0.01, 0.03, 0.91, 0.01, ...] → the model thinks it's a 3.
        # The digit with the highest probability is the model's final prediction.
        keras.layers.Dense(units=10, activation="softmax"),

    ])

    # Print a table showing each layer's name, output shape, and parameter count.
    # Parameters = the weights and biases the network learns during training.
    model.summary()

    print("\nModel built successfully!")
    return model


# =============================================================================
# SECTION 2: TRAIN THE MODEL
# =============================================================================

def train_model(model, x_train, y_train, epochs=10, batch_size=32, validation_split=0.1):
    """
    Compiles and trains the model on the preprocessed MNIST training data.

    Training overview:
      - The model sees the training images in small batches (32 at a time).
      - After each batch, it compares its predictions to the true labels,
        calculates how wrong it was (the "loss"), and adjusts its internal
        weights slightly to do better next time. This adjustment is called
        "backpropagation."
      - One full pass through ALL the training data is called an "epoch."
      - We train for multiple epochs so the model can keep improving.

    Args:
        model:            The Keras model returned by build_model().
        x_train:          Preprocessed training images → shape (60000, 784)
        y_train:          Training labels (integers 0–9) → shape (60000,)
        epochs:           How many full passes through the training data (default: 10).
        batch_size:       Number of images processed before each weight update (default: 32).
        validation_split: Fraction of training data held back to measure
                          performance on unseen data during training (default: 10%).

    Returns:
        history: A Keras History object containing loss and accuracy recorded
                 at the end of each epoch. Useful for plotting training curves.
    """

    print("\nCompiling the model...")

    # --- Compile: configure HOW the model will learn ---
    # Before training, we must tell Keras three things:

    # 1. optimizer="adam"
    #    The optimizer decides how to adjust the model's weights after each batch.
    #    Adam (Adaptive Moment Estimation) is a popular, reliable choice — it
    #    automatically adjusts the learning rate for each weight individually,
    #    which usually leads to faster and more stable training than plain
    #    gradient descent.

    # 2. loss="sparse_categorical_crossentropy"
    #    The loss function measures how wrong the model's predictions are.
    #    We use "sparse_categorical_crossentropy" because:
    #      - "categorical": we're classifying into multiple categories (10 digits).
    #      - "sparse": our labels are plain integers (e.g., 3), NOT one-hot encoded
    #        vectors (e.g., [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).
    #    A lower loss = better predictions.

    # 3. metrics=["accuracy"]
    #    This is just for reporting — it tells Keras to also track what percentage
    #    of predictions are correct during training, so we can monitor progress.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("  Optimizer : Adam")
    print("  Loss      : sparse_categorical_crossentropy")
    print("  Metrics   : accuracy")

    print(f"\nTraining the model for {epochs} epochs "
          f"(batch size: {batch_size}, validation split: {validation_split*100:.0f}%)...")
    print("-" * 60)

    # --- Fit: run the actual training loop ---
    # model.fit() handles everything:
    #   - splits data into batches
    #   - runs forward pass (predictions)
    #   - calculates loss
    #   - runs backward pass (adjusts weights)
    #   - repeats for each batch and each epoch
    history = model.fit(
        x_train,                        # Input images
        y_train,                        # Correct labels
        epochs=epochs,                  # How many full passes through training data
        batch_size=batch_size,          # How many images per weight update
        validation_split=validation_split,  # Hold out 10% of training data for validation
        verbose=1                       # Print progress bar + metrics after each epoch
    )

    print("-" * 60)
    print("Training complete!")

    # --- Plot the training history ---
    # Visualising how loss and accuracy changed over epochs helps us spot
    # problems like overfitting (training accuracy >> validation accuracy).
    _plot_training_history(history)

    return history


def _plot_training_history(history):
    """
    (Private helper) Plots loss and accuracy curves for training and validation.

    Two charts are created side by side:
      Left:  Training loss vs. Validation loss over each epoch.
      Right: Training accuracy vs. Validation accuracy over each epoch.

    A good model should show both curves decreasing/increasing together.
    If training improves but validation plateaus, the model is overfitting
    (memorising the training data rather than generalising).

    Args:
        history: The Keras History object returned by model.fit().
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=14)

    # --- Left chart: Loss ---
    axes[0].plot(history.history["loss"],     label="Training Loss",   color="steelblue")
    axes[0].plot(history.history["val_loss"], label="Validation Loss", color="tomato", linestyle="--")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Right chart: Accuracy ---
    axes[1].plot(history.history["accuracy"],     label="Training Accuracy",   color="steelblue")
    axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy", color="tomato", linestyle="--")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png")  # Save chart to disk
    plt.show()
    print("  Saved: training_history.png")


# =============================================================================
# SECTION 3: SAVE THE MODEL
# =============================================================================

def save_model(model, filepath="model.h5"):
    """
    Saves the trained model to disk in HDF5 (.h5) format.

    Why save the model?
      Training takes time. By saving the model once it's trained, you can
      reload it later (with keras.models.load_model("model.h5")) and make
      predictions instantly — without retraining from scratch.

    The .h5 file stores everything needed to use the model later:
      - The architecture (layers, units, activations)
      - The learned weights and biases
      - The optimizer state

    Args:
        model:    The trained Keras model to save.
        filepath: Where to save the file (default: "model.h5").
    """

    print(f"\nSaving trained model to '{filepath}'...")
    model.save(filepath)
    print(f"  Model saved successfully → {filepath}")
    print("  To reload it later:  model = keras.models.load_model('model.h5')")


# =============================================================================
# SECTION 4: MAIN — RUN THE FULL PIPELINE
# =============================================================================

if __name__ == "__main__":
    # This block runs only when you execute this file directly, e.g.:
    #   python src/model.py
    # It won't run when another file imports functions from here.

    # We import data_loader from the same src/ directory.
    # If you're running from the project root, Python needs to find the module.
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))  # Add src/ to the module search path
    from data_loader import load_data, preprocess_data

    print("=" * 60)
    print("  MNIST DIGIT CLASSIFIER — FULL TRAINING PIPELINE")
    print("=" * 60)

    # --- Step 1: Load the raw data ---
    x_train_raw, y_train, x_test_raw, y_test = load_data()

    # --- Step 2: Preprocess the data (normalize + flatten) ---
    x_train, x_test = preprocess_data(x_train_raw, x_test_raw)

    # --- Step 3: Build the model ---
    model = build_model()

    # --- Step 4: Train the model ---
    # epochs=10 means the model will see the full training set 10 times.
    # Feel free to increase this (e.g., epochs=20) for potentially higher accuracy.
    history = train_model(model, x_train, y_train, epochs=10)

    # --- Step 5: Evaluate on the test set ---
    # The test set was never seen during training, so this is a fair measure
    # of how well the model generalises to brand-new, unseen handwriting.
    print("\nEvaluating model on the test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_accuracy * 100:.2f}%")

    # --- Step 6: Save the trained model ---
    save_model(model, filepath="model.h5")

    # --- Step 7: Final summary ---
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Training samples : {x_train.shape[0]}")
    print(f"  Test samples     : {x_test.shape[0]}")
    print(f"  Input shape      : {x_train.shape[1]} features (28×28 pixels, flattened)")
    print(f"  Output classes   : 10 (digits 0–9)")
    print(f"  Epochs trained   : 10")
    print(f"  Test Accuracy    : {test_accuracy * 100:.2f}%")
    print(f"  Model saved to   : model.h5")
    print("=" * 60)
    print("\nmodel.py finished. Your trained model is ready for evaluation and demo!")
