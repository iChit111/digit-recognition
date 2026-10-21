# =============================================================================
# phase1_data_summary.py
# Location: src/phase1_data_summary.py
#
# Purpose: DEV Mini Project — PHASE 1: Data Loading & Summary.
# Loads the dataset from a CSV file and gives a complete overview of it:
#   1. Dataset shape (rows, columns)
#   2. List of all columns
#   3. Data types of each column
#   4. Missing values per column
#   5. Statistical summary (describe())
#
# WHERE THE CSV COMES FROM:
# EMNIST ByClass ships through tensorflow-datasets as arrays, not as a CSV.
# The full training split (697,932 images x 784 pixels) is far too big for a
# pandas-based EDA and for the scikit-learn models in Phase 5 (SVM / KNN would
# take hours). So on the first run this script:
#   a) loads the data with data_loader.load_data() (orientation already fixed),
#   b) takes a STRATIFIED random sample of SAMPLE_SIZE images — "stratified"
#      means every one of the 62 classes keeps the same share it has in the
#      full dataset, so the sample is representative,
#   c) saves it as data/emnist_byclass_sample.csv.
# Every later run (and every later phase) simply reads that CSV.
#
# CSV layout (one row per image):
#   label      -> integer class 0-61 (see src/label_map.py)
#   char       -> the actual character, e.g. '7', 'A', 'q'
#   pixel0 ... pixel783 -> the 28x28 image flattened row by row, values 0-255
#
# Run from the project root with:
#     python src/phase1_data_summary.py            # uses the CSV if it exists
#     python src/phase1_data_summary.py --rebuild  # re-creates the CSV
# =============================================================================

# --- Step 1: Import the libraries we need ---
import argparse                             # For the optional --rebuild flag
import os
import sys
import textwrap                             # For indenting printed tables

import numpy as np                          # For working with arrays
import pandas as pd                         # For working with tabular (CSV) data
from sklearn.model_selection import train_test_split  # For the stratified sample

sys.path.append(os.path.dirname(__file__))  # So Python can find our own modules
from label_map import NUM_CLASSES, LABEL_TO_CHAR


# =============================================================================
# SETTINGS (change values here, not deep inside the functions)
# =============================================================================

SEED = 42                                   # Same sample every run
SAMPLE_SIZE = 50_000                        # Rows in the CSV (~97 MB on disk)
DATA_DIR = "data"
CSV_PATH = "data/emnist_byclass_sample.csv"

PIXEL_COLUMNS = [f"pixel{i}" for i in range(28 * 28)]   # pixel0 ... pixel783


# =============================================================================
# SECTION 1: CREATE THE CSV (first run only)
# =============================================================================

def build_sample_csv():
    """
    Draws a stratified sample from the EMNIST ByClass training split and
    saves it to CSV_PATH.

    Only the official TRAINING split is sampled. The official test split is
    left untouched, so it is still available as a truly unseen test set.
    """
    # Imported here, not at the top: data_loader pulls in TensorFlow, which is
    # slow to start. If the CSV already exists we never need it.
    from data_loader import load_data

    x_train, y_train, _, _ = load_data()

    print(f"\nDrawing a stratified sample of {SAMPLE_SIZE} images...")
    # train_test_split is normally used to split data, but with
    # train_size=SAMPLE_SIZE and stratify=y it doubles as a stratified sampler:
    # we keep the first part and throw the rest away.
    x_sample, _, y_sample, _ = train_test_split(
        x_train, y_train,
        train_size=SAMPLE_SIZE,
        random_state=SEED,
        stratify=y_train,
    )

    # Build the table: label, char, then the 784 pixel columns.
    df = pd.DataFrame(x_sample.reshape(len(x_sample), -1), columns=PIXEL_COLUMNS)
    df.insert(0, "label", y_sample)
    df.insert(1, "char", [LABEL_TO_CHAR[int(y)] for y in y_sample])

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"  Saved: {CSV_PATH}  ({os.path.getsize(CSV_PATH) / 1e6:.1f} MB)")

    # Quick check that the sample really is representative: compare each
    # class's share in the sample with its share in the full training split.
    full_share = np.bincount(y_train, minlength=NUM_CLASSES) / len(y_train)
    sample_share = np.bincount(y_sample, minlength=NUM_CLASSES) / len(y_sample)
    max_diff = np.abs(full_share - sample_share).max() * 100
    print(f"  Largest class-share difference vs. full dataset: {max_diff:.3f} percentage points")


# =============================================================================
# SECTION 2: LOAD THE DATASET FROM CSV
# =============================================================================

def load_csv():
    """
    Loads the dataset from CSV_PATH into a pandas DataFrame.

    We tell pandas the column types up front. Pixel values are 0-255, so
    uint8 (1 byte each) is enough; letting pandas guess would give int64
    (8 bytes each) and use 8x the memory for no benefit.

    Returns:
        df: DataFrame with columns label, char, pixel0 ... pixel783
    """
    print(f"Loading dataset from '{CSV_PATH}'...")
    dtypes = {"label": "int64", "char": "string"}
    dtypes.update({col: "uint8" for col in PIXEL_COLUMNS})
    df = pd.read_csv(CSV_PATH, dtype=dtypes)
    print("  Loaded successfully!")
    return df


# =============================================================================
# SECTION 3: SUMMARIZE THE DATASET
# =============================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def show_shape_and_columns(df):
    """1. Shape and 2. list of all columns."""
    print_header("1. DATASET SHAPE")
    print(f"  Dataset Shape: {df.shape}  -> {df.shape[0]} rows (images), "
          f"{df.shape[1]} columns")

    print_header("2. COLUMNS")
    # Printing all 786 names one by one would fill several screens, so we
    # print the two descriptive columns, then the pixel range, then confirm
    # the pixel columns are complete and in order.
    other_columns = [c for c in df.columns if c not in PIXEL_COLUMNS]
    pixel_columns = [c for c in df.columns if c in PIXEL_COLUMNS]
    print(f"  Columns: {other_columns} + [{pixel_columns[0]}, {pixel_columns[1]}, "
          f"..., {pixel_columns[-1]}]")
    print(f"  Descriptive columns : {len(other_columns)}  {other_columns}")
    print(f"  Pixel columns       : {len(pixel_columns)}  "
          f"(28 x 28 image flattened row by row)")
    print(f"  Pixel columns complete and in order: {pixel_columns == PIXEL_COLUMNS}")


def show_info_and_dtypes(df):
    """Basic info (df.info()) and 3. data types."""
    print_header("BASIC INFORMATION  (df.info())")
    df.info()   # With 786 columns pandas prints a compact summary

    print_header("3. DATA TYPES")
    print("  Data type of the descriptive columns:")
    print(textwrap.indent(df[["label", "char"]].dtypes.to_string(), "    "))
    print("\n  Number of columns per data type:")
    print(textwrap.indent(df.dtypes.astype(str).value_counts().to_string(), "    "))
    print(f"\n  Memory used by the DataFrame: "
          f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")


def show_missing_values(df):
    """4. Missing values per column."""
    print_header("4. MISSING VALUES")
    missing = df.isnull().sum()

    # Show the count for the descriptive columns explicitly, and summarize the
    # 784 pixel columns (listing 784 zeros would hide anything interesting).
    print(f"  label  : {missing['label']}")
    print(f"  char   : {missing['char']}")
    pixel_missing = missing[PIXEL_COLUMNS]
    print(f"  pixel columns with at least one missing value: "
          f"{(pixel_missing > 0).sum()} of {len(PIXEL_COLUMNS)}")
    print(f"  Total missing values in the whole dataset: {missing.sum()}")

    if missing.sum() == 0:
        print("  -> No missing values. Phase 2 will confirm this and needs no imputation.")
    else:
        print("  -> Columns with missing values:")
        print(missing[missing > 0].to_string())


def show_statistical_summary(df):
    """5. Statistical summary using describe()."""
    print_header("5. STATISTICAL SUMMARY  (describe())")

    # --- 5a. Numeric target column ---
    print("  a) label (numeric class id 0-61):")
    print(df[["label"]].describe().T.to_string())

    # --- 5b. Categorical column ---
    # For text columns describe() gives count / unique / most frequent value.
    print("\n  b) char (categorical):")
    print(df[["char"]].describe().T.to_string())

    # --- 5c. A few representative pixel columns ---
    # describe() on all 784 pixel columns is an unreadable 8 x 784 table, so
    # we pick pixels from the corner, edge and centre of the image to show
    # how differently they behave.
    def pixel_name(row, col):
        return f"pixel{row * 28 + col}"
    sample_pixels = {
        "corner (0,0)":  pixel_name(0, 0),
        "edge (0,14)":   pixel_name(0, 14),
        "inner (7,7)":   pixel_name(7, 7),
        "centre (14,14)": pixel_name(14, 14),
    }
    print("\n  c) Selected pixel columns (0 = black background, 255 = full ink):")
    summary = df[list(sample_pixels.values())].describe().T
    summary.index = [f"{name} {col}" for name, col in sample_pixels.items()]
    print(summary.round(2).to_string())

    # --- 5d. All pixel values pooled together ---
    pixels = df[PIXEL_COLUMNS].to_numpy()
    print("\n  d) All 784 pixel columns pooled together:")
    print(f"     Min / Max pixel value       : {pixels.min()} / {pixels.max()}")
    print(f"     Mean pixel value            : {pixels.mean():.2f}")
    print(f"     Share of pixels that are 0  : {(pixels == 0).mean() * 100:.1f}%  "
          f"(background)")
    always_zero = int((pixels.max(axis=0) == 0).sum())
    print(f"     Pixel columns that are 0 in every image: {always_zero}  "
          f"(carry no information)")

    # --- 5e. Class balance at a glance ---
    counts = df["char"].value_counts()
    print("\n  e) Images per class:")
    print(f"     Classes present : {df['label'].nunique()} of {NUM_CLASSES}")
    print(f"     Most common     : '{counts.index[0]}' with {counts.iloc[0]} images")
    print(f"     Least common    : '{counts.index[-1]}' with {counts.iloc[-1]} images")
    print(f"     Imbalance ratio : {counts.iloc[0] / counts.iloc[-1]:.1f}x  "
          f"(examined in detail in Phase 3)")


# =============================================================================
# SECTION 4: MAIN — RUN EVERYTHING IN ORDER
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEV Mini Project - Phase 1: Data Loading & Summary")
    parser.add_argument("--rebuild", action="store_true",
                        help="Re-create the sample CSV even if it already exists")
    args = parser.parse_args()

    # Step 1: Make sure the CSV exists (built from EMNIST on the first run)
    if args.rebuild or not os.path.exists(CSV_PATH):
        build_sample_csv()

    # Step 2: Load the dataset from CSV
    df = load_csv()

    # Step 3: Summaries
    show_shape_and_columns(df)
    show_info_and_dtypes(df)
    show_missing_values(df)
    show_statistical_summary(df)

    print("\nphase1_data_summary.py finished. Next: Phase 2 (preprocessing & transformation).")
