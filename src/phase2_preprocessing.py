# =============================================================================
# phase2_preprocessing.py
# Location: src/phase2_preprocessing.py
#
# Purpose: DEV Mini Project — PHASE 2: Data Preprocessing & Transformation (ETL).
# Takes the Phase 1 CSV and cleans + transforms it for analysis and modeling:
#   1. Handle missing values            (dropna / fillna)
#   2. Remove invalid (blank) images
#   3. Remove duplicate records         (exact and conflicting-label duplicates)
#   4. Type conversions                 (string -> category)
#   5. Feature creation                 (10 shape features computed from pixels)
#   6. Encode categorical variables     (LabelEncoder + OneHotEncoder)
#   7. Normalize / standardize          (pixels / 255, StandardScaler on features)
#
# Output: data/emnist_processed.csv — the cleaned dataset with the new
# features, used by Phases 3, 4 and 5.
#
# WHY THE SAVED CSV IS NOT SCALED:
# Pixel normalization (divide by 255) is a fixed formula, so it can be applied
# anywhere. StandardScaler is different: it LEARNS a mean and standard
# deviation from the data. If we fitted it here on all 50,000 rows, the test
# rows used in Phase 5 would have influenced the scaling (data leakage), and
# test scores would look better than they really are. So this phase
# demonstrates both scalers and reports their effect, but saves UNSCALED
# values; Phase 5 fits the scaler on the training split only.
#
# Run from the project root (after Phase 1) with:
#     python src/phase2_preprocessing.py
# =============================================================================

# --- Step 1: Import the libraries we need ---
import os
import sys

import numpy as np                          # For working with arrays
import pandas as pd                         # For working with tabular data
import matplotlib.pyplot as plt             # For the feature illustration figure
from matplotlib.patches import Rectangle    # To draw bounding boxes on images
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

sys.path.append(os.path.dirname(__file__))  # So Python can find our own modules
from label_map import DIGIT_LABELS, UPPERCASE_LABELS
from phase1_data_summary import CSV_PATH, PIXEL_COLUMNS, load_csv


# =============================================================================
# SETTINGS
# =============================================================================

PROCESSED_CSV_PATH = "data/emnist_processed.csv"
RESULTS_DIR = "results"
FEATURES_PLOT_PATH = "results/phase2_features_explained.png"

# The 10 features created in Step 5, in the order they appear in the CSV.
FEATURE_COLUMNS = [
    "ink_pixel_count",   # how many pixels contain any ink
    "mean_intensity",    # average brightness of the whole image
    "bbox_width",        # width of the smallest box around the ink
    "bbox_height",       # height of that box
    "aspect_ratio",      # bbox_width / bbox_height
    "ink_density",       # share of the bounding box that is inked
    "centroid_x",        # horizontal centre of mass of the ink (0 = left)
    "centroid_y",        # vertical centre of mass of the ink (0 = top)
    "lr_symmetry",       # 1 = left half mirrors right half perfectly
    "tb_symmetry",       # 1 = top half mirrors bottom half perfectly
]


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# =============================================================================
# SECTION 1: CLEANING
# =============================================================================

def handle_missing_values(df):
    """
    Step 1: Handle missing values.

    Strategy (applied even though Phase 1 found none, so the pipeline stays
    safe if the data source ever changes):
      - A row without a label is useless for supervised learning -> dropna().
      - A missing pixel is most likely background -> fillna(0) (black).
    """
    print_header("STEP 1: HANDLE MISSING VALUES")
    before = len(df)
    missing_total = int(df.isnull().sum().sum())
    print(f"  Missing values found: {missing_total}")

    df = df.dropna(subset=["label", "char"])            # rows without a label
    df[PIXEL_COLUMNS] = df[PIXEL_COLUMNS].fillna(0)     # missing pixel = background

    print(f"  Rows dropped (no label)      : {before - len(df)}")
    print(f"  Pixel values filled with 0   : {int(missing_total)}")
    print(f"  Missing values remaining     : {int(df.isnull().sum().sum())}")
    return df


def remove_blank_images(df):
    """
    Step 2: Remove invalid records — images with no ink at all.
    A blank image cannot be a character, and it would also break the shape
    features in Step 5 (a bounding box of nothing has no width or height).
    """
    print_header("STEP 2: REMOVE BLANK (INVALID) IMAGES")
    is_blank = df[PIXEL_COLUMNS].to_numpy().max(axis=1) == 0
    print(f"  Blank images found and removed: {int(is_blank.sum())}")
    return df[~is_blank]


def remove_duplicates(df):
    """
    Step 3: Remove duplicate records.

    Two kinds of duplicates are checked:
      a) EXACT duplicates: same pixels AND same label. Keep the first copy.
      b) CONFLICTING duplicates: same pixels but DIFFERENT labels. These are
         ambiguous (one drawing cannot be two characters), so every copy is
         removed instead of guessing which label is right.
    """
    print_header("STEP 3: REMOVE DUPLICATE RECORDS")
    before = len(df)

    # Comparing 784 columns row by row is slow, so we turn each image into a
    # single hashable value (its raw bytes) and compare those instead.
    df = df.copy()
    df["_pixels_key"] = [row.tobytes() for row in df[PIXEL_COLUMNS].to_numpy()]

    # a) Exact duplicates
    exact = df.duplicated(subset=["_pixels_key", "label"], keep="first")
    print(f"  a) Exact duplicates (same image, same label)      : {int(exact.sum())}")
    df = df[~exact]

    # b) Conflicting duplicates
    labels_per_image = df.groupby("_pixels_key")["label"].transform("nunique")
    conflicting = labels_per_image > 1
    print(f"  b) Conflicting duplicates (same image, diff label): {int(conflicting.sum())}")
    df = df[~conflicting]

    df = df.drop(columns="_pixels_key").reset_index(drop=True)
    print(f"  Rows: {before} -> {len(df)}  ({before - len(df)} removed)")
    return df


# =============================================================================
# SECTION 2: TRANSFORMATION
# =============================================================================

def convert_types(df):
    """
    Step 4: Type conversions.

    'char' is loaded as text, and the new 'group' column (digit / uppercase /
    lowercase) is derived from the label. Both have a small fixed set of
    values, so we convert them to pandas 'category' type: less memory, and it
    tells every later step (and seaborn in Phase 3) that these are categories,
    not free text. (This dataset has no dates, so no string -> datetime
    conversion is needed.)
    """
    print_header("STEP 4: TYPE CONVERSIONS")

    # Derive the character group from the label ranges in label_map.py.
    df["group"] = np.select(
        [df["label"].isin(DIGIT_LABELS), df["label"].isin(UPPERCASE_LABELS)],
        ["digit", "uppercase"],
        default="lowercase",
    )
    print(f"  Created column 'group' from label: {df['group'].value_counts().to_dict()}")

    for col in ["char", "group"]:
        before_type = df[col].dtype
        before_mem = df[col].memory_usage(deep=True)
        df[col] = df[col].astype("category")
        after_mem = df[col].memory_usage(deep=True)
        print(f"  {col:<6}: {str(before_type):<8} -> category   "
              f"(memory {before_mem / 1e3:.0f} KB -> {after_mem / 1e3:.0f} KB)")
    return df


def compute_shape_features(pixels):
    """
    Computes the 10 shape features for every image.

    Args:
        pixels: array of shape (N, 784), values 0-255.
    Returns:
        DataFrame with one column per name in FEATURE_COLUMNS.
    """
    images = pixels.reshape(-1, 28, 28).astype(np.float32)
    ink = images > 0                                  # True where there is any ink
    total_brightness = images.sum(axis=(1, 2))

    # --- Bounding box ---
    # rows_with_ink[i, r] is True if row r of image i contains ink.
    rows_with_ink = ink.any(axis=2)
    cols_with_ink = ink.any(axis=1)
    # argmax returns the FIRST True; searching the reversed array gives the last.
    top    = rows_with_ink.argmax(axis=1)
    bottom = 27 - rows_with_ink[:, ::-1].argmax(axis=1)
    left   = cols_with_ink.argmax(axis=1)
    right  = 27 - cols_with_ink[:, ::-1].argmax(axis=1)
    bbox_height = bottom - top + 1
    bbox_width  = right - left + 1

    # --- Centre of mass (brightness-weighted average position) ---
    positions = np.arange(28)
    centroid_y = (images.sum(axis=2) * positions).sum(axis=1) / total_brightness
    centroid_x = (images.sum(axis=1) * positions).sum(axis=1) / total_brightness

    # --- Symmetry ---
    # Compare the image with its mirror image. If they are identical the
    # difference is 0 and symmetry is 1. Dividing by twice the total
    # brightness keeps the score between 0 and 1.
    lr_diff = np.abs(images - images[:, :, ::-1]).sum(axis=(1, 2))
    tb_diff = np.abs(images - images[:, ::-1, :]).sum(axis=(1, 2))

    ink_pixel_count = ink.sum(axis=(1, 2))
    return pd.DataFrame({
        "ink_pixel_count": ink_pixel_count,
        "mean_intensity":  images.mean(axis=(1, 2)),
        "bbox_width":      bbox_width,
        "bbox_height":     bbox_height,
        "aspect_ratio":    bbox_width / bbox_height,
        "ink_density":     ink_pixel_count / (bbox_width * bbox_height),
        "centroid_x":      centroid_x,
        "centroid_y":      centroid_y,
        "lr_symmetry":     1 - lr_diff / (2 * total_brightness),
        "tb_symmetry":     1 - tb_diff / (2 * total_brightness),
    })


def create_features(df):
    """
    Step 5: Feature creation.

    The raw data is 784 separate pixel values; none of them alone means much.
    These features summarize the SHAPE of each character in a few numbers a
    human can interpret — e.g. '1' and 'l' are tall and narrow (low
    aspect_ratio), 'O' and '0' are highly symmetric, '.'-like strokes have low
    ink counts. They are explored in Phase 3 and tested statistically in Phase 4.
    """
    print_header("STEP 5: FEATURE CREATION")
    features = compute_shape_features(df[PIXEL_COLUMNS].to_numpy())
    df = pd.concat([df, features], axis=1)

    print(f"  Created {len(FEATURE_COLUMNS)} new features: {FEATURE_COLUMNS}")
    print("\n  Summary of the new features:")
    print(df[FEATURE_COLUMNS].describe().T.round(3).to_string())

    # Interpretable sanity check: average aspect ratio per group.
    print("\n  Example — mean aspect_ratio (width / height) per character:")
    for ch in ["1", "l", "I", "0", "O", "m", "w"]:
        print(f"    '{ch}': {df.loc[df['char'] == ch, 'aspect_ratio'].mean():.3f}")
    return df


def encode_categoricals(df):
    """
    Step 6: Encode categorical variables.

    - LabelEncoder on 'char' (the target): turns each character into an
      integer, which is what scikit-learn classifiers expect for the target.
      LabelEncoder sorts the values, and in ASCII order '0'-'9' < 'A'-'Z' <
      'a'-'z' — the same order as EMNIST's own labels — so the result should
      equal the existing 'label' column exactly. We check that.
    - OneHotEncoder on 'group' (a nominal feature): there is no order between
      digit / uppercase / lowercase, so giving them the numbers 0/1/2 would
      wrongly suggest one is "bigger" than another. One-hot encoding creates
      one 0/1 column per group instead.
    """
    print_header("STEP 6: ENCODE CATEGORICAL VARIABLES")

    # --- LabelEncoder on the target ---
    label_encoder = LabelEncoder()
    char_encoded = label_encoder.fit_transform(df["char"].astype(str))
    matches = bool((char_encoded == df["label"].to_numpy()).all())
    print("  a) LabelEncoder on 'char' (target):")
    print(f"     Classes learned : {len(label_encoder.classes_)}  "
          f"({''.join(label_encoder.classes_[:12])}...{''.join(label_encoder.classes_[-5:])})")
    print(f"     Example         : 'A' -> {label_encoder.transform(['A'])[0]}, "
          f"'a' -> {label_encoder.transform(['a'])[0]}")
    print(f"     Identical to the EMNIST 'label' column: {matches}")
    if not matches:
        raise ValueError("LabelEncoder order does not match label_map.py — check the data.")

    # --- OneHotEncoder on the group ---
    one_hot = OneHotEncoder(sparse_output=False, dtype=np.uint8)
    group_encoded = one_hot.fit_transform(df[["group"]])
    group_columns = list(one_hot.get_feature_names_out(["group"]))
    df = pd.concat(
        [df, pd.DataFrame(group_encoded, columns=group_columns, index=df.index)],
        axis=1,
    )
    print("\n  b) OneHotEncoder on 'group' (nominal feature):")
    print(f"     New columns: {group_columns}")
    example = df.groupby("group", observed=True)[group_columns].first()
    print("     Encoding of each group:")
    print("     " + example.to_string().replace("\n", "\n     "))
    return df, group_columns


def demonstrate_scaling(df):
    """
    Step 7: Normalize and standardize numerical data.

    - Pixels: NORMALIZED to 0.0-1.0 by dividing by 255 (all pixels share the
      same 0-255 scale, so one fixed formula is right for all of them).
    - Shape features: STANDARDIZED with StandardScaler (mean 0, std 1). They
      are on very different scales (ink_pixel_count is in the hundreds,
      symmetry is between 0 and 1); without this, distance-based models like
      KNN and SVM in Phase 5 would be dominated by the largest-valued feature.

    See the note at the top of the file: the results are shown here but not
    saved, to avoid data leakage.
    """
    print_header("STEP 7: NORMALIZE / STANDARDIZE")

    pixels = df[PIXEL_COLUMNS].to_numpy()
    pixels_normalized = pixels.astype(np.float32) / 255.0
    print("  a) Pixels normalized (x / 255):")
    print(f"     Before: min {pixels.min()}, max {pixels.max()}, mean {pixels.mean():.2f}")
    print(f"     After : min {pixels_normalized.min():.1f}, max {pixels_normalized.max():.1f}, "
          f"mean {pixels_normalized.mean():.3f}")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[FEATURE_COLUMNS])
    comparison = pd.DataFrame({
        "mean before": df[FEATURE_COLUMNS].mean(),
        "std before":  df[FEATURE_COLUMNS].std(),
        "mean after":  features_scaled.mean(axis=0),
        "std after":   features_scaled.std(axis=0),
    }).round(3)
    # -0.0 prints confusingly; it is just rounding of a tiny negative number.
    comparison = comparison + 0.0
    print("\n  b) Shape features standardized (StandardScaler):")
    print("     " + comparison.to_string().replace("\n", "\n     "))
    print("\n  Note: the saved CSV keeps UNSCALED values. Phase 5 fits the scaler on")
    print("        the training split only, so the test set cannot leak into it.")


# =============================================================================
# SECTION 3: ILLUSTRATE THE NEW FEATURES
# =============================================================================

def plot_feature_illustration(df):
    """
    Saves a figure showing a few images with their bounding box (red) and
    centre of mass (cyan dot), plus the feature values — useful in the report
    to explain what the engineered features actually measure.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    examples = ["1", "0", "A", "m", "g", "W"]

    fig, axes = plt.subplots(1, len(examples), figsize=(3 * len(examples), 3.8))
    fig.suptitle("Engineered Shape Features (red = bounding box, cyan = centre of mass)",
                 fontsize=13)

    for ax, ch in zip(axes, examples):
        row = df[df["char"] == ch].iloc[0]
        image = row[PIXEL_COLUMNS].to_numpy(dtype=np.uint8).reshape(28, 28)
        ink_rows = np.where((image > 0).any(axis=1))[0]
        ink_cols = np.where((image > 0).any(axis=0))[0]

        ax.imshow(image, cmap="gray")
        # Rectangle takes the top-left corner; -0.5 aligns it to pixel edges.
        ax.add_patch(Rectangle((ink_cols[0] - 0.5, ink_rows[0] - 0.5),
                               row["bbox_width"], row["bbox_height"],
                               fill=False, edgecolor="red", linewidth=1.5))
        ax.plot(row["centroid_x"], row["centroid_y"], "o", color="cyan", markersize=6)
        ax.set_title(f"'{ch}'\naspect {row['aspect_ratio']:.2f} | ink {row['ink_pixel_count']}\n"
                     f"LR-sym {row['lr_symmetry']:.2f} | TB-sym {row['tb_symmetry']:.2f}",
                     fontsize=9)
        ax.axis("off")

    # rect leaves the top 12% free so the suptitle doesn't overlap the titles.
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(FEATURES_PLOT_PATH, dpi=120)
    plt.close(fig)
    print(f"\n  Saved: {FEATURES_PLOT_PATH}")


# =============================================================================
# SECTION 4: MAIN — RUN THE ETL PIPELINE IN ORDER
# =============================================================================

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"'{CSV_PATH}' not found. Run 'python src/phase1_data_summary.py' first."
        )

    # EXTRACT
    df = load_csv()
    shape_before = df.shape

    # TRANSFORM — cleaning
    df = handle_missing_values(df)
    df = remove_blank_images(df)
    df = remove_duplicates(df)

    # TRANSFORM — new columns, encoding, scaling
    df = convert_types(df)
    df = create_features(df)
    df, group_columns = encode_categoricals(df)
    demonstrate_scaling(df)
    plot_feature_illustration(df)

    # LOAD — save the processed dataset for the next phases
    column_order = ["label", "char", "group"] + group_columns + FEATURE_COLUMNS + PIXEL_COLUMNS
    df = df[column_order]
    df.to_csv(PROCESSED_CSV_PATH, index=False, float_format="%.5g")

    print_header("PHASE 2 SUMMARY (ETL)")
    print(f"  Shape before : {shape_before}")
    print(f"  Shape after  : {df.shape}")
    print(f"  Rows removed : {shape_before[0] - df.shape[0]}  (missing / blank / duplicate)")
    print(f"  Columns added: {df.shape[1] - shape_before[1]}  "
          f"(group + {len(group_columns)} one-hot + {len(FEATURE_COLUMNS)} features)")
    print(f"  Saved        : {PROCESSED_CSV_PATH}  "
          f"({os.path.getsize(PROCESSED_CSV_PATH) / 1e6:.1f} MB)")
    print("\nphase2_preprocessing.py finished. Next: Phase 3 (exploratory data analysis).")
