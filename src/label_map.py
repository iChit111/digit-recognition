# =============================================================================
# label_map.py
# Location: src/label_map.py
#
# Purpose: Single source of truth for the EMNIST ByClass 62-class label
# mapping. Person 2 (model.py) and Person 3 (evaluate.py / demo.py) should
# import from HERE rather than hand-rolling their own digit/letter ordering.
#
# WHY THIS FILE EXISTS (read this before you assume A-Z-a-z alphabetical order):
# EMNIST ByClass does NOT use a simple "0-9, A-Z, a-z" ordering by accident of
# fate — it uses that ordering ON PURPOSE, but it comes from NIST's official
# byclass mapping file (emnist-byclass-mapping.txt), not from someone eyeballing
# the alphabet. In this particular split, it happens to line up with the
# intuitive ordering:
#
#   Labels  0-9  : digits '0'-'9'
#   Labels 10-35 : uppercase 'A'-'Z'
#   Labels 36-61 : lowercase 'a'-'z'
#
# This is documented in the original EMNIST paper (Cohen et al., 2017,
# "EMNIST: an extension of MNIST to handwritten letters", arXiv:1702.05373)
# and matches the label ordering used by tensorflow-datasets' emnist/byclass
# config. We hardcode it here explicitly, keyed off Python's own string
# module, rather than parsing a mapping file at runtime — one less thing that
# can silently fail if a download path changes.
#
# DO NOT reorder this. If you're ever unsure whether it's still correct for
# a given TFDS version, decode a few known digit images (e.g. index 0 in the
# 'train' split is well documented to be a '5' in most EMNIST/MNIST-lineage
# datasets) and visually confirm against LABEL_TO_CHAR before trusting it.
# =============================================================================

import string

NUM_CLASSES = 62

# --- Build the ordered list of 62 characters -------------------------------
# '0'-'9' (10) + 'A'-'Z' (26) + 'a'-'z' (26) = 62
_CHARACTERS = list(string.digits) + list(string.ascii_uppercase) + list(string.ascii_lowercase)

assert len(_CHARACTERS) == NUM_CLASSES, (
    f"Expected {NUM_CLASSES} characters, built {len(_CHARACTERS)}. "
    f"Something is very wrong with this file — do not proceed."
)

# --- Public lookup tables ----------------------------------------------------

# label_map.LABEL_TO_CHAR[7]  -> '7'
# label_map.LABEL_TO_CHAR[10] -> 'A'
# label_map.LABEL_TO_CHAR[36] -> 'a'
LABEL_TO_CHAR = {i: ch for i, ch in enumerate(_CHARACTERS)}

# label_map.CHAR_TO_LABEL['A'] -> 10
CHAR_TO_LABEL = {ch: i for i, ch in enumerate(_CHARACTERS)}

# Convenience groupings, useful for per-group accuracy breakdowns in evaluate.py
DIGIT_LABELS     = list(range(0, 10))   # 0-9
UPPERCASE_LABELS = list(range(10, 36))  # 10-35
LOWERCASE_LABELS = list(range(36, 62))  # 36-61


def label_to_char(label):
    """Convert an integer class label (0-61) to its character ('0'-'9', 'A'-'Z', 'a'-'z')."""
    return LABEL_TO_CHAR[int(label)]


def char_to_label(char):
    """Convert a character to its integer class label (0-61)."""
    return CHAR_TO_LABEL[char]


def get_class_names():
    """
    Returns the full ordered list of 62 class names as strings, index-aligned
    with the integer labels (index 0 -> '0', index 10 -> 'A', index 36 -> 'a').
    Useful for sklearn's classification_report(target_names=...) in evaluate.py.
    """
    return list(_CHARACTERS)


if __name__ == "__main__":
    # Quick self-check when run directly: python src/label_map.py
    print(f"Total classes: {NUM_CLASSES}")
    print(f"Digits (0-9)      : {[LABEL_TO_CHAR[i] for i in DIGIT_LABELS]}")
    print(f"Uppercase (10-35) : {[LABEL_TO_CHAR[i] for i in UPPERCASE_LABELS]}")
    print(f"Lowercase (36-61) : {[LABEL_TO_CHAR[i] for i in LOWERCASE_LABELS]}")
    print(f"\nSanity checks:")
    print(f"  label_to_char(0)  = {label_to_char(0)!r}  (expect '0')")
    print(f"  label_to_char(10) = {label_to_char(10)!r}  (expect 'A')")
    print(f"  label_to_char(36) = {label_to_char(36)!r}  (expect 'a')")
    print(f"  char_to_label('Z') = {char_to_label('Z')}  (expect 35)")
    print(f"  char_to_label('z') = {char_to_label('z')}  (expect 61)")