import pandas as pd
from pathlib import Path

# ----------------------------
# 1. Load raw dataset
# ----------------------------
raw_path = Path("../data/raw/liar_train.csv")
df = pd.read_csv(raw_path)

print("Original shape:", df.shape)

# ----------------------------
# 2. Keep only what we need
# ----------------------------
df = df[["statement", "label"]].copy()

# ----------------------------
# 3. Map labels to binary
# ----------------------------
# Original LIAR labels:
# 0 = false
# 1 = true
# 2 = barely-true
# 3 = half-true
# 4 = mostly-true
# 5 = pants-fire

label_map = {
    0: 0,  # false → FAKE
    5: 0,  # pants-fire → FAKE
    1: 1,  # true → REAL
    4: 1   # mostly-true → REAL
    # 2 and 3 are intentionally excluded
}

df["label"] = df["label"].map(label_map)

# ----------------------------
# 4. Drop ambiguous rows
# ----------------------------
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int) # Ensures the label column is integer type

# ----------------------------
# 5. Basic text cleanup
# ----------------------------
df["statement"] = df["statement"].astype(str).str.strip() # Ensure all statemetns are strings and strips whitespaces 
df = df[df["statement"] != ""]

# ----------------------------
# 6. Remove duplicates (optional but good)
# ----------------------------
df = df.drop_duplicates(subset=["statement"])

print("Cleaned shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())

# ----------------------------
# 7. Save processed dataset
# ----------------------------
output_dir = Path("../data/processed")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "liar_binary_clean.csv"
df.rename(columns={"statement": "text"}).to_csv(output_path, index=False)

print(f"\nSaved clean dataset to: {output_path}")
