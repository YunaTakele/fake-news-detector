import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load the cleaned dataset you created in Step 5
df = pd.read_csv("../data/processed/liar_binary_clean.csv")
print("Loaded cleaned:", df.shape)

# Stratify keeps the same 0/1 ratio in every split
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,        #make the second output 20% of the data 
    random_state=42,      # suffle the data the same way every time
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,        # split the 20% into 10% val, 10% test
    random_state=42,      # suffle the data the same way every time
    stratify=temp_df["label"]
)

print("Train:", train_df.shape, " label balance:\n", train_df["label"].value_counts(normalize=True))
print("Val:  ", val_df.shape,   " label balance:\n", val_df["label"].value_counts(normalize=True))
print("Test: ", test_df.shape,  " label balance:\n", test_df["label"].value_counts(normalize=True))

# Save splits
out_dir = Path("../data/processed")
out_dir.mkdir(exist_ok=True)

train_df.to_csv(out_dir / "liar_binary_train.csv", index=False)
val_df.to_csv(out_dir / "liar_binary_val.csv", index=False)
test_df.to_csv(out_dir / "liar_binary_test.csv", index=False)

print("\nSaved:")
print(" -", out_dir / "liar_binary_train.csv")
print(" -", out_dir / "liar_binary_val.csv")
print(" -", out_dir / "liar_binary_test.csv")
