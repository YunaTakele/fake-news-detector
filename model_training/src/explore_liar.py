import pandas as pd

# Load the training split (this is the main one)
df = pd.read_csv("../data/raw/liar_train.csv")

print("SHAPE (rows, columns):")
print(df.shape)

print("\nCOLUMNS:")
print(list(df.columns))

print("\nLABEL DISTRIBUTION:")
print(df["label"].value_counts())

print("\nSAMPLE ROWS (statement + label):")
print(df[["statement", "label"]].head(8).to_string(index=False))
