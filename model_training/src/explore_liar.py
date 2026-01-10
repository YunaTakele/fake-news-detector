import pandas as pd #import the pandas library for data manipulation and analysis

# Load the training split (this is the main one)
df = pd.read_csv("../data/raw/liar_train.csv")

print("SHAPE (rows, columns):")
print(df.shape)

print("\nCOLUMNS:")
print(list(df.columns))

print("\nLABEL DISTRIBUTION:") #this is where you check how many labels there are and how many count for each label 
print(df["label"].value_counts())

print("\nSAMPLE ROWS (statement + label):")
print(df[["statement", "label"]].head(8).to_string(index=False)) #print the first 8 rows of the statement and label columns 
