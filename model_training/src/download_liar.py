from datasets import load_dataset
from pathlib import Path

def main():
    # This is where raw dataset files will live
    output_dir = Path("../data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ask Hugging Face for the LIAR dataset
    dataset = load_dataset("liar")

    # Save each split (train / validation / test) as CSV
    for split in dataset:
        df = dataset[split].to_pandas()
        file_path = output_dir / f"liar_{split}.csv"
        df.to_csv(file_path, index=False)
        print(f"Saved {split} -> {file_path} | rows={len(df)}")

if __name__ == "__main__":
    main()
