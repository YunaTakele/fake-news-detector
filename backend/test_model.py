from transformers import pipeline

def main():
    MODEL = "jy46604790/Fake-News-Bert-Detect"
    clf = pipeline("text-classification", model=MODEL, tokenizer=MODEL)

    samples = [
        "The president of Mars announced a new law today.",
        "Scientists confirm that drinking water is essential for life."
    ]

    for text in samples:
        result = clf(text)[0]
        print("TEXT:", text)
        print("PREDICTION:", result)
        print("-" * 40)

if __name__ == "__main__":
    main()
