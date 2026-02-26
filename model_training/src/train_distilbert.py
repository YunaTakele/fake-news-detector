"""
train_distilbert.py
-------------------
Fine-tunes DistilBERT on the binary LIAR dataset for fake news detection.

Output:
    model_training/models/distilbert-fakenews/   <- trained model + tokenizer
    (plug this folder into your FastAPI backend)

Run from: model_training/src/
    python train_distilbert.py
"""

# ── 0. Imports ─────────────────────────────────────────────────────────────────
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,                          # AutoTokenizer avoids the chat-template 404 bug
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

# ── 1. Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("../data/processed")
MODEL_OUT = Path("../models/distilbert-fakenews")
MODEL_OUT.mkdir(parents=True, exist_ok=True)

# ── 2. Load the CSV splits you already created ──────────────────────────────────
print("📂  Loading dataset splits...")
dataset = load_dataset("csv", data_files={
    "train":      str(DATA_DIR / "liar_binary_train.csv"),
    "validation": str(DATA_DIR / "liar_binary_val.csv"),
    "test":       str(DATA_DIR / "liar_binary_test.csv"),
})
print(dataset)

# ── 3. Tokenizer ─────────────────────────────────────────────────────────────────
# Using AutoTokenizer fixes the "additional_chat_templates 404" crash you hit
# with DistilBertTokenizerFast.from_pretrained() in newer transformers versions.
print("\n🔤  Loading tokenizer...")
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) #from_pretrained() goes out to HuggingFace and downloads the tokenizer files for distilbert-base-uncased

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",   # pad all sequences to the same length
        truncation=True,        # cut sequences longer than max_length
        max_length=256,         # 256 tokens is enough for short news headlines
    )

print("🔄  Tokenizing dataset (this takes ~30 seconds)...")
tokenized = dataset.map(tokenize, batched=True)

# Rename + reformat so PyTorch Trainer understands the data
tokenized = tokenized.remove_columns(["text"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

print("✅  Tokenization done:")
print(tokenized)

# ── 4. Model ──────────────────────────────────────────────────────────────────────
# num_labels=2  →  binary classifier (0 = FAKE, 1 = REAL)
print("\n🤖  Loading DistilBERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

# ── 5. Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    """Called by Trainer after every eval step to report accuracy + F1."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# ── 6. Training Arguments ─────────────────────────────────────────────────────────
# These settings are tuned for a CPU/MPS (Mac) machine.
# If you have a GPU (or use Colab), you can increase batch size to 32 and reduce epochs.
training_args = TrainingArguments(
    output_dir=str(MODEL_OUT / "checkpoints"),  # where to save checkpoints
    num_train_epochs=3,                          # 3 epochs is standard for fine-tuning
    per_device_train_batch_size=16,              # lower if you get out-of-memory errors
    per_device_eval_batch_size=32,
    learning_rate=2e-5,                          # standard fine-tuning LR for BERT models
    weight_decay=0.01,                           # helps prevent overfitting
    warmup_ratio=0.1,                            # gradually ramp up LR for first 10% of steps
    eval_strategy="epoch",                       # evaluate at the end of every epoch
    save_strategy="epoch",                       # save checkpoint every epoch
    load_best_model_at_end=True,                 # keep the best checkpoint
    metric_for_best_model="f1",
    logging_steps=50,                            # print loss every 50 steps
    report_to="none",                            # disable wandb / tensorboard logging
)

# ── 7. Trainer ────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
)

# ── 8. Train ──────────────────────────────────────────────────────────────────────
print("\n🚀  Starting training...")
print("    (~15-30 min on CPU | ~3-5 min on GPU/MPS)")
print("    You'll see loss printed every 50 steps.\n")
trainer.train()

# ── 9. Evaluate on Test Set ───────────────────────────────────────────────────────
print("\n📊  Evaluating on held-out test set...")
results = trainer.evaluate(eval_dataset=tokenized["test"])
print(f"\n✅  Test Results:")
print(f"    Accuracy : {results['eval_accuracy']:.4f}")
print(f"    F1 Score : {results['eval_f1']:.4f}")

# ── 10. Save Final Model + Tokenizer ──────────────────────────────────────────────
# This saves everything you need to load the model in FastAPI later.
print(f"\n💾  Saving model to: {MODEL_OUT}")
trainer.save_model(str(MODEL_OUT))
tokenizer.save_pretrained(str(MODEL_OUT))

print(f"""
🎉  Done! Your trained model is saved at:
    {MODEL_OUT.resolve()}

    It contains:
    - config.json          (model architecture)
    - model.safetensors    (trained weights)
    - tokenizer files      (vocab + tokenizer config)

    To load it in FastAPI later:
        from transformers import pipeline
        classifier = pipeline("text-classification", model="{MODEL_OUT.resolve()}")
        result = classifier("Breaking: vaccines cause autism")
        # → [{{'label': 'LABEL_0', 'score': 0.94}}]  (LABEL_0 = FAKE, LABEL_1 = REAL)
""")
