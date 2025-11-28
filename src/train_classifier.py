import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# ==== 1. LOAD ERROR BANK ====
examples = []
with open("data/error_bank.jsonl") as f:
    for line in f:
        examples.append(json.loads(line))

# Pick FIRST label for now
for ex in examples:
    ex["label"] = ex["error_type"][0]

# Numeric label mapping
label_list = sorted(set(ex["label"] for ex in examples))
label_to_id = {lbl: i for i, lbl in enumerate(label_list)}
num_labels = len(label_list)

for ex in examples:
    ex["label_id"] = label_to_id[ex["label"]]

print("Labels:", label_to_id)

# ==== 2. CREATE DATASET ====
dataset = Dataset.from_list(examples)

# Keep raw text for preprocessing
raw_dataset = dataset

# ==== 3. TOKENIZER ====
model_name = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(model_name)

def preprocess(batch):
    text = batch["question"] + " " + batch["model_output"]
    out = tok(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    out["labels"] = batch["label_id"]
    return out

dataset = raw_dataset.map(preprocess)

# IMPORTANT: remove string fields that break Trainer
dataset = dataset.remove_columns(["label", "error_type", "question", "model_output", "correct_output"])

# Train/test split
dataset = dataset.train_test_split(test_size=0.3)

# ==== 4. CLASSIFIER ====
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
)

# ==== 5. TRAINING SETTINGS ====
args = TrainingArguments(
    output_dir="clf_output",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=1,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tok,
)

# ==== 6. TRAIN ====
trainer.train()

# ==== 7. SAVE ====
trainer.save_model("error_classifier")
tok.save_pretrained("error_classifier")

print("\n🎉 Classifier training complete!")