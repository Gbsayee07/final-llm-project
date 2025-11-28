import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

MODEL_NAME = "gpt2"
DATA_FILE = "data/finetune_data.jsonl"

print("Loading dataset…")
dataset = load_dataset("json", data_files=DATA_FILE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------------
# tokenize() merges prompt+completion
# and creates labels = input_ids
# -------------------------------
def tokenize(batch):
    texts = [p + c for p, c in zip(batch["prompt"], batch["completion"])]
    tok = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tok["labels"] = tok["input_ids"].copy()
    return tok

tokenized = dataset.map(tokenize, batched=True)

print("Loading GPT-2 model…")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
)

print("Applying LoRA adapters…")
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_fc", "c_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_cfg)

training_args = TrainingArguments(
    output_dir="models/lora_adapter",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
)

print("Training LoRA model…")
trainer.train()

print("Saving adapter…")
model.save_pretrained("models/lora_adapter")

print("Done!")