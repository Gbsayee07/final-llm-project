import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import json

device = "cpu"   # You can try "mps" or "cuda" if available

# === Load GPT-2 model (student model to score) ===
lm_name = "gpt2"
lm_tok = AutoTokenizer.from_pretrained(lm_name)
lm_tok.pad_token = lm_tok.eos_token

lm_model = AutoModelForCausalLM.from_pretrained(lm_name)
lm_model.to(device)
lm_model.eval()

# === Load trained classifier ===
clf_name = "error_classifier"
clf_tok = AutoTokenizer.from_pretrained(clf_name)
clf_model = AutoModelForSequenceClassification.from_pretrained(clf_name)
clf_model.to(device)
clf_model.eval()

labels = ["incorrect explanation", "incorrect reasoning", "nonsense"]

def classify_error(question, output):
    text = question + " " + output
    enc = clf_tok(text, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        logits = clf_model(**enc).logits
        pred = logits.argmax(dim=-1).item()

    return labels[pred]

def generate_answer(question):
    enc = lm_tok(question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = lm_model.generate(**enc, max_new_tokens=50)
    return lm_tok.decode(out[0], skip_special_tokens=True)

# === Run scoring ===
questions = [
    "What is RoPE in transformers?",
    "Why do Transformers need positional encoding?",
    "How does attention behave with very long sequences?"
]

results = []

for q in questions:
    output = generate_answer(q)
    etype = classify_error(q, output)

    print("\nQ:", q)
    print("Model Output:", output)
    print("Detected Error:", etype)

    results.append({
        "question": q,
        "model_output": output,
        "error_type": etype
    })

# Save results
with open("aeb_scored.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print("\nSaved AEB scores to aeb_scored.jsonl")