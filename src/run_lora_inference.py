from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import textwrap

BASE = "gpt2"
ADAPTER = "models/lora_adapter"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model…")
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("Loading LoRA adapter…")
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

def generate(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(out[0], skip_special_tokens=True)

    # Extract only what comes after "Correct Answer:"
    if "Correct Answer:" in full_text:
        continuation = full_text.split("Correct Answer:", 1)[1].strip()
    else:
        continuation = full_text.strip()

    # Clean unicode artifacts
    continuation = continuation.lstrip("�•·—- ")

    if not continuation:
        continuation = "[Model produced no continuation]"

    print("\n".join(textwrap.wrap(continuation, width=80)))
    print("=" * 80)

# -------------------------
# TEST CASES
# -------------------------
test_cases = [
    (
        "What is RoPE in transformers?",
        "RoPE transforms a model into a fixed value."
    ),
    (
        "Why do transformers need positional encoding?",
        "Transformers use positional encoding to optimize memory usage."
    ),
    (
        "How does attention behave with very long sequences?",
        "Attention does not associate data to memory regions."
    ),
]

instruction = (
    "Correct the following incorrect model answer. "
    "Write a short, accurate explanation (1–2 sentences) based only on true ML knowledge.\n\n"
)

print("\nRunning LoRA-enhanced model…\n")

for q, wrong in test_cases:
    prompt = (
        "Correct the following incorrect model answer.\n\n"
        f"Question: {q}\n"
        f"Incorrect Answer: {wrong}\n"
        "Correct Answer:"
    )
    generate(prompt)