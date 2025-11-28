from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading GPT-2…")

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float32,   # Mac CPU = fp32
    device_map=None
)

prompt = "What is Rotary Position Embedding (RoPE) in transformers? Answer in 2 sentences.\nA:"

inputs = tok(prompt, return_tensors="pt")

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False
    )

print("\nMODEL OUTPUT:")
print(tok.decode(out[0], skip_special_tokens=True))