from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tok("Test the environment.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=20)

print(tok.decode(out[0], skip_special_tokens=True))