import json, os

os.makedirs("data", exist_ok=True)

examples = [
    {
        "question": "What is RoPE in transformers?",
        "model_output": "RoPE is a method of storing and manipulating data for a system of transformers.",
        "correct_output": "RoPE rotates query/key vectors using sinusoidal frequencies so attention depends on relative positions and generalizes to long sequences.",
        "error_type": ["incorrect explanation", "hallucination"]
    },
    {
        "question": "Why do Transformers need positional encoding?",
        "model_output": "Because they're not real.",
        "correct_output": "Transformers process tokens in parallel and need positional encoding to track word order and structure.",
        "error_type": ["nonsense", "failure to answer"]
    },
    {
        "question": "How does attention behave with very long sequences?",
        "model_output": "The sequence repeats many times.",
        "correct_output": "Attention weakens for distant tokens and computation grows quadratically, making long-range context harder.",
        "error_type": ["incorrect reasoning"]
    }
]

with open("data/error_bank.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print("Created data/error_bank.jsonl successfully.")