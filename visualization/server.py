from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# =====================
# FLASK APP
# =====================

app = Flask(__name__, static_folder="static")

# =====================
# LOAD MODEL
# =====================

print("Loading model...")

BASE = "gpt2"
ADAPTER = "../models/lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float32,
    device_map="cpu"
)

model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

# =====================
# FRONTEND
# =====================

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

# =====================
# DATA FILES
# =====================

@app.route("/data/<path:filename>")
def serve_data(filename):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return send_from_directory(data_dir, filename)

# =====================
# API: RUN MODEL
# =====================

@app.route("/run", methods=["POST"])
def run_model():
    data = request.json
    prompt = data.get("prompt", "")

    if not prompt.strip():
        return jsonify({"response": "(No input provided)"})

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({"response": generated})

# =====================
# CHECKPOINT LIST
# =====================

@app.route("/list_checkpoints")
def list_checkpoints():
    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "models/lora_adapter")
    files = []

    for root, _, filenames in os.walk(ckpt_dir):
        for f in filenames:
            files.append(os.path.join(root, f).replace(ckpt_dir + "/", ""))

    return jsonify({"files": files})

# =====================
# RUN SERVER
# =====================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)