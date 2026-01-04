# Error-Aware LoRA Fine-Tuning of GPT-2 for Automated Answer Correction

This project fine-tunes a **GPT-2 base** model using **LoRA (Low-Rank Adaptation)** to perform **error correction**.  
We build an **Error Bank** (question + incorrect answer + corrected answer), convert it into a fine-tuning dataset, and provide a **Flask-based web visualization** to explore the dataset and test the model interactively.

## Project Highlights
- Error Bank creation (incorrect → corrected pairs)
- LoRA adapter fine-tuning on GPT-2 base
- Web dashboard:
  - Error Bank overview
  - Fine-tune dataset preview
  - Saved checkpoints viewer
  - Error type visualization
  - Prompt length histogram
  - Interactive LoRA model tester

## Results (Summary)
- In-domain full correction accuracy improved from **41% → 57%**
- Out-of-domain improvement was small (**18% → 22%**)
- Common failure modes: repetition, circular explanations, superficial rewrites

## Repository Structure
- `visualization/` - Flask server + frontend UI
- `data/` - dataset files (error bank + fine-tune dataset)
- `models/` - LoRA adapter files (do not include large base model weights)
- `report/` - final report

## Setup (Local)
### 1) Create + activate virtual environment
```bash
python3 -m venv aeb_env
source aeb_env/bin/activate
