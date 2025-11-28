import json

ERROR_BANK = "data/error_bank.jsonl"
OUTPUT_FILE = "data/finetune_data.jsonl"

def load_errors():
    errors = []
    with open(ERROR_BANK, "r") as f:
        for line in f:
            item = json.loads(line)

            # Use correct keys from your dataset
            q = item["question"]
            wrong = item["model_output"]
            correct = item["correct_output"]

            # Build training pair
            prompt = (
                "Correct the following incorrect model answer.\n\n"
                f"Question: {q}\n"
                f"Incorrect Answer: {wrong}\n"
                f"Correct Answer:"
            )

            errors.append({
                "prompt": prompt,
                "completion": " " + correct  # leading space helps training
            })
    return errors


def main():
    errors = load_errors()

    with open(OUTPUT_FILE, "w") as f:
        for ex in errors:
            f.write(json.dumps(ex) + "\n")

    print(f"Created {OUTPUT_FILE} with {len(errors)} examples.")


if __name__ == "__main__":
    main()