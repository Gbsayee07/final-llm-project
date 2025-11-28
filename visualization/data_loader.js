// Utility: load JSONL file
async function loadJSONL(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error("File not found: " + path);
    const text = await response.text();
    return text.trim().split("\n").map(line => JSON.parse(line));
}

// Load error_bank.jsonl
async function loadErrorBank() {
    const data = await loadJSONL("/data/error_bank.jsonl");

    let out = "";
    data.forEach((item, i) => {
        out += `#${i+1}\nQuestion: ${item.question}\nWrong: ${item.model_output}\nCorrect: ${item.correct_output}\n\n`;
    });

    document.getElementById("error-bank-output").textContent = out;
}

// Load finetune_data.jsonl
async function loadFinetuneData() {
    const data = await loadJSONL("/data/finetune_data.jsonl");

    let out = "";
    data.forEach((item, i) => {
        out += `#${i+1}\n${item.prompt}${item.completion}\n\n`;
    });

    document.getElementById("finetune-output").textContent = out;
}