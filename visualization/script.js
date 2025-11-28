// ===============================
// Real Backend Model Execution
// ===============================
async function runModel() {
    const prompt = document.getElementById("prompt-input").value.trim();
    const output = document.getElementById("model-output");

    if (!prompt) {
        output.textContent = "Please enter a prompt.";
        return;
    }

    output.textContent = "Running your trained LoRA model...\nPlease wait...";

    try {
        const response = await fetch("/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt })
        });

        const data = await response.json();
        output.textContent = data.response;
    }
    catch (err) {
        output.textContent = "❌ Error connecting to backend.\n" + err;
    }
}


// ===============================
// JSONL file loaders (unchanged)
// ===============================
async function loadJSONL(path) {
    const response = await fetch(path);
    const text = await response.text();
    return text.trim().split("\n").map(line => JSON.parse(line));
}

async function loadErrorBank() {
    const data = await loadJSONL("data/error_bank.jsonl");

    let out = "";
    data.forEach((item, i) => {
        out += `#${i+1}\nQuestion: ${item.question}\nWrong: ${item.model_output}\nCorrect: ${item.correct_output}\n\n`;
    });

    document.getElementById("error-bank-output").textContent = out;
}

async function loadFinetuneData() {
    const data = await loadJSONL("data/finetune_data.jsonl");

    let out = "";
    data.forEach((item, i) => {
        out += `#${i+1}\n${item.prompt}${item.completion}\n\n`;
    });

    document.getElementById("finetune-output").textContent = out;
}

// ------------------------------------------
// CHART 1: ERROR TYPE DISTRIBUTION
// ------------------------------------------
async function drawErrorTypeChart() {
    const data = await loadJSONL("/data/error_bank.jsonl");

    const counts = {};

    data.forEach(item => {
        let types = item.error_type;

        // If missing, skip
        if (!types) return;

        // If it's a string → convert to array
        if (typeof types === "string") {
            types = [types];
        }

        // If it's not an array → skip
        if (!Array.isArray(types)) return;

        // Count each error type
        types.forEach(t => {
            counts[t] = (counts[t] || 0) + 1;
        });
    });

    const labels = Object.keys(counts);
    const values = Object.values(counts);

    const canvas = document.getElementById("error-type-chart");
    const ctx = canvas.getContext("2d");

    if (window.errorChart) {
        window.errorChart.destroy();
    }

    window.errorChart = new Chart(ctx, {
        type: "pie",
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: [
                    "#4e79a7", "#f28e2b", "#e15759",
                    "#76b7b2", "#59a14f", "#edc948"
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: "bottom"
                }
            }
        }
    });
}


// ------------------------------------------
// CHART 2: PROMPT LENGTH HISTOGRAM
// ------------------------------------------
async function drawPromptHistogram() {
    const data = await loadJSONL("/data/finetune_data.jsonl");

    const lengths = data.map(item =>
        (item.prompt + item.completion).length
    );

    const ctx = document.getElementById("prompt-length-chart");

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: lengths.map((_, i) => i),
            datasets: [{
                label: "Prompt Length",
                data: lengths,
                backgroundColor: "#36a2eb"
            }]
        },
        options: {
            scales: {
                x: { display: false },
                y: { beginAtZero: true }
            }
        }
    });
}

async function listCheckpoints() {
    const response = await fetch("/list_checkpoints");
    const data = await response.json();

    document.getElementById("checkpoint-output").textContent =
        data.files.join("\n");
}