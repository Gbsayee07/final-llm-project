function runExample(id) {
    let question, wrong, outputBox;

    if (id === 1) {
        question = "What is RoPE in transformers?";
        wrong = "RoPE transforms a model into a fixed value.";
        outputBox = document.getElementById("out1");
    }
    if (id === 2) {
        question = "Why do transformers need positional encoding?";
        wrong = "It improves memory usage.";
        outputBox = document.getElementById("out2");
    }
    if (id === 3) {
        question = "How does attention behave with long sequences?";
        wrong = "Attention does not associate data to memory regions.";
        outputBox = document.getElementById("out3");
    }

    outputBox.textContent = "Running…";

    fetch("http://localhost:5000/run", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ question: question, wrong: wrong })
    })
    .then(res => res.json())
    .then(data => {
        outputBox.textContent = data.output;
    })
    .catch(err => {
        outputBox.textContent = "Error: " + err;
    });
}