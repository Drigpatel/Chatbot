const base = "http://localhost:8000";
const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");

function append(msg, who = "bot") {
  const d = document.createElement("div");
  d.className = "msg " + who;
  d.innerText = msg;
  chatEl.appendChild(d);
  chatEl.scrollTop = chatEl.scrollHeight;
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  append(text, "user");
  inputEl.value = "";

  // Step 1: Validate input (semantic correctness)
  append("Validating question...");
  const validationRes = await fetch(base + "/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text }),
  });
  const validationData = await validationRes.json();

  const validation = validationData.validation;
  chatEl.lastChild.innerText = validation.is_valid
    ? "Validation: Valid question"
    : "Invalid: " + validation.reason;

  if (!validation.is_valid) {
    append("Try rewriting your question and send again.");
    return;
  }

  // Step 2: Refine input
  append("Refining question...");
  const refineRes = await fetch(base + "/refine", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: text, feedback: "" }),
  });
  const refineData = await refineRes.json();

  const refined = refineData.refined_answer;
  append("Refined: " + refined.revised_question);

  // Step 3: Similarity check
  append("Checking for similar questions...");
  const simRes = await fetch(base + "/similarity", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: refined.revised_question }),
  });
  const simData = await simRes.json();

  if (simData.similar && simData.similar.length > 0) {
    append("Similar questions found:");
    simData.similar.forEach((item) => {
      append(
        "- " + item.question + " (Match Score: " + item.score.toFixed(2) + ")"
      );
    });
  } else {
    append("No similar questions found.");
  }

  append("Would you like to refine further? Provide feedback or type ACCEPT.");
}

sendBtn.addEventListener("click", sendMessage);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
