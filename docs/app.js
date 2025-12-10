const base = "https://chatbot-rfh0.onrender.com";
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

// Typing animation
function showTyping() {
  const wrap = document.createElement("div");
  wrap.id = "typing";
  wrap.className = "typing";

  wrap.innerHTML = `
    <div class="dot"></div>
    <div class="dot"></div>
    <div class="dot"></div>
  `;

  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function hideTyping() {
  const t = document.getElementById("typing");
  if (t) t.remove();
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  append(text, "user");
  inputEl.value = "";

  // Validate
  showTyping();
  const valRes = await fetch(base + "/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text }),
  });
  hideTyping();

  const valData = await valRes.json();
  const validation = valData.validation;

  if (!validation.is_valid) {
    append("Invalid question: " + validation.reason);
    return;
  }
  append("Validation: Valid question");

  // Refine
  showTyping();
  const refineRes = await fetch(base + "/refine", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: text, feedback: "" }),
  });
  hideTyping();

  const refineData = await refineRes.json();
  const refined = refineData.refined_answer;

  append("Refined: " + refined.revised_question);

  // Similarity
  showTyping();
  const simRes = await fetch(base + "/similarity", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: refined.revised_question }),
  });
  hideTyping();

  const simData = await simRes.json();

  if (simData.similar?.length > 0) {
    append("Similar questions found:");
    simData.similar.forEach((s) =>
      append("- " + s.meta.question + " (score " + s.score.toFixed(2) + ")")
    );
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
