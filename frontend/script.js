const form = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const chatBox = document.getElementById("chat-box");
const typingIndicator = document.getElementById("typing-indicator");

function addMessage(message, className) {
  const div = document.createElement("div");
  div.className = "message " + className;
  div.textContent = message;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = userInput.value.trim();
  if (!message) return;

  addMessage(message, "user");
  userInput.value = "";
  typingIndicator.style.display = "block";

  try {
    const response = await fetch("https://your-render-url.onrender.com/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });

    const data = await response.json();
    addMessage(data.reply || "No response", "ai");
  } catch (err) {
    addMessage("Failed to get AI response.", "ai");
  } finally {
    typingIndicator.style.display = "none";
  }
});