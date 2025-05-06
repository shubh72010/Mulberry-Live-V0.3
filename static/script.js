const form = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const chatBox = document.getElementById("chat-box");
const typingIndicator = document.getElementById("typing-indicator");
const micBtn = document.getElementById("mic-btn");

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
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });

    const data = await response.json();
    const reply = data.response || "No response.";
    addMessage(reply, "ai");

    // Text-to-speech
    const speech = new SpeechSynthesisUtterance(reply);
    speech.lang = "en-US";
    window.speechSynthesis.speak(speech);

  } catch (err) {
    console.error("Error talking to AI:", err);
    addMessage("Failed to get AI response.", "ai");
  } finally {
    typingIndicator.style.display = "none";
  }
});

// Speech-to-text (Mic input)
if ("webkitSpeechRecognition" in window) {
  const recognition = new webkitSpeechRecognition();
  recognition.lang = "en-US";
  recognition.continuous = false;
  recognition.interimResults = false;

  micBtn.addEventListener("click", () => {
    recognition.start();
    micBtn.classList.add("listening");
  });

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    userInput.value = transcript;
    micBtn.classList.remove("listening");
  };

  recognition.onerror = (event) => {
    console.error("Speech recognition error:", event.error);
    micBtn.classList.remove("listening");
  };

  recognition.onend = () => {
    micBtn.classList.remove("listening");
  };
} else {
  micBtn.disabled = true;
  micBtn.title = "Speech recognition not supported in this browser.";
}