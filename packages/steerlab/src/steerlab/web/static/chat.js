// The visitor chat: one conversation per tab (its id lives in sessionStorage), replies streamed over POST.
// Text from the model or the visitor is always inserted with textContent.
"use strict";

const messages = document.getElementById("messages");
const scroller = document.getElementById("scroller");
const input = document.getElementById("input");
const send = document.getElementById("send");

function conversationId() {
  let id = null;
  try { id = sessionStorage.getItem("conversation"); } catch (_) { /* storage blocked */ }
  if (!id) {
    // crypto.randomUUID exists only on secure pages (https or localhost); a visitor on the LAN opens plain http.
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    id = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("").replace(/(.{8})(.{4})(.{4})(.{4})/, "$1-$2-$3-$4-");
    try { sessionStorage.setItem("conversation", id); } catch (_) { /* a reload starts over */ }
  }
  return id;
}

let conversation = conversationId();
let version = null;
let busy = false;
let available = true;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollDown() { scroller.scrollTop = scroller.scrollHeight; }

function notice(text, kind) {
  messages.appendChild(el("div", "notice" + (kind ? " " + kind : ""), text));
  scrollDown();
}

function render(turns) {
  messages.replaceChildren(el("div", "notice", "New conversation"));
  for (const turn of turns) messages.appendChild(el("div", turn.role === "user" ? "u" : "a", turn.content));
  scrollDown();
}

function setEnabled() {
  const on = available && !busy;
  input.disabled = !available;
  send.disabled = !on;
}

async function loadTranscript() {
  const response = await fetch(`/api/conversation/${encodeURIComponent(conversation)}`);
  if (!response.ok) return;
  const body = await response.json();
  version = body.version;
  render(body.messages);
}

// One SSE stream per message: read `event:`/`data:` blocks from the POST response body.
async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const data = block.split("\n").filter((line) => line.startsWith("data: ")).map((line) => line.slice(6)).join("\n");
      if (data) yield JSON.parse(data);
    }
  }
}

async function sendMessage(text) {
  busy = true;
  setEnabled();
  const bubble = el("div", "u", text);
  messages.appendChild(bubble);
  const waiting = el("div", "writing", "Sending…");
  messages.appendChild(waiting);
  scrollDown();
  let reply = null;
  let caret = null;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation, text }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      bubble.remove();
      input.value = text;
      notice(typeof body.detail === "string" ? body.detail : "That message could not be sent.", "error");
      return;
    }
    for await (const event of readEvents(response)) {
      if (event.type === "queued") {
        waiting.textContent = event.position > 1 ? `You're number ${event.position} in line…` : "Assistant is writing…";
      } else if (event.type === "start") {
        waiting.textContent = "Assistant is writing…";
        reply = el("div", "a");
        caret = el("span", "caret");
        reply.appendChild(caret);
        messages.insertBefore(reply, waiting);
      } else if (event.type === "token" && reply) {
        reply.insertBefore(document.createTextNode(event.text), caret);
        if (scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80) scrollDown();
      } else if (event.type === "error") {
        notice(event.message, "error");
      } else if (event.type === "end") {
        if (caret) caret.remove();
        if (event.reason === "reset") {
          render([]);
          notice("Conversation reset");
        } else if (event.reason === "full" || event.reason === "failed") {
          if (reply) reply.remove();
          bubble.remove(); // the server dropped this turn, so the transcript matches what the model remembers
        }
      }
    }
  } catch (_) {
    notice("The connection dropped. Try again.", "error");
  } finally {
    waiting.remove();
    busy = false;
    setEnabled();
    const status = await fetch(`/api/status?conversation=${encodeURIComponent(conversation)}`).then((r) => r.json()).catch(() => null);
    if (status) version = status.version;
    input.focus();
  }
}

async function poll() {
  if (busy) return;
  const status = await fetch(`/api/status?conversation=${encodeURIComponent(conversation)}`).then((r) => r.json()).catch(() => null);
  if (!status) return;
  const wasAvailable = available;
  available = status.status === "ready";
  if (!available && wasAvailable) notice("The assistant is restarting");
  if (available && !wasAvailable) notice("The assistant is back");
  if (version !== null && status.version !== version) {
    const had = messages.querySelectorAll(".u, .a").length;
    await loadTranscript();
    if (had && !messages.querySelectorAll(".u, .a").length) notice("Conversation reset");
  }
  version = status.version;
  setEnabled();
}

document.getElementById("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || busy || !available) return;
  input.value = "";
  input.style.height = "";
  sendMessage(text);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    document.getElementById("composer").requestSubmit();
  }
});
input.addEventListener("input", () => {
  input.style.height = "";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
});

document.getElementById("reset").addEventListener("click", async () => {
  if (busy) return;
  await fetch("/api/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation }),
  });
  render([]);
  const status = await fetch(`/api/status?conversation=${encodeURIComponent(conversation)}`).then((r) => r.json()).catch(() => null);
  if (status) version = status.version;
  input.focus();
});

loadTranscript().then(poll);
setInterval(poll, 2000);
setEnabled();
