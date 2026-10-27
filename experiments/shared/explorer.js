// Reply explorer: the model's real reply at each steering strength, read from a results file. Nothing is generated
// in the browser. Usage: mountExplorer(container, "../results/explorer.json")
export async function mountExplorer(container, url) {
  let data;
  try {
    data = await (await fetch(url)).json();
  } catch {
    container.querySelector(".explorer-fallback")?.removeAttribute("hidden");
    return;
  }
  const questions = [...new Set(data.replies.map((r) => r.question))];
  const strengths = data.strengths;
  let question = questions[0];
  let index = strengths.indexOf(1.75) >= 0 ? strengths.indexOf(1.75) : Math.floor(strengths.length / 2);

  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const root = el("div", "explorer");
  const tabs = el("div", "explorer-tabs");
  tabs.setAttribute("role", "group");
  tabs.setAttribute("aria-label", "Question");
  const buttons = questions.map((q) => {
    const b = el("button", "explorer-tab", q);
    b.type = "button";
    b.addEventListener("click", () => { question = q; render(); });
    tabs.append(b);
    return b;
  });
  const controls = el("div", "explorer-controls");
  const label = el("label", "explorer-label");
  const labelText = el("span", "", "Steering strength");
  const slider = el("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = String(strengths.length - 1);
  slider.step = "1";
  slider.value = String(index);
  slider.addEventListener("input", () => { index = Number(slider.value); render(); });
  const value = el("output", "explorer-value");
  label.append(labelText, slider, value);
  controls.append(label);
  const chat = el("div", "chat");
  const sys = el("div", "turn system");
  sys.append(el("span", "role", "System"), el("p", "msg", `English only (prompt ${data.prompt})`));
  const user = el("div", "turn user");
  const userMsg = el("p", "msg");
  user.append(el("span", "role", "User"), userMsg);
  const model = el("div", "turn model");
  const modelMsg = el("p", "msg");
  model.append(el("span", "role", "Model"), modelMsg);
  chat.append(sys, user, model);
  const stats = el("p", "explorer-stats");
  root.append(tabs, controls, chat, stats);
  container.prepend(root);

  const cjk = /[㐀-鿿　-〿＀-￯]+/g;
  function render() {
    const strength = strengths[index];
    const row = data.replies.find((r) => r.question === question && r.strength === strength);
    buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.textContent === question)));
    value.textContent = strength === 0 ? "off" : `×${strength}`;
    userMsg.textContent = question;
    modelMsg.replaceChildren();
    const who = el("span", "who", data.model.split("/").pop());
    who.append(el("span", `tag ${strength ? "steered" : "plain"}`, strength ? `steered · ×${strength}, layer ${data.layer}` : "no steering"));
    modelMsg.append(who);
    let last = 0;
    for (const match of row.text.matchAll(cjk)) {
      modelMsg.append(document.createTextNode(row.text.slice(last, match.index)));
      modelMsg.append(el("em", "", match[0]));
      last = match.index + match[0].length;
    }
    modelMsg.append(document.createTextNode(row.text.slice(last)));
    stats.textContent = `Chinese share ${Math.round(row.cjk * 100)}% · ${row.fluent ? "fluent" : "not fluent"}` +
      `${row.loops ? " · loops" : ""} · ${data.max_new_tokens} tokens, greedy`;
  }
  render();
}
