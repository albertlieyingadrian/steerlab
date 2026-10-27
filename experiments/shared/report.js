// Light/dark switch, contents rail and heading links for steerlab research notes. Works without it: light theme,
// and headings keep their ids.
(() => {
  const root = document.documentElement;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "theme-toggle";
  const label = () => {
    const dark = root.dataset.theme === "dark";
    button.textContent = dark ? "Light mode" : "Dark mode";
    button.setAttribute("aria-pressed", String(dark));
  };
  button.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem("steerlab-theme", root.dataset.theme); } catch {}
    label();
  });
  label();
  document.body.append(button);
})();

(() => {
  const headings = [...document.querySelectorAll("article h2[id]")].filter((h) => !h.closest(".key"));
  for (const h of document.querySelectorAll("article h2[id], article h3[id]")) {
    if (h.closest(".key")) continue;
    const a = document.createElement("a");
    a.className = "anchor";
    a.href = `#${h.id}`;
    a.setAttribute("aria-label", `Link to this section: ${h.textContent.trim()}`);
    a.textContent = "#";
    h.prepend(a);
  }
  const list = document.querySelector(".toc ol");
  if (!list || !headings.length) return;
  for (const h of headings) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = `#${h.id}`;
    a.textContent = h.dataset.short || h.textContent.replace(/^#/, "").replace(/^\s*\d+\s*/, "").trim();
    li.append(a);
    list.append(li);
  }
  const links = new Map([...list.querySelectorAll("a")].map((a) => [a.getAttribute("href").slice(1), a]));
  const mark = (id) => {
    for (const [key, a] of links) a.setAttribute("aria-current", String(key === id));
  };
  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible.length) mark(visible[0].target.id);
    },
    { rootMargin: "0px 0px -70% 0px" },
  );
  headings.forEach((h) => observer.observe(h));
  mark(headings[0].id);
})();
