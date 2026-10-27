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
  const label = (h) => h.dataset.short || h.textContent.replace(/^#/, "").replace(/^\s*[\dA]+(\.\d+)?\s*/, "").trim();
  const links = new Map();
  const groups = [];  // [{ li, toggle, sub, ids: [h2 id, h3 ids...] }]
  for (const h of headings) {
    const li = document.createElement("li");
    const row = document.createElement("div");
    row.className = "toc-row";
    const a = document.createElement("a");
    a.href = `#${h.id}`;
    a.textContent = label(h);
    links.set(h.id, a);
    // the h3s that belong to this section: every h3 after it, up to the next h2
    const subs = [];
    for (let el = h.nextElementSibling; el && el.tagName !== "H2"; el = el.nextElementSibling) {
      if (el.tagName === "H3" && el.id) subs.push(el);
    }
    const group = { li, toggle: null, sub: null, ids: [h.id, ...subs.map((s) => s.id)] };
    if (subs.length) {
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "toc-toggle";
      toggle.setAttribute("aria-label", `Show the subsections of ${a.textContent}`);
      toggle.setAttribute("aria-expanded", "false");
      toggle.textContent = "▸";
      const sub = document.createElement("ol");
      sub.className = "toc-sub";
      sub.hidden = true;
      for (const s of subs) {
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.href = `#${s.id}`;
        link.textContent = s.dataset.short || s.textContent.replace(/^#/, "").replace(/^\s*\d+\.\d+\s*/, "").trim();
        links.set(s.id, link);
        item.append(link);
        sub.append(item);
      }
      toggle.addEventListener("click", () => setOpen(group, sub.hidden, true));
      group.toggle = toggle;
      group.sub = sub;
      row.append(toggle);
    } else {
      const spacer = document.createElement("span");
      spacer.className = "toc-toggle-space";
      row.append(spacer);
    }
    row.append(a);
    li.append(row);
    if (group.sub) li.append(group.sub);
    list.append(li);
    groups.push(group);
  }
  function setOpen(group, open, byHand = false) {
    if (!group.sub) return;
    group.sub.hidden = !open;
    group.toggle.setAttribute("aria-expanded", String(open));
    group.toggle.textContent = open ? "▾" : "▸";
    if (byHand) group.pinned = open;  // a reader's choice stays until they change it
  }
  const mark = (id) => {
    for (const [key, a] of links) a.setAttribute("aria-current", String(key === id));
    for (const group of groups) {
      const here = group.ids.includes(id);
      if (here) setOpen(group, true);
      else if (!group.pinned) setOpen(group, false);
    }
  };
  const tracked = [...document.querySelectorAll("article h2[id], article h3[id]")].filter((h) => links.has(h.id));
  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible.length) mark(visible[0].target.id);
    },
    { rootMargin: "0px 0px -70% 0px" },
  );
  tracked.forEach((h) => observer.observe(h));
  mark(headings[0].id);
})();

// Example carousels: one example at a time with ← / → and a counter. Without JavaScript every slide shows in order.
(() => {
  for (const carousel of document.querySelectorAll(".carousel")) {
    const slides = [...carousel.querySelectorAll(":scope > .slide")];
    if (slides.length < 2) continue;
    let index = 0;
    const nav = document.createElement("div");
    nav.className = "carousel-nav";
    const prev = document.createElement("button");
    const next = document.createElement("button");
    const counter = document.createElement("span");
    const label = document.createElement("span");
    prev.type = next.type = "button";
    prev.textContent = "←";
    next.textContent = "→";
    prev.setAttribute("aria-label", "Previous example");
    next.setAttribute("aria-label", "Next example");
    counter.className = "count";
    counter.setAttribute("aria-live", "polite");
    label.className = "slide-label";
    nav.append(prev, counter, next, label);
    carousel.append(nav);
    carousel.classList.add("is-live");
    carousel.tabIndex = 0;
    const show = (i) => {
      index = (i + slides.length) % slides.length;
      slides.forEach((slide, j) => { slide.hidden = j !== index; });
      counter.textContent = `${index + 1} / ${slides.length}`;
      label.textContent = slides[index].dataset.label || "";
    };
    prev.addEventListener("click", () => show(index - 1));
    next.addEventListener("click", () => show(index + 1));
    carousel.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft") { show(index - 1); event.preventDefault(); }
      if (event.key === "ArrowRight") { show(index + 1); event.preventDefault(); }
    });
    show(0);
  }
})();

// Demo videos autoplay muted and looped; readers who ask for reduced motion get them paused, with controls.
if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  for (const video of document.querySelectorAll("video[autoplay]")) { video.removeAttribute("autoplay"); video.pause(); }
}

// Every code block gets a Copy button, so readers can paste the clips straight into a terminal or a notebook.
for (const pre of document.querySelectorAll("article pre")) {
  const code = pre.querySelector("code") || pre;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "copy";
  button.textContent = "Copy";
  button.setAttribute("aria-label", "Copy this code");
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(code.innerText.replace(/\n$/, ""));
      button.textContent = "Copied";
    } catch {
      button.textContent = "Select and copy";
    }
    setTimeout(() => { button.textContent = "Copy"; }, 1600);
  });
  pre.classList.add("has-copy");
  pre.append(button);
}
