/* steerlab inspect: the shell. Renders the room nav from /api/rooms, routes the URL to a room and mounts it.
 *
 * URL = /<room>/<path...>?<params>#<hash>; "/" is the Trace room. Each room is an ES module static/rooms/<id>.js:
 *   export function mount(el, api, route)   draw into el (an empty <div> the shell owns)
 *   export function unmount()               stop timers and listeners (the shell then removes el)
 *   export function update(route)           optional: a new deep link inside the same room, without a remount
 * route = { room, path: ["qwen3-8b", "L8", "n344+"], params: URLSearchParams, hash, url }.
 * api   = { get(path), post(path, body), navigate(url, {replace}), href(room, ...path), el, show, model, rooms }
 *         get/post take "lens/run" (or "/api/lens/run") and return parsed JSON, throwing Error(detail) on a non-2xx.
 * Any same-origin <a href="/units/..."> inside the page navigates in place (no reload); modified clicks open as usual.
 * Keys: g then a room's key (g t Trace, g l Lens, ...); ? lists them. Rooms' own keys must ignore events when
 * document.documentElement.dataset.room is not theirs.
 * Model text must be inserted with textContent (api.el), never as HTML.
 */
const $ = (id) => document.getElementById(id);
const root = document.documentElement;

function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return node;
}

let statusTimer = null;
function show(message, bad = false) {
  const box = $("status");
  box.textContent = message;
  box.classList.toggle("bad", bad);
  box.classList.add("on");
  clearTimeout(statusTimer);
  statusTimer = setTimeout(() => box.classList.remove("on"), bad ? 5000 : 2200);
}

async function call(path, init) {
  const response = await fetch(`api/${String(path).replace(/^\/?(api\/)?/, "")}`, init);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `request failed (${response.status})`);
  }
  return data;
}

const S = { rooms: [], model: "", current: null, module: null, host: null, loading: 0 };

function parse(url = location) {
  const u = new URL(url, location.origin);
  const parts = u.pathname.split("/").filter(Boolean).map(decodeURIComponent);
  return { room: parts[0] || "trace", path: parts.slice(1), params: u.searchParams, hash: u.hash.slice(1), url: u.pathname + u.search + u.hash };
}

const api = {
  get: (path) => call(path),
  post: (path, body) => call(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) }),
  navigate(url, { replace = false } = {}) {
    if (replace) history.replaceState(null, "", url);
    else history.pushState(null, "", url);
    route();
  },
  href: (room, ...path) => "/" + [room, ...path].map((p) => encodeURIComponent(String(p)).replace(/%2B/g, "+")).join("/"),
  el,
  show,
  get model() { return S.model; },
  get rooms() { return S.rooms; },
};

function renderNav() {
  const nav = $("rooms");
  nav.replaceChildren(...S.rooms.map((room) => el("a", {
    href: room.id === "trace" ? "/" : `/${room.id}`, "data-room": room.id, title: `${room.title} (g then ${room.key})`,
  }, room.title, el("kbd", { "aria-hidden": "true", text: room.key }))));
  nav.hidden = false;
}

function markNav(id) {
  for (const a of $("rooms").children) {
    if (a.dataset.room === id) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
}

function ensureCss(room) {
  if (!room.css || document.querySelector(`link[data-room-css="${room.id}"]`)) return;
  document.head.append(el("link", { rel: "stylesheet", href: `static/rooms/${room.id}.css`, "data-room-css": room.id }));
}

async function leave() {
  if (S.module?.unmount) { try { S.module.unmount(); } catch (error) { console.error(error); } }
  S.module = null;
  S.host?.remove();
  S.host = null;
}

async function route() {
  const r = parse();
  const room = S.rooms.find((x) => x.id === r.room);
  if (S.current === r.room && S.module) {
    if (S.module.update) { S.module.update(r); return; }
  }
  if (S.current === r.room && r.room === "trace") return;
  const token = ++S.loading;
  await leave();
  S.current = r.room;
  root.dataset.room = r.room;
  markNav(r.room);
  const trace = r.room === "trace";
  $("room-trace").hidden = !trace;
  $("room-host").hidden = trace;
  if (trace) {
    document.title = "Layer explorer";
    window.dispatchEvent(new Event("steerlab:trace"));
    return;
  }
  const host = el("div", { class: `room room-${r.room}` });
  $("room-host").replaceChildren(host);
  S.host = host;
  if (!room || !room.js) {
    document.title = "Not found · steerlab inspect";
    host.append(el("section", { class: "panel" }, el("h2", { text: room ? room.title : "No such room" }),
      el("p", { class: "note", text: room ? "This room has no page yet." : `There is no "${r.room}" room in this build. Rooms: ${S.rooms.map((x) => x.title).join(", ")}.` })));
    return;
  }
  document.title = `${room.title} · steerlab inspect`;
  ensureCss(room);
  try {
    const module = await import(`./rooms/${room.id}.js`);
    if (token !== S.loading) return;
    S.module = module;
    await module.mount(host, api, r);
  } catch (error) {
    console.error(error);
    host.append(el("p", { class: "err", text: `The ${room.title} room could not start: ${error.message}` }));
  }
}

function wireLinks() {
  document.addEventListener("click", (e) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = e.target.closest?.("a[href]");
    if (!a || a.target || a.hasAttribute("download")) return;
    const u = new URL(a.href, location.href);
    if (u.origin !== location.origin || /^\/(static|api)\//.test(u.pathname)) return;
    if (u.pathname === location.pathname && u.search === location.search && u.hash) return;  // in-page anchor
    e.preventDefault();
    api.navigate(u.pathname + u.search + u.hash);
  });
  window.addEventListener("popstate", route);
}

function wireKeys() {
  let armed = 0;
  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "") || document.activeElement?.isContentEditable;
    if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
    if (armed && Date.now() - armed < 1500) {
      armed = 0;
      const room = S.rooms.find((x) => x.key === e.key.toLowerCase());
      if (room) { e.preventDefault(); e.stopImmediatePropagation(); api.navigate(room.id === "trace" ? "/" : `/${room.id}`); }
      return;
    }
    if (e.key === "g") { armed = Date.now(); return; }
    if (e.key === "?") show(S.rooms.map((x) => `g ${x.key} ${x.title}`).join(" · "));
  }, true);
}

async function init() {
  if (root.dataset.source) return;  // the static export: Trace only, no server
  try {
    const data = await api.get("rooms");
    S.rooms = data.rooms;
    S.model = data.model;
  } catch (error) {
    console.error(error);
    return;
  }
  renderNav();
  if ($("model").textContent === "loading…" && parse().room !== "trace") $("model").textContent = S.model;
  wireLinks();
  wireKeys();
  await route();
}

init();
