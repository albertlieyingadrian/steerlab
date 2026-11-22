/* The Model room: what is loaded, its shape, and which lenses, catalog and SAEs steerlab has for it. */
let alive = false;

const fmt = (n) => Number(n).toLocaleString("en");
function short(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  return fmt(n);
}

export async function mount(root, api) {
  alive = true;
  const { el } = api;
  const panel = el("section", { class: "panel", "aria-labelledby": "h-model" }, el("h2", { id: "h-model", text: "Model" }),
    el("p", { class: "empty", text: "loading…" }));
  root.append(panel);
  let info, cached;
  try {
    [info, cached] = await Promise.all([api.get("model/info"), api.get("model/cached").catch(() => ({ cached: [] }))]);
  } catch (error) {
    panel.lastChild.replaceWith(el("p", { class: "err", text: error.message }));
    return;
  }
  if (!alive) return;

  const fact = (k, v, sub) => el("div", { class: "fact" }, el("div", { class: "k", text: k }), el("div", { class: "v" }, v, sub ? el("small", { text: ` ${sub}` }) : null));
  const gqa = info.n_kv_heads !== info.n_heads ? `${info.n_kv_heads} key/value` : "";
  panel.lastChild.replaceWith(
    el("p", { class: "sub" }, el("code", { text: info.model }), ` · ${info.family} · ${info.dtype} on ${info.device}`),
    el("div", { class: "facts" },
      fact("layers", info.n_layers),
      fact("heads / layer", info.n_heads, gqa),
      fact("neurons / layer", fmt(info.d_mlp)),
      fact("MLP", info.mlp, info.poles === 2 ? "two poles" : ""),
      fact("units", short(info.units), `${short(info.neurons)} neurons × ${info.poles} + ${fmt(info.heads_total)} heads`),
      fact("d_model", fmt(info.d_model), `d_head ${info.d_head}`),
      fact("vocab", fmt(info.vocab), info.tied_embeddings ? "tied" : ""),
      fact("parameters", short(info.params))),
  );

  const lensRows = Object.entries(info.lenses).map(([name, v]) => el("tr", {},
    el("td", { class: "id", text: name === "logit" ? "logit" : `${name}-lens` }),
    el("td", {}, el("span", { class: `badge ${v.status === "ready" ? "ready" : v.status === "downloadable" ? "control" : ""}`, text: v.status })),
    el("td", { class: "words", text: v.source || v.note || "" })));
  const lenses = el("section", { class: "panel", "aria-labelledby": "h-lenses" },
    el("h2", { id: "h-lenses", text: "Lenses" }),
    el("p", { class: "hint", text: "Read a vector as words. The logit lens uses the model's own unembedding; the J, R and S lenses are fitted Jacobian lenses that also read early layers and the input side." }),
    el("table", {}, el("thead", {}, el("tr", {}, el("th", { text: "lens" }), el("th", { text: "status" }), el("th", { text: "source" }))), el("tbody", {}, lensRows)),
    el("p", { class: "note" }, el("a", { href: "/lens", text: "Open the Lens room" })));

  const cat = info.catalog;
  const catalog = el("section", { class: "panel", "aria-labelledby": "h-catalog" },
    el("h2", { id: "h-catalog", text: "Neuron catalog" }),
    el("p", { class: "sub" }, el("span", { class: `badge ${cat.covered ? "ready" : ""}`, text: cat.covered ? "covered" : "not covered" }),
      ` ${cat.source}: labels, top contexts and token drivers for every neuron pole and head.`),
    cat.covered ? null : el("p", { class: "hint", text: `Covered models: ${cat.models.join(", ")}. Others are computed on demand in the Units room.` }));

  const saes = el("section", { class: "panel", "aria-labelledby": "h-saes" }, el("h2", { id: "h-saes", text: "Sparse autoencoders" }),
    info.saes.length
      ? el("table", {}, el("thead", {}, el("tr", {}, el("th", { text: "release" }), el("th", { text: "kind" }), el("th", { class: "num", text: "layers" }))),
        el("tbody", {}, info.saes.map((s) => el("tr", {}, el("td", { class: "id", text: s.release }), el("td", { text: s.kind }), el("td", { class: "num", text: s.layers.join(", ") })))))
      : el("p", { class: "empty", text: "steerlab has no SAE for this model (Qwen-Scope: Qwen3-1.7B, Qwen3-8B; Gemma Scope: Gemma 2 2B/9B, Gemma 3 1B; Goodfire: Llama-3.1-8B)." }));

  const switcher = el("section", { class: "panel", "aria-labelledby": "h-switch" }, el("h2", { id: "h-switch", text: "Switch model" }),
    el("p", { class: "hint", text: "Every room shares the loaded model, so switching means a restart:" }),
    el("pre", { class: "cmd", text: "steerlab inspect --model <id>" }),
    cached.cached.length ? el("details", {}, el("summary", { text: `${cached.cached.length} models in the local Hugging Face cache` }),
      el("ul", { class: "cached" }, cached.cached.map((id) => el("li", {}, el("code", { text: id }), id === info.model ? " (loaded)" : "")))) : null);

  const rooms = el("section", { class: "panel", "aria-labelledby": "h-rooms" }, el("h2", { id: "h-rooms", text: "Rooms" }),
    el("table", {}, el("tbody", {}, info.rooms.map((r) => el("tr", {},
      el("td", {}, el("a", { href: r.id === "trace" ? "/" : `/${r.id}`, text: r.title })),
      el("td", { class: "num" }, el("kbd", { text: "g" }), el("kbd", { text: r.key })))))));

  root.append(el("div", { class: "model-grid" }, lenses, catalog, saes, rooms, switcher));
}

export function unmount() { alive = false; }
