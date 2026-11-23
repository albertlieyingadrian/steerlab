// The filterable hack gallery. Reads window.HACKS (data.js, built by make_report.py) and renders one card per entry
// into #gallery. Without JavaScript the static matrix (Table 2) and the entries.jsonl link still carry everything.
(function () {
  const data = window.HACKS;
  const root = document.getElementById("gallery");
  if (!data || !root) return;
  root.classList.add("is-live");
  const staticList = root.querySelector(".hg-static-list");
  if (staticList) staticList.remove(); // the no-JavaScript list; its ids move to the live cards
  const list = root.querySelector(".hg-list");
  const count = root.querySelector(".hg-count");
  const f = {
    kind: root.querySelector("#hg-kind"),
    caught: root.querySelector("#hg-caught"),
    cheap: root.querySelector("#hg-cheap"),
    q: root.querySelector("#hg-q"),
  };
  const DET = ["loop", "copy", "length", "ledger"];
  const open = new Set(location.hash.startsWith("#hack-") ? [location.hash.slice(6)] : []);
  const WORD = { full: "caught", partial: "partly", miss: "missed", "n/a": "does not apply" };
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  const kinds = [...new Set(data.entries.map((e) => e.kind))];
  kinds.forEach((k) => f.kind.insertAdjacentHTML("beforeend", `<option value="${esc(k)}">${esc(k)}</option>`));
  // reward-type chips: one pressed at a time ("all" clears); counts are entries of that type
  const RT = ["taste/judge", "style proxy", "verifiable task", "selection rule", "monitor/probe"];
  const chips = root.querySelector(".hg-rt-chips");
  let rt = "";
  const chip = (v, label, n) => `<button type="button" data-rt="${esc(v)}" aria-pressed="${v === rt}">${esc(label)}<span class="n">${n}</span></button>`;
  chips.innerHTML = chip("", "all types", data.entries.length) +
    RT.map((t) => chip(t, t, data.entries.filter((e) => e.reward_type === t).length)).join("");
  function setRt(v) {
    rt = v;
    chips.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.rt === v)));
  }
  chips.addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    setRt(b.dataset.rt === rt ? "" : b.dataset.rt);
    render();
  });
  const checks = [...new Set(data.entries.map((e) => e.lab_check))].sort();
  checks.forEach((k) => f.caught.insertAdjacentHTML("beforeend", `<option value="${esc(k)}">${esc(k)}</option>`));

  function card(e) {
    // keep a card open across re-renders once a reader opens it
    // (the toggle listener below records it)
    const dets = e.detectors
      ? `<ul class="hg-dets" aria-label="Cheap checks">${DET.map((n) => {
          const d = e.detectors[n];
          const rate = d.hack_rate != null ? ` <span class="rate">${d.hack_rate.toFixed(2)} / ${d.clean_rate.toFixed(2)}</span>` : "";
          const star = d.shared_with_label ? ' <span class="small" title="the hack label was made by a rule of the same kind as this check">*</span>' : "";
          return `<li class="v-${d.verdict.replace("/", "")}"><span class="n">${n}</span> ${WORD[d.verdict]}${star}${rate}</li>`;
        }).join("")}</ul>`
      : `<p class="hg-none">No artefacts: ${esc(e.why_no_artefact)}</p>`;
    const samples = e.artefacts
      ? `<details class="hg-samples"><summary>Samples (${e.artefacts.n_hack} hack, ${e.artefacts.n_clean} clean rows)</summary>
           <p class="lbl">Hack</p><pre>${esc(e.artefacts.hack_sample)}</pre>
           <p class="lbl">Clean</p><pre>${esc(e.artefacts.clean_sample)}</pre>
           <p class="small">Samples from: ${esc(e.artefacts.from.join(", "))}. Groups: ${esc(e.artefacts.label_source)}. ${e.artefacts.file ? `File: <a href="../${esc(e.artefacts.file)}">${esc(e.artefacts.file)}</a>` : esc(e.artefacts.private)}</p>
         </details>`
      : "";
    const badge = e.detectors ? `<span class="hg-badge b-${e.best}">${{ full: "cheap check caught it", partial: "partly caught", miss: "no cheap check caught it" }[e.best]}</span>` : `<span class="hg-badge b-none">no samples</span>`;
    return `<li class="hg-card" id="hack-${esc(e.id)}"><details${open.has(e.id) ? " open" : ""}>
      <summary><span class="hg-tag"><span class="hg-rt">${esc(e.reward_type)}</span> · ${esc(e.kind)} · ${esc(e.domain)}</span><span class="hg-title">${esc(e.title)}</span>${badge}</summary>
      <dl>
        <dt>Reward or ruler</dt><dd>${esc(e.reward)}</dd>
        <dt>Reward type</dt><dd>${esc(e.reward_type)}${e.reward_type_note ? ` <span class="small">(${esc(e.reward_type_note)})</span>` : ""}</dd>
        <dt>What it did</dt><dd>${esc(e.hack)}</dd>
        <dt>Evidence</dt><dd>“${esc(e.quote)}” <a href="${esc(e.href)}">source note ↗</a></dd>
        <dt>Caught by</dt><dd>${esc(e.caught_by)} <span class="small">(${esc(e.when)})</span></dd>
        <dt>Missed by</dt><dd>${esc((e.missed_by || []).join("; "))}</dd>
        ${e.notes ? `<dt>Also</dt><dd>${esc(e.notes)}</dd>` : ""}
      </dl>
      ${dets}${samples}
    </details></li>`;
  }

  function haystack(e) {
    const a = e.artefacts || {};
    return [e.title, e.domain, e.kind, e.reward_type, e.reward_type_note, e.reward, e.hack, e.quote, e.caught_by, e.lab_check, e.when, e.notes,
      (e.missed_by || []).join(" "), e.why_no_artefact, a.hack_sample, a.clean_sample, a.label_source]
      .filter(Boolean).join(" ").toLowerCase();
  }

  function render() {
    const q = f.q.value.trim().toLowerCase();
    const shown = data.entries.filter((e) => {
      if (rt && e.reward_type !== rt) return false;
      if (f.kind.value && e.kind !== f.kind.value) return false;
      if (f.caught.value && e.lab_check !== f.caught.value) return false;
      if (f.cheap.value === "art" && !e.detectors) return false;
      if (f.cheap.value && f.cheap.value !== "art" && e.best !== f.cheap.value) return false;
      if (q && !haystack(e).includes(q)) return false;
      return true;
    });
    list.innerHTML = shown.map(card).join("");
    count.textContent = `${shown.length} of ${data.entries.length} entries`;
  }
  Object.values(f).forEach((el) => el.addEventListener("input", render));
  list.addEventListener("toggle", (ev) => {
    const li = ev.target.closest && ev.target.closest(".hg-card");
    if (!li || ev.target.parentElement !== li) return;
    const id = li.id.slice(5);
    if (ev.target.open) open.add(id); else open.delete(id);
  }, true);
  render();
  function openHash() {
    if (!location.hash.startsWith("#hack-")) return;
    const id = location.hash.slice(6);
    open.add(id);
    if (!document.getElementById("hack-" + id) && data.entries.some((e) => e.id === id)) {
      Object.values(f).forEach((el) => (el.value = "")); // the card is filtered out: clear the filters
      setRt("");
      render();
    }
    const el = document.getElementById("hack-" + id);
    if (!el) return;
    el.querySelector("details").open = true;
    el.scrollIntoView();
  }
  window.addEventListener("hashchange", openHash);
  openHash();
  window.addEventListener("load", () => setTimeout(openHash, 400));
})();
