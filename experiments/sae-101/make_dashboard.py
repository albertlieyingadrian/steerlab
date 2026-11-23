"""
Build the feature dashboard: one static HTML page (dashboard/index.html) with, for each latent, its automatic label
(if step 5c ran) and how well that label checked out, how often it fires, a histogram of its activations, the tokens
its decoder direction pushes up and down, and its top 20 snippets with every token shaded by the latent's activation.

The data is embedded in the page, so it works from a file, on GitHub Pages, or anywhere else, with no server.

    .venv/bin/python experiments/sae-101/make_dashboard.py --size full
"""

import argparse
import html
import json

import sae101 as s

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=list(s.SIZES), default="full")
ap.add_argument("--repo", default="albertlieadrian/qwen2.5-0.5b-sae-101")
args = ap.parse_args()
res = s.RESULTS / args.size
it = s.read_json(res / "interpret.json")
al = s.read_json(res / "autolabels.json") or {"latents": [], "summary": None}
labels = {r["id"]: r for r in al["latents"]}

rows = []
for lat in it["latents"]:
    lab = labels.get(lat["id"], {})
    rows.append(
        {
            "id": lat["id"],
            "chosen": lat.get("chosen"),
            "freq": lat["freq"],
            "max": lat["max"],
            "label": lab.get("label"),
            "acc": lab.get("accuracy"),
            "acc_wrong": lab.get("accuracy_wrong_label"),
            "up": lat["logit_up"][:8],
            "down": lat["logit_down"][:8],
            "hist": lat["hist"],
            "top": [{"t": sn["tokens"], "a": sn["acts"], "at": sn["at"], "v": sn["value"]} for sn in lat["top"]],
        }
    )
summary = al["summary"]
note = (
    f"Labels by {html.escape(summary['model'])}, checked on held-out snippets: mean accuracy "
    f"{summary['accuracy']:.2f} with the latent's own label against {summary['control']:.2f} with another latent's."
    if summary
    else "No automatic labels (step 5c was not run)."
)

page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAE 101 feature dashboard</title>
<meta name="description" content="Top snippets, logit lens and activation histograms for latents of a TopK SAE on Qwen2.5-0.5B's residual stream.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,650;12..96,700&family=JetBrains+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<script>try{if(localStorage.getItem("steerlab-theme")==="dark")document.documentElement.dataset.theme="dark"}catch(e){}</script>
<style>
:root{--canvas:#fff;--ink:#1d272a;--soft:#586468;--faint:#8a9497;--rule:#e3e7e8;--panel:#f5f6f4;--accent:#c9561c;
--shade:201,86,28;--mono:"JetBrains Mono",ui-monospace,"SF Mono",Menlo,monospace;--sans:"Source Serif 4",Georgia,serif;
--display:"Bricolage Grotesque","Avenir Next","Segoe UI",sans-serif}
:root[data-theme="dark"]{--canvas:#121516;--ink:#e8ecec;--soft:#a7b1b3;--faint:#7b8689;--rule:#2a3133;--panel:#1b2022;--accent:#e07a3f;--shade:224,122,63}
*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink);font:16px/1.55 var(--sans)}
header{padding:20px 16px 12px;border-bottom:1px solid var(--rule);position:relative}h1{font:700 22px/1.2 var(--display);margin:0 0 4px}
h2{font-family:var(--display)}.theme{position:absolute;top:18px;right:16px;font:12px var(--mono);border:1px solid var(--rule);background:none;color:var(--soft);border-radius:999px;padding:3px 10px;cursor:pointer}
header p{margin:0;color:var(--soft);font-size:13px;max-width:70rem}a{color:inherit}
.wrap{display:grid;grid-template-columns:260px 1fr}
nav{border-right:1px solid var(--rule);overflow-y:auto;max-height:calc(100vh - 20px);position:sticky;top:0}
nav button{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid var(--rule);background:none;color:inherit;
padding:8px 14px;font:13px/1.35 var(--sans);cursor:pointer}nav button:hover{background:var(--panel)}
nav button[aria-current="true"]{background:var(--panel);box-shadow:inset 3px 0 0 var(--accent)}
nav .id{font-family:var(--mono);font-size:12px;color:var(--soft)}nav .lab{display:block}
main{padding:18px 20px 40px;min-width:0}h2{font-size:18px;margin:0 0 4px}.sub{color:var(--soft);font-size:13px;margin:0 0 14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-bottom:16px}
.card{border:1px solid var(--rule);border-radius:6px;padding:10px 12px}.card h3{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--soft);margin:0 0 8px}
.chip{display:inline-block;font-family:var(--mono);font-size:12px;background:var(--panel);border-radius:4px;padding:1px 6px;margin:2px 3px 2px 0;white-space:pre}
.snip{font-family:var(--mono);font-size:12.5px;line-height:1.9;padding:4px 0;border-bottom:1px solid var(--rule);overflow-wrap:anywhere}
.snip span{border-radius:2px;white-space:pre-wrap}.snip .at{outline:1.5px solid var(--accent);font-weight:600}.snip .v{color:var(--soft);font-size:11px;margin-right:8px}
.hist{max-width:320px}.caveat{font-size:12px;color:var(--soft);margin:8px 0 0}
.tag{font-size:11px;color:var(--accent);font-weight:600}
@media (max-width:720px){.wrap{grid-template-columns:1fr}nav{max-height:220px;position:static;border-right:0;border-bottom:1px solid var(--rule)}main{padding:14px 16px 32px}}
</style>
</head>
<body>
<header>
<h1>SAE 101 feature dashboard</h1>
<button class="theme" type="button" id="theme">Dark mode</button>
<p>TopK SAE on Qwen2.5-0.5B, residual stream after block 11 (<a href="https://huggingface.co/__REPO__">__REPO__</a>).
Latents drawn at random (seed 0) from those firing on 1 in 100,000 to 1 in 10 tokens, plus a few chosen for a concept in
section 4.5 of the note (marked). __NOTE__ <a href="../report/">Back to the tutorial</a>.</p>
</header>
<div class="wrap"><nav id="list" aria-label="Latents"></nav><main id="detail"></main></div>
<script>
const DATA = __DATA__;
const list = document.getElementById("list"), detail = document.getElementById("detail");
const esc = s => s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const vis = t => t.replace(/\\n/g, "⏎");
function hist(h){
  const w = 240, ht = 70, n = h.counts.length, m = Math.max(...h.counts, 1);
  const bars = h.counts.map((c,i)=>`<rect x="${i*w/n+1}" y="${ht-ht*c/m}" width="${w/n-2}" height="${ht*c/m}" fill="var(--accent)"></rect>`).join("");
  return `<svg class="hist" viewBox="0 0 ${w} ${ht+16}" width="100%" role="img" aria-label="histogram of activations">${bars}
  <text x="0" y="${ht+13}" font-size="10" fill="var(--faint)">${h.edges[0]}</text>
  <text x="${w}" y="${ht+13}" font-size="10" fill="var(--faint)" text-anchor="end">${h.edges[h.edges.length-1]}</text></svg>`;
}
function show(i){
  const d = DATA[i];
  [...list.children].forEach((b,j)=>b.setAttribute("aria-current", j===i));
  const lab = d.label ? esc(d.label) : "(no automatic label)";
  const check = d.acc != null ? `label check: ${Math.round(d.acc*10)}/10 with this label, ${Math.round(d.acc_wrong*10)}/10 with another latent's` : "";
  const snips = d.top.map(s => {
    const m = Math.max(...s.a, 1e-6);
    return `<div class="snip"><span class="v">${s.v.toFixed(2)}</span>` + s.t.map((t,j)=>
      `<span class="${j===s.at?"at":""}" style="background:rgba(var(--shade),${(0.55*Math.max(0,s.a[j])/m).toFixed(2)})" title="${s.a[j]}">${esc(vis(t))}</span>`).join("") + `</div>`;
  }).join("");
  detail.innerHTML = `<h2>Latent #${d.id}: ${lab}</h2>
  <p class="sub">${d.chosen ? `<span class="tag">${esc(d.chosen)}</span> · ` : ""}fires on ${(100*d.freq).toPrecision(2)}% of tokens · strongest activation ${d.max.toFixed(2)}${check ? " · " + check : ""}</p>
  <div class="grid">
   <div class="card"><h3>Activation when it fires</h3>${hist(d.hist)}</div>
   <div class="card"><h3>Pushes these next tokens up</h3>${d.up.map(t=>`<span class="chip">${esc(vis(t))}</span>`).join("")}
   <h3 style="margin-top:10px">and these down</h3>${d.down.map(t=>`<span class="chip">${esc(vis(t))}</span>`).join("")}
   <p class="caveat">Logit lens: mostly noise this far from the output (block 11 of 24); see section 4.4 of the note.</p></div>
  </div>
  <div class="card"><h3>Top 20 snippets (shade = activation; outlined = the top token)</h3>${snips}</div>`;
  history.replaceState(null, "", "#" + d.id);
}
DATA.forEach((d,i)=>{
  const b = document.createElement("button");
  b.innerHTML = `<span class="id">#${d.id}${d.chosen?" · chosen":""}</span><span class="lab">${d.label?esc(d.label):"latent "+d.id}</span>`;
  b.onclick = () => show(i);
  list.appendChild(b);
});
const themeBtn = document.getElementById("theme");
function setTheme(dark){ if(dark) document.documentElement.dataset.theme="dark"; else delete document.documentElement.dataset.theme;
  themeBtn.textContent = dark ? "Light mode" : "Dark mode"; try{localStorage.setItem("steerlab-theme", dark?"dark":"light")}catch(e){} }
themeBtn.onclick = () => setTheme(document.documentElement.dataset.theme !== "dark");
themeBtn.textContent = document.documentElement.dataset.theme === "dark" ? "Light mode" : "Dark mode";
const start = DATA.findIndex(d => "#" + d.id === location.hash);
show(start >= 0 ? start : 0);
</script>
</body>
</html>
"""
page = (
    page.replace("__DATA__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/"))
    .replace("__REPO__", html.escape(args.repo))
    .replace("__NOTE__", note)
)
out = s.HERE / "dashboard" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(page)
print(
    f"1. Wrote dashboard/index.html: {len(rows)} latents ({sum(1 for r in rows if r['chosen'])} chosen for a concept), "
    f"{out.stat().st_size / 1e6:.1f} MB"
)
print("\nSelf-checks:")
s.check("every drawn latent has 20 snippets", all(len(r["top"]) == 20 for r in rows if not r["chosen"]))
s.check(f"the page is under 5 MB ({out.stat().st_size / 1e6:.1f} MB)", out.stat().st_size < 5e6)
