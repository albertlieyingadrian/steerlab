"""
The templated known-answer set (the template check and the slop known-answer check): four page templates written for this study, each filled with content
for all 12 categories in two variants (6 x 12 x 2 = 144 pages). Pages from one template differ only in words, so a
working index must put them close together and give them a high slop number.

Templates A and B are the dev set (tuning); C and D are held out. E and F were written on 2026-11-21, after the
slop split was approved, in two other generators' markup (Bootstrap, Tailwind), as fresh unseen held-out templates. No third-party content: images are inline SVG shapes.
Writes CACHE/html/tpl_<T>_<category>_<v>.html and CACHE/jobs_tpl.jsonl (file-mode render jobs).
Run: .venv/bin/python experiments/design-index/templates.py
"""

from __future__ import annotations

import json

from common import CACHE, page_id

CONTENT = {
    "finance": [
        (
            "Harbor Credit Union",
            "Banking that stays local",
            ["Free checking", "Low-rate auto loans", "Mobile deposits"],
            "Open an account",
        ),
        (
            "Ledgerly",
            "Money tools for small teams",
            ["Instant invoices", "Expense cards", "Cash-flow forecasts"],
            "Start free",
        ),
    ],
    "health": [
        (
            "Maple Family Clinic",
            "Care for every age",
            ["Same-day visits", "Vaccinations", "Telehealth"],
            "Book a visit",
        ),
        (
            "Brightsmile Dental",
            "Gentle dentistry, clear prices",
            ["Cleanings", "Implants", "Emergency care"],
            "Request an appointment",
        ),
    ],
    "law": [
        (
            "Okafor & Lin LLP",
            "Counsel you can reach",
            ["Immigration", "Family law", "Contracts"],
            "Free consultation",
        ),
        (
            "Westbrook Legal",
            "Injury claims, handled",
            ["No win, no fee", "Medical liaison", "Court representation"],
            "Tell us your case",
        ),
    ],
    "restaurant": [
        (
            "Trattoria Sole",
            "Wood-fired since 1998",
            ["Seasonal menu", "Private dining", "Natural wines"],
            "Reserve a table",
        ),
        (
            "Corner Crumb Bakery",
            "Bread at 6 a.m. daily",
            ["Sourdough", "Pastries", "Custom cakes"],
            "Order ahead",
        ),
    ],
    "pets": [
        (
            "Paws & Co.",
            "Everything your pet needs",
            ["Raw food", "Toys", "Grooming"],
            "Shop now",
        ),
        (
            "Riverside Vets",
            "Kind care for animals",
            ["Check-ups", "Surgery", "Dental"],
            "Book a vet",
        ),
    ],
    "software": [
        (
            "Stackpilot",
            "Ship infrastructure faster",
            ["Preview environments", "Audit logs", "SSO"],
            "Get started",
        ),
        (
            "Taskwise",
            "Projects without the chaos",
            ["Boards", "Timelines", "Automations"],
            "Try it free",
        ),
    ],
    "fashion": [
        (
            "Linen Lane",
            "Slow fashion, made well",
            ["Organic cotton", "Free repairs", "Small batches"],
            "Shop the collection",
        ),
        (
            "Ember Jewellery",
            "Handmade in small runs",
            ["Rings", "Necklaces", "Custom pieces"],
            "Browse pieces",
        ),
    ],
    "nonprofit": [
        (
            "Green Valley Trust",
            "Protecting the river",
            ["Clean-ups", "School programs", "Habitat restoration"],
            "Donate",
        ),
        (
            "Second Chance Rescue",
            "Every dog deserves a home",
            ["Adopt", "Foster", "Volunteer"],
            "Meet the dogs",
        ),
    ],
    "realestate": [
        (
            "Northside Realty",
            "Homes in the neighbourhood",
            ["Buy", "Sell", "Valuations"],
            "Find a home",
        ),
        (
            "Keystone Builders",
            "Custom homes, built right",
            ["Design", "Build", "Warranty"],
            "Plan your build",
        ),
    ],
    "education": [
        (
            "Hillcrest Academy",
            "Small classes, big futures",
            ["K-12", "Arts", "Sports"],
            "Schedule a tour",
        ),
        (
            "Lingua Hub",
            "Speak a new language",
            ["Evening classes", "Private tutors", "Exam prep"],
            "Join a class",
        ),
    ],
    "travel": [
        (
            "The Quay Hotel",
            "Rooms by the harbour",
            ["Sea views", "Breakfast included", "Spa"],
            "Check availability",
        ),
        (
            "Ridge Tours",
            "Small-group hikes",
            ["Day trips", "Multi-day treks", "Private guides"],
            "Book a tour",
        ),
    ],
    "studio": [
        (
            "Form & Field Architects",
            "Buildings that fit their place",
            ["Residential", "Civic", "Interiors"],
            "See our work",
        ),
        (
            "Northlight Photo",
            "Portraits and weddings",
            ["Portraits", "Weddings", "Commercial"],
            "Check dates",
        ),
    ],
}

SVG = (
    '<svg viewBox="0 0 400 260" width="100%" height="100%" preserveAspectRatio="xMidYMid slice">'
    '<rect width="400" height="260" fill="{a}"/><circle cx="300" cy="80" r="60" fill="{b}"/>'
    '<rect x="40" y="150" width="220" height="70" rx="12" fill="{b}" opacity=".6"/></svg>'
)

TEMPLATES = {
    # A: centred hero, three feature cards (the "SaaS landing" shape)
    "A": """<!doctype html><html><head><meta charset="utf-8"><title>{name}</title><style>
body{{margin:0;font-family:Helvetica,Arial,sans-serif;color:#1d2433;background:#fff}}
nav{{display:flex;justify-content:space-between;align-items:center;padding:20px 64px;border-bottom:1px solid #eee}}
nav b{{font-size:20px}} nav a{{margin-left:28px;color:#1d2433;text-decoration:none}}
.hero{{text-align:center;padding:96px 20% 72px}} h1{{font-size:56px;margin:0 0 16px}} p{{font-size:20px;color:#5b6475}}
.btn{{display:inline-block;background:#4f46e5;color:#fff;padding:14px 28px;border-radius:8px;text-decoration:none;margin-top:24px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;padding:0 64px 80px}}
.card{{border:1px solid #e5e7eb;border-radius:12px;padding:28px}} .card h3{{margin:0 0 8px}}
</style></head><body><nav><b>{name}</b><div><a href="#">About</a><a href="#">Services</a><a href="#">Contact</a></div></nav>
<section class="hero"><h1>{tagline}</h1><p>{name} - {f0}, {f1} and {f2}.</p><a class="btn" href="#">{cta}</a></section>
<section class="cards"><div class="card"><h3>{f0}</h3><p>Simple and reliable.</p></div><div class="card"><h3>{f1}</h3>
<p>Built around you.</p></div><div class="card"><h3>{f2}</h3><p>Ready when you are.</p></div></section></body></html>""",
    # B: split hero (text left, picture right), dark band of features
    "B": """<!doctype html><html><head><meta charset="utf-8"><title>{name}</title><style>
body{{margin:0;font-family:Georgia,serif;color:#222;background:#faf7f2}}
header{{display:flex;justify-content:space-between;padding:24px 80px;font-size:15px;letter-spacing:.08em;text-transform:uppercase}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:48px;padding:48px 80px;align-items:center}}
h1{{font-size:52px;line-height:1.1;margin:0}} .lead{{font-size:19px;color:#555;margin:20px 0}}
.btn{{background:#b4532a;color:#fff;padding:12px 26px;border:none;font-size:16px}} .pic{{height:380px;border-radius:4px;overflow:hidden}}
.band{{background:#2b2b2b;color:#f2efe9;display:flex;justify-content:space-around;padding:48px 80px}} .band div{{max-width:240px}}
</style></head><body><header><span>{name}</span><span>Home · About · Visit</span></header>
<section class="split"><div><h1>{tagline}</h1><p class="lead">{f0}. {f1}. {f2}.</p><button class="btn">{cta}</button></div>
<div class="pic">"""
    + SVG.format(a="#e9dccb", b="#b4532a")
    + """</div></section>
<section class="band"><div><h3>{f0}</h3>Since day one.</div><div><h3>{f1}</h3>Done with care.</div><div><h3>{f2}</h3>Just ask.</div></section>
</body></html>""",
    # C: full-bleed dark hero with overlay text, then a two-column list
    "C": """<!doctype html><html><head><meta charset="utf-8"><title>{name}</title><style>
body{{margin:0;font-family:Verdana,sans-serif;background:#0f172a;color:#e2e8f0}}
.top{{position:absolute;top:0;left:0;right:0;display:flex;justify-content:space-between;padding:22px 56px;z-index:2}}
.hero{{position:relative;height:640px;display:flex;align-items:flex-end;padding:0 56px 96px}} .hero .bg{{position:absolute;inset:0}}
.hero .t{{position:relative;max-width:640px}} h1{{font-size:64px;margin:0;color:#fff}}
.btn{{display:inline-block;margin-top:28px;border:2px solid #38bdf8;color:#38bdf8;padding:12px 24px;border-radius:999px}}
.list{{display:grid;grid-template-columns:1fr 1fr;gap:16px 64px;padding:64px 56px}} .list div{{border-top:1px solid #334155;padding-top:16px}}
</style></head><body><div class="top"><strong>{name}</strong><span>Menu</span></div>
<section class="hero"><div class="bg">"""
    + SVG.format(a="#1e293b", b="#38bdf8")
    + """</div><div class="t"><h1>{tagline}</h1>
<span class="btn">{cta}</span></div></section><section class="list"><div>{f0}</div><div>{f1}</div><div>{f2}</div><div>Contact {name}</div></section>
</body></html>""",
    # D: left sidebar navigation, content column with a stat row
    "D": """<!doctype html><html><head><meta charset="utf-8"><title>{name}</title><style>
body{{margin:0;font-family:"Trebuchet MS",sans-serif;display:grid;grid-template-columns:260px 1fr;min-height:100vh;background:#fff}}
aside{{background:#14532d;color:#ecfdf5;padding:40px 28px}} aside div{{margin:14px 0}} aside b{{font-size:22px;display:block;margin-bottom:40px}}
main{{padding:72px 80px}} h1{{font-size:44px;color:#14532d;margin:0 0 12px}} .sub{{font-size:18px;color:#4b5563}}
.stats{{display:flex;gap:40px;margin:48px 0}} .stats div{{font-size:15px;color:#4b5563}} .stats b{{display:block;font-size:36px;color:#111}}
.btn{{background:#facc15;padding:14px 30px;border-radius:2px;font-weight:bold}}
</style></head><body><aside><b>{name}</b><div>{f0}</div><div>{f1}</div><div>{f2}</div><div>Contact</div></aside>
<main><h1>{tagline}</h1><p class="sub">Welcome to {name}.</p><div class="stats"><div><b>12</b>years</div><div><b>4.9</b>rating</div>
<div><b>3k</b>clients</div></div><span class="btn">{cta}</span></main></body></html>""",
    # E (fresh held-out, 2026-11-21): Bootstrap 5 markup from its CDN (navbar, jumbotron, card row, footer)
    "E": """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name}</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet"></head>
<body><nav class="navbar navbar-expand-lg navbar-dark bg-dark"><div class="container"><a class="navbar-brand" href="#">{name}</a>
<ul class="navbar-nav ms-auto"><li class="nav-item"><a class="nav-link" href="#">{f0}</a></li><li class="nav-item"><a class="nav-link" href="#">{f1}</a></li>
<li class="nav-item"><a class="nav-link" href="#">{f2}</a></li><li class="nav-item"><a class="btn btn-warning ms-2" href="#">{cta}</a></li></ul></div></nav>
<main><div class="container py-5"><div class="row align-items-center g-5"><div class="col-lg-6"><h1 class="display-5 fw-bold">{tagline}</h1>
<p class="lead text-muted">{name} offers {f0}, {f1} and {f2}.</p><div class="d-flex gap-2"><a class="btn btn-primary btn-lg" href="#">{cta}</a>
<a class="btn btn-outline-secondary btn-lg" href="#">Learn more</a></div></div><div class="col-lg-6"><div class="ratio ratio-16x9 bg-secondary-subtle rounded-3"></div></div></div></div>
<div class="bg-light py-5"><div class="container"><div class="row row-cols-1 row-cols-md-3 g-4">
<div class="col"><div class="card h-100 shadow-sm"><div class="card-body"><h5 class="card-title">{f0}</h5><p class="card-text">Trusted by our community.</p></div></div></div>
<div class="col"><div class="card h-100 shadow-sm"><div class="card-body"><h5 class="card-title">{f1}</h5><p class="card-text">Clear and simple.</p></div></div></div>
<div class="col"><div class="card h-100 shadow-sm"><div class="card-body"><h5 class="card-title">{f2}</h5><p class="card-text">Here when you need us.</p></div></div></div>
</div></div></div></main><footer class="container py-4 text-muted border-top">&copy; {name}</footer></body></html>""",
    # F (fresh held-out, 2026-11-21): Tailwind from its CDN, nested flex wrappers as page builders emit them
    "F": """<!doctype html><html><head><meta charset="utf-8"><title>{name}</title><script src="https://cdn.tailwindcss.com/3.4.5"></script></head>
<body class="bg-stone-50 text-stone-900"><div class="min-h-screen flex flex-col"><div class="w-full"><div class="max-w-6xl mx-auto px-6">
<div class="flex items-center justify-between h-20"><div class="flex items-center gap-3"><div class="w-9 h-9 rounded-full bg-emerald-600"></div>
<div class="text-lg font-semibold">{name}</div></div><div class="hidden md:flex items-center gap-8 text-sm"><div>{f0}</div><div>{f1}</div><div>{f2}</div></div></div></div></div>
<div class="w-full"><div class="max-w-6xl mx-auto px-6 pt-16 pb-24"><div class="grid md:grid-cols-12 gap-10 items-end"><div class="md:col-span-7">
<div class="text-xs uppercase tracking-widest text-emerald-700">{f0}</div><div class="mt-4 text-6xl font-serif leading-tight">{tagline}</div></div>
<div class="md:col-span-5"><div class="text-stone-600">{name}: {f1}, {f2}.</div><div class="mt-6 flex gap-3"><div class="px-5 py-3 bg-stone-900 text-white rounded-md">{cta}</div>
<div class="px-5 py-3 border border-stone-300 rounded-md">Contact</div></div></div></div>
<div class="mt-16 grid grid-cols-2 md:grid-cols-4 gap-4"><div class="h-40 bg-emerald-100 rounded-xl"></div><div class="h-40 bg-stone-200 rounded-xl"></div>
<div class="h-40 bg-amber-100 rounded-xl"></div><div class="h-40 bg-stone-300 rounded-xl"></div></div></div></div></div></body></html>""",
}


def main():
    out = CACHE / "html"
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for t, tpl in TEMPLATES.items():
        for cat, variants in CONTENT.items():
            for v, (name, tagline, feats, cta) in enumerate(variants):
                html = tpl.format(
                    name=name,
                    tagline=tagline,
                    f0=feats[0],
                    f1=feats[1],
                    f2=feats[2],
                    cta=cta,
                )
                fn = f"tpl_{t}_{cat}_{v}.html"
                (out / fn).write_text(html)
                jobs.append(
                    {
                        "pid": page_id("tpl", fn),
                        "mode": "file",
                        "url": f"file:///vol/html/{fn}",
                        "domain": fn,
                        "template": t,
                        "category": cat,
                    }
                )
    (CACHE / "jobs_tpl.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(len(jobs), "template pages")


if __name__ == "__main__":
    main()
