"""Write the 30 synthetic cold-email briefs (data/briefs.json).

Every prospect, company and person is invented. Freight facts (shipping season, equipment) come
from public commodity guides; no shipper names are used. Each brief carries:
- `facts`: the only claims the email may make about the sender, offer and relationship (the honesty ledger);
- `relationship`: cold | one_prior | customer, so "invented history" can be checked;
- `situation`: first_email | follow_up, so examples can be situation-matched.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# (commodity, shipping months [from, to], equipment) from the public commodity guides
FREIGHT = [
    ("cherries", (6, 8), "reefer"),
    ("potatoes", (9, 6), "reefer or vented van"),
    ("spinach", (1, 12), "reefer"),
    ("drywall", (3, 11), "flatbed or dry van"),
    ("aggregates", (3, 11), "end-dump or hopper"),
    ("hay", (6, 11), "flatbed or dry van"),
    ("bottled water", (4, 9), "dry van"),
    ("tires", (1, 12), "dry van"),
    ("mattresses", (1, 12), "dry van"),
    ("paper products", (1, 12), "dry van"),
]
FREIGHT_PROSPECTS = [
    ("Ridgeline Orchards", "Yakima, WA", "logistics manager", "Dana"),
    ("Red Butte Growers", "Idaho Falls, ID", "shipping lead", "Marcus"),
    ("Salinas Greenleaf Farms", "Salinas, CA", "transportation coordinator", "Priya"),
    ("Keystone Wallboard Supply", "Allentown, PA", "branch operations manager", "Tom"),
    ("Granite Hollow Quarry", "Bend, OR", "dispatch supervisor", "Elena"),
    ("Prairie Bale Co.", "Garden City, KS", "owner", "Walt"),
    ("Clearspring Beverage", "Chattanooga, TN", "distribution manager", "Kim"),
    ("Midstate Tire Distributors", "Columbus, OH", "warehouse manager", "Rafael"),
    ("Restwell Bedding", "High Point, NC", "supply chain lead", "Joanne"),
    ("Northfold Paper Goods", "Green Bay, WI", "traffic manager", "Steve"),
]
FREIGHT_SENDER = {
    "company": "Lanebridge Logistics",
    "role": "account executive",
    "offer": "asset-light truckload brokerage with a vetted carrier network",
}

INDUSTRIAL = [
    (
        "Helix Precision Machining",
        "Rockford, IL",
        "plant manager",
        "Greg",
        "contract CNC machining for low-volume aluminum parts",
        "their second shift was announced on a local jobs board",
        [
            "ISO 9001 certified shop",
            "5-axis capacity of 40 spindle-hours a week",
            "first-article inspection report with every new part",
        ],
    ),
    (
        "Brightline Fabrication",
        "Erie, PA",
        "procurement manager",
        "Lisa",
        "MRO supply program with vending cabinets on the shop floor",
        "they opened a new bay last quarter",
        [
            "vending cabinets installed at no upfront cost",
            "next-day replenishment within 50 miles",
            "consolidated monthly invoicing",
        ],
    ),
    (
        "Corrigan Pumps",
        "Tulsa, OK",
        "maintenance supervisor",
        "Hector",
        "predictive maintenance sensors for rotating equipment",
        "they posted for a reliability engineer",
        [
            "wireless vibration sensors that mount without downtime",
            "a 90-day pilot on up to 10 assets",
            "alerts go to email or SMS",
        ],
    ),
    (
        "Arden Plastics",
        "Grand Rapids, MI",
        "operations director",
        "Nadia",
        "used injection-molding presses, refurbished and warrantied",
        "they won a new automotive contract",
        [
            "presses from 150 to 500 tons in stock",
            "a 6-month parts-and-labor warranty",
            "installation crews in the Midwest",
        ],
    ),
    (
        "Westfield Gear Works",
        "Springfield, MO",
        "quality manager",
        "Paul",
        "third-party CMM inspection services",
        "they are preparing for a customer audit",
        ["same-week inspection slots", "reports formatted to AS9102", "pickup and drop-off within 100 miles"],
    ),
    (
        "Summit Coatings",
        "Lancaster, PA",
        "EHS manager",
        "Rosa",
        "industrial ventilation audits",
        "a new state air-permit rule takes effect next year",
        [
            "audits done in one day on site",
            "a written report within two weeks",
            "engineers with 10+ years in finishing shops",
        ],
    ),
    (
        "Tri-County Castings",
        "Muncie, IN",
        "purchasing lead",
        "Derek",
        "foundry consumables: ceramic filters and sleeves",
        "they switched to a new alloy line",
        [
            "filters stocked in three regional warehouses",
            "free samples for a first trial pour",
            "volume pricing at 20 cases",
        ],
    ),
    (
        "Larkspur Packaging",
        "Fresno, CA",
        "plant engineer",
        "Mei",
        "retrofit kits for case packers",
        "their line runs 24/7 in harvest season",
        [
            "kits fit the three most common case packer models",
            "install during a single planned shutdown",
            "remote support during the first month",
        ],
    ),
    (
        "Ironbridge Steel Service",
        "Birmingham, AL",
        "operations manager",
        "Carl",
        "forklift fleet rentals with maintenance included",
        "they announced a warehouse expansion",
        [
            "short-term rentals from one month",
            "maintenance and repairs included in the rate",
            "replacement unit within 24 hours if one goes down",
        ],
    ),
    (
        "Pioneer Valve",
        "Houston, TX",
        "production manager",
        "Angela",
        "robotic welding cells on a lease-to-own plan",
        "they listed open welder roles for months",
        [
            "cells sized for parts up to 2 meters",
            "lease-to-own over 36 months",
            "operator training for two staff included",
        ],
    ),
]
INDUSTRIAL_SENDER = {"company": "Northgate Industrial Partners", "role": "territory sales manager"}

CONSTRUCTION = [
    (
        "Bayview Builders",
        "Tacoma, WA",
        "project manager",
        "Kevin",
        "temporary fencing and site security rental",
        "they broke ground on a mixed-use project downtown",
        [
            "fencing delivered within 48 hours",
            "monthly rental with no minimum term",
            "camera towers available as an add-on",
        ],
    ),
    (
        "Rio Grande Concrete",
        "Albuquerque, NM",
        "estimator",
        "Luis",
        "ready-mix delivery from a new batch plant",
        "they bid on the county road project",
        ["new batch plant 12 miles from the county yard", "Saturday pours available", "fleet of 18 mixer trucks"],
    ),
    (
        "Oakline Roofing",
        "Charlotte, NC",
        "owner",
        "Brenda",
        "roofing underlayment at contractor pricing",
        "storm season is coming",
        ["two warehouses in the Carolinas", "same-day will-call pickup", "net-30 terms after one order"],
    ),
    (
        "Granger Electric",
        "Omaha, NE",
        "operations manager",
        "Mike",
        "equipment rental: scissor lifts and boom lifts",
        "they posted they are hiring journeyman electricians",
        [
            "lifts delivered and picked up free within 30 miles",
            "weekly and monthly rates",
            "units serviced before every rental",
        ],
    ),
    (
        "Cedar Point Excavation",
        "Boise, ID",
        "superintendent",
        "Jake",
        "GPS machine-control systems for excavators",
        "they bought two new excavators this year",
        ["systems fit most major excavator brands", "two days of operator training", "a 30-day trial on one machine"],
    ),
    (
        "Harbor Plumbing & Mechanical",
        "Baltimore, MD",
        "purchasing manager",
        "Denise",
        "PEX and fittings supply with jobsite delivery",
        "they won a 120-unit apartment contract",
        [
            "jobsite delivery before 7 a.m.",
            "stock held for a project for up to 60 days",
            "a dedicated inside sales rep",
        ],
    ),
    (
        "Summit Drywall Contractors",
        "Denver, CO",
        "field operations lead",
        "Ray",
        "drywall labor staffing for peak months",
        "they are behind on a hospital job",
        ["crews of 4 to 12 hangers and finishers", "workers carry OSHA-10 cards", "a one-week minimum"],
    ),
    (
        "Lone Star Paving",
        "Austin, TX",
        "general manager",
        "Carla",
        "asphalt equipment maintenance contracts",
        "their paving season is about to peak",
        [
            "mobile mechanics who service on site",
            "a 4-hour response window in the metro",
            "flat monthly pricing per machine",
        ],
    ),
    (
        "Greenway Landscaping Co.",
        "Raleigh, NC",
        "owner",
        "Tony",
        "fleet telematics for trucks and trailers",
        "they added a second crew this spring",
        [
            "GPS trackers that install in 15 minutes",
            "no contract longer than 12 months",
            "alerts for after-hours vehicle use",
        ],
    ),
    (
        "Northshore Framing",
        "Duluth, MN",
        "project coordinator",
        "Erin",
        "engineered lumber packages with takeoff service",
        "they are framing a 40-home subdivision",
        [
            "free takeoffs from plans within 3 business days",
            "deliveries sequenced by build phase",
            "a yard 20 minutes from the site",
        ],
    ),
]
CONSTRUCTION_SENDER = {"company": "Keystone Build Supply", "role": "outside sales rep"}

# relationship / situation pattern across each vertical's 10 briefs: 6 cold first emails,
# 2 follow-ups after one unanswered email, 2 notes to an existing customer
PATTERN = [("cold", "first_email")] * 6 + [("one_prior", "follow_up")] * 2 + [("customer", "first_email")] * 2

REL_FACTS = {
    "cold": "This is the first contact: there is no prior email, call or relationship.",
    "one_prior": "The sender emailed once two weeks ago and got no reply. There has been no call or meeting.",
    "customer": "The prospect's company already uses the sender's company for a different service; this email is about a new one.",
}


def season(m):
    a, b = m
    return "year-round" if (a, b) == (1, 12) else f"{MONTHS[a - 1]} to {MONTHS[b - 1]}"


def build():
    briefs = []
    for i, ((commodity, months, equip), (co, loc, role, first)) in enumerate(
        zip(FREIGHT, FREIGHT_PROSPECTS, strict=False)
    ):
        rel, sit = PATTERN[i]
        facts = [
            f"{FREIGHT_SENDER['company']} offers {FREIGHT_SENDER['offer']}.",
            f"{commodity.capitalize()} usually ship {season(months)}; the usual equipment is {equip}.",
            f"{FREIGHT_SENDER['company']} can quote {equip} capacity out of the {loc.split(', ')[1]} area.",
            REL_FACTS[rel],
        ]
        briefs.append(
            dict(
                id=f"fr{i + 1:02d}",
                vertical="freight",
                relationship=rel,
                situation=sit,
                sender=FREIGHT_SENDER | {"name": "the sender"},
                prospect=dict(company=co, location=loc, role=role, first_name=first),
                trigger=f"their {commodity} shipping season ({season(months)}) is approaching",
                offer=f"truckload capacity ({equip}) for their {commodity} shipments",
                cta="a 15-minute call next week",
                facts=facts,
            )
        )
    for vert, rows, sender, tag in (
        ("industrial", INDUSTRIAL, INDUSTRIAL_SENDER, "in"),
        ("construction", CONSTRUCTION, CONSTRUCTION_SENDER, "co"),
    ):
        for i, (co, loc, role, first, offer, trigger, offer_facts) in enumerate(rows):
            rel, sit = PATTERN[i]
            facts = (
                [f"{sender['company']} sells {offer}."]
                + [f"The offer includes: {f}." for f in offer_facts]
                + [REL_FACTS[rel]]
            )
            briefs.append(
                dict(
                    id=f"{tag}{i + 1:02d}",
                    vertical=vert,
                    relationship=rel,
                    situation=sit,
                    sender=sender | {"name": "the sender"},
                    prospect=dict(company=co, location=loc, role=role, first_name=first),
                    trigger=trigger,
                    offer=offer,
                    cta="a short call or a reply with the right contact",
                    facts=facts,
                )
            )
    return briefs


if __name__ == "__main__":
    out = HERE / "data" / "briefs.json"
    b = build()
    out.write_text(json.dumps(b, indent=1))
    print(len(b), "briefs ->", out)
