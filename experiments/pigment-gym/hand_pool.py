"""
A tiny hand-rated pool: 12 sketches for the subject "a sunset over green hills", written by hand
with a KNOWN quality order. The benchmark (scripts 1-3) uses them to sanity-check CLIP; the reward check (4_rank_check.py) uses them to check
that the reward function ranks paintings the way a person would (Spearman correlation).

`rank` 1 is best. The four sketches that should fail the gate share rank 10.5 (the average of
ranks 9-12), because "broken" has no internal order.

The HF reproduction's version of this is a 178-painting reference pool rated by hand. Their line:
"What the pool contains is the reward function." Twelve is enough to catch a broken reward, not
enough to tune a good one.
"""

from brush_tools import PART_DIR

SUBJECT = "a sunset over green hills"

SKELETON_TOP = """function setup() {
  createCanvas(400, 400, WEBGL);
  brush.scaleBrushes(2);
  noLoop();
}

function wash(hex, opacity, points, bleed) {
  brush.noStroke();
  brush.fill(hex, opacity);
  brush.fillBleed(bleed);
  brush.fillTexture(0.5, 0.4);
  brush.beginShape(0.4);
  for (const [x, y] of points) brush.vertex(x, y);
  brush.endShape(true);
}

function draw() {
  translate(-width / 2, -height / 2);
  background("#f4efe4");
  randomSeed(3);
"""
SKELETON_END = "\n}\n"


def sketch(body):
    return SKELETON_TOP + body + SKELETON_END


SUNSET_HILLS = sketch("""
  // 1. sky bands, warm to cool
  wash("#f6bd60", 170, [[0, 0], [400, 0], [400, 120], [0, 140]], 0.3);
  wash("#f28482", 140, [[0, 110], [400, 95], [400, 220], [0, 230]], 0.3);
  wash("#b388c4", 90, [[0, 0], [400, 0], [400, 60], [0, 50]], 0.3);
  // 2. a big low sun, half behind the hills
  brush.noStroke();
  brush.fill("#f4a259", 220);
  brush.fillBleed(0.3);
  brush.circle(250, 225, 55);
  // 3. three hills, far to near
  wash("#a7c4a0", 190, [[0, 230], [110, 190], [230, 225], [330, 195], [400, 210], [400, 300], [0, 300]], 0.15);
  wash("#6a994e", 210, [[0, 280], [140, 235], [280, 270], [400, 245], [400, 340], [0, 340]], 0.15);
  wash("#386641", 220, [[0, 330], [180, 290], [400, 320], [400, 400], [0, 400]], 0.12);
""")

SIMPLE_SUNSET = sketch("""
  // 1. one warm sky
  wash("#f6c177", 150, [[0, 0], [400, 0], [400, 260], [0, 260]], 0.2);
  // 2. sun
  brush.noStroke();
  brush.fill("#ee6c4d", 210);
  brush.circle(200, 180, 45);
  // 3. one green hill
  wash("#6a994e", 200, [[0, 300], [200, 230], [400, 300], [400, 400], [0, 400]], 0.15);
""")

HILLS_NO_SUN = sketch("""
  // 1. a pale blue midday sky, no sun
  wash("#a9d6e5", 150, [[0, 0], [400, 0], [400, 250], [0, 250]], 0.2);
  // 2. hills
  wash("#90a955", 200, [[0, 250], [130, 200], [260, 240], [400, 205], [400, 320], [0, 320]], 0.15);
  wash("#4f772d", 210, [[0, 310], [200, 270], [400, 310], [400, 400], [0, 400]], 0.12);
""")

SUN_ONLY = sketch("""
  // just a sun on bare paper
  brush.noStroke();
  brush.fill("#f4a259", 220);
  brush.fillBleed(0.3);
  brush.fillTexture(0.5, 0.4);
  brush.circle(200, 200, 70);
""")

CONFETTI = sketch("""
  // forty random dots in random colours: paint, but no picture
  const colours = ["#e63946", "#457b9d", "#ffb703", "#8338ec", "#06d6a0", "#222222"];
  brush.noStroke();
  brush.fillTexture(0.5, 0.4);
  for (let i = 0; i < 40; i++) {
    brush.fill(colours[i % colours.length], 200);
    brush.circle(random(20, 380), random(20, 380), random(6, 20));
  }
""")

MUDDY = sketch("""
  // everything the same brown, laid on thick
  wash("#6b4f3a", 230, [[0, 0], [400, 0], [400, 400], [0, 400]], 0.1);
  brush.noStroke();
  brush.fill("#4a3728", 230);
  brush.circle(130, 160, 60);
  brush.circle(270, 240, 70);
""")

TINY_DOT = sketch("""
  // one small mark, just above the gate's 0.5% coverage floor
  brush.noStroke();
  brush.fill("#e07a5f", 230);
  brush.fillTexture(0.5, 0.4);
  brush.circle(200, 200, 28);
""")

SYNTAX_ERROR = sketch("""
  wash("#f6bd60", 170, [[0, 0], [400, 0], [400, 200], [0, 200]], 0.3;
  brush.circle(250, 225, 55);
""")

# CLIP reads text in images very well, so writing the word is the classic reward hack.
TEXT_HACK = sketch("""
  wash("#f6bd60", 170, [[0, 0], [400, 0], [400, 200], [0, 200]], 0.3);
  textSize(48);
  text("SUNSET OVER HILLS", 20, 200);
""")

BARE_PRIMITIVES = """function setup() { createCanvas(400, 400); noLoop(); }
function draw() {
  background("#f4efe4");
  noStroke();
  fill("#f6bd60"); rect(0, 0, 400, 250);
  fill("#f4a259"); ellipse(250, 200, 110, 110);
  fill("#6a994e"); ellipse(200, 420, 700, 300);
}
"""

POOL = [
    {"name": "meadow at sunset", "rank": 1, "code": (PART_DIR / "sketches" / "meadow.js").read_text()},
    {"name": "sunset over 3 hills", "rank": 2, "code": SUNSET_HILLS},
    {"name": "simple sunset, 1 hill", "rank": 3, "code": SIMPLE_SUNSET},
    {"name": "hills, no sun", "rank": 4, "code": HILLS_NO_SUN},
    {"name": "sun only", "rank": 5, "code": SUN_ONLY},
    {"name": "random confetti", "rank": 6, "code": CONFETTI},
    {"name": "muddy brown", "rank": 7, "code": MUDDY},
    {"name": "one tiny dot", "rank": 8, "code": TINY_DOT},
    {"name": "syntax error", "rank": 10.5, "code": SYNTAX_ERROR},
    {"name": "writes 'SUNSET' (hack)", "rank": 10.5, "code": TEXT_HACK},
    {"name": "bare p5 shapes", "rank": 10.5, "code": BARE_PRIMITIVES},
    {"name": "blank paper", "rank": 10.5, "code": sketch('  brush.noStroke();\n  brush.fill("#f4efe4", 10);\n  brush.circle(200, 200, 30);')},
]
